import argparse
import sys
import datetime
import time
import math
import json
from pathlib import Path
import wandb

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models as torchvision_models

from utils import train_tools, misc, optims, metrics
from voc12 import datasets, augmentations
from arguments import common_arguments
import vision_transformer as vits

cudnn.benchmark = True


def parse_train_arguments():
    parser = argparse.ArgumentParser('ViT_WSL', parents=[common_arguments()])

    # --------------------------------------  Dataset/Augmentations  ---------------------------------------------------
    parser.add_argument("--base_size", default=512, type=int)
    parser.add_argument("--scale_range", default=(0.8, 1.2), type=tuple)

    # ----------------------------------  Training/Optimization parameters  --------------------------------------------
    parser.add_argument('--use_fp16', type=misc.bool_flag, default=False, help="""Whether to use half-precision for 
    training. Improves training time/memory requirements, but can provoke instability and slight decay of performance.
    We recommend disabling mixed precision if the loss is unstable, if reducing the patch size or if training with 
    bigger ViTs.""")
    parser.add_argument('--train_batch_size', default=2, type=int, help="""Batch-size: Number of distinct images loaded 
    on one batch during training.""")
    parser.add_argument('--val_batch_size', default=2, type=int, help="""Batch-size: Number of distinct images loaded on 
    one batch during validation.""")

    # -------------------------------------  Optimizer/Schedulers parameters  ------------------------------------------
    parser.add_argument("--lr", default=1e-5, type=float, help="""Learning rate at the end of linear warmup (highest LR 
    used during training)""")
    parser.add_argument('--min_lr', type=float, default=1e-7, help="""Target LR at the end of optimization.We use a 
    cosine LR schedule with linear warmup.""")
    parser.add_argument('--weight_decay', type=float, default=0.1, help="""Initial value of the weight decay. 
    With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=1.0, help="""Final value of the weight decay.
     We use a cosine schedule for WD and using a larger decay by the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3., help="""Maximal parameter gradient norm if using gradient
     clipping. Clipping with norm .3 ~ 1.0 can help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument("--warmup_epochs", default=0, type=int, help="""Number of epochs for the linear learning-rate
    warm up.""")
    parser.add_argument('--optimizer', default='adamw', type=str, choices=['adamw', 'sgd'], help="""Type of optimizer. 
    We recommend using ADAMW with ViTs.""")

    args = parser.parse_args()

    # Checkpoint Directory
    head_type = 'mlp_head' if args.mlp_head else 'linear_head'
    args.exp_name = f'{args.pretrain_method}{args.img_size}/{args.arch}/patch{args.patch_size}/input{args.crop_size}/' \
                    f'{head_type}/epoch{args.max_epochs}/{args.exp_name}'
    args.ckpt_dir = Path(args.ckpt_dir, args.project_name, args.exp_name)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    return args


def train():
    # Parse arguments
    args = parse_train_arguments()

    # Fix RNG seeds for reproducibility
    misc.fix_random_seeds(args.seed)

    # Set parameters
    cudnn.benchmark = True

    # Write the arguments into the JSON file
    print("git:\n  {}\n".format(misc.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    with open(args.ckpt_dir.joinpath('args.json'), "w") as args_file:
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        json.dump(config, args_file, indent=4)

    # ======================== DATASET ============================
    train_dataset = datasets.VOC12ImageDataset(
        args.train_list, voc12_root=args.voc12_root,
        transform=transforms.Compose([
            transforms.RandomAffine(degrees=0, scale=args.scale_range),
            transforms.RandomCrop(size=args.crop_size, pad_if_needed=True),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    val_dataset = datasets.VOC12ImageDataset(
        args.val_list, voc12_root=args.voc12_root,
        transform=transforms.Compose([
            transforms.CenterCrop(args.base_size),
            transforms.ToTensor(),
            augmentations.ValCrop(args.patch_size),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    print(f"Data loaded: there are {len(train_dataset)} training images and {len(val_dataset)} validation images.")

    # ================================  NETWORKS  ===============================
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](args=args)
    # if the network is a XCiT
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        model = torch.hub.load(
            'facebookresearch/xcit:main', args.arch, pretrained=True, drop_path_rate=args.drop_path_rate
        )
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
    else:
        print(f"Unknown architecture: {args.arch}")
        model = None

    model = model.cuda()
    print(f"The model are built on {args.arch} network.")

    # ================================  LOSS  ===============================
    criterion = nn.BCEWithLogitsLoss().cuda()

    # ================================  OPTIMIZER  ===============================
    param_groups = model.get_params_groups()
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(param_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(param_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = optims.LARS(param_groups)  # to use with convnet and large batches
    else:
        optimizer = None

    # for mixed precision training
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
    else:
        fp16_scaler = None

    # ================================  SCHEDULERS  ===============================
    lr_schedule = optims.cosine_scheduler(
        args.lr,        # linear scaling rule
        args.min_lr,
        args.max_epochs, len(train_dataloader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = optims.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.max_epochs, len(train_dataloader),
    )
    print(f"Loss, optimizer and schedulers are ready....")

    # ================================  RESUME TRAINING  ===============================
    to_restore = {"epoch": args.start_epoch}
    train_tools.restart_from_checkpoint(
        args.ckpt_dir.joinpath(f"checkpoint_{args.start_epoch:03}.pth"),
        run_variables=to_restore,
        model=model,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        classification_loss=criterion
    )
    start_epoch = to_restore["epoch"]

    # ================================  PARALLEL TRAINING  ===============================
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    # ================================  START TRAINING  ===============================
    start_time = time.perf_counter()
    wandb.watch(model, criterion=criterion, log_freq=1000, log='all', log_graph=True)
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.max_epochs):

        # ============ Training ... ============
        train_stats = train_one_epoch(
            epoch=epoch, model=model, criterion=criterion, data_loader=train_dataloader,  optimizer=optimizer,
            lr_schedule=lr_schedule, wd_schedule=wd_schedule, args=args
        )

        # ============ Validation ... ============
        val_stats, metric_stats = validate_one_epoch(
            epoch=epoch, model=model, criterion=criterion, dataloader=val_dataloader, args=args
        )

        # ============ Logging ... ============
        save_dict = {
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'classification_loss': criterion.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()

        train_tools.save_on_master(save_dict, args.ckpt_dir.joinpath('last_checkpoint.pth'))
        if args.save_ckpt_freq and epoch % args.save_ckpt_freq == 0:
            train_tools.save_on_master(save_dict, args.ckpt_dir.joinpath(f'checkpoint_{epoch + 1:03}.pth'))

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}': v for k, v in val_stats.items()},
            **{k: v for k, v in metric_stats.items()},
            'epoch': epoch
        }
        if train_tools.is_main_process():
            wandb.log(log_stats)
            with args.ckpt_dir.joinpath("log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(epoch, model, criterion, data_loader, optimizer, lr_schedule, wd_schedule, args):
    model.train()
    metric_logger = metrics.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.max_epochs)
    for iters, pack in enumerate(metric_logger.log_every(data_loader, 10, header)):

        # global training iteration
        global_iteration = len(data_loader) * epoch + iters

        # update weight decay and learning rate according to their schedule
        for idx, param_group in enumerate(optimizer.param_groups):
            if idx == 0:    # regularized
                param_group["lr"] = lr_schedule[global_iteration]
                param_group["weight_decay"] = wd_schedule[global_iteration]
            elif idx == 1:  # not_regularized
                param_group["lr"] = lr_schedule[global_iteration]
            elif idx == 2:  # scratch_weights
                param_group["lr"] = lr_schedule[global_iteration] * 10
                param_group["weight_decay"] = wd_schedule[global_iteration]
            elif idx == 3:  # scratch_biases
                param_group["lr"] = lr_schedule[global_iteration] * 10
            else:
                print('There is no such params group')

        # Parse inputs
        image_name = pack['name']
        image = pack['image'].cuda(non_blocking=True)
        class_label = pack['class_label'].cuda(non_blocking=True)

        # Forward pass + compute loss
        class_output = model(image)
        loss = criterion(class_output, class_label)
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        # Model Update
        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad:
            optims.clip_gradients(model, clip=args.clip_grad)
        optimizer.step()

        # Logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        wandb.log({'batch/loss': loss.item()})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    train_stats = {key: meter.global_avg for key, meter in metric_logger.meters.items()}
    return train_stats


def validate_one_epoch(epoch, model, criterion, dataloader, args):
    model.eval()
    header = 'Epoch: [{}/{}]'.format(epoch, args.max_epochs)
    metric_logger = metrics.MetricLogger(delimiter="  ")
    mlcm = metrics.MultiLabelConfusionMatrix(
        num_classes=args.num_classes, device=torch.device('cuda'), normalized=False
    )

    for iters, pack in enumerate(metric_logger.log_every(dataloader, 10, header)):
        with torch.no_grad():
            images = pack['image'].cuda(non_blocking=True)
            label = pack['class_label'].cuda(non_blocking=True)

            batch_size, _, height, width = images.shape

            # Forward pass + compute loss
            class_output = model(images)
            loss = criterion(class_output, label)

            # logging
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            mlcm.update(output=(class_output, label))

    # cm = mlcm.compute()
    # print(cm)

    metric_stats = mlcm.compute_precision_recall()
    val_stats = {key: meter.global_avg for key, meter in metric_logger.meters.items()}
    print("Averaged stats:", metric_logger)

    model.train()
    return val_stats, metric_stats
