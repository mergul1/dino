import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data.dataset import Subset
from torch.utils.data.dataloader import DataLoader
from torch.backends import cudnn
import torch.nn.functional as F

import torchvision
from torchvision import models as torchvision_models
from torchvision import transforms

# import vision_transformer as vits
import cam_generator as vits
from utils import misc, train_tools
from voc12 import datasets, augmentations
from arguments import common_arguments


def parse_inference_arguments():
    parser = argparse.ArgumentParser('ViT_WSL', parents=[common_arguments()])

    # --------------------------------------  Dataset  ---------------------------------------------------
    parser.add_argument("--infer_list", default="voc12/train.txt", type=str)
    parser.add_argument("--chainer_eval_set", default='val', choices=['train', 'val'], type=str)
    parser.add_argument('--eval_batch_size', default=1, type=int, help="""Batch-size: Number of distinct images loaded 
    on one batch during evaluation.""")

    # ----------------------------------  CAM or Pseudo-mask computation  ----------------------------------------------
    parser.add_argument('--eval_epoch', default=1, type=int, help="The checkpoints for evaluation")

    # Output directory
    parser.add_argument("--out_cam", default='out_cam', type=str, help="...")
    parser.add_argument("--out_crf", default='out_crf', type=str, help="...")
    parser.add_argument("--out_pred", default='out_pred', type=str, help="...")

    # Post-process parameters
    parser.add_argument('--multi_scales', default=(1.0,), type=tuple, help="Multi-scale factors for evaluation")
    parser.add_argument("--alpha", default=4, type=int, help="..")
    parser.add_argument("--low_alpha", default=4, type=int, help="..")
    parser.add_argument("--high_alpha", default=16, type=int, help="..")
    parser.add_argument('--mask_guided_crf', default=True, type=misc.bool_flag, help="...")

    # Aggregation parameters
    parser.add_argument("--method", default='attention_rollout', type=str,
                        choices=['full', 'second_layer', 'attn_gradcam',
                                 'relevance_rollout', 'attention_rollout', 'gradient_rollout',
                                 'attribution_rollout', 'getam_rollout'],
                        help='')
    parser.add_argument("--aggregation_type", default='sum', type=str,
                        choices=['rollout1', 'rollout2', 'sum', 'mean', 'hadamard'],
                        help="""How to aggregate relevances of the transformer layers""")
    parser.add_argument('--start_layer', default=11, type=int, help="""From which transformer layer will be used 
        in computation CAMs""")
    parser.add_argument('--double_gradient', default=False, type=misc.bool_flag, help="...")

    # Random Walk parameters
    parser.add_argument('--is_refined', default=False, type=misc.bool_flag, help="...")
    parser.add_argument("--logt", default=0, type=int, help='Number of iterations in random walk (in log space)')
    parser.add_argument("--beta", default=2, type=int, help='Exponent number for transition probability in random walk')

    args = parser.parse_args()

    # Project Directory
    head_type = 'mlp_head' if args.mlp_head else 'linear_head'
    args.exp_name = f'{args.pretrain_method}{args.img_size}/{args.arch}/patch{args.patch_size}/input{args.crop_size}/' \
                    f'{head_type}/epoch{args.max_epochs}/{args.exp_name}'
    args.ckpt_dir = Path(args.ckpt_dir, args.project_name, args.exp_name)
    args.output_dir = Path(args.output_dir, args.project_name, args.exp_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Result Directory
    args.result_dir = args.output_dir.joinpath(
        'results', args.method, args.aggregation_type, f'{args.start_layer}', f'ms:{args.multi_scales}'
    )
    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.out_cam = args.result_dir.joinpath(args.out_cam)
    args.out_cam.mkdir(parents=True, exist_ok=True)
    args.out_pred = args.result_dir.joinpath(args.out_pred)
    args.out_pred.mkdir(parents=True, exist_ok=True)
    args.out_crf = args.result_dir.joinpath(args.out_crf)
    args.out_crf.mkdir(parents=True, exist_ok=True)

    return args


def get_attention(process_id, model, dataset, args):
    # Parameters
    num_gpus = torch.cuda.device_count()

    # Dataloader
    data_loader = DataLoader(dataset[process_id], shuffle=False, num_workers=0, pin_memory=True)

    # Run model and inference for each GPU
    with torch.cuda.device(process_id):
        # Transfer the model into the current device
        model.cuda()

        # Run inference
        for idx, data in enumerate(data_loader):
            # Extract image, label and affinity matrices (if exists)
            img_name = data['name'][0]
            label = data['class_label']
            valid_classes = torch.nonzero(label, as_tuple=True)[0]

            # Extract Attentions for each scale and flip
            resized_attentions = []
            if args.mask_guided:
                pass
            else:
                for i, img in enumerate(data['image_list']):
                    _, _, height, width = img.shape
                    num_patches = (height // args.patch_size, width // args.patch_size)

                    # Compute Class Attention Map
                    cam = model.get_cam(args, img.cuda(non_blocking=True), label=label.cuda(non_blocking=True))

                    # # Compute Attention
                    # attentions = model.get_attentions(img.cuda(non_blocking=True))
                    # num_heads = attentions[-1].shape[1]
                    # num_depths = len(attentions)
                    # class_attns = attentions[-6][:, :, 0, 1:].reshape(1, -1, num_patches[0], num_patches[1])

                    # fig, axs = plt.subplots(nrows=num_depths, ncols=num_heads)
                    for r, attn in enumerate(cam):
                        # class_attns = attn[0, :, 0, 1:].reshape(-1, num_patches[0], num_patches[1])
                        class_attns = F.interpolate(
                            class_attns.unsqueeze(dim=1), size=(height, width), mode='bilinear'
                        ).squeeze(dim=1)
                        fname = os.path.join(f"attn_{r}.png")
                        plt.imsave(fname=fname, arr=class_attns.mean(dim=0).cpu(), format='png')

                        # for c, attn_head in enumerate(class_attns):
                        #     attn_head = (attn_head - attn_head.min()) / (attn_head.max()-attn_head.min())
                        #     axs[r, c].imshow(attn_head.cpu(), cmap='plasma')
                        #     axs[r, c].set_title(f'{r}-{c}')
                        #     # plt.imshow(attn_head.cpu())
                        #     plt.show()
                    # for ax in axs.flat:
                    #     ax.label_outer()

                    # plt.show()
                    print('deneme')

                    # resized_attns = []
                    # for class_attention in class_attns:
                    #     scaled_attn = F.interpolate(
                    #         class_attention.unsqueeze(dim=1), size=(height, width), mode='bilinear'
                    #     ).squeeze().relu()
                    #     resized_attns.append(scaled_attn)
                    #     torchvision.utils.save_image(
                    #         torchvision.utils.make_grid(img, normalize=True, scale_each=True), 'img.png'
                    #     )
                    #     for j in range(num_heads):
                    #         fname = os.path.join("attn-head" + str(j) + ".png")
                    #         plt.imsave(fname=fname, arr=scaled_attn[j].cpu(), format='png')
                    #         print(f"{fname} saved.")
                    #
                    #     print('deneme')
                    #


def make_pseudo_gt():

    # ================================ PARAMETERS =============================
    args = parse_inference_arguments()
    num_gpus = torch.cuda.device_count()

    # ================================ DATASETS =============================
    if args.mask_guided:
        infer_dataset = ...
    else:
        infer_dataset = datasets.VOC12ImageDatasetMSF(
            args.infer_list, voc12_root=args.voc12_root, scales=args.multi_scales, is_flip=True,
            transform=transforms.Compose([
                # transforms.CenterCrop(args.base_size),
                transforms.ToTensor(),
                augmentations.ValCrop(args.patch_size),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        )

    infer_datasets = [Subset(infer_dataset, np.arange(i, len(infer_dataset), num_gpus)) for i in range(num_gpus)]
    for i, dataset in enumerate(infer_datasets):
        print(f"Data loaded for {i}th loader: there are {len(dataset)} inference images.")

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

    # ================================ LOAD MODEL CHECKPOINTS ... ================================
    train_tools.restart_from_checkpoint(
        args.ckpt_dir.joinpath(f"checkpoint_{args.eval_epoch:03}.pth"),
        model=model
    )
    model = model.cuda()
    model.eval()

    # ================================ MAKE CAM ===================================
    get_attention(process_id=0, model=model, dataset=infer_datasets, args=args)

    torch.cuda.empty_cache()
