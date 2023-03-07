import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import torch
from torch.utils.data.dataset import Subset
from torch.utils.data.dataloader import DataLoader
from torch.backends import cudnn
import torch.nn.functional as F
import torch.multiprocessing as mp

import torchvision
from torchvision import models as torchvision_models
from torchvision import transforms

from chainercv.datasets import voc_semantic_segmentation_label_names, voc_semantic_segmentation_label_colors
from chainercv.visualizations import vis_semantic_segmentation

# import vision_transformer as vits
import cam_generator as vits
from utils import misc, train_tools
from voc12 import datasets, augmentations
from arguments import common_arguments
from utils.visualizations import show_cam_on_image, show_cams_on_image
from utils.cam_utils import crf_with_alpha, cam2label, eval_cam

import matplotlib
matplotlib.use(backend='Qt5Agg')

is_show = False

# alpha -> prob
# 0.1   -> 0.835
# 0.2   -> 0.75
# 0.3   -> 0.7
# 0.4   -> 0.65
# 0.5   -> 0.62
# 0.8   -> 0.54
#   1   -> 0.5
#   2   -> 0.38
#   4   -> 0.275
#   8   -> 0.188
#  16   -> 0.122
#  32   -> 0.078


def parse_inference_arguments():
    parser = argparse.ArgumentParser('ViT_WSL', parents=[common_arguments()])

    # --------------------------------------  Dataset  ---------------------------------------------------
    parser.add_argument("--infer_list", default="voc12/train.txt", type=str)
    parser.add_argument('--eval_batch_size', default=1, type=int, help="""Batch-size: Number of distinct images loaded 
    on one batch during evaluation.""")
    parser.add_argument("--chainer_eval_set", default='train', choices=['train', 'val'], type=str)

    # ----------------------------------  CAM or Pseudo-mask computation  ----------------------------------------------
    parser.add_argument('--eval_epoch', default=10, type=int, help="The checkpoints for evaluation")
    parser.add_argument("--base_size", default=224, type=int)

    # Output directory
    parser.add_argument("--out_cam", default='out_cam', type=str, help="...")
    parser.add_argument("--out_crf", default='out_crf', type=str, help="...")
    parser.add_argument("--out_pred", default='out_pred', type=str, help="...")

    # Post-process parameters
    parser.add_argument('--multi_scales', default=(1.0,), type=tuple,
                        help="Multi-scale factors for evaluation")
    parser.add_argument('--is_flip', default=True, type=misc.bool_flag,
                        help="Flag for whether flipped image is evaluated for pseudo-mask generation")
    parser.add_argument("--alpha", default=2, type=int,
                        help="alpha (power) value to estimate background map from foreground maps, normal case")
    parser.add_argument("--low_alpha", default=1, type=int,
                        help="alpha (power) value to estimate background map from foreground maps, strong background")
    parser.add_argument("--high_alpha", default=4, type=int,
                        help="alpha (power) value to estimate background map from foreground maps, weak background")
    parser.add_argument("--bg_thr", default=None, type=float,
                        help="Background threshold for prediction, [0.0, 1.0]")
    parser.add_argument('--mask_guided_crf', default=False, type=misc.bool_flag,
                        help="Flag for whether the depth image is included in CRF or not")

    # Aggregation parameters
    parser.add_argument("--method", default='gradient_rollout', type=str,
                        choices=['full', 'second_layer', 'attn_gradcam',
                                 'getam_rollout', 'attention_rollout', 'gradient_rollout',
                                 'attribution_rollout', 'relevance_rollout'],
                        help='')
    parser.add_argument("--agg_type", default='sum', type=str,
                        choices=['rollout1', 'rollout2', 'sum', 'mean', 'hadamard'],
                        help="""How to aggregate relevances of the transformer layers""")
    parser.add_argument('--start_layer', default=10, type=int,
                        help="""From which transformer layer will be used in computation CAMs""")
    parser.add_argument('--start_layer_attns', default=0, type=int,
                        help="""From which transformer layer will be used in computation of affinity via Attentions""")
    parser.add_argument('--double_gradient', default=False, type=misc.bool_flag,
                        help="...")

    # Random Walk parameters
    parser.add_argument('--is_refined', default=True, type=misc.bool_flag,
                        help="...")
    parser.add_argument("--logt", default=6, type=int,
                        help='Number of iterations in random walk (in log space)')
    parser.add_argument("--beta", default=2, type=int,
                        help='Exponent number for transition probability in random walk')
    parser.add_argument('--semantic_weight', default=1.0, type=float,
                        help="Weight of semantic affinity w.r.t low-level affinity")

    # Debugging
    parser.add_argument('--is_show', default=False, type=misc.bool_flag, help='...')

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
        'results', args.method, f'agg_{args.agg_type}', f'start_layer_{args.start_layer}', f'ms:{args.multi_scales}',
        f'semantic_weight:{args.semantic_weight}'
    )
    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.out_cam = args.result_dir.joinpath(args.out_cam)
    args.out_cam.mkdir(parents=True, exist_ok=True)
    args.out_pred = args.result_dir.joinpath(args.out_pred)
    args.out_pred.mkdir(parents=True, exist_ok=True)
    args.out_crf = args.result_dir.joinpath(args.out_crf)
    args.out_crf.mkdir(parents=True, exist_ok=True)

    return args


def get_pseudo_mask(process_id, model, dataset, args):
    # Parameters
    num_gpus = torch.cuda.device_count()
    print(f"plot interactive: {plt.isinteractive()}")

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
            valid_classes = torch.nonzero(label[0], as_tuple=True)[0]

            img_path = datasets.get_img_path(img_name, args.voc12_root)
            orig_img = np.array(Image.open(img_path).convert('RGB'))
            orig_img_size = orig_img.shape[:2]

            print(f"{idx}/{len(data_loader)} \t {img_name} \t {orig_img.shape}")

            if args.mask_guided_crf:
                depth_img_path = datasets.get_depth_path(img_name, args.voc12_root)
                depth_img = np.asarray(cv2.imread(depth_img_path[1]))
                guidance_img = np.concatenate((orig_img, depth_img[:, :, 0:1]), axis=-1)
                #     guidance_img = np.concatenate((orig_img, depth_img), axis=-1)
            else:
                guidance_img = orig_img

            # Extract Attentions for each scale and flip
            resized_cams = []
            if args.mask_guided:
                for i, (img, affinity) in enumerate(zip(data['image_list'], data['affinity'])):
                    _, _, height, width = img.shape
                    num_patches = (height // args.patch_size, width // args.patch_size)

                    # Compute Class Attention Map (CAM)
                    cam = model.get_cam(
                        args, img.cuda(non_blocking=True), label=label.cuda(non_blocking=True), orig_img=orig_img,
                        low_affinity=affinity.cuda(non_blocking=True)
                    )

                    # CLip CAM
                    scale = args.multi_scales[i // 2]
                    orig_rescaled_size = [round(s * scale) for s in orig_img_size]
                    cam = cam[:, :, :orig_rescaled_size[0], :orig_rescaled_size[1]]

                    # Flip CAM, if necessary
                    cam = cam.flip(-1) if i % 2 else cam   # sum or max?

                    # Scale CAM
                    cam = F.interpolate(cam, orig_img_size, mode='bilinear', align_corners=False)
                    resized_cams.append(cam.cpu())

            else:
                for i, img in enumerate(data['image_list']):
                    _, _, height, width = img.shape
                    num_patches = (height // args.patch_size, width // args.patch_size)

                    # Compute Class Attention Map (CAM)
                    cam = model.get_cam(
                        args, img.cuda(non_blocking=True), label=label.cuda(non_blocking=True), orig_img=orig_img
                    )

                    # CLip CAM
                    scale = args.multi_scales[i // 2]
                    orig_rescaled_size = [round(s * scale) for s in orig_img_size]
                    cam = cam[:, :, :orig_rescaled_size[0], :orig_rescaled_size[1]]

                    # Flip CAM, if necessary
                    cam = cam.flip(-1) if i % 2 else cam   # sum or max?

                    # Scale CAM
                    cam = F.interpolate(cam, orig_img_size, mode='bilinear', align_corners=False)
                    resized_cams.append(cam.cpu())

            # Show CAMs
            if is_show:
                show_cams_on_image(orig_img, resized_cams, valid_classes)

            # Integrate multi-scale CAMs
            cam = torch.sum(torch.cat(resized_cams, dim=0), dim=0, keepdim=True)

            # Normalize CAMs
            cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))    # subtract min value
            cam /= (F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5)
            norm_cam = cam[0].cpu().numpy()

            # Show CAMs
            if is_show:
                show_cam_on_image(orig_img, norm_cam.take(valid_classes, axis=0))

            # Save raw CAMs
            if args.out_cam is not None:
                show_cam_on_image(
                    orig_img, norm_cam.take(valid_classes, axis=0), save_path=args.out_cam.joinpath(img_name + '.png')
                )

                np.savez_compressed(
                    args.out_cam.joinpath(img_name),
                    valid_classes=valid_classes, cam=np.take(norm_cam, valid_classes, axis=0)
                )

            # Raw CAM and prediction mask
            raw_cam, raw_pred = cam2label(norm_cam, valid_classes, alpha=args.alpha, bg_thr=args.bg_thr)
            if is_show:
                ax, legend_handles = vis_semantic_segmentation(
                    orig_img.transpose((2, 0, 1)), raw_pred,
                    label_names=voc_semantic_segmentation_label_names,
                    label_colors=voc_semantic_segmentation_label_colors,
                    alpha=0.7, all_label_names_in_legend=True
                )
                ax.legend(handles=legend_handles, bbox_to_anchor=(1, 1), loc=2)
                plt.show(block=True)

            if args.out_pred is not None:
                plt.imsave(args.out_pred.joinpath(img_name + '.png'), raw_pred.astype(np.uint8))

            # Predict the segmentation mask with CRF post-processing
            if args.out_crf is not None:
                # # constant background
                # crf_constant = crf_with_alpha(
                #     guidance_img, norm_cam=norm_cam, valid_classes=valid_classes, bg_thr=args.bg_thr,
                #     mask_guided=args.mask_guided_crf
                # )
                # crf_constant_pred = np.argmax(crf_constant, axis=0)
                # if is_show:
                #     ax, legend_handles = vis_semantic_segmentation(
                #         orig_img.transpose((2, 0, 1)), crf_constant_pred,
                #         label_names=voc_semantic_segmentation_label_names,
                #         label_colors=voc_semantic_segmentation_label_colors,
                #         alpha=0.7, all_label_names_in_legend=True
                #     )
                #     ax.legend(handles=legend_handles, bbox_to_anchor=(1, 1), loc=2)
                #     plt.show(block=True)

                # alpha
                crf_alpha = crf_with_alpha(
                    guidance_img, norm_cam=norm_cam, valid_classes=valid_classes, alpha=args.alpha,
                    mask_guided=args.mask_guided_crf
                )
                crf_alpha_pred = np.argmax(crf_alpha, axis=0)
                if is_show:
                    ax, legend_handles = vis_semantic_segmentation(
                        orig_img.transpose((2, 0, 1)), crf_alpha_pred,
                        label_names=voc_semantic_segmentation_label_names,
                        label_colors=voc_semantic_segmentation_label_colors,
                        alpha=0.7, all_label_names_in_legend=True
                    )
                    ax.legend(handles=legend_handles, bbox_to_anchor=(1, 1), loc=2)
                    plt.show(block=True)

                # low alpha
                crf_la = crf_with_alpha(
                    guidance_img, norm_cam=norm_cam, valid_classes=valid_classes, alpha=args.low_alpha,
                    mask_guided=args.mask_guided_crf
                )
                crf_la_pred = np.argmax(crf_la, axis=0)
                if is_show:
                    ax, legend_handles = vis_semantic_segmentation(
                        orig_img.transpose((2, 0, 1)), crf_la_pred,
                        label_names=voc_semantic_segmentation_label_names,
                        label_colors=voc_semantic_segmentation_label_colors,
                        alpha=0.7, all_label_names_in_legend=True
                    )
                    ax.legend(handles=legend_handles, bbox_to_anchor=(1, 1), loc=2)
                    plt.show(block=True)

                # High alpha
                crf_ha = crf_with_alpha(
                    guidance_img, norm_cam=norm_cam, valid_classes=valid_classes, alpha=args.high_alpha,
                    mask_guided=args.mask_guided_crf
                )
                crf_ha_pred = np.argmax(crf_ha, axis=0)
                if is_show:
                    ax, legend_handles = vis_semantic_segmentation(
                        orig_img.transpose((2, 0, 1)), crf_ha_pred,
                        label_names=voc_semantic_segmentation_label_names,
                        label_colors=voc_semantic_segmentation_label_colors,
                        alpha=0.7, all_label_names_in_legend=True
                    )
                    ax.legend(handles=legend_handles, bbox_to_anchor=(1, 1), loc=2)
                    plt.show(block=True)

                # Save the segmentation predictions
                np.savez_compressed(
                    args.out_crf.joinpath(img_name),
                    valid_classes=valid_classes, no_crf=raw_pred,   # refined=refined_pred,
                    crf=crf_alpha_pred, crf_la=crf_la_pred, crf_ha=crf_ha_pred
                )


def make_pseudo_gt():
    # ================================ PARAMETERS =============================
    args = parse_inference_arguments()
    num_gpus = torch.cuda.device_count()

    # ================================ DATASETS =============================
    if args.mask_guided:
        infer_dataset = datasets.VOC12ClsDepthDatasetMSF(
            args.infer_list, voc12_root=args.voc12_root, scales=args.multi_scales, is_flip=args.is_flip,
            scale_factor=args.patch_size, feature_type=args.feature_type,
            sigma_xy=args.sigma_xy, sigma_rgb=args.sigma_rgb, sigma_depth=args.sigma_depth,
            transform=transforms.Compose([
                transforms.ToTensor(),
                augmentations.ValCrop(args.patch_size),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        )
    else:
        infer_dataset = datasets.VOC12ImageDatasetMSF(
            args.infer_list, voc12_root=args.voc12_root, scales=args.multi_scales, is_flip=args.is_flip,
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

    print(f"The model are built on {args.arch} network.")

    # ================================ LOAD MODEL CHECKPOINTS ... ================================
    train_tools.restart_from_checkpoint(
        args.ckpt_dir.joinpath(f"checkpoint_{args.eval_epoch:03}.pth"),
        model=model
    )
    model = model.cuda()
    model.eval()

    # ================================ MAKE CAM ===================================
    # get_pseudo_mask(process_id=0, model=model, dataset=infer_datasets, args=args)
    mp.spawn(get_pseudo_mask, nprocs=num_gpus, args=(model, infer_datasets, args), join=True)

    torch.cuda.empty_cache()


def eval_pseudo_gt():
    # ================================ PARAMETERS =============================
    args = parse_inference_arguments()
    num_gpus = torch.cuda.device_count()

    eval_cam(args)
