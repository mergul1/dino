import argparse

import torch
from torchvision import models as torchvision_models

from utils import misc

torchvision_archs = sorted(
    name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__") and callable(torchvision_models.__dict__[name])
)


def common_arguments():
    parser = argparse.ArgumentParser('ViT_WSL', add_help=False)

    # --------------------------------------  Dataset  -----------------------------------------------------------------
    parser.add_argument("--voc12_root", default="/home/mergul/Desktop/VOC2012", type=str)
    parser.add_argument("--num_classes", default=20, type=int)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)

    parser.add_argument("--crop_size", default=480, type=int)

    parser.add_argument('--max_epochs', default=10, type=int, help="Number of epochs of training.")
    parser.add_argument('--num_workers', default=0, type=int, help="Number of data loading workers per GPU.")

    # -------------------------------------  Model parameters  ---------------------------------------------------------
    parser.add_argument('--pretrain_method', default='dino', type=str,
                        choices=['dino', 'deit', 'deit_distilled', 'timm', 'cait', 'xcit'], help="Training methodology")
    parser.add_argument('--img_size', default=224, type=int, choices=[224, 384, 448], help="""Image size during the 
    training process""")
    parser.add_argument('--patch_size', default=8, type=int, help="""Size in pixels of input square patches 
    - default 16 (for 16x16 patches)""")
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'xcit'] + torchvision_archs
                                + torch.hub.list("facebookresearch/xcit:main"),
                        help="""Name of architecture.""")
    parser.add_argument('--mlp_head', default=True, type=misc.bool_flag, help="""Whether MLP head or Linear head will 
    be used during the classification""")
    parser.add_argument('--drop_path_rate', type=float, default=0., help="""stochastic depth rate""")
    parser.add_argument('--attn_drop_rate', type=float, default=0., help="""attention drop rate""")
    parser.add_argument('--drop_rate', type=float, default=0., help="""drop rate""")
    parser.add_argument('--mask_guided', default=False, type=misc.bool_flag, help="...")

    # ---------------------------------------  Resume Training/Pre-training  -------------------------------------------
    parser.add_argument('--start_epoch', default=0, type=int, help="""From what epochs the training start at resume""")
    parser.add_argument('--pretrained', default=True, type=misc.bool_flag, help="Whether to pretrained or not")
    parser.add_argument('--pretrained_weights', default='', type=str, help="""Path to pretrained weights to load,
    if exists locally.""")
    parser.add_argument("--checkpoint_key", default="model", type=str, help="""Key to use in the checkpoint
    (example: "model")""")

    # ----------------------------------  Checkpointing and Logging  ---------------------------------------------------
    parser.add_argument('--ckpt_dir', default="checkpoints", type=str, help="Path to save the logs and checkpoints.")
    parser.add_argument('--output_dir', default="output", type=str, help="Path to save the logs and outputs")
    parser.add_argument('--save_ckpt_freq', default=1, type=int, help="Save checkpoint every x epochs.")

    parser.add_argument('--wandb', default="online", type=str, choices=["online", "offline", "disabled"], help="..")
    parser.add_argument('--project_name', default="DINO_Classification", type=str, help="The project name")

    # -----------------------------------------------  Misc  -----------------------------------------------------------
    parser.add_argument('--seed', default=0, type=int, help="""Random seed.""")
    parser.add_argument('--mode', default="infer", type=str, choices=["train", "val", "eval", "infer"], help=""".""")
    parser.add_argument('--exp_name', default="exp_wd0.1-1.0_per-layer-lr", type=str,
                        help="The experiment name under the project")

    return parser
