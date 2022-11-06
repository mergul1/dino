import argparse
from pathlib import Path
import random
from omegaconf import OmegaConf

import wandb
import tkinter
import matplotlib.pyplot as plt
import numpy as np

import torch

from utils import train_tools, misc, optims, metrics
from voc12 import datasets, augmentations
from arguments import common_arguments
import vision_transformer as vits

from inference import make_pseudo_gt


from train import train

# wandb.login(key='2dd6ce2a6575a98800c8c3c4a9ce3b0d18d4bb76')
# import matplotlib
# matplotlib.rcParams['backend'] = 'QtAgg'

plt.switch_backend('TkAgg')
print(plt.get_backend())
plt.ioff()
print(plt.isinteractive())


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ViT_WSL', parents=[common_arguments()])
    args = parser.parse_args()

    # Run
    if args.mode == 'train':
        run = wandb.init(
            project=args.project_name, name=args.exp_name, job_type=args.mode, config=vars(args), mode=args.wandb
        )
        wandb.run.log_code('.')
        train()
        wandb.finish()
    elif args.mode == 'val':
        pass
    elif args.mode == 'eval':
        pass
    elif args.mode == 'infer':
        make_pseudo_gt()
    else:
        pass

