import argparse
import wandb

from arguments import common_arguments
from inference import make_pseudo_gt, eval_pseudo_gt
from train import train

# wandb.login(key='2dd6ce2a6575a98800c8c3c4a9ce3b0d18d4bb76')


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
        eval_pseudo_gt()
    elif args.mode == 'infer':
        make_pseudo_gt()
        eval_pseudo_gt()
    else:
        pass

