import sys
import os
import math
import warnings

import torch
import torch.distributed as dist


# ================================  PARALLEL TRAINING ===============================
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """ This function disables printing when not in master process """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


# ================================  RESUME TRAINING & FINE TUNING ===============================
def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """ Re-start from checkpoint """
    if not os.path.isfile(ckp_path):
        print(f'\n Not found the specified checkpoint at {ckp_path}')
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file.
    # value is the object to load.
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                # remove `module.` prefix
                checkpoint[key] = {k.replace("module.", ""): v for k, v in checkpoint[key].items()}

                # load value to the module
                msg = value.load_state_dict(checkpoint[key], strict=True)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def load_pretrained(model, args):
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]

        # remove `module.` prefix, if exists
        state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {key.replace("backbone.", ""): value for key, value in state_dict.items()}

        # Load the pretrained weight in state dict into the model
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))

    else:
        print("Since no pretrained weights have been provided, we load the reference pretrained weights.")

        url = None
        # Select url path
        if args.pretrain_method == "dino":
            if args.arch == "vit_small":
                if args.patch_size == 16:
                    url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
                elif args.patch_size == 8:
                    # url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
                    # model used for visualizations in our paper
                    url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
                else:
                    raise ValueError(
                        f"There is no such a pretrained model for {args.pretrain_method}-{args.arch}-{args.patch_size}")
            elif args.arch == "vit_base":
                if args.patch_size == 16:
                    url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
                elif args.patch_size == 8:
                    url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
                else:
                    raise ValueError(
                        f"There is no such a pretrained model for {args.pretrain_method}-{args.arch}-{args.patch_size}")
            elif args.arch == "xcit_small_12_p16":
                url = "dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth"
            elif args.arch == "xcit_small_12_p8":
                url = "dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth"
            elif args.arch == "xcit_medium_24_p16":
                url = "dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth"
            elif args.arch == "xcit_medium_24_p8":
                url = "dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth"
            elif args.arch == "resnet50":
                url = "dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
            else:
                raise ValueError(
                    f"There is no such a pretrained model for {args.pretrain_method}-{args.arch}-{args.patch_size}")
            url = "https://dl.fbaipublicfiles.com/dino/" + url
        elif args.pretrain_method == "deit":
            if args.arch == "vit_tiny":
                if args.patch_size == 16:
                    url = "deit_tiny_patch16_224-a1311bcf.pth"
                else:
                    raise ValueError(
                        f"There is no such a pretrained model for {args.pretrain_method}-{args.arch}-{args.patch_size}")
            elif args.arch == "vit_small":
                if args.patch_size == 16:
                    url = "deit_small_patch16_224-cd65a155.pth"
                else:
                    raise ValueError(
                        f"There is no such a pretrained model for {args.pretrain_method}-{args.arch}-{args.patch_size}")
            elif args.arch == "vit_base":
                if args.patch_size == 16 and args.img_size == 384:
                    url = "deit_base_patch16_384-8de9b5d1.pth"
                elif args.patch_size == 16 and args.img_size == 224:
                    url = "deit_base_patch16_224-b5f2ef4d.pth"
                else:
                    raise ValueError(
                        f"There is no such a pretrained model for {args.pretrain_method}-{args.arch}-{args.patch_size}")
            else:
                raise ValueError(
                    f"There is no such a pretrained model for {args.pretrain_method}-{args.arch}-{args.patch_size}")
            url = "https://dl.fbaipublicfiles.com/deit/" + url
        elif args.pretrain_method == "deit_distilled":
            if args.arch == "vit_tiny":
                if args.patch_size == 16:
                    url = "/deit_tiny_distilled_patch16_224-b40b3cf7.pth"
                else:
                    raise ValueError(
                        f"There is no such a pretrained model for {args.pretrain_method}-{args.arch}-{args.patch_size}")
            elif args.arch == "vit_small":
                if args.patch_size == 16:
                    url = "deit_small_distilled_patch16_224-649709d9.pth"
                else:
                    raise ValueError(
                        f"There is no such a pretrained model for {args.pretrain_method}-{args.arch}-{args.patch_size}")
            elif args.arch == "vit_base":
                if args.patch_size == 16 and args.img_size == 224:
                    url = "deit_base_distilled_patch16_224-df68dfff.pth"
                elif args.patch_size == 16 and args.img_size == 384:
                    url = "deit_base_distilled_patch16_384-d0272ac0.pth"
                else:
                    raise ValueError(
                        f"There is no such a pretrained model for {args.pretrain_method}-{args.arch}-{args.patch_size}")
            else:
                raise ValueError(
                    f"There is no such a pretrained model for {args.pretrain_method}-{args.arch}-{args.patch_size}")
            url = "https://dl.fbaipublicfiles.com/deit/" + url
        elif args.pretrain_method == "cait":
            if args.arch == "S24" and args.img_size == 224:
                url = "S24_224.pth"
            elif args.arch == "S24_384" and args.img_size == 384:
                url = "S24_384.pth"
            elif args.arch == "S36_384" and args.img_size == 384:
                url = "S36_384.pth"
            elif args.arch == "M36_384" and args.img_size == 384:
                url = "M36_384.pth"
            elif args.arch == "M48_448" and args.img_size == 448:
                url = "M48_448.pth"
            elif args.arch == "XS24_384" and args.img_size == 384:
                url = "XS24_384.pth"
            elif args.arch == "XXS24_384" and args.img_size == 384:
                url = "XXS24_384.pth"
            elif args.arch == "XXS36_384" and args.img_size == 384:
                url = "XXS36_384.pth"
            elif args.arch == "XXS24_224" and args.img_size == 224:
                url = "XXS24_224.pth"
            elif args.arch == "XXS36_224" and args.img_size == 224:
                url = "XXS36_224.pth"
            else:
                raise ValueError(
                    f"There is no such a pretrained model for {args.pretrain_method}-{args.arch}-{args.patch_size}")
            url = "https://dl.fbaipublicfiles.com/deit/" + url
        elif args.pretrain_method == "timm":
            if args.arch == "vit_small":
                if args.patch_size == 16:
                    url = "v0.1-weights/vit_small_p16_224-15ec54c9.pth"
                else:
                    raise ValueError(
                        f"There is no such a pretrained model for {args.pretrain_method}-{args.arch}-{args.patch_size}")
            elif args.arch == "vit_base":
                if args.patch_size == 16:
                    url = "v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth"
                else:
                    raise ValueError(
                        f"There is no such a pretrained model for {args.pretrain_method}-{args.arch}-{args.patch_size}")
            elif args.arch == "vit_large":
                if args.patch_size == 16:
                    url = "v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth"
                else:
                    raise ValueError(
                        f"There is no such a pretrained model for {args.pretrain_method}-{args.arch}-{args.patch_size}")
            else:
                raise ValueError(
                    f"There is no such a pretrained model for {args.pretrain_method}-{args.arch}-{args.patch_size}")
            url = "https://github.com/rwightman/pytorch-image-models/releases/download/" + url
        else:
            raise ValueError(
                f"There is no such a pretrained model for {args.pretrain_method}-{args.arch}-{args.patch_size}")

        # Load the pretrained weights into the model
        if url is not None:
            print("Pretrained file has not been found in local place. It will be downloaded from internet")
            checkpoint = torch.hub.load_state_dict_from_url(url=url)

            try:
                checkpoint_model = checkpoint[args.checkpoint_key]
            except KeyError:
                checkpoint_model = checkpoint

            deleted_layers_name = []
            for key in checkpoint_model.keys():
                if 'head' in key and checkpoint_model[key].shape != model.state_dict()[key].shape:
                    deleted_layers_name.append(key)
            for key in deleted_layers_name:
                print(f"Removing key {key} from pretrained checkpoint")
                del checkpoint_model[key]

            msg = model.load_state_dict(checkpoint_model, strict=False)
            print('Pretrained weights found at {} in the hub and loaded with msg: {}'.format(url, msg))
        else:
            print("There is no reference weights available for this model => We use random weights.")


# ================================  INITIALIZATION ===============================
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.", stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and then using the inverse CDF
        # for the normal distribution. Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
