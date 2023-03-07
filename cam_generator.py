from functools import partial
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from vision_transformer import VisionTransformer
from utils.misc import min_max_normalizer, min_max_normalizer_clip
from utils.train_tools import load_pretrained
from utils.visualizations import ShowAttentions, show_cam_list


def layer_aggregation(layer_matrices_list, aggregation_type='rollout1', start_layer=0, is_norm=False):
    def _min_max_normalizer(tensor_list):
        for idx, tensor in enumerate(tensor_list):
            tensor -= tensor.min()
            tensor /= (tensor.max() + 1e-8)
            tensor_list[idx] = tensor
        return tensor_list

    batch_size, num_tokens, _ = layer_matrices_list[-1].shape
    identity = torch.eye(num_tokens, device=layer_matrices_list[-1].device).expand(batch_size, -1, -1)

    if aggregation_type == 'rollout1':
        valid_layer_list = [layer_matrices + identity for layer_matrices in layer_matrices_list[start_layer:]]
        # valid_layer_list = [layer_matrices / layer_matrices.sum(dim=-1, keepdim=True) for layer_matrices in valid_layer_list]
        rollout = valid_layer_list[0]
        for idx, layer_matrices in enumerate(valid_layer_list[1:]):
            rollout = layer_matrices.bmm(rollout)
        class_cam = rollout[:, 0, 1:]
        return class_cam

    elif aggregation_type == 'rollout2':
        rollout = identity.clone()
        for idx, layer_matrices in enumerate(layer_matrices_list[start_layer:]):
            rollout += layer_matrices.bmm(rollout)
            context_rollout = rollout - identity
            context_rollout /= (context_rollout.sum(dim=-1, keepdim=True) + 1e-5)
            rollout = context_rollout + identity
        class_cam = rollout[:, 0, 1:]
        return class_cam

    elif aggregation_type in ['sum', 'mean', 'hadamard']:
        cam_list = [layer_matrices[:, 0, 1:] for layer_matrices in layer_matrices_list[start_layer:]]
        if is_norm:
            cam_list = _min_max_normalizer(cam_list)
        if aggregation_type == 'sum':
            class_cam = torch.sum(torch.stack(cam_list, dim=1), dim=1)
        elif aggregation_type == 'mean':
            class_cam = torch.mean(torch.stack(cam_list, dim=1), dim=1)
        elif aggregation_type == 'hadamard':
            class_cam = torch.prod(torch.stack(cam_list, dim=1), dim=1)
        else:
            raise ValueError(f'There is no such an aggregation type: {aggregation_type}')
        return class_cam, cam_list

    else:
        raise ValueError(f'There is no such an aggregation type: {aggregation_type}')


class CAMGenerator(VisionTransformer):
    def __init__(self, *args, **kwargs):

        super(CAMGenerator, self).__init__(*args, **kwargs)
        self.orig_img = None

    def get_class_attention(self, x, args):
        # Evaluation mode
        self.eval()

        # Forward Propagation
        class_score = self.forward(x, register_hook=True)

        # Get Attention Maps
        attns = [blk.attn.get_attn().detach() for idx, blk in enumerate(self.blocks) if idx >= args.start_layer]

        # Get Class Attention Maps
        class_attns = [
            attn[:, :, 0, 1:].mean(dim=1).reshape(-1, self.num_patch_height, self.num_patch_width) for attn in attns
        ]

        #
        class_attn = torch.stack(class_attns[args.start_layer:], dim=1).sum(dim=1)
        resized_class_attn = F.interpolate(
            class_attn.unsqueeze(dim=1), scale_factor=self.patch_size, mode='bilinear', align_corners=False
        ).squeeze(dim=1).relu()

        return resized_class_attn, class_score

    def get_patchwise_affinity(self, args):
        attns = [blk.attn.get_attn().detach() for idx, blk in enumerate(self.blocks) if idx >= args.start_layer_attns]
        attn_list = [attn.relu().mean(dim=1) for attn in attns]
        affinity = torch.mean(torch.stack(attn_list, dim=0), dim=0)  # B x N X N
        patchwise_affinity = affinity[:, 1:, 1:]
        return patchwise_affinity

    def get_attention_maps(self, args):
        # Get attentions maps by averaging though heads for each layers
        attn_list = [None] * len(self.blocks)
        for idx, blk in enumerate(self.blocks):
            if idx >= args.start_layer_attns:
                attn = blk.attn.get_attn().detach()
                attn_list[idx] = attn.relu().mean(dim=1)
        # ShowAttentions(self.orig_img, attns[-2].mean(dim=1), self.patch_size)
        return attn_list

    def get_gradient_maps(self, args):
        # Get gradients maps
        grad_list = [None] * len(self.blocks)
        for idx, blk in enumerate(self.blocks):
            if idx >= args.start_layer:
                grad = blk.attn.get_attn_gradients().detach()
                grad_list[idx] = grad.relu().mean(dim=1)
                # grad_list[idx] = grad.mean(dim=1).relu()
        # ShowAttentions(self.orig_img, attns[-2].mean(dim=1), self.patch_size)
        return grad_list

    def get_getam(self, args):
        getam_list = [None] * len(self.blocks)
        for idx, blk in enumerate(self.blocks):
            if idx >= args.start_layer:
                # Get the attentions maps and attention-gradient maps for each layers
                attn = blk.attn.get_attn().detach()
                grad = blk.attn.get_attn_gradients().detach()

                # ncols = 3
                # fig, axs = plt.subplots(2, 3)
                # for i in range(6):
                #     axs[i//ncols, i % ncols].imshow(grad[0, i, 1:, 1:].relu().detach().cpu())
                #     axs[i//ncols, i % ncols].axes.xaxis.set_ticks([])
                #     axs[i//ncols, i % ncols].axes.yaxis.set_ticks([])
                #     axs[i//ncols, i % ncols].set_title(f'Head: {i}')
                # ShowAttentions(self.orig_img, attn.mean(dim=1, keepdims=True), self.patch_size)
                # ShowAttentions(self.orig_img, attn, self.patch_size)

                # Compute the gradient-weighted attention map
                getam = attn * grad
                if args.double_gradient:
                    getam_list[idx] = (getam.relu() * grad.relu()).mean(dim=1)
                else:
                    getam_list[idx] = getam.relu().mean(dim=1)

        return getam_list

    @staticmethod
    def refinement_with_affinity(class_cam, semantic_affinity, low_affinity, beta=1, logt=0, semantic_weight=1.0):

        # patchwise_affinity: layer mean of head mean of attention map
        batch_size, num_tokens, _ = low_affinity.shape
        identity = torch.eye(num_tokens, device=low_affinity.device).expand(batch_size, -1, -1)
        low_affinity = low_affinity + identity
        low_affinity = low_affinity.softmax(dim=-1)

        affinity = (1 - semantic_weight) * low_affinity + semantic_weight * semantic_affinity
        affinity = affinity.softmax(dim=-1)

        # patch_attention = attention_rollout[0, 1:, 1:]
        # affinity = patch_attention + patch_attention.t()
        # show_affinity(None, affinity.cpu(), feat_size)
        # affinity /= affinity.diag().unsqueeze(dim=1)
        # affinity /= affinity.max(dim=-1, keepdim=True)[0]
        # affinity = torch.clamp(affinity, min=0.0, max=1.0)

        # affinity = torch.pow(affinity, beta)
        # transition_prob = affinity / (affinity.sum(dim=-1, keepdim=True) + 1e-6)    # avoid nan
        # for _ in range(logt):
        #     transition_prob = torch.matmul(transition_prob, transition_prob)

        class_cam = torch.matmul(affinity, class_cam.permute(1, 0))
        # class_cam = torch.matmul(transition_prob, class_cam.permute(1, 0))

        return class_cam

    def get_cam(self, args, img, label=None, orig_img=None, low_affinity=None):
        self.eval()

        batch_size, _, height, width = img.shape
        self.num_patch_height = height // self.patch_size
        self.num_patch_width = width // self.patch_size
        self.orig_img = orig_img

        # Classification output prediction and infer the pseudo-label for classification
        with torch.no_grad():
            # If label is not given
            if label is None:
                print(f'True label is not given. The prediction label is inferred and used for pseudo mask generation')
                class_score = self.forward(img, register_hook=False)
                label = class_score.ge(0.0).to(torch.float)

            # Number of classes in the image
            num_targets = torch.count_nonzero(label[0]).item()
            valid_classes = torch.nonzero(label[0], as_tuple=True)[0]

        # Class Attention Map
        cam = torch.zeros((batch_size, self.num_classes, height, width), device=img.device)
        for idx, class_idx in enumerate(valid_classes):
            # Forward propagation
            class_score = self.forward(img, register_hook=True)

            if args.method in ['attention_rollout', ]:
                attn_list = self.get_attention_maps(args)
                class_attention_map = layer_aggregation(
                    attn_list, aggregation_type=args.agg_type, start_layer=args.start_layer
                )
            else:
                # Backward propagation
                self.zero_grad()
                out_grad = F.one_hot(valid_classes, args.num_classes)
                valid_class_score = out_grad.select(dim=0, index=idx).unsqueeze(dim=0) * class_score
                valid_class_score.sum().backward()

                # Compute Class Attention Maps
                if args.method == 'getam_rollout':
                    getam_list = self.get_getam(args)
                    class_attention_map, cam_list = layer_aggregation(
                        getam_list, aggregation_type=args.agg_type, start_layer=args.start_layer
                    )
                    if args.is_show and args.agg_type in ['mean', 'sum', 'product']:
                        show_cam_list(
                            class_attention_map, cam_list, height=self.num_patch_height, width=self.num_patch_width
                        )

                elif args.method == 'gradient_rollout':
                    grad_list = self.get_gradient_maps(args)
                    class_attention_map, cam_list = layer_aggregation(
                        grad_list, aggregation_type=args.agg_type, start_layer=args.start_layer
                    )
                    if args.is_show and args.agg_type in ['mean', 'sum', 'product']:
                        show_cam_list(
                            class_attention_map, cam_list, height=self.num_patch_height, width=self.num_patch_width
                        )

                else:
                    class_attention_map = None
                    print(f'There is no such a method: {args.method}')

            if args.is_refined:
                semantic_affinity = self.get_patchwise_affinity(args)
                class_attention_map = self.refinement_with_affinity(
                    class_attention_map, semantic_affinity, low_affinity, args.beta, args.logt, args.semantic_weight
                )

            # Reshape and resize the class attention map
            class_attention_map = class_attention_map.reshape(-1, self.num_patch_height, self.num_patch_width)
            resized_class_map = F.interpolate(
                class_attention_map.unsqueeze(dim=1), scale_factor=self.patch_size, mode='bilinear', align_corners=False
            ).squeeze(dim=1).relu()

            # Normalize the class_wise CAMs
            normalized_map = min_max_normalizer(resized_class_map)
            cam[0, class_idx] = normalized_map
            if args.is_show:
                fig, axs = plt.subplots(nrows=1, ncols=2)
                axs[0].imshow(orig_img)
                axs[1].imshow(normalized_map[0].cpu())
                plt.show(block=True)

            # Memory cleaning
            del class_score, normalized_map, resized_class_map, class_attention_map
            torch.cuda.empty_cache()

        return cam


def vit_tiny(args, **kwargs):
    model = CAMGenerator(
        img_size=args.img_size, patch_size=args.patch_size, num_classes=args.num_classes, embed_dim=192, depth=12,
        num_heads=3, mlp_ratio=4, qkv_bias=True, mlp_head=args.mlp_head, mask_guided=args.mask_guided,
        drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate, drop_path_rate=args.drop_path_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    if args.pretrained:
        load_pretrained(model, args)
    return model


def vit_small(args, **kwargs):
    model = CAMGenerator(
        img_size=args.img_size, patch_size=args.patch_size, num_classes=args.num_classes, embed_dim=384, depth=12,
        num_heads=6, mlp_ratio=4, qkv_bias=True, mlp_head=args.mlp_head, mask_guided=args.mask_guided,
        drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate, drop_path_rate=args.drop_path_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    if args.pretrained:
        load_pretrained(model, args)
    return model


def vit_base(args, **kwargs):
    model = CAMGenerator(
        img_size=args.img_size, patch_size=args.patch_size, num_classes=args.num_classes, embed_dim=768, depth=12,
        num_heads=12, mlp_ratio=4, qkv_bias=True, mlp_head=args.mlp_head, mask_guided=args.mask_guided,
        drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate, drop_path_rate=args.drop_path_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    if args.pretrained:
        load_pretrained(model, args)
    return model


def vit_large(args, **kwargs):
    model = CAMGenerator(
        img_size=args.img_size, patch_size=args.patch_size, num_classes=args.num_classes, embed_dim=1024, depth=24,
        num_heads=16, mlp_ratio=4, qkv_bias=True, mlp_head=args.mlp_head, mask_guided=args.mask_guided,
        drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate, drop_path_rate=args.drop_path_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    if args.pretrained:
        load_pretrained(model, args)
    return model
