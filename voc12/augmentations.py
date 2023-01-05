import random
import math
import numbers
from collections.abc import Sequence

from PIL import Image, ImageFilter, ImageOps
import numpy as np
import scipy.ndimage as nd
from scipy.spatial.distance import pdist, squareform

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode


def check_channels(img):
    if img.ndim == 3:
        if img.shape[-3] in [1, 3, 4]:
            num_channels = img.shape[-3]
            is_channel_first = True
        elif img.shape[-1] in [1, 3, 4]:
            num_channels = img.shape[-1]
            is_channel_first = False
        else:
            raise ValueError('This tensor is not an image composed of 1, 3, 4 channels')
    elif img.ndim == 2:
        num_channels = 1
        is_channel_first = None
    else:
        raise ValueError('This is not 1 or 3 dimensional tensor')

    return num_channels, is_channel_first


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return size, size

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class GaussianBlur(object):
    """ Apply Gaussian Blur to the PIL image. """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max)))


class Solarization(object):
    """ Apply Solarization to the PIL image."""
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=1.0),
            normalize,
        ])
        # second global crop
        self.global_transform2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = [self.global_transform1(image), self.global_transform2(image)]
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops


# ======================== SEGMENTATION ============================
class ComposeMultiField:
    """Composes several transforms together.
    Args:
        transform_list (list of ``Transform`` objects): list of transforms to compose.

    Example:
        transforms.Compose([
            transforms.CenterCrop(10),
            transforms.ToTensor()
            ])
    """

    def __init__(self, transform_list):
        self.transforms = transform_list

    def __call__(self, args):
        for t in self.transforms:
            args = t(args)
        return args

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += f'\n {t}'
        format_string += '\n)'
        return format_string


class RandomHorizontalFlipMultiField(nn.Module):
    """ Horizontally flip the given image and the depth image randomly with a given probability.
    If the image and the depth image is torch Tensor, it is expected to have [..., H, W] shape, where ... means
    an arbitrary number of leading dimensions.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, field_list):
        """
        Args:
            field_list (List of PIL Image or Tensor): Images to be flipped.

        Returns:
            List of PIL Image or Tensor: Randomly flipped images.
        """
        if torch.rand(1) < self.p:
            flipped_list = []
            for field in field_list:
                num_channels, is_channel_first = check_channels(field)
                is_hwc = (num_channels == 3) and (not is_channel_first)
                if is_hwc:
                    field = field.permute(2, 0, 1)

                if num_channels == 1:
                    field = field.unsqueeze(dim=0)

                field = F.hflip(field)

                if num_channels == 1:
                    field = field.squeeze(dim=0)

                if is_hwc:
                    field = field.permute(1, 2, 0)

                flipped_list.append(field)

            return flipped_list
        return field_list

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomCropMultiField(nn.Module):
    """ Crop the given images at a random location.
     If the images is torch Tensor, it is expected to have [..., H, W] shape, where ... means an arbitrary number of
     leading dimensions, but if non-constant padding is used, the input is expected to have at most 2 leading dimensions

    Args:
        crop_size (sequence or int): Desired output size of the crop. If size is an int instead of sequence like (h, w),
            a square crop (size, size) is made. If provided a sequence of length 1, it will be interpreted
            as (size[0], size[0]).
        padding (int or sequence, optional): Optional padding on each border of the image. Default is None.
            If a single int is provided this is used to pad all borders. If sequence of length 2 is provided
            this is the padding on left/right and top/bottom respectively. If a sequence of length 4 is provided
            this is the padding for the left, top, right and bottom borders respectively.

            .. note::
                In torchscript mode padding as single int is not supported, use a sequence of length 1: ``[padding, ]``.
        pad_if_needed (boolean): It will pad the image if smaller than the desired size to avoid raising an exception.
            Since cropping is done after padding, the padding seems to be done at a random offset.
        fill (number or str or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of length 3,
            it is used to fill R, G, B channels respectively. This value is only used when the padding_mode is constant.
            Only number is supported for torch Tensor.
            Only int or str or tuple value is supported for PIL Image.
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
            Default is constant.

            - constant: pads with a constant value, this value is specified with fill

            - edge: pads with the last value at the edge of the image. If input a 5D torch Tensor,
                the last 3 dimensions will be padded instead of the last 2

            - reflect: pads with reflection of image without repeating the last value on the edge. For example,
              padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode will result in [3, 2, 1, 2, 3, 4, 3, 2]

            - symmetric: pads with reflection of image repeating the last value on the edge. For example,
              padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3]
    """

    @staticmethod
    def get_params(input_size, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            input_size (tuple): Input size of the images.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = input_size
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError("Required crop size {} is larger then input image size {}".format((th, tw), (h, w)))

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1, )).item()
        j = torch.randint(0, w - tw + 1, size=(1, )).item()
        return i, j, th, tw

    def __init__(self, crop_size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()

        self.crop_size = tuple(_setup_size(crop_size, error_msg="Please provide only two dimensions (h, w) for size."))

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, field_list):
        """
        Args:
            field_list (list of PIL Image or Tensor): Images to be cropped.

        Returns:
            List of PIL Image or Tensor: Cropped images.
        """

        # Pad the images, if necessary.
        padded_fields = []
        for field in field_list:
            num_channels, is_channel_first = check_channels(field)
            is_hwc = (num_channels == 3) and (not is_channel_first)
            if is_hwc:
                field = field.permute(2, 0, 1)

            if self.padding is not None:
                field = F.pad(field, self.padding, self.fill, self.padding_mode)

            width, height = F.get_image_size(field)
            # pad the width if needed
            if self.pad_if_needed and width < self.crop_size[1]:
                padding = [self.crop_size[1] - width, 0]
                field = F.pad(field, padding, self.fill, self.padding_mode)

            # pad the height if needed
            if self.pad_if_needed and height < self.crop_size[0]:
                padding = [0, self.crop_size[0] - height]
                field = F.pad(field, padding, self.fill, self.padding_mode)

            if is_hwc:
                field = field.permute(1, 2, 0)

            padded_fields.append(field)

        width, height = F.get_image_size(padded_fields[0])
        i, j, h, w = self.get_params(input_size=(width, height), output_size=self.crop_size)

        # Crop the images
        cropped_fields = []
        for field in padded_fields:
            num_channels, is_channel_first = check_channels(field)
            is_hwc = (num_channels == 3) and (not is_channel_first)
            if is_hwc:
                field = field.permute(2, 0, 1)

            field = F.crop(field, i, j, h, w)

            if is_hwc:
                field = field.permute(1, 2, 0)

            cropped_fields.append(field)

        return cropped_fields

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(self.crop_size, self.padding)


class ResizeMultiField(nn.Module):
    """ Resize the input images to the given size.
    The images can be a PIL Image or a torch Tensor, in which case it is expected to have [..., H, W] shape,
    where ... means an arbitrary number of leading dimensions.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like (h, w), output size will be matched to
            this. If size is an int, smaller edge of the image will be matched to this number. i.e, if height > width,
            then image will be rescaled to (size * height / width, size).
            In torchscript mode padding as single int is not supported, use a tuple or list of length 1: ``[size, ]``.

        interpolation (int, optional): Desired interpolation enum defined by `filters`_.
            Default is ``PIL.Image.BILINEAR``. If input is Tensor, only ``PIL.Image.NEAREST``, ``PIL.Image.BILINEAR``
            and ``PIL.Image.BICUBIC`` are supported.
    """

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        super(ResizeMultiField, self).__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")

        self.size = size
        self.interpolation = interpolation

    def forward(self, field_list):
        """
        Args:
            field_list (List of PIL Image or Tensor): Images to be scaled.

        Returns:
            List of PIL Image or Tensor: Rescaled images.
        """

        resized_fields = []
        for field in field_list:
            num_channels, is_channel_first = check_channels(field)
            is_hwc = num_channels == 3 and not is_channel_first
            if is_hwc:
                field = field.permute(2, 0, 1)

            if num_channels == 1:
                field = field.unsqueeze(dim=0)

            field = F.resize(field, self.size, self.interpolation)

            if num_channels == 1:
                field = field.squeeze(dim=0)

            if is_hwc:
                field = field.permute(1, 2, 0)

            resized_fields.append(field)

        return resized_fields

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, self.interpolation)


class RandomScaleMultiField(nn.Module):
    """ Randomly scale the input images.
    The images can be a PIL Image or a torch Tensor, in which case it is expected to have [..., H, W] shape,
    where ... means an arbitrary number of leading dimensions.

    Args:
        scale_range (sequence): Scale range. (min_scale, max_scale)

        interpolation (int, optional): Desired interpolation enum defined by `filters`_.
            Default is ``PIL.Image.BILINEAR``. If input is Tensor, only ``PIL.Image.NEAREST``, ``PIL.Image.BILINEAR``
            and ``PIL.Image.BICUBIC`` are supported.
    """

    def __init__(self, scale_range, interpolation=F.InterpolationMode.BILINEAR):
        super(RandomScaleMultiField, self).__init__()
        if not isinstance(scale_range, (Sequence,)):
            raise TypeError("Size should be sequence. Got {}".format(type(scale_range)))
        if isinstance(scale_range, Sequence) and len(scale_range) not in (2,):
            raise ValueError("If size is a sequence, it should have 2 values")

        self.scale_range = scale_range
        self.interpolation = interpolation

    def forward(self, field_list):
        """
        Args:
            field_list (List of PIL Image or Tensor): Images to be scaled.

        Returns:
            List of PIL Image or Tensor: Rescaled images.
        """

        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        width, height = F.get_image_size(field_list[0])
        target_size = [int(height * scale), int(width * scale)]

        resized_fields = []
        for field in field_list:
            num_channels, is_channel_first = check_channels(field)
            is_hwc = num_channels == 3 and not is_channel_first
            if is_hwc:
                field = field.permute(2, 0, 1)

            if num_channels == 1:
                field = field.unsqueeze(dim=0)
                field = F.resize(field, target_size, F.InterpolationMode.NEAREST)
            else:
                field = F.resize(field, target_size, self.interpolation)

            if num_channels == 1:
                field = field.squeeze(dim=0)

            if is_hwc:
                field = field.permute(1, 2, 0)

            resized_fields.append(field)

        return resized_fields

    def __repr__(self):
        return self.__class__.__name__ + f'(scale_range={self.scale_range}, interpolation={self.interpolation})'


class ValCrop:
    """ Crop for validation."""
    def __init__(self, patch_size, fill_value=0):
        self.patch_size = patch_size
        self.fill_value = fill_value

    def __call__(self, x):
        val_height = int(math.ceil(x.shape[-2] / self.patch_size) * self.patch_size)
        val_width = int(math.ceil(x.shape[-1] / self.patch_size) * self.patch_size)
        pad_height = val_height - x.shape[-2]
        pad_width = val_width - x.shape[-1]

        return F.pad(x, padding=[0, 0, pad_width, pad_height], fill=self.fill_value)


# ======================== AFFINITY ============================
def create_position_feats(shape, yx_scale=None):
    cord_range = [range(s) for s in shape]
    mesh = np.array(np.meshgrid(*cord_range, indexing='ij'), dtype=np.float32)

    if yx_scale is not None:
        mesh = mesh * (1.0 / yx_scale)

    return mesh


def create_yxrgb(img, yx_scale=None, rgb_scale=None):
    mesh = create_position_feats(img.shape[-2:], yx_scale)

    if rgb_scale is not None:
        img = img * (1.0 / rgb_scale)

    feats = np.concatenate([mesh, img], axis=0)
    return feats


def create_yxd(depth, yx_scale=None, depth_scale=None):
    mesh = create_position_feats(depth.shape[-2:], yx_scale)

    if depth_scale is not None:
        depth = depth * (1.0 / depth_scale)

    feats = np.concatenate([mesh, np.expand_dims(depth, axis=0)], axis=0)
    return feats


def create_yxrgbd(img, depth, yx_scale=None, rgb_scale=None, depth_scale=None):
    mesh = create_position_feats(img.shape[-2:], yx_scale)

    if rgb_scale is not None:
        img = img.astype(np.float32) * (1.0 / rgb_scale)

    if depth_scale is not None:
        depth = depth * (1.0 / depth_scale)

    feats = np.concatenate([mesh, img, np.expand_dims(depth, axis=0)], axis=0)
    return feats


def compute_similarity(
        orig_img, inverse_depth, croppings=None, feature_type='RGBD',
        scale_factor=8, sigma_xy=80, sigma_rgb=13, sigma_depth=6000
):
    height, width, _ = orig_img.shape
    feature_size = (math.ceil(height/scale_factor), math.ceil(width/scale_factor))
    feature_numel = feature_size[0] * feature_size[1]

    # Resize image and depth
    zooming_factor = (feature_size[0]/height, feature_size[1]/width)
    img_reduced = nd.zoom(orig_img, zoom=zooming_factor+(1,), order=3).astype(np.float32).transpose((2, 0, 1))
    depth_reduced = nd.zoom(inverse_depth, zoom=zooming_factor, order=3).astype(np.float32)

    if feature_type == 'RGBD':
        # Extract RGBD features
        features = create_yxrgbd(
            img_reduced, depth_reduced,
            yx_scale=sigma_xy/scale_factor, rgb_scale=sigma_rgb, depth_scale=sigma_depth
        )
    elif feature_type == "RGB":
        # Extract RGB features
        features = create_yxrgb(img_reduced, yx_scale=sigma_xy / scale_factor, rgb_scale=sigma_rgb)
    elif feature_type == 'D':
        features = create_yxd(depth_reduced, yx_scale=sigma_xy / scale_factor, depth_scale=sigma_depth)
    else:
        raise ValueError(f'There is no such a feature type: {feature_type}')

    distance_matrix = pdist(np.reshape(features, (features.shape[0], feature_numel)).transpose(), metric='sqeuclidean')
    similarity_map = squareform(np.exp(-distance_matrix / 2.0))

    if croppings is not None:
        croppings_reduced = nd.zoom(croppings, zoom=zooming_factor, order=0)
        crop_tile = np.tile(np.reshape(croppings_reduced, (feature_numel, -1)), (1, feature_numel))
        crop_mask = crop_tile * crop_tile.transpose()
        similarity_map = similarity_map * crop_mask

    # show_sim(idx=100, fg_sim=similarity_map, feat_size=feature_size, im_reduced=img_reduced, depth=depth_reduced)
    show_similarity(img_reduced, similarity_map, feature_size, depth_reduced)
    return similarity_map


def show_similarity(img_reduced, sim, feat_size, depth):
    import cv2
    import matplotlib.pyplot as plt
    def show_map(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            sim_map = sim[y * feat_size[1] + x].reshape(feat_size[0], feat_size[1])
            plt.figure('Affinity Map'), plt.imshow(sim_map)
            plt.figure('Affinity Map2'), plt.imshow(sim_map > 0.1)

            plt.figure('Depth'), plt.imshow(depth)
            plt.show()

    cv2.namedWindow('image', 2)
    cv2.imshow('image', cv2.cvtColor(img_reduced.transpose(1, 2, 0).astype('uint8'), cv2.COLOR_RGB2BGR))
    cv2.setMouseCallback('image', show_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
