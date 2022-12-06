import random
import math
import PIL.Image
import os.path
import numpy as np
import argparse
from collections.abc import Sequence
import numbers

import cv2
import matplotlib.pyplot as plt
from PIL import Image
import scipy.ndimage as nd

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F

import utils

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
            'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

CAT_NAME_TO_NUM = dict(zip(CAT_LIST, range(len(CAT_LIST))))

IMG_FOLDER_NAME = "JPEGImages"
ANNOT_FOLDER_NAME = "Annotations"
DEPTH_FOLDER_NAME = "Depth"


def load_image_label_from_xml(img_name, voc12_root):
    from xml.dom import minidom

    el_list = minidom.parse(os.path.join(voc12_root, ANNOT_FOLDER_NAME, img_name + '.xml')).getElementsByTagName('name')

    multi_cls_lab = np.zeros(20, np.float32)

    for el in el_list:
        cat_name = el.firstChild.data
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab


def load_image_label_list_from_xml(img_name_list, voc12_root):
    return [load_image_label_from_xml(img_name, voc12_root) for img_name in img_name_list]


def load_image_label_list_from_npy(img_name_list):
    cls_labels_dict = np.load('voc12/cls_labels.npy', allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]


def get_img_path(img_name, voc12_root):
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.jpg')


def get_depth_path(img_name, voc12_root):
    pfm_file_path = os.path.join(voc12_root, DEPTH_FOLDER_NAME, 'midas_v3.0', img_name + '.pfm')
    png_file_path = os.path.join(voc12_root, DEPTH_FOLDER_NAME, 'midas_v3.0', img_name + '.png')
    return pfm_file_path, png_file_path


def load_img_name_list(dataset_path):
    img_gt_name_list = open(dataset_path).read().splitlines()
    img_name_list = [img_gt_name.split(' ')[0][-15:-4] for img_gt_name in img_gt_name_list]
    return img_name_list


class VOC12ImageDataset(Dataset):
    def __init__(self, img_name_list_path, voc12_root, transform=None, transform2=None):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.transform = transform
        self.transform2 = transform2
        self.croppings = 0
        self.label_list = load_image_label_list_from_npy(self.img_name_list)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        # Get the name of images
        name = self.img_name_list[idx]

        # Read image as PIL
        img = PIL.Image.open(get_img_path(name, self.voc12_root)).convert("RGB")

        # Realize the transform ops
        if self.transform:
            img = self.transform(img)

        label = torch.from_numpy(self.label_list[idx])

        return name, img, label


class VOC2012ClsDepthDataset(Dataset):
    def __init__(
            self, image_name_list_path, voc12_root, transform=None, transform2=None, scale_factor=8, mask_guided=False,
            color_included=False
    ):
        self.img_name_list = load_img_name_list(image_name_list_path)
        self.voc12_root = voc12_root
        self.transform = transform
        self.transform2 = transform2
        self.scale_factor = scale_factor
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        self.mask_guided = mask_guided
        self.color_included = color_included

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        # Get the name of images
        name = self.img_name_list[idx]

        # Get classification labels (one-hot encoding)
        label = torch.from_numpy(self.label_list[idx])

        # Read image as PIL
        img = PIL.Image.open(get_img_path(name, self.voc12_root)).convert("RGB")
        orig_img = torch.from_numpy(np.array(img))
        cropping_map = torch.from_numpy(np.ones_like(orig_img)[:, :, 0])

        if self.transform:
            img = self.transform(img)

        if self.mask_guided:
            # Read inverse depth image
            inverse_depth_float, _ = utils.read_pfm(get_depth_path(name, self.voc12_root)[0])
            inverse_depth = torch.from_numpy(cv2.imread(get_depth_path(name, self.voc12_root)[1], cv2.IMREAD_GRAYSCALE))

            # th = 10.0
            # inverse_depth[inverse_depth <= th] = th
            # depth = (1 / inverse_depth * 255.0).astype('uint8')
            # guidance = np.concatenate((orig_img, np.expand_dims(depth, axis=-1)), axis=-1)

            # plt.figure('inverse_depth'), plt.imshow(inverse_depth), plt.show()
            # plt.figure('image'), plt.imshow(orig_img), plt.show()

            input_list = [img, orig_img, cropping_map, inverse_depth]
        else:
            input_list = [img, orig_img, cropping_map]

        if self.transform2:
            input_list = self.transform2(input_list)

        if self.mask_guided:
            if self.color_included:
                sim_matrix = compute_similarity(
                    input_list[1], input_list[3], croppings=input_list[2], scale_factor=self.scale_factor
                )
            else:
                sim_matrix = compute_similarity_from_depth(
                    inverse_depth=input_list[3], orig_img=input_list[1], croppings=input_list[2],
                    scale_factor=self.scale_factor
                )

            sim_matrix = torch.from_numpy(sim_matrix)
            return {
                'name': name,
                'img': input_list[0],
                'label': label,
                'affine_matrices': sim_matrix,
                'orig_img': input_list[1]
            }
        else:
            return {
                'name': name,
                'img': input_list[0],
                'label': label,
                'orig_img': input_list[1]
            }


class VOC12ClsDatasetMSF(VOC12ImageDataset):
    def __init__(self, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1):
        super().__init__(img_name_list_path, voc12_root, transform=None)
        self.inter_transform = inter_transform
        self.scales = scales
        self.unit = unit

    def __getitem__(self, idx):
        name, img, label = super().__getitem__(idx)
        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        ms_img_list = []
        for s in self.scales:
            if s == 1.0:
                s_img = img
            else:
                target_size = (round(rounded_size[0]*s), round(rounded_size[1]*s))
                s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append((ms_img_list[i].flip(-1)))

        return {'name': name, 'img_list': msf_img_list, 'label': label}


class VOC12ClsDepthDatasetMSF(VOC12ImageDataset):
    def __init__(
            self, img_name_list_path, voc12_root, scales, inter_transform=None, depth_transform=None, unit=1,
            mask_guided=True, patch_size=16
    ):
        super().__init__(img_name_list_path, voc12_root, transform=None)
        self.inter_transform = inter_transform
        self.depth_transform = depth_transform
        self.scales = scales
        self.unit = unit
        self.mask_guided = mask_guided
        self.patch_size = patch_size

    def __getitem__(self, idx):
        name, img, label = super().__getitem__(idx)
        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        # Read inverse depth image
        inverse_depth_float, _ = utils.read_pfm(get_depth_path(name, self.voc12_root)[0])
        inverse_depth = cv2.imread(get_depth_path(name, self.voc12_root)[1], cv2.IMREAD_GRAYSCALE)

        ms_orig_img_list = []
        ms_depth_list = []
        ms_img_list = []

        for s in self.scales:
            if s == 1.0:
                s_orig_img = img
                s_depth = inverse_depth
                s_depth = self.depth_transform(s_depth)
                s_orig_img = self.depth_transform(s_orig_img)
            else:
                target_size = (round(rounded_size[0]*s), round(rounded_size[1]*s))
                s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
                s_depth = torch.tensor(cv2.resize(inverse_depth, target_size, cv2.INTER_CUBIC))
                s_depth = self.depth_transform(s_depth)
                s_orig_img = self.depth_transform(torch.from_numpy(np.array(s_img).transpose((2, 0, 1)))).permute(1, 2, 0)
            ms_orig_img_list.append(s_orig_img)
            ms_depth_list.append(s_depth)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        msf_img_list = []
        msf_depth_list = []
        msf_orig_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(ms_img_list[i].flip(-1))

            msf_depth_list.append(ms_depth_list[i])
            msf_depth_list.append(ms_depth_list[i].flip(-1))

            msf_orig_img_list.append(ms_orig_img_list[i])
            msf_orig_img_list.append(ms_orig_img_list[i].flip(2))

        msf_sim_matrices = []
        if self.mask_guided:
            for i, img in enumerate(msf_orig_img_list):
                msf_sim_matrices.append(
                    torch.from_numpy(compute_similarity_from_depth(msf_depth_list[i], img, scale_factor=self.patch_size)))

        return {'name': name, 'img_list': msf_img_list, 'mask_list': msf_sim_matrices, 'label': label}


def show_sim(idx, fg_sim, feat_size, im_reduced, depth):
    plt.figure('im'), plt.imshow(im_reduced.transpose(1, 2, 0).astype('uint8'))
    # plt.figure('idepth_float'), plt.imshow(cv2.resize(idepth_float, (feat_size[1], feat_size[0])))
    plt.figure('depth'), plt.imshow(cv2.resize(depth, (feat_size[1], feat_size[0])))

    fg = np.reshape(fg_sim[idx], (feat_size[0], feat_size[1]))
    plt.figure('sim')
    plt.imshow(fg)
    plt.plot(idx % feat_size[1], idx // feat_size[0], 'o', color=(1.0, 0.0, 0.0), alpha=0.8)
    plt.show()


def show_similarity(img_reduced, sim, feat_size, depth):
    def show_map(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            sim_map = sim[y * feat_size[1] + x].reshape(feat_size[0], feat_size[1])
            cv2.namedWindow('map', 2)
            cv2.imshow('map', sim_map)

            cv2.namedWindow('depth', 2)
            cv2.imshow('depth', depth)

    cv2.namedWindow('image', 2)
    cv2.imshow('image', cv2.cvtColor(img_reduced.transpose(1, 2, 0).astype('uint8'), cv2.COLOR_RGB2BGR))
    cv2.setMouseCallback('image', show_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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


def create_yxrgbd(img, depth, yx_scale=None, rgb_scale=None, depth_scale=None):
    mesh = create_position_feats(img.shape[-2:], yx_scale)

    if rgb_scale is not None:
        img = img.astype(np.float32) * (1.0 / rgb_scale)

    if depth_scale is not None:
        depth = depth * (1.0 / depth_scale)

    feats = np.concatenate([mesh, img, np.expand_dims(depth, axis=0)], axis=0)
    return feats


def compute_similarity_from_depth(inverse_depth, orig_img=None, croppings=None, scale_factor=8, sigma_depth=20):
    height, width = inverse_depth.shape
    feature_size = (math.ceil(height/scale_factor), math.ceil(width/scale_factor))
    feature_numel = feature_size[0] * feature_size[1]

    img_reduced = nd.zoom(orig_img, zoom=(feature_size[0]/height, feature_size[1]/width, 1), order=3).astype(np.float32).transpose((2, 0, 1))
    depth_reduced = nd.zoom(inverse_depth, zoom=(feature_size[0]/height, feature_size[1]/width), order=3).astype(np.float32)

    depth_feature = depth_reduced * (1.0 / sigma_depth)
    feature_matrix = np.tile(np.reshape(depth_feature, (feature_numel, -1)), (1, feature_numel))
    feature_matrix_t = np.transpose(feature_matrix, (1, 0))
    similarity_map = np.exp(-(feature_matrix - feature_matrix_t) ** 2 / 2)
    if croppings is not None:
        croppings = nd.zoom(croppings, zoom=(feature_size[0] / height, feature_size[1] / width), order=0)
        crop_tile = np.tile(np.reshape(croppings, (feature_numel, -1)), (1, feature_numel))
        similarity_map = similarity_map * crop_tile

    # show_sim(20 * 40 + 25, similarity_map, feature_size, img_reduced, depth_reduced, depth_reduced)
    # show_similarity(img_reduced, similarity_map, feature_size, depth_reduced/255.0)

    return similarity_map


def compute_similarity(
        orig_img, inverse_depth, croppings=None, scale_factor=8, sigma_xy=80, sigma_rgb=13, sigma_depth=20
):
    height, width, _ = orig_img.shape
    feature_size = (round(height/scale_factor + 0.5), round(width/scale_factor + 0.5))
    feature_numel = feature_size[0] * feature_size[1]

    img_reduced = nd.zoom(orig_img, zoom=(feature_size[0]/height, feature_size[1]/width, 1), order=3).astype(np.float32).transpose((2, 0, 1))
    depth_reduced = nd.zoom(inverse_depth, zoom=(feature_size[0]/height, feature_size[1]/width), order=3).astype(np.float32)

    features = create_yxrgbd(img_reduced, depth_reduced, yx_scale=sigma_xy/scale_factor, rgb_scale=sigma_rgb, depth_scale=sigma_depth)
    feature_matrix = np.tile(np.reshape(features, (features.shape[0], feature_numel, -1)), (1, 1, feature_numel))
    feature_matrix_t = np.transpose(feature_matrix, (0, 2, 1))
    similarity_map = np.exp(-np.sum((feature_matrix - feature_matrix_t) ** 2, axis=0) / 2)
    if croppings is not None:
        croppings = nd.zoom(croppings, zoom=(feature_size[0]/height, feature_size[1]/width), order=0)
        crop_tile = np.tile(np.reshape(croppings, (feature_numel, -1)), (1, feature_numel))
        similarity_map = similarity_map * crop_tile

    # show_sim(idx=100, fg_sim=similarity_map, feat_size=feature_size, im_reduced=img_reduced, depth=depth_reduced)
    # show_similarity(img_reduced, similarity_map, feature_size, depth_reduced/255.0)
    return similarity_map


class ValCrop:
    """Crop for validation."""

    def __init__(self, patch_size, fill_value=0):
        self.patch_size = patch_size
        self.fill_value = fill_value

    def __call__(self, x):
        val_height = int(math.ceil(x.shape[-2] / self.patch_size) * self.patch_size)
        val_width = int(math.ceil(x.shape[-1] / self.patch_size) * self.patch_size)
        pad_height = val_height - x.shape[-2]
        pad_width = val_width - x.shape[-1]

        return F.pad(x, padding=[0, 0, pad_width, pad_height], fill=self.fill_value)


class ComposeMultiField:
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        transforms.Compose([
            transforms.CenterCrop(10),
            transforms.ToTensor()
            ])

    .. note::
        In order to script the transformations, please use ``torch.nn.Sequential`` as below.
        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, field_list):
        for t in self.transforms:
            field_list = t(field_list)
        return field_list

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomHorizontalFlipMultiField(nn.Module):
    """ Horizontally flip the given image and the depth image randomly with a given probability.
    If the image and the depth image is torch Tensor, it is expected to have [..., H, W] shape,
    where ... means an arbitrary number of leading dimensions.

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
                is_hwc = num_channels == 3 and not is_channel_first
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
    """Crop the given images at a random location.
    If the images is torch Tensor, it is expected to have [..., H, W] shape,
    where ... means an arbitrary number of leading dimensions, but if non-constant padding is used,
    the input is expected to have at most 2 leading dimensions

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
            is_hwc = num_channels == 3 and not is_channel_first
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
            is_hwc = num_channels == 3 and not is_channel_first
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
    """Resize the input images to the given size.
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

    def __init__(self, size, interpolation=F.InterpolationMode.BILINEAR):
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
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


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
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(scale_range={0},interpolation={1})'.format(self.scale_range, interpolate_str)


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return size, size

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', add_help=False)
    # Dataset
    parser.add_argument("--voc12_root", default='/home/mergul/Desktop/VOC2012', type=str)
    parser.add_argument("--num_classes", default=20, type=int)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)

    parser.add_argument("--crop_size", default=480, type=int)
    parser.add_argument("--base_size", default=512, type=int)

    args = parser.parse_args()

    dataset = VOC2012ClsDepthDataset(args.val_list, voc12_root=args.voc12_root,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                     ]),
                                     transform2=ComposeMultiField([
                                         RandomScaleMultiField(scale_range=(0.5, 1.5)),
                                         # ResizeMultiField(500),
                                         RandomCropMultiField(crop_size=args.crop_size, pad_if_needed=True),
                                         RandomHorizontalFlipMultiField(),
                                     ]))

    # transform = transforms.Compose([
    #     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    #     np.asarray,
    #     imutils.Normalize(),
    #     imutils.HWC2CHW()
    # ]),
    # transform2 = imutils.ComposeDepth([
    #     # imutils.RandomResizeLongDepth(256, 512),
    #     imutils.RandomScaleDepth(0.5, 1.5),
    #     imutils.RandomCropDepth(args.crop_size),
    #     imutils.RandomHorizontalFlipDepth()
    # ])

    dataset.__getitem__(100)
