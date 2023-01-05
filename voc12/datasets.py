import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from voc12 import augmentations

import matplotlib
matplotlib.use(backend='Qt5Agg')

IMG_FOLDER_NAME = "JPEGImages"
ANNOT_FOLDER_NAME = "Annotations"
DEPTH_FOLDER_NAME = "Depth"

CAT_LIST = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
CAT_NAME_TO_NUM = dict(zip(CAT_LIST, range(len(CAT_LIST))))


def get_depth_path(img_name, voc12_root):
    pfm_file_path = os.path.join(voc12_root, 'Depth', 'midas_v3.0', 'pfm', img_name + '.pfm')
    png_file_path = os.path.join(voc12_root, 'Depth', 'midas_v3.0', 'png', img_name + '.png')
    return pfm_file_path, png_file_path


def get_img_path(img_name, voc12_root):
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.jpg')


def load_image_label_list_from_npy(img_name_list, cls_gt_path):
    cls_labels_dict = np.load(cls_gt_path, allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]


def load_img_name_list(dataset_path):
    img_gt_name_list = open(dataset_path).read().splitlines()
    img_name_list = [img_gt_name.split(' ')[0][-15:-4] for img_gt_name in img_gt_name_list]
    return img_name_list


class VOC12ImageDataset(Dataset):
    def __init__(self, img_name_list_path, voc12_root, transform=None, cls_gt_path=None):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.transform = transform
        self.cls_gt_path = cls_gt_path if cls_gt_path else 'voc12/cls_labels.npy'
        self.label_list = load_image_label_list_from_npy(self.img_name_list, self.cls_gt_path)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        # Get the name of images
        name = self.img_name_list[idx]

        # Read image as PIL
        img = Image.open(get_img_path(name, self.voc12_root)).convert("RGB")

        # Realize the transform ops
        if self.transform:
            img = self.transform(img)

        # Classification label with multi-hot encoding
        class_label = torch.from_numpy(self.label_list[idx])

        return {'name': name, 'image': img, 'class_label': class_label}


class VOC12ImageDatasetMSF(Dataset):
    def __init__(self, img_name_list_path, voc12_root, scales=(1.,), is_flip=True, cls_gt_path=None, transform=None):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.scales = scales
        self.is_flip = is_flip
        self.transform = transform
        self.cls_gt_path = cls_gt_path if cls_gt_path else 'voc12/cls_labels.npy'
        self.label_list = load_image_label_list_from_npy(self.img_name_list, self.cls_gt_path)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        # Get the name of images
        name = self.img_name_list[idx]

        # Read image as PIL
        img = Image.open(get_img_path(name, self.voc12_root)).convert("RGB")
        width, height = img.size
        orig_img = np.array(img)

        if width > 400 and height > 400:
            (new_width, new_height) = (int(width * 0.75), int(height * 0.75))
            img = img.resize((new_width, new_height))
            print(f'{name} is resized to {(new_width, new_height)}')

        # Multi-scale and flip
        multi_scales_flipped_images = []
        for scale in self.scales:
            if scale == 1.0:
                scaled_image = img
            else:
                target_size = (round(width * scale), round(height * scale))
                scaled_image = img.resize(size=target_size, resample=Image.CUBIC)
            multi_scales_flipped_images.append(scaled_image)

            # Flip, if necessary
            if self.is_flip:
                multi_scales_flipped_images.append(scaled_image.transpose(Image.FLIP_LEFT_RIGHT))

        # Classification label with multi-hot encoding
        class_label = torch.from_numpy(self.label_list[idx])

        # Realize the transform ops
        if self.transform:
            multi_scales_flipped_images = [self.transform(img) for img in multi_scales_flipped_images]

        return {'name': name, 'image_list': multi_scales_flipped_images, 'class_label': class_label, 'orig_img': orig_img}


class VOC12ClsDepthDataset(Dataset):
    def __init__(
            self, image_name_list_path, voc12_root, transform=None, transform2=None, cls_gt_path=None,
            mask_guided=True, scale_factor=8, feature_type='RGBD', sigma_xy=None, sigma_rgb=13, sigma_depth=6000
    ):
        self.img_name_list = load_img_name_list(image_name_list_path)
        self.voc12_root = voc12_root
        self.transform = transform
        self.transform2 = transform2

        self.cls_gt_path = cls_gt_path if cls_gt_path else 'voc12/cls_labels.npy'
        self.label_list = load_image_label_list_from_npy(self.img_name_list, self.cls_gt_path)

        self.mask_guided = mask_guided
        self.scale_factor = scale_factor
        self.feature_type = feature_type
        self.sigma_xy = sigma_xy
        self.sigma_rgb = sigma_rgb
        self.sigma_depth = sigma_depth

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        # Get the name of images
        name = self.img_name_list[idx]

        # Get classification labels (one-hot encoding)
        class_label = torch.from_numpy(self.label_list[idx])

        # Read image as PIL
        img = Image.open(get_img_path(name, self.voc12_root)).convert('RGB')
        orig_img = torch.from_numpy(np.asarray(img))
        orig_size = orig_img.shape

        # Create cropping mask
        cropping_mask = torch.from_numpy(np.ones_like(orig_img)[:, :, 0])

        if self.transform:
            img = self.transform(img)

        if self.mask_guided:
            # Read inverse depth maps from png file (16 bit)
            inverse_depth = Image.open(get_depth_path(name, self.voc12_root)[1])
            inverse_depth = torch.from_numpy(np.array(inverse_depth))
            # plt.figure('inverse_depth'), plt.imshow(inverse_depth)

            input_list = [img, orig_img, cropping_mask, inverse_depth]
        else:
            input_list = [img, orig_img, cropping_mask]

        if self.transform2:
            input_list = self.transform2(input_list)

        if self.mask_guided:
            sigma_xy = (orig_size[0] + orig_size[1])/2 if self.sigma_xy is None else self.sigma_xy
            affinity = augmentations.compute_similarity(
                orig_img=input_list[1], inverse_depth=input_list[3], croppings=input_list[2],
                feature_type=self.feature_type, scale_factor=self.scale_factor,
                sigma_xy=sigma_xy, sigma_rgb=self.sigma_rgb, sigma_depth=self.sigma_depth
            )
            return {
                'name': name,
                'img': input_list[0],
                'class_label': class_label,
                'orig_img': input_list[1],
                'affinity': affinity
            }
        else:
            return {'name': name, 'img': input_list[0], 'class_label': class_label, 'orig_img': input_list[1]}


if __name__ == '__main__':
    print(plt.get_backend())
    train_list = 'train_aug.txt'
    voc12 = '/home/mergul/Desktop/VOC2012'
    gt_path = 'cls_labels.npy'
    crop_size = 480
    # transforms = transforms.Compose([
    #     # transforms.CenterCrop(512),
    #     transforms.ToTensor(),
    #     augmentations.ValCrop(patch_size=16),
    #     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    # ])
    # dataset = VOC12ImageDatasetMSF(train_list, voc12, scales=(1.0, 1.5), cls_gt_path=gt_path, transform=transforms)
    # dataset = VOC12ClsDepthDataset(train_list, voc12, cls_gt_path=gt_path, transform=transforms)
    # data = dataset.__getitem__(50)
    # print(data['image_list'][0].shape)
    # print(data['image_list'][2].shape)
    # print(data['image_list'][0].mean((1, 2)))
    # print(data['image_list'][2].mean((1, 2)))

    depth_dataset = VOC12ClsDepthDataset(train_list, voc12_root=voc12, cls_gt_path=gt_path,
                                         scale_factor=4, feature_type='RGBD', sigma_xy=None, sigma_rgb=130, sigma_depth=6500,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                         ]),
                                         transform2=augmentations.ComposeMultiField([
                                             # augmentations.RandomScaleMultiField(scale_range=(0.8, 1.2)),
                                             # ResizeMultiField(500),
                                             augmentations.RandomCropMultiField(crop_size=crop_size, pad_if_needed=True),
                                             augmentations.RandomHorizontalFlipMultiField()
                                         ])
                                         )

    data = depth_dataset.__getitem__(9978)

