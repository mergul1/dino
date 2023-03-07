import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

import matplotlib
matplotlib.use(backend='Qt5Agg')


# create heatmap from mask on image
def show_cam_on_image(img, masks, alpha=0.5, save_path=None):
    heatmaps = [
        cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB) for mask in masks
    ]
    img_cams = [
        torch.tensor(cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)).permute(2, 0, 1) for heatmap in heatmaps
    ]
    grid = make_grid(img_cams).permute(1, 2, 0)
    if save_path:
        cv2.imwrite(str(save_path), cv2.cvtColor(grid.numpy(), cv2.COLOR_RGB2BGR))
    else:
        plt.imshow(grid)
        plt.xticks([]), plt.yticks([])
        plt.show(block=True)


def show_cams_on_image(img, masks, valid_classes=None, alpha=0.5, save_path=None):
    num_mask = len(masks)
    if valid_classes is not None:
        masks = [mask[0][valid_classes] for mask in masks]

    heatmaps = [
        [cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * mask_c), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)]
        for mask in masks for mask_c in mask
    ]
    img_cams = [
        torch.tensor(cv2.addWeighted(img, alpha, heatmap_c, 1 - alpha, 0)).permute(2, 0, 1)
        for heatmap in heatmaps for heatmap_c in heatmap
    ]
    grid = make_grid(img_cams, nrow=num_mask).permute(1, 2, 0)

    if save_path:
        cv2.imwrite(save_path, grid)
    else:
        plt.imshow(grid)
        plt.xticks([]), plt.yticks([])
        plt.show(block=True)


# def show_cam_on_image(img, mask, save_path):
#     img = np.float32(img) / 255.
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
#     heatmap = np.float32(heatmap) / 255.0
#     cam = heatmap + img
#     cam = cam / np.max(cam)
#     cam = np.uint8(255 * cam)
#     cv2.imwrite(save_path, cam)


def show_cam_list(class_cam, cam_list, height, width):
    plt.figure('class cam')
    plt.imshow(class_cam.cpu().reshape(height, width))

    ncols = 4 if len(cam_list) >= 4 else len(cam_list)
    nrows = len(cam_list) // ncols
    fig, axs = plt.subplots(nrows, ncols)
    for i in range(nrows):
        for j in range(ncols):
            cam = cam_list[i * ncols + j].cpu().reshape(height, width)
            if nrows == 1:
                if ncols > 1:
                    axs[j].imshow(cam)
                    axs[j].axes.xaxis.set_ticks([])
                    axs[j].axes.yaxis.set_ticks([])

                else:
                    axs.imshow(cam)
                    axs.axes.xaxis.set_ticks([])
                    axs.axes.yaxis.set_ticks([])

            else:
                axs[i, j].imshow(cam)
                axs[i, j].axes.xaxis.set_ticks([])
                axs[i, j].axes.yaxis.set_ticks([])
    plt.show(block=True)


def show_attentions(attns, num_patch_height, num_patch_width, title='Attentions'):
    num_heads, num_patches, num_patches = attns.shape
    attns = attns.detach().cpu()
    class_attn = attns[:, 0, 1:].reshape(-1, num_patch_height, num_patch_width)
    grid = make_grid(class_attn.unsqueeze(dim=1).expand(-1, 3, -1, -1), nrow=3, normalize=True, scale_each=True)

    plt.title(title)
    plt.imshow(grid.permute(1, 2, 0)), plt.show(block=True)


def show_affinity(img, affinity, feat_size):
    def show_map(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            sim_map = affinity[y * feat_size[1] + x].reshape(feat_size[0], feat_size[1])
            # cv2.namedWindow('Affinity Map', 2)
            # cv2.imshow('Affinity Map', sim_map)
            plt.imshow(sim_map), plt.show(block=True)

    if img is None:
        img = np.zeros((feat_size[0], feat_size[1]), dtype=np.uint8)

    cv2.namedWindow('Image', 2)
    cv2.imshow('Image', img.astype('uint8'))
    cv2.setMouseCallback('Image', show_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class ShowAttentions:
    def __init__(self, img, attns, scale_factor):
        self.img = img
        if len(attns.shape) == 4:
            self.attns = attns[0, :, 1:, 1:].detach().cpu()
        elif len(attns.shape) == 3:
            self.attns = attns[:, 1:, 1:].detach().cpu()

        self.scale_factor = scale_factor
        self.num_patch_height = math.ceil(img.shape[0] / scale_factor)
        self.num_patch_weight = math.ceil(img.shape[1] / scale_factor)
        self.num_heads, self.num_tokens, _ = self.attns.shape

        fig, ax = plt.subplots()
        ax.imshow(self.img)
        self.cid = fig.canvas.mpl_connect('button_press_event', self.pick_point)
        plt.show(block=True)

    def pick_point(self, event):
        orig_x, orig_y = int(event.xdata), int(event.ydata)
        print(f"orig_x: {orig_x} - orig_y:{orig_y}")

        x = int(event.xdata // self.scale_factor)
        y = int(event.ydata // self.scale_factor)
        print(f"x: {x} - y:{y}")

        nrows = 2 if self.num_heads > 2 else 1
        ncols = self.num_heads // nrows
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
        if nrows * ncols > 1:
            for idx, attn in enumerate(self.attns):
                head_attns = attn[y * self.num_patch_weight + x].reshape(self.num_patch_height, self.num_patch_weight)
                head_attns = F.interpolate(head_attns.unsqueeze(dim=0).unsqueeze(dim=0), size=self.img.shape[:2], mode='bilinear').squeeze(dim=0).squeeze(dim=0)
                attns = (head_attns - head_attns.min()) / (head_attns.max() - head_attns.min())

                axs[idx//ncols, idx % ncols].plot(event.xdata, event.ydata, color='red', marker='*', markersize=5)
                axs[idx//ncols, idx % ncols].imshow(self.img)
                axs[idx//ncols, idx % ncols].imshow(attns, alpha=0.5)
                axs[idx//ncols, idx % ncols].axes.xaxis.set_ticks([])
                axs[idx//ncols, idx % ncols].axes.yaxis.set_ticks([])
                axs[idx//ncols, idx % ncols].set_title(f'Head: {idx}')
        else:
            head_attns = self.attns[0, y * self.num_patch_weight + x].reshape(self.num_patch_height, self.num_patch_weight)
            head_attns = F.interpolate(head_attns.unsqueeze(dim=0).unsqueeze(dim=0), size=self.img.shape[:2], mode='bilinear').squeeze(dim=0).squeeze(dim=0)
            attns = (head_attns - head_attns.min()) / (head_attns.max() - head_attns.min())

            axs.plot(event.xdata, event.ydata, color='red', marker='*', markersize=5)
            axs.imshow(self.img)
            axs.imshow(attns, alpha=0.5)
            axs.axes.xaxis.set_ticks([])
            axs.axes.yaxis.set_ticks([])
            axs.set_title(f'Mean')

        plt.show(block=True)

