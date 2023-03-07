import os
import json
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

import utils.misc

from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import eval_semantic_segmentation
from chainercv.datasets import voc_semantic_segmentation_label_names, voc_semantic_segmentation_label_colors

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, unary_from_softmax


def cam2label(cam, valid_classes, alpha=2, bg_thr=None):
    # Get foreground score map
    fg_score = np.take(cam, valid_classes, axis=0)
    num_valid_classes, height, width = fg_score.shape

    # Estimate the background score map from foreground score map
    if bg_thr is None:
        bg_score = np.power(1.0 - np.max(fg_score, axis=0, keepdims=True), alpha)
    else:
        bg_score = np.ones((1, height, width)) * bg_thr

    # Concatenate bg-fg score map
    raw_cam = np.concatenate((bg_score, cam), axis=0)

    # Normalize the channel-wise
    # raw_cam /= raw_cam.sum(axis=0, keepdims=True)

    # Predict the segmentation mask without CRF or any post-processing
    pred = np.argmax(raw_cam, axis=0)

    return raw_cam, pred


def crf_with_alpha(orig_img, cam, valid_classes, alpha=None, mask_guided=False):
    # Get foreground score map
    fg_score = np.take(cam, valid_classes, axis=0)
    _, height, width = fg_score.shape

    # Estimate the background score map from foreground score map
    bg_score = np.power(1.0 - np.max(fg_score, axis=0, keepdims=True), alpha)

    # Concatenate bg-fg score map
    bg_fg_score = np.concatenate((bg_score, fg_score), axis=0)

    # Normalize the channel-wise
    # bg_fg_score /= bg_fg_score.sum(axis=0, keepdims=True)

    # CRF inference
    if mask_guided:
        valid_crf_score = crf_inference_rgbd(orig_img, bg_fg_score, num_labels=bg_fg_score.shape[0])
    else:
        valid_crf_score = crf_inference(orig_img, bg_fg_score, num_labels=bg_fg_score.shape[0])

    # Convert the valid_crf into the crf_cam
    crf_cam = np.zeros_like(cam)
    crf_cam = np.concatenate((valid_crf_score[:1], crf_cam), axis=0)
    for idx, key in enumerate(valid_classes):
        crf_cam[key + 1] = valid_crf_score[idx + 1]

    return crf_cam


def load_file(file_path):
    # open a file and return the contents
    cam_dict = dict(np.load(file_path[0], allow_pickle=True))
    return cam_dict, file_path[1]


def read_files(path, dataset):
    # prepare all of paths
    paths = [(os.path.join(path, filepath + '.npz'), filepath)for filepath in dataset.ids]

    # create the thread pool
    with ThreadPoolExecutor(100) as executor:

        # submit all tasks
        futures = [executor.submit(load_file, p) for p in paths]

        pred_dict = dict()

        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            data, fname = future.result()
            pred_dict[fname] = data

            # report progress
            print(f'.loaded {fname}')
    print('Done')

    return pred_dict


def eval_cam(args):
    # Dataset
    dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
    labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]

    pred_dict = read_files(args.out_crf, dataset)

    # # Read the segmentation prediction results
    # preds = []
    # preds_alpha = []
    # preds_low_alpha = []
    # preds_high_alpha = []
    # preds_reliable = []
    # preds_refined = []
    # for idx, img_name in enumerate(dataset.ids):
    #     print(f'{idx}: {img_name}')
    #     cam_dict = np.load(args.out_crf.joinpath(img_name + '.npz'), allow_pickle=True)
    #
    #     preds.append(cam_dict['no_crf'])
    #     preds_alpha.append(cam_dict['crf'])
    #     preds_low_alpha.append(cam_dict['crf_la'])
    #     preds_high_alpha.append(cam_dict['crf_ha'])
    #     # preds_reliable.append(_reliable_pred(cam_dict['crf_la'], cam_dict['crf_ha']))
    #     # preds_refined.append(cam_dict['refined'])
    #
    # # Performance evaluation
    # prediction_types = {'no_crf': preds,
    #                     'crf': preds_alpha,
    #                     'low_alpha': preds_low_alpha,
    #                     'high_alpha': preds_high_alpha,
    #                     'reliable': preds_reliable,
    #                     'refined': preds_refined
    #                     }
    pred_types = dict()
    for key in pred_dict[list(pred_dict.keys())[0]].keys():
        pred_types[key] = [pred_dict[idx][key] for idx in dataset.ids]

    performance_dict = dict()
    miou_performance_dict = dict()
    for key, pred in pred_types.items():
        if key in ['valid_classes']:
            continue

        # Evaluate for 'key' type prediction
        performance = eval_semantic_segmentation(pred_labels=pred, gt_labels=labels)

        # Metrics for json serialization
        metrics_dict = dict()
        for metric_type, metric_value in performance.items():
            if metric_value.size > 1:
                metrics_dict[metric_type] = {
                    k: float(format(v * 100, '0.2f'))
                    for k, v in zip(voc_semantic_segmentation_label_names, metric_value)
                }
            else:
                metrics_dict[metric_type] = float(format(metric_value * 100, '0.2f'))
                if metric_type == 'miou':
                    miou_performance_dict[key] = float(format(metric_value * 100, '0.2f'))

        performance_dict[key] = metrics_dict

    with open(os.path.join(args.result_dir, f'detail_performance_{args.chainer_eval_set}.json'), 'w') as json_file:
        json.dump(performance_dict, json_file, indent=4)

    with open(os.path.join(args.result_dir, f'miou_performance_{args.chainer_eval_set}.json'), 'w') as json_file:
        json.dump(miou_performance_dict, json_file, indent=4)


# ############################################# CRF INFERENCE ##########################################################
def crf_inference(img, probs, t=10, scale_factor=1, num_labels=21, sigma_xy=40, sigma_rgb=13):
    height, width = img.shape[:2]

    d = dcrf.DenseCRF2D(width, height, num_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    img_c = np.ascontiguousarray(np.copy(img))

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=sigma_xy/scale_factor, srgb=sigma_rgb, rgbim=img_c, compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((num_labels, height, width))


def crf_inference_rgbd(img, probs, t=10, scale_factor=1, num_labels=21, sigma_xy=40, sigma_rgb=13, sigma_depth=15):
    height, width = img.shape[:2]

    d = dcrf.DenseCRF2D(width, height, num_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    img_c = np.ascontiguousarray(np.copy(img))
    pairwise_energy = create_pairwise_bilateral(sdims=(sigma_xy/scale_factor, sigma_xy/scale_factor),
                                                schan=(sigma_rgb, sigma_rgb, sigma_rgb, sigma_depth),
                                                img=img_c,
                                                chdim=2)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseEnergy(pairwise_energy, compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((num_labels, height, width))


def crf_inference_inf(img, probs, t=10, scale_factor=1, num_labels=21, sigma_xy=83, sigma_rgb=5):
    height, width = img.shape[:2]

    d = dcrf.DenseCRF2D(width, height, num_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    img_c = np.ascontiguousarray(np.copy(img))

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=sigma_xy/scale_factor, srgb=sigma_rgb, rgbim=img_c, compat=4)
    Q = d.inference(t)

    return np.array(Q).reshape((num_labels, height, width))


def crf_inference_label(img, labels, t=10, num_labels=21, gt_prob=0.7, sigma_xy=50, sigma_rgb=5):
    height, width = img.shape[:2]

    d = dcrf.DenseCRF2D(width, height, num_labels)

    unary = unary_from_labels(labels, num_labels, gt_prob=gt_prob, zero_unsure=False)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=sigma_xy, srgb=sigma_rgb, rgbim=np.ascontiguousarray(np.copy(img)), compat=10)

    Q = d.inference(t)

    return np.argmax(np.array(Q).reshape((num_labels, height, width)), axis=0)

