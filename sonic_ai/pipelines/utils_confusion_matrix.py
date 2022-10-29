import os
import platform

import matplotlib.pyplot as plt
import mmcv
import numpy as np
from PIL import ImageFont
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator
from mmcv.ops import nms

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps


def get_font(size=12):
    font_path = ''
    if platform.system() == 'Windows':
        font_path = 'simhei.ttf'
    elif platform.system() == 'Linux':
        font_path = 'NotoSansCJK-Regular.ttc'
    elif platform.system() == 'Darwin':
        font_path = 'Hiragino Sans GB.ttc'
    font = ImageFont.truetype(font_path, size)
    font = FontProperties(fname=font.path, size=size)
    return font


def calculate_confusion_matrix(
        num_classes,
        results,
        ann_infos,
        score_thr=0,
        nms_iou_thr=None,
        tp_iou_thr=0.5):
    confusion_matrix = np.zeros(shape=[num_classes + 1, num_classes + 1])
    prog_bar = mmcv.ProgressBar(len(results))
    for idx, per_img_res in enumerate(results):
        if isinstance(per_img_res, tuple):
            res_bboxes, _ = per_img_res
        else:
            res_bboxes = per_img_res
        ann = ann_infos[idx]
        gt_bboxes = ann['bboxes']
        labels = ann['labels']
        analyze_per_img_dets(
            confusion_matrix, gt_bboxes, labels, res_bboxes, score_thr,
            tp_iou_thr, nms_iou_thr)
        prog_bar.update()
    return confusion_matrix


def analyze_per_img_dets(
        confusion_matrix,
        gt_bboxes,
        gt_labels,
        result,
        score_thr=0,
        tp_iou_thr=0.5,
        nms_iou_thr=None):
    true_positives = np.zeros_like(gt_labels)
    for det_label, det_bboxes in enumerate(result):
        if nms_iou_thr:
            det_bboxes, _ = nms(
                det_bboxes[:, :4],
                det_bboxes[:, -1],
                nms_iou_thr,
                score_threshold=score_thr)
        ious = bbox_overlaps(det_bboxes[:, :4], gt_bboxes)
        for i, det_bbox in enumerate(det_bboxes):
            score = det_bbox[4]
            det_match = 0
            if score >= score_thr:
                for j, gt_label in enumerate(gt_labels):
                    if ious[i, j] >= tp_iou_thr:
                        det_match += 1
                        if gt_label == det_label:
                            true_positives[j] += 1  # TP
                        confusion_matrix[gt_label, det_label] += 1
                if det_match == 0:  # BG FP
                    confusion_matrix[-1, det_label] += 1
    for num_tp, gt_label in zip(true_positives, gt_labels):
        if num_tp == 0:  # FN
            confusion_matrix[gt_label, -1] += 1


def plot_confusion_matrix(
        confusion_matrix,
        labels,
        save_dir=None,
        title='Normalized Confusion Matrix',
        color_theme='plasma'):
    # normalize the confusion matrix
    per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
    confusion_matrix = \
        confusion_matrix.astype(np.float32) / per_label_sums * 100

    font = get_font(size=12)
    num_classes = len(labels)
    fig, ax = plt.subplots(
        figsize=(1.5 * num_classes, 1.5 * num_classes * 0.8), dpi=180)
    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(confusion_matrix, cmap=cmap)
    plt.colorbar(mappable=im, ax=ax)

    title_font = {'weight': 'bold', 'size': 12}
    ax.set_title(title, fontdict=title_font)
    label_font = {'size': 10}
    plt.ylabel('Ground Truth Label', fontdict=label_font)
    plt.xlabel('Prediction Label', fontdict=label_font)

    # draw locator
    xmajor_locator = MultipleLocator(1)
    xminor_locator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(xmajor_locator)
    ax.xaxis.set_minor_locator(xminor_locator)
    ymajor_locator = MultipleLocator(1)
    yminor_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(ymajor_locator)
    ax.yaxis.set_minor_locator(yminor_locator)

    # draw grid
    ax.grid(True, which='minor', linestyle='-')

    plt.xticks(np.arange(num_classes), labels, fontproperties=font)
    plt.yticks(np.arange(num_classes), labels, fontproperties=font)

    ax.tick_params(
        axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

    # draw confution matrix value
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                '{}%'.format(
                    int(confusion_matrix[
                        i,
                        j]) if not np.isnan(confusion_matrix[i, j]) else -1),
                ha='center',
                va='center',
                color='w',
                size=7)

    ax.set_ylim(len(confusion_matrix) - 0.5, -0.5)  # matplotlib>3.1.1

    fig.tight_layout()
    if save_dir is not None:
        plt.savefig(
            os.path.join(save_dir, 'confusion_matrix.png'), format='png')
