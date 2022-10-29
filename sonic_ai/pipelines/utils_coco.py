import contextlib
import io
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from mmdet.datasets.api_wrappers import COCOeval


def xyxy2xywh(bbox):
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0],
        _bbox[3] - _bbox[1],
    ]


def _proposal2json(results, img_ids, length):
    json_results = []
    for idx in range(length):
        img_id = img_ids[idx]
        bboxes = results[idx]
        for i in range(bboxes.shape[0]):
            data = dict()
            data['image_id'] = img_id
            data['bbox'] = xyxy2xywh(bboxes[i])
            data['score'] = float(bboxes[i][4])
            data['category_id'] = 1
            json_results.append(data)
    return json_results


def _det2json(results, img_ids, cat_ids, length):
    json_results = []
    for idx in range(length):
        img_id = img_ids[idx]
        result = results[idx]
        for label in range(len(result)):
            bboxes = result[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = cat_ids[label]
                json_results.append(data)
    return json_results


def _segm2json(results, img_ids, cat_ids, length):
    bbox_json_results = []
    segm_json_results = []
    for idx in range(length):
        img_id = img_ids[idx]
        det, seg = results[idx]
        for label in range(len(det)):
            # bbox results
            bboxes = det[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = cat_ids[label]
                bbox_json_results.append(data)

            # segm results
            # some detectors use different scores for bbox and mask
            if isinstance(seg, tuple):
                segms = seg[0][label]
                mask_score = seg[1][label]
            else:
                segms = seg[label]
                mask_score = [bbox[4] for bbox in bboxes]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(mask_score[i])
                data['category_id'] = cat_ids[label]
                if isinstance(segms[i]['counts'], bytes):
                    segms[i]['counts'] = segms[i]['counts'].decode()
                data['segmentation'] = segms[i]
                segm_json_results.append(data)
    return bbox_json_results, segm_json_results


def results2json(results, outfile_prefix, img_ids, cat_ids, length):
    result_files = dict()
    if isinstance(results[0], list):
        json_results = _det2json(results, img_ids, cat_ids, length)
        result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        result_files['proposal'] = f'{outfile_prefix}.bbox.json'
        mmcv.dump(json_results, result_files['bbox'])
    elif isinstance(results[0], tuple):
        json_results = _segm2json(results, img_ids, cat_ids, length)
        result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        result_files['proposal'] = f'{outfile_prefix}.bbox.json'
        result_files['segm'] = f'{outfile_prefix}.segm.json'
        mmcv.dump(json_results[0], result_files['bbox'])
        mmcv.dump(json_results[1], result_files['segm'])
    elif isinstance(results[0], np.ndarray):
        json_results = _proposal2json(results, img_ids, length)
        result_files['proposal'] = f'{outfile_prefix}.proposal.json'
        mmcv.dump(json_results, result_files['proposal'])
    else:
        raise TypeError('invalid type of results')
    return result_files


def fast_eval_recall(
        results, proposal_nums, iou_thrs, coco, img_ids, logger=None):
    gt_bboxes = []
    for i in range(len(img_ids)):
        ann_ids = coco.get_ann_ids(img_ids=img_ids[i])
        ann_info = coco.load_anns(ann_ids)
        if len(ann_info) == 0:
            gt_bboxes.append(np.zeros((0, 4)))
            continue
        bboxes = []
        for ann in ann_info:
            if ann.get('ignore', False) or ann['iscrowd']:
                continue
            x1, y1, w, h = ann['bbox']
            bboxes.append([x1, y1, x1 + w, y1 + h])
        bboxes = np.array(bboxes, dtype=np.float32)
        if bboxes.shape[0] == 0:
            bboxes = np.zeros((0, 4))
        gt_bboxes.append(bboxes)

    recalls = eval_recalls(
        gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
    ar = recalls.mean(axis=1)
    return ar


def format_results(results, img_ids, cat_ids, length, jsonfile_prefix=None):
    assert isinstance(results, list), 'results must be a list'
    assert len(results) == length, (format(len(results), length))

    if jsonfile_prefix is None:
        tmp_dir = tempfile.TemporaryDirectory()
        jsonfile_prefix = osp.join(tmp_dir.name, 'results')
    else:
        tmp_dir = None
    result_files = results2json(
        results, jsonfile_prefix, img_ids, cat_ids, length)
    return result_files, tmp_dir


def evaluate_det_segm(
        results,
        result_files,
        coco_gt,
        metrics,
        coco,
        cat_ids,
        img_ids,
        logger=None,
        classwise=False,
        proposal_nums=(100, 300, 1000),
        iou_thrs=None,
        metric_items=None):
    if iou_thrs is None:
        iou_thrs = np.linspace(
            .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    if metric_items is not None:
        if not isinstance(metric_items, list):
            metric_items = [metric_items]

    eval_results = OrderedDict()
    for metric in metrics:
        msg = f'Evaluating {metric}...'
        if logger is None:
            msg = '\n' + msg
        print_log(msg, logger=logger)

        if metric == 'proposal_fast':
            if isinstance(results[0], tuple):
                raise KeyError(
                    'proposal_fast is not supported for '
                    'instance segmentation result.')
            ar = fast_eval_recall(
                results,
                proposal_nums,
                coco,
                img_ids,
                iou_thrs,
                logger='silent')
            log_msg = []
            for i, num in enumerate(proposal_nums):
                eval_results[f'AR@{num}'] = ar[i]
                log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
            log_msg = ''.join(log_msg)
            print_log(log_msg, logger=logger)
            continue

        iou_type = 'bbox' if metric == 'proposal' else metric
        if metric not in result_files:
            raise KeyError(f'{metric} is not in results')
        try:
            predictions = mmcv.load(result_files[metric])
            if iou_type == 'segm':
                # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                # When evaluating mask AP, if the results contain bbox,
                # cocoapi will use the box area instead of the mask area
                # for calculating the instance area. Though the overall AP
                # is not affected, this leads to different
                # small/medium/large mask AP results.
                for x in predictions:
                    x.pop('bbox')
                warnings.simplefilter('once')
                warnings.warn(
                    'The key "bbox" is deleted for more accurate mask AP '
                    'of small/medium/large instances since v2.12.0. This '
                    'does not change the overall mAP calculation.',
                    UserWarning)
            coco_det = coco_gt.loadRes(predictions)
        except IndexError:
            print_log(
                'The testing results of the whole dataset is empty.',
                logger=logger,
                level=logging.ERROR)
            break

        cocoEval = COCOeval(coco_gt, coco_det, iou_type)
        cocoEval.params.catIds = cat_ids
        cocoEval.params.imgIds = img_ids
        cocoEval.params.maxDets = list(proposal_nums)
        cocoEval.params.iouThrs = iou_thrs
        # mapping of cocoEval.stats
        coco_metric_names = {
            'mAP': 0,
            'mAP_50': 1,
            'mAP_75': 2,
            'mAP_s': 3,
            'mAP_m': 4,
            'mAP_l': 5,
            'AR@100': 6,
            'AR@300': 7,
            'AR@1000': 8,
            'AR_s@1000': 9,
            'AR_m@1000': 10,
            'AR_l@1000': 11
        }
        if metric_items is not None:
            for metric_item in metric_items:
                if metric_item not in coco_metric_names:
                    raise KeyError(
                        f'metric item {metric_item} is not supported')

        if metric == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.evaluate()
            cocoEval.accumulate()

            # Save coco summarize print information to logger
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            print_log('\n' + redirect_string.getvalue(), logger=logger)

            if metric_items is None:
                metric_items = [
                    'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000', 'AR_m@1000',
                    'AR_l@1000'
                ]

            for item in metric_items:
                val = float(f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                eval_results[item] = val
        else:
            cocoEval.evaluate()
            cocoEval.accumulate()

            # Save coco summarize print information to logger
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            print_log('\n' + redirect_string.getvalue(), logger=logger)

            if classwise:  # Compute per-category AP
                # Compute per-category AP
                # from https://github.com/facebookresearch/detectron2/
                precisions = cocoEval.eval['precision']
                # precision: (iou, recall, cls, area range, max dets)
                assert len(cat_ids) == precisions.shape[2]

                results_per_category = []
                for idx, catId in enumerate(cat_ids):
                    # area range index 0: all area ranges
                    # max dets index -1: typically 100 per image
                    nm = coco.loadCats(catId)[0]
                    precision = precisions[:, :, idx, 0, -1]
                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float('nan')
                    results_per_category.append(
                        (f'{nm["name"]}', f'{float(ap):0.3f}'))

                num_columns = min(6, len(results_per_category) * 2)
                results_flatten = list(itertools.chain(*results_per_category))
                headers = ['category', 'AP'] * (num_columns // 2)
                results_2d = itertools.zip_longest(
                    *[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                table_data = [headers]
                table_data += [result for result in results_2d]
                table = AsciiTable(table_data)
                print_log('\n' + table.table, logger=logger)

            if metric_items is None:
                metric_items = [
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                ]

            for metric_item in metric_items:
                key = f'{metric}_{metric_item}'
                val = float(
                    f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}')
                eval_results[key] = val
            ap = cocoEval.stats[:6]
            eval_results[f'{metric}_mAP_copypaste'] = (
                f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                f'{ap[4]:.3f} {ap[5]:.3f}')

    return eval_results


def evaluate(
        results,
        coco,
        img_ids,
        cat_names,
        length,
        metric='bbox',
        logger=None,
        jsonfile_prefix=None,
        classwise=False,
        proposal_nums=(100, 300, 1000),
        iou_thrs=None,
        metric_items=None):
    metrics = metric if isinstance(metric, list) else [metric]
    allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
    for metric in metrics:
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')

    coco_gt = coco
    cat_ids = coco_gt.get_cat_ids(cat_names=cat_names)

    result_files, tmp_dir = format_results(
        results, img_ids, cat_ids, length, jsonfile_prefix)
    eval_results = evaluate_det_segm(
        results, result_files, coco_gt, metrics, coco, cat_ids, img_ids,
        logger, classwise, proposal_nums, iou_thrs, metric_items)

    if tmp_dir is not None:
        tmp_dir.cleanup()
    return eval_results
