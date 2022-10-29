import logging
import os
import traceback
from pathlib import Path

import numpy as np
from torch import distributed as dist

from mmdet.datasets.builder import PIPELINES
from mmdet.utils.logger import get_root_logger
from sonic_ai.pipelines.utils_confusion_matrix import calculate_confusion_matrix, plot_confusion_matrix
from .utils_coco import evaluate
from .utils_labelme import copy_json_and_img, bb_intersection_over_union


@PIPELINES.register_module()
class CocoEvaluate:

    def __init__(self, metric='bbox', classwise=False, iou_thrs=None):
        self.metric = metric
        self.classwise = classwise
        self.iou_thrs = iou_thrs

    def __call__(self, results, *args, **kwargs):
        pred_results = results['pred_results']
        coco = results['coco']
        category_list = results['category_list']
        img_ids = results['img_ids']
        length = results['length']

        logger = get_root_logger()
        logger.propagate = False

        results['eval_results'] = evaluate(
            pred_results,
            coco,
            img_ids,
            category_list,
            length,
            logger=logger,
            metric=self.metric,
            classwise=self.classwise,
            iou_thrs=self.iou_thrs)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class ShowScores:

    def __init__(self, threshold=0.5, *args, **kwargs):
        self.threshold = threshold

    def __call__(self, results, *args, **kwargs):
        pred_results = results['pred_results']
        category_list = results['category_list']
        category_counter = results['category_counter']

        logger = get_root_logger()
        logger.propagate = False
        try:
            if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
                s = '\n'
                for cat_id, cat_name in enumerate(category_list):
                    arr = np.concatenate(
                        [x[cat_id] if not isinstance(x, tuple) else x[0][cat_id] for x in pred_results])
                    scores = arr[:, -1]
                    score_thresholds = [0.1, 0.3, 0.5]
                    scores_iou = [x.mean() if len(x) else 0 for x in [scores[scores > y] for y in score_thresholds]]
                    for i, score_threshold in enumerate(score_thresholds):
                        s += f'>{score_threshold}: {scores_iou[i]:.2f}, '
                    s += f"【{cat_name}】 true:{category_counter.get(cat_name, 0)}, " \
                         f"pred:{len(scores[scores > self.threshold])}\n"
                if s[-1] == '\n':
                    s = s[:-1]
                logger.info(f'统计置信度分布，threshold = {self.threshold}\n'
                            f'{s}')
        except:
            logger.warning(f'计算置信度分布失败：\n{traceback.format_exc()}')

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class CopyErrorCases:

    def __init__(self, threshold=0.5, shuffle=True, seed=42, max_copy_num=100):
        self.shuffle = shuffle
        self.seed = seed
        self.max_copy_num = max_copy_num
        self.threshold = threshold

        self.escape_path_list = []
        self.error_path_list = []
        self.overkill_path_list = []

    def __call__(self, results, *args, **kwargs):
        pred_results = results['pred_results']
        json_data_list = results['json_data_list']
        self.category_list = results['category_list']
        self.category_map = results['category_map']
        self.copy_pred_bad_case_path = results['copy_pred_bad_case_path']
        self.coco = results['coco']
        self.right_labels = results['right_labels']
        self.timestamp = results['timestamp']

        escape_path_list = []
        error_path_list = []
        overkill_path_list = []

        logger = get_root_logger()
        logger.propagate = False

        if isinstance(pred_results[0], tuple):
            for idx, value in enumerate(pred_results):
                pred_results[idx] = value[0]

        assert len(pred_results) == len(json_data_list)

        for idx, pred_result in enumerate(pred_results):
            true_result = self.coco.loadAnns(self.coco.getAnnIds([idx]))
            excape_flag, error_flag, overkill_flag = self.match_label(true_result, pred_result)

            json_path = json_data_list[idx]['json_path']
            if excape_flag:
                escape_path_list.append(json_path)
                continue
            if error_flag:
                error_path_list.append(json_path)
                continue
            if overkill_flag:
                overkill_path_list.append(json_path)
                continue

        self.escape_path_list = escape_path_list
        self.error_path_list = error_path_list
        self.overkill_path_list = overkill_path_list

        logger.info(
            f"统计过漏检，threshold = {self.threshold}\n"
            f"验证集数量：{len(pred_results)}\n"
            f'漏检数量：{len(escape_path_list)}\n'
            f'误判数量：{len(error_path_list)}\n'
            f'过检数量：{len(overkill_path_list)}')
        return results

    def match_label(self, true_result, pred_result):
        escape_flag = False
        overkill_flag = False
        error_flag = False

        logger = get_root_logger()
        logger.propagate = False

        right_labels = []
        for right_label in self.right_labels:
            if right_label in self.category_list:
                right_labels.append(self.category_list.index(right_label))

        for true_shapes in true_result:
            true_label = true_shapes['category_id']
            bbox = true_shapes['bbox']
            true_bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            if true_label in right_labels:
                continue
            iou_dic = {}
            for pre_label, pre_bboxes in enumerate(pred_result):
                pre_bboxes = pre_bboxes[pre_bboxes[:, -1] > self.threshold]
                for pre_bbox in pre_bboxes:
                    pre_bbox = [pre_bbox[0], pre_bbox[1], pre_bbox[2], pre_bbox[3]]
                    iou = bb_intersection_over_union(true_bbox, pre_bbox)
                    if iou > 0:
                        if pre_label not in iou_dic.keys():
                            iou_dic[pre_label] = iou
                        else:
                            if iou > iou_dic[pre_label]:
                                iou_dic[pre_label] = iou
            iou_lis = sorted(iou_dic.items(), key=lambda kv: kv[1], reverse=True)
            if len(iou_lis) == 0:
                escape_flag = True
                break
            else:
                match_label = iou_lis[0][0]
                if match_label == true_label:
                    continue
                else:
                    error_flag = True

        if escape_flag or error_flag:
            return escape_flag, error_flag, overkill_flag

        for pre_label, pre_bboxes in enumerate(pred_result):
            if pre_label in right_labels:
                continue
            pre_bboxes = pre_bboxes[pre_bboxes[:, -1] > 0.5]
            for pre_bbox in pre_bboxes:
                match = False
                pre_bbox = [pre_bbox[0], pre_bbox[1], pre_bbox[2], pre_bbox[3]]
                for true_shapes in true_result:
                    true_label = true_shapes['category_id']
                    if true_label != pre_label:
                        continue
                    else:
                        bbox = true_shapes['bbox']
                        true_bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                        if bb_intersection_over_union(true_bbox, pre_bbox) > 0:
                            match = True
                if not match:
                    overkill_flag = True
                    break
            if overkill_flag:
                break

        return escape_flag, error_flag, overkill_flag

    def __del__(self):
        logger: logging.Logger = get_root_logger()
        logger.propagate = False

        escape_path_list = self.escape_path_list
        error_path_list = self.error_path_list
        overkill_path_list = self.overkill_path_list

        state = np.random.get_state()
        if self.shuffle:
            np.random.seed(self.seed)

        if self.shuffle:
            np.random.shuffle(escape_path_list)
            np.random.shuffle(error_path_list)
            np.random.shuffle(overkill_path_list)

        np.random.set_state(state)

        if not hasattr(self, 'timestamp'):
            return
        else:
            date = self.timestamp

        logger.info(f'CopyErrorCases')

        copy_json_and_img(escape_path_list[:self.max_copy_num], Path(self.copy_pred_bad_case_path, date, '漏检'))
        copy_json_and_img(error_path_list[:self.max_copy_num], Path(self.copy_pred_bad_case_path, date, '误判'))
        copy_json_and_img(overkill_path_list[:self.max_copy_num], Path(self.copy_pred_bad_case_path, date, '过检'))

        logger.info(f'CopyErrorCases 拷贝完成, 拷贝的文件路径在{str(Path(self.copy_pred_bad_case_path, date))}')

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class CreateConfusionMatrix:

    def __init__(
            self,
            score_thr=0,
            nms_iou_thr=None,
            tp_iou_thr=0.5,
            color_theme='plasma',
            title='Normalized Confusion Matrix'):
        self.score_thr = score_thr
        self.nms_iou_thr = nms_iou_thr
        self.tp_iou_thr = tp_iou_thr
        self.color_theme = color_theme
        self.title = title
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')

    def __call__(self, results, *args, **kwargs):
        category_list = results['category_list']
        ann_infos = results['ann_infos']
        pred_results = results['pred_results']
        timestamp = results['timestamp']
        copy_pred_bad_case_path = results['copy_pred_bad_case_path']

        save_confusion_matrix_dir = Path(copy_pred_bad_case_path, timestamp)

        logger = get_root_logger()
        logger.propagate = False
        try:
            logger.info("生成混淆矩阵中")
            os.makedirs(save_confusion_matrix_dir, exist_ok=True)
            confusion_matrix = calculate_confusion_matrix(
                len(category_list), pred_results, ann_infos, self.score_thr, self.nms_iou_thr, self.tp_iou_thr)

            plot_confusion_matrix(
                confusion_matrix,
                tuple(category_list) + ('background', ),
                save_dir=save_confusion_matrix_dir,
                color_theme=self.color_theme,
                title=self.title)
            logger.info(f'生成混淆矩阵成功，文件路径为{str(Path(save_confusion_matrix_dir,"confusion_matrix.png"))}')
        except:
            logger.warning(f'生成混淆矩阵失败：\n{traceback.format_exc()}')

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
