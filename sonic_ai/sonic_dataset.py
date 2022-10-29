import os
import tempfile
import time
import traceback

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.pipelines.compose import Compose
from mmdet.utils import get_root_logger


@DATASETS.register_module()
class SonicDataset(CocoDataset):

    def __init__(
            self,
            dataset_path_list,
            label_path,
            init_pipeline,
            filter_empty_gt=True,
            timestamp=None,
            eval_pipeline=None,
            copy_pred_bad_case_path='/data/14-调试数据/过漏检数据',
            right_labels=None,
            start=None,
            end=None,
            times=None,
            ignore_labels=None,
            ok_dataset_path_list=None,
            *args,
            **kwargs):

        if right_labels is None:
            right_labels = ['OK', "mark孔OK"]
        self.right_labels = right_labels
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.ann_file = os.path.join(self.tmp_dir.name, "ann_file.json")
        self.ann_infos = {}

        self.data = dict(
            dataset_path_list=dataset_path_list,
            label_path=label_path,
            ann_save_path=self.ann_file,
            start=start,
            end=end,
            times=times,
            ignore_labels=ignore_labels,
            ok_dataset_path_list=ok_dataset_path_list)
        init_pipeline_compose = Compose(init_pipeline)
        init_pipeline_compose(self.data)

        self.copy_pred_bad_case_path = copy_pred_bad_case_path

        super().__init__(
            classes=self.data['category_list'],
            ann_file=self.ann_file,
            filter_empty_gt=filter_empty_gt,
            *args,
            **kwargs)

        if timestamp is not None:
            self.timestamp = timestamp
        else:
            self.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        self.eval_pipeline_compose = None
        if self.test_mode:
            self.eval_pipeline_compose = Compose(eval_pipeline)
            for idx in range(len(self)):
                self.get_ann_info(idx)

    def load_annotations(self, ann_file):
        data_infos = super().load_annotations(ann_file)
        self.tmp_dir.cleanup()
        return data_infos

    def get_ann_info(self, idx):
        ann = super().get_ann_info(idx)
        self.ann_infos[idx] = ann
        return ann

    def _parse_ann_info(self, img_info, ann_info):
        filename = img_info['filename']
        if isinstance(img_info['filename'], list):
            img_info['filename'] = img_info['filename'][0]
        ann = super()._parse_ann_info(img_info, ann_info)
        img_info['filename'] = filename
        return ann

    def prepare_train_img(self, idx):
        data = super().prepare_train_img(idx)
        if data is not None:
            ori_shape = data['img_metas'].data['ori_shape'][:2] if data.get(
                'img_metas', None) else data['ori_shape'][:2]
            if ori_shape != (self.data_infos[idx]['height'],
                             self.data_infos[idx]['width']):
                logger = get_root_logger()
                logger.propagate = False
                logger.warning(
                    f"图片路径{data['img_metas'].data['ori_filename']}的宽高{ori_shape}与json记录的宽高{(self.data_infos[idx]['height'], self.data_infos[idx]['width'])}不一致，请检查"
                )
                data = None
        return data

    def evaluate(self, *args, **kwargs):
        logger = get_root_logger()
        logger.propagate = False

        results = dict(
            pred_results=args[0],
            img_ids=self.img_ids,
            coco=self.coco,
            ann_infos=self.ann_infos,
            length=len(self),
            right_labels=self.right_labels,
            category_list=self.data['category_list'],
            category_map=self.data['category_map'],
            category_counter=self.data['category_counter'],
            json_data_list=self.data['json_data_list'],
            dataset_path_list=self.data['dataset_path_list'],
            copy_pred_bad_case_path=self.copy_pred_bad_case_path,
            timestamp=self.timestamp)
        try:
            self.eval_pipeline_compose(results)
        except:
            logger.warning(f'eval_pipeline执行出现出错：\n{traceback.format_exc()}')

        return results['eval_results']
