import time

from sonic_ai.pipelines.init_pipeline import LoadCategoryList

_base_ = [
    '../base/mask_rcnn_r50_fpn.py', '../base/default_runtime.py',
    '../base/schedule_1x.py', '../base/base_sonic_dataset.py'
]

label_path = r"/home/xys/Data/label.txt"

dataset_path_list = ["/home/xys/Data/images_ct"]

save_model_path = r'/home/xys/Data/模型备份'
project_path = '医学影像_单通道'
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

num_classes = len(
    LoadCategoryList()(results={
        'label_path': label_path
    })['category_list'])

data = dict(
    persistent_workers=True,
    workers_per_gpu=4,
    train=dict(
        label_path=label_path,
        dataset_path_list=dataset_path_list,
    ),
    val=dict(
        label_path=label_path,
        dataset_path_list=dataset_path_list,
        timestamp=timestamp),
    test=dict(
        label_path=label_path,
        dataset_path_list=dataset_path_list,
        timestamp=timestamp,
    ))

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes),
        mask_head=dict(num_classes=num_classes)))

log_config = dict(interval=50)

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.pth'

LoadCategoryList = None

runner = dict(
    save_model_path=f"{save_model_path}/{project_path}", timestamp=timestamp, max_epochs=12)
