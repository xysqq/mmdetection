import time

from sonic_ai.pipelines.init_pipeline import LoadCategoryList

_base_ = [
    '../base/mask_rcnn_r50_fpn.py', '../base/default_runtime.py',
    '../base/schedule_sonic.py', '../base/base_sonic_dataset.py'
]

label_path = r"D:\Data\label.ini"

dataset_path_list = ["D:/Data/ct_images"]

badcase_path = r'D:/Data/过检漏检'
save_model_path = r'D:/Data/模型备份'
project_path = '医学影像'
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

num_classes = len(
    LoadCategoryList()(results={
        'label_path': label_path
    })['category_list'])

data = dict(
    persistent_workers=False,
    workers_per_gpu=0,
    train=dict(
        label_path=label_path,
        dataset_path_list=dataset_path_list,
    ),
    val=dict(
        label_path=label_path,
        dataset_path_list=dataset_path_list,
        copy_pred_bad_case_path=f"{badcase_path}/{project_path}",
        timestamp=timestamp),
    test=dict(
        label_path=label_path,
        dataset_path_list=dataset_path_list,
        copy_pred_bad_case_path=f"{badcase_path}/{project_path}",
        timestamp=timestamp,
    ))

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes),
        mask_head=dict(num_classes=num_classes)))

log_config = dict(interval=50)

load_from = r"D:\Data\pretrained_model\mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.pth"

LoadCategoryList = None

optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)

runner = dict(
    save_model_path=f"{save_model_path}/{project_path}", timestamp=timestamp,  max_epochs=12)


