_base_ = [
    '../base/mask_rcnn_r18_fpn.py',
    '../base/default_runtime.py',
    '../base/schedule_1x.py',
    '../base/base_dataset.py'
]
data_root = ''
dataset_type = 'CocoDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

category_list = ['rle_large', 'rle_small', 'rle_stomach']

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True,  poly2mask=True),
    dict(
        type='Resize',
        img_scale=(512, 512),
        ratio_range=[0.75, 1.25],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    persistent_workers=False,
    train=dict(
        classes=category_list,
        type=dataset_type,
        ann_file=data_root + 'train_json.json',
        img_prefix='',
        pipeline=train_pipeline),
    val=dict(
        classes=category_list,
        type=dataset_type,
        ann_file=data_root + 'test_json.json',
        img_prefix='',
        pipeline=test_pipeline),
    test=dict(
        classes=category_list,
        type=dataset_type,
        ann_file=data_root + 'test_json.json',
        img_prefix='',
        pipeline=test_pipeline))

evaluation = dict(metric=['bbox', 'segm'])
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=len(category_list)),
        mask_head=dict(num_classes=len(category_list))))

log_config = dict(interval=25)

load_from = 'checkpoint/mask_rcnn_r18_fpn_2x_coco_bbox_mAP-0.329__segm_mAP-0.301_20210909_112615-9c1ca240.pth'
fp16 = dict(loss_scale=512.)
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
runner = dict(type='EpochBasedRunner', max_epochs=12)
