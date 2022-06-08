custom_imports = dict(
    imports=[
        'sonic_ai.labelme_dataset', 'sonic_ai.encrypt_epoch_based_runner',
        'sonic_ai.load_3D_image_from_file'
    ],
    allow_failed_imports=True)

dataset_type = 'LabelmeDataset'
category_map = {}
dataset_path_list = []
img_scale = (640, 640)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=img_scale,
        ratio_range=[0.75, 1.25],
        keep_ratio=True),
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
        img_scale=img_scale,
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
    workers_per_gpu=4,
    persistent_workers=True,
    train=dict(
        type=dataset_type,
        dataset_path_list=dataset_path_list,
        start=0,
        end=0.8,
        category_map=category_map,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        dataset_path_list=dataset_path_list,
        start=0.8,
        end=1.0,
        category_map=category_map,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        dataset_path_list=dataset_path_list,
        start=0.8,
        end=1.0,
        category_map=category_map,
        pipeline=test_pipeline))

evaluation = dict(metric=['bbox', 'segm'])
