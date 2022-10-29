custom_imports = dict(
    imports=[
        'sonic_ai.encrypt_epoch_based_runner',
        'sonic_ai.load_3D_image_from_file', 'sonic_ai.sonic_dataset',
        'sonic_ai.sonic_epoch_based_runner',
        'sonic_ai.pipelines.init_pipeline', 'sonic_ai.pipelines.eval_pipeline',
        'sonic_ai.pipelines.save_pipeline',
        'sonic_ai.pipelines.after_run_pipeline',
        'sonic_ai.sonic_after_run_hook',
        'sonic_ai.pipelines.dataset_pipeline'
    ],
    allow_failed_imports=True)

dataset_type = 'SonicDataset'

img_scale = (512, 512)

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

train_init_pipeline = [
    dict(type='CopyData2Local', target_dir='/data/公共数据缓存', run_rsync=True),
    dict(type='LoadCategoryList', ignore_labels=['屏蔽']),
    dict(type='LoadPathList'),
    dict(type='SplitData', start=0, end=0.8, key='json_path_list'),
    dict(type='LoadJsonDataList'),
    dict(type='LoadLabelmeDataset'),
    dict(type='StatCategoryCounter'),
    dict(type='CopyData', times=1),
    dict(type='Labelme2Coco'),
    # dict(type='LoadOKPathList'),
    # dict(type='ShuffleCocoImage'),
    dict(type='SaveJson'),
]

test_init_pipeline = [
    dict(type='CopyData2Local', target_dir='/data/公共数据缓存', run_rsync=False),
    dict(type='LoadCategoryList', ignore_labels=['屏蔽']),
    dict(type='LoadPathList'),
    dict(type='SplitData', start=0.8, end=1, key='json_path_list'),
    dict(type='LoadJsonDataList'),
    dict(type='LoadLabelmeDataset'),
    dict(type='Labelme2Coco'),
    dict(type='StatCategoryCounter'),
    dict(type='SaveJson'),
]

eval_pipeline = [
    dict(
        type='CocoEvaluate', metric=['bbox'], classwise=True, iou_thrs=[0, 0]),
    dict(type='ShowScores'),
]

data = dict(
    persistent_workers=True,
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        pipeline=train_pipeline,
        init_pipeline=train_init_pipeline),
    train_dataloader=dict(class_aware_sampler=dict(num_sample_class=1)),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        init_pipeline=test_init_pipeline,
        eval_pipeline=eval_pipeline),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        init_pipeline=test_init_pipeline,
        eval_pipeline=eval_pipeline))

fp16 = dict(loss_scale=512.)

evaluation = dict(metric=['bbox'])
