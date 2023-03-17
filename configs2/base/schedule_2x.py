# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
auto_scale_lr = dict(enable=True, base_batch_size=16)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
custom_hooks = [dict(type='SonicAfterRunHook'), dict(type='NumClassCheckHook')]

save_pipeline = [
    dict(
        type='SaveEachEpochModel',
        save_each_epoch=True,
        encrypt_each_epoch=False,
        save_latest=True,
        encrypt_latest=False),
    dict(type='SaveLatestModel', encrypt=False),
]

after_run_pipeline = [
    dict(type='SaveLog', create_briefing=True),
]

runner = dict(
    type='SonicEpochBasedRunner',
    save_pipeline=save_pipeline,
    after_run_pipeline=after_run_pipeline,
    max_epochs=24,
)

