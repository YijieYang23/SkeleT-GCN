model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='DGSTGCN',
        in_channels=3,
        graph_cfg=dict(layout='expressive-keypoints', mode='random', num_filter=8, init_off=.04, init_std=.02),
        gcn_ratio=0.125,
        gcn_ctr='T',
        gcn_ada='T',
        tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
        tcn_num_joints=65
        ),
    cls_head=dict(type='GCNHead', num_classes=120, in_channels=256))

dataset_type = 'PoseDataset'
ann_file = 'data/ntu_rgb/2d_skeleton/ntu120_expressive-keypoints.pkl'
train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='expressive-keypoints', feats=['bm']),
    dict(type='UniformSample', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='expressive-keypoints', feats=['bm']),
    dict(type='UniformSample', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='expressive-keypoints', feats=['bm']),
    dict(type='UniformSample', clip_len=100, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=4),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='xsub_train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='xsub_val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='xsub_val'))

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 24
checkpoint_config = dict(interval=1, max_keep_ckpts=5)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

# runtime settings
log_level = 'INFO'
work_dir = './work_dirs/vanilla_dgstgcn/ntu120_xsub/{placeholder}'
