model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='SkeleT_GCN',
        init_graph_cfg=dict(layout='expressive-keypoints', mode='skelet_spatial'),
        in_channels=3,
        pooling_stages=[5, 8],
        pooling_joints=[27, 11],
        num_groups=[1, 1, 1, 1, 2, 2, 2, 4, 4, 4],
        num_person=2,
        data_bn_type='VC',
        same_transformation=True,
        same_transformation_importance=True,
        M_adaptive='init',
        num_subsets=3,
        split_way='split',
        concat_way='concat+conv',
        gcn_type='unit_gcn',
        tcn_type='mstcn',
        gcn_adaptive='init'
    )
    ,
    cls_head=dict(type='GCNHead', num_classes=60, in_channels=256))

dataset_type = 'PoseDataset'
ann_file = 'data/ntu_rgb/2d_skeleton/ntu60_expressive-keypoints.pkl'
train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='expressive-keypoints', feats=['jm']),
    dict(type='UniformSample', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='expressive-keypoints', feats=['jm']),
    dict(type='UniformSample', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='expressive-keypoints', feats=['jm']),
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
work_dir = './work_dirs/skelet_stgcn++/ntu60_xsub/{placeholder}'
