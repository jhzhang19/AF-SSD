_base_ = [
    '../_base_/models/fusion_3dssd_base.py', '../_base_/datasets/kitti-3d-car.py',
    '../_base_/default_runtime.py','../_base_/schedules/cosine.py'
]

# _base_ = [
#     '../_base_/models/fusion_3dssd_base.py', '../_base_/datasets/kitti-3d-car.py',
#     '../_base_/default_runtime.py','../_base_/schedules/mmdet_schedule_1x.py'
# ]

# dataset settings
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Car']
point_cloud_range = [0, -40, -5, 70, 40, 3]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
input_modality = dict(use_lidar=True, use_camera=True)
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(filter_by_difficulty=[-1], filter_by_min_points=dict(Car=5)),
    classes=class_names,
    sample_groups=dict(Car=15))

file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel', path_mapping=dict(data='s3://kitti_data/'))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=file_client_args),
    dict(
        type='Resize',
        img_scale=[(640, 192), (2560, 768)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),

    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0],
        global_rot_range=[0.0, 0.0],
        rot_range=[-1.0471975511965976, 1.0471975511965976]),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.9, 1.1]),
    dict(type='BackgroundPointsFilter', bbox_enlarge_range=(0.5, 2.0, 0.5)),
    dict(type='IndoorPointSample', num_points=16384),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d']),
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1280, 384),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='Resize', multiscale_mode='value', keep_ratio=True),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='IndoorPointSample', num_points=16384),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img'])
        ])
]

eval_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadImageFromFile'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points', 'img'])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(dataset=dict(pipeline=train_pipeline)),
    val=dict(pipeline=eval_pipeline),
    test=dict(
        pipeline=test_pipeline
        
        ))

evaluation = dict(interval=3, pipeline=eval_pipeline)

# model settings
model = dict(
    bbox_head=dict(
        num_classes=1,
        bbox_coder=dict(
            type='AnchorFreeBBoxCoder', num_dir_bins=12, with_rot=True)))

# optimizer
lr = 0.000075
# 记得该继承文件
# optimizer = dict(type='SGD', lr=0.000005, momentum=0.9, weight_decay=0.0001)

# optimizer_config = dict(grad_clip=None)
# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=5000,
#     warmup_ratio=0.0001,
#     step=[8, 11])
# runner = dict(type='EpochBasedRunner', max_epochs=15)

# Training settings
optimizer = dict(lr=lr,  weight_decay=0.001)
# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# lr = 0.0003 # max learning rate
# optimizer = dict(type='AdamW', lr=lr, weight_decay=0)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# lr_config = dict(policy='step', warmup=None, step=[80, 120])
# # runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=30)

# yapf:disable
log_config = dict(
    interval=30,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

load_from = '/mmdetection3d/data/mmdetection/work_dir_res18/latest.pth'
# load_from = 'https://download.openmmlab.com/mmdetection3d/pretrain_models/mvx_faster_rcnn_detectron2-caffe_20e_coco-pretrain_gt-sample_kitti-3-class_moderate-79.3_20200207-a4a6a3c7.pth'  # noqa
#load_from = 'https://download.openmmlab.com/mmdetection3d/v0.1.0_models/3dssd/3dssd_kitti-3d-car_20210602_124438-b4276f56.pth'