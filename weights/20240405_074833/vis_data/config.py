_backend_args = None
_multiscale_resize_transforms = [
    dict(
        _scope_='mmyolo',
        transforms=[
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        _scope_='mmyolo',
        transforms=[
            dict(scale=(
                320,
                320,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    320,
                    320,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        _scope_='mmyolo',
        transforms=[
            dict(scale=(
                960,
                960,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    960,
                    960,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
]
act_cfg = dict(inplace=True, type='SiLU')
auto_scale_lr = dict(base_batch_size=256, enable=True)
backend_args = None
base_lr = 0.004
batch_shapes_cfg = dict(
    _scope_='mmyolo',
    batch_size=32,
    extra_pad_ratio=0.5,
    img_size=640,
    size_divisor=32,
    type='BatchShapePolicy')
custom_hooks = [
    dict(
        _scope_='mmyolo',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        strict_load=False,
        type='EMAHook',
        update_buffers=True),
    dict(
        _scope_='mmyolo',
        switch_epoch=280,
        switch_pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.1,
                    2.0,
                ),
                resize_type='mmdet.Resize',
                scale=(
                    640,
                    640,
                ),
                type='mmdet.RandomResize'),
            dict(crop_size=(
                640,
                640,
            ), type='mmdet.RandomCrop'),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    640,
                    640,
                ),
                type='mmdet.Pad'),
            dict(type='mmdet.PackDetInputs'),
        ],
        type='mmdet.PipelineSwitchHook'),
]
data_root = '/data/coco/coco_minitrain_25k/images'
dataset_type = 'YOLOv5CocoDataset'
deepen_factor = 0.6666666666666666
default_hooks = dict(
    checkpoint=dict(
        _scope_='mmyolo', interval=10, max_keep_ckpts=3,
        type='CheckpointHook'),
    logger=dict(_scope_='mmyolo', interval=50, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmyolo', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmyolo', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmyolo', type='IterTimerHook'),
    visualization=dict(_scope_='mmyolo', type='mmdet.DetVisualizationHook'))
default_scope = 'mmyolo'
dsl_topk = 13
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_scale = (
    640,
    640,
)
img_scales = [
    (
        640,
        640,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]
in_channels = [
    320,
    640,
    1280,
]
in_down_ratio = 2
in_expand_ratio = 3
kernel_sizes = [
    1,
    (
        3,
        3,
    ),
    (
        3,
        3,
    ),
]
launcher = 'none'
layers_num = 3
load_from = '/data/weights/yoloms-se_syncbn_fast_8xb8-300e_coco-b0dc997d.pth'
log_level = 'INFO'
log_processor = dict(
    _scope_='mmyolo', by_epoch=True, type='LogProcessor', window_size=50)
loss_bbox_weight = 2.0
loss_cls_weight = 1.0
lr_start_factor = 1e-05
max_epochs = 300
max_keep_ckpts = 3
mid_channels = [
    160,
    320,
    640,
]
mid_expand_ratio = 2
mixup_max_cached_images = 20
model = dict(
    _scope_='mmyolo',
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        arch='C3-K3579',
        attention_cfg=dict(type='SE'),
        deepen_factor=0.6666666666666666,
        in_expand_ratio=3,
        layers_num=3,
        mid_expand_ratio=2,
        norm_cfg=dict(type='BN'),
        type='YOLOMS',
        widen_factor=0.8),
    bbox_head=dict(
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        head_module=dict(
            act_cfg=dict(inplace=True, type='LeakyReLU'),
            feat_channels=240,
            featmap_strides=[
                8,
                16,
                32,
            ],
            in_channels=240,
            norm_cfg=dict(type='BN'),
            num_classes=80,
            pred_kernel_size=1,
            share_conv=True,
            stacked_convs=2,
            type='RTMDetSepBNHeadModule',
            widen_factor=0.8),
        loss_bbox=dict(loss_weight=2.0, type='mmdet.DIoULoss'),
        loss_cls=dict(
            beta=2.0,
            loss_weight=1.0,
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True),
        prior_generator=dict(
            offset=0, strides=[
                8,
                16,
                32,
            ], type='mmdet.MlvlPointGenerator'),
        type='RTMDetHead'),
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        std=[
            57.375,
            57.12,
            58.395,
        ],
        type='YOLOv5DetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        deepen_factor=0.6666666666666666,
        in_channels=[
            320,
            640,
            1280,
        ],
        in_down_ratio=2,
        in_expand_ratio=3,
        kernel_sizes=[
            1,
            (
                3,
                3,
            ),
            (
                3,
                3,
            ),
        ],
        layers_num=3,
        mid_channels=[
            160,
            320,
            640,
        ],
        mid_expand_ratio=2,
        norm_cfg=dict(type='BN'),
        out_channels=240,
        type='YOLOMSPAFPN',
        widen_factor=0.8),
    test_cfg=dict(
        max_per_img=300,
        multi_label=True,
        nms=dict(iou_threshold=0.65, type='nms'),
        nms_pre=30000,
        score_thr=0.001),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            iou_calculator=dict(type='mmdet.BboxOverlaps2D'),
            num_classes=80,
            topk=13,
            type='BatchDynamicSoftLabelAssigner'),
        debug=False,
        pos_weight=-1),
    type='YOLODetector')
model_test_cfg = dict(
    max_per_img=300,
    multi_label=True,
    nms=dict(_scope_='mmyolo', iou_threshold=0.65, type='nms'),
    nms_pre=30000,
    score_thr=0.001)
mosaic_max_cached_images = 40
norm_cfg = dict(_scope_='mmyolo', type='BN')
num_classes = 80
num_epochs_stage2 = 20
optim_wrapper = dict(
    _scope_='mmyolo',
    optimizer=dict(lr=0.004, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
out_channels = 240
param_scheduler = [
    dict(
        _scope_='mmyolo',
        begin=0,
        by_epoch=False,
        end=1000,
        start_factor=1e-05,
        type='LinearLR'),
    dict(
        T_max=150,
        _scope_='mmyolo',
        begin=150,
        by_epoch=True,
        convert_to_iter_based=True,
        end=300,
        eta_min=0.0002,
        type='CosineAnnealingLR'),
]
persistent_workers = True
qfl_beta = 2.0
random_resize_ratio_range = (
    0.1,
    2.0,
)
resume = True
save_checkpoint_intervals = 10
strides = [
    8,
    16,
    32,
]
test_cfg = dict(_scope_='mmyolo', type='TestLoop')
test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        _scope_='mmyolo',
        ann_file='/data/annotations/instances_val2017.json',
        batch_shapes_cfg=dict(
            batch_size=32,
            extra_pad_ratio=0.5,
            img_size=640,
            size_divisor=32,
            type='BatchShapePolicy'),
        data_prefix=dict(img='val2017/'),
        data_root='/data/coco/coco_minitrain_25k/images',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='YOLOv5CocoDataset'),
    drop_last=False,
    num_workers=10,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    _scope_='mmyolo',
    ann_file='data/coco/annotations/instances_val2017.json',
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')
test_pipeline = [
    dict(_scope_='mmyolo', backend_args=None, type='LoadImageFromFile'),
    dict(_scope_='mmyolo', scale=(
        640,
        640,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        _scope_='mmyolo',
        allow_scale_up=False,
        pad_val=dict(img=114),
        scale=(
            640,
            640,
        ),
        type='LetterResize'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(
        _scope_='mmyolo',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'pad_param',
        ),
        type='mmdet.PackDetInputs'),
]
train_ann_file = '/data/annotations/instances_minitrain2017.json'
train_batch_size_per_gpu = 8
train_cfg = dict(
    _scope_='mmyolo',
    dynamic_intervals=[
        (
            280,
            1,
        ),
    ],
    max_epochs=300,
    type='EpochBasedTrainLoop',
    val_interval=10)
train_data_prefix = 'train2017/'
train_dataloader = dict(
    batch_size=8,
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        _scope_='mmyolo',
        ann_file='/data/annotations/instances_minitrain2017.json',
        data_prefix=dict(img='train2017/'),
        data_root='/data/coco/coco_minitrain_25k/images',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                img_scale=(
                    640,
                    640,
                ),
                max_cached_images=40,
                pad_val=114.0,
                type='Mosaic',
                use_cached=True),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.1,
                    2.0,
                ),
                resize_type='mmdet.Resize',
                scale=(
                    1280,
                    1280,
                ),
                type='mmdet.RandomResize'),
            dict(crop_size=(
                640,
                640,
            ), type='mmdet.RandomCrop'),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    640,
                    640,
                ),
                type='mmdet.Pad'),
            dict(max_cached_images=20, type='YOLOv5MixUp', use_cached=True),
            dict(type='mmdet.PackDetInputs'),
        ],
        type='YOLOv5CocoDataset'),
    num_workers=10,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_num_workers = 10
train_pipeline = [
    dict(_scope_='mmyolo', backend_args=None, type='LoadImageFromFile'),
    dict(_scope_='mmyolo', type='LoadAnnotations', with_bbox=True),
    dict(
        _scope_='mmyolo',
        img_scale=(
            640,
            640,
        ),
        max_cached_images=40,
        pad_val=114.0,
        type='Mosaic',
        use_cached=True),
    dict(
        _scope_='mmyolo',
        keep_ratio=True,
        ratio_range=(
            0.1,
            2.0,
        ),
        resize_type='mmdet.Resize',
        scale=(
            1280,
            1280,
        ),
        type='mmdet.RandomResize'),
    dict(_scope_='mmyolo', crop_size=(
        640,
        640,
    ), type='mmdet.RandomCrop'),
    dict(_scope_='mmyolo', type='mmdet.YOLOXHSVRandomAug'),
    dict(_scope_='mmyolo', prob=0.5, type='mmdet.RandomFlip'),
    dict(
        _scope_='mmyolo',
        pad_val=dict(img=(
            114,
            114,
            114,
        )),
        size=(
            640,
            640,
        ),
        type='mmdet.Pad'),
    dict(
        _scope_='mmyolo',
        max_cached_images=20,
        type='YOLOv5MixUp',
        use_cached=True),
    dict(_scope_='mmyolo', type='mmdet.PackDetInputs'),
]
train_pipeline_stage2 = [
    dict(_scope_='mmyolo', backend_args=None, type='LoadImageFromFile'),
    dict(_scope_='mmyolo', type='LoadAnnotations', with_bbox=True),
    dict(
        _scope_='mmyolo',
        keep_ratio=True,
        ratio_range=(
            0.1,
            2.0,
        ),
        resize_type='mmdet.Resize',
        scale=(
            640,
            640,
        ),
        type='mmdet.RandomResize'),
    dict(_scope_='mmyolo', crop_size=(
        640,
        640,
    ), type='mmdet.RandomCrop'),
    dict(_scope_='mmyolo', type='mmdet.YOLOXHSVRandomAug'),
    dict(_scope_='mmyolo', prob=0.5, type='mmdet.RandomFlip'),
    dict(
        _scope_='mmyolo',
        pad_val=dict(img=(
            114,
            114,
            114,
        )),
        size=(
            640,
            640,
        ),
        type='mmdet.Pad'),
    dict(_scope_='mmyolo', type='mmdet.PackDetInputs'),
]
tta_model = dict(
    _scope_='mmyolo',
    tta_cfg=dict(max_per_img=300, nms=dict(iou_threshold=0.65, type='nms')),
    type='mmdet.DetTTAModel')
tta_pipeline = [
    dict(_scope_='mmyolo', backend_args=None, type='LoadImageFromFile'),
    dict(
        _scope_='mmyolo',
        transforms=[
            [
                dict(
                    transforms=[
                        dict(scale=(
                            640,
                            640,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                640,
                                640,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            320,
                            320,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                320,
                                320,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            960,
                            960,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                960,
                                960,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
            ],
            [
                dict(prob=1.0, type='mmdet.RandomFlip'),
                dict(prob=0.0, type='mmdet.RandomFlip'),
            ],
            [
                dict(type='mmdet.LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'pad_param',
                        'flip',
                        'flip_direction',
                    ),
                    type='mmdet.PackDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_ann_file = '/data/annotations/instances_val2017.json'
val_batch_size_per_gpu = 32
val_cfg = dict(_scope_='mmyolo', type='ValLoop')
val_data_prefix = 'val2017/'
val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        _scope_='mmyolo',
        ann_file='/data/annotations/instances_val2017.json',
        batch_shapes_cfg=dict(
            batch_size=32,
            extra_pad_ratio=0.5,
            img_size=640,
            size_divisor=32,
            type='BatchShapePolicy'),
        data_prefix=dict(img='val2017/'),
        data_root='/data/coco/coco_minitrain_25k/images',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='YOLOv5CocoDataset'),
    drop_last=False,
    num_workers=10,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    _scope_='mmyolo',
    ann_file='data/coco/annotations/instances_val2017.json',
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')
val_interval_stage2 = 1
val_num_workers = 10
vis_backends = [
    dict(_scope_='mmyolo', type='LocalVisBackend'),
]
visualizer = dict(
    _scope_='mmyolo',
    name='visualizer',
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
weight_decay = 0.05
widen_factor = 0.8
work_dir = '/data/weights'
