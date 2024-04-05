default_scope = 'mmyolo'
default_hooks = dict(
    timer=dict(type='IterTimerHook', _scope_='mmyolo'),
    logger=dict(type='LoggerHook', interval=50, _scope_='mmyolo'),
    param_scheduler=dict(type='ParamSchedulerHook', _scope_='mmyolo'),
    checkpoint=dict(
        type='CheckpointHook', interval=10, max_keep_ckpts=3,
        _scope_='mmyolo'),
    sampler_seed=dict(type='DistSamplerSeedHook', _scope_='mmyolo'),
    visualization=dict(type='mmdet.DetVisualizationHook', _scope_='mmyolo'))
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend', _scope_='mmyolo')]
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer',
    _scope_='mmyolo')
log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, _scope_='mmyolo')
log_level = 'INFO'
load_from = '/data/weights/yoloms-se_syncbn_fast_8xb8-300e_coco-b0dc997d.pth'
resume = True
backend_args = None
_backend_args = None
tta_model = dict(
    type='mmdet.DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.65), max_per_img=300),
    _scope_='mmyolo')
img_scales = [(640, 640), (320, 320), (960, 960)]
_multiscale_resize_transforms = [
    dict(
        type='Compose',
        transforms=[
            dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
            dict(
                type='LetterResize',
                scale=(640, 640),
                allow_scale_up=False,
                pad_val=dict(img=114))
        ],
        _scope_='mmyolo'),
    dict(
        type='Compose',
        transforms=[
            dict(type='YOLOv5KeepRatioResize', scale=(320, 320)),
            dict(
                type='LetterResize',
                scale=(320, 320),
                allow_scale_up=False,
                pad_val=dict(img=114))
        ],
        _scope_='mmyolo'),
    dict(
        type='Compose',
        transforms=[
            dict(type='YOLOv5KeepRatioResize', scale=(960, 960)),
            dict(
                type='LetterResize',
                scale=(960, 960),
                allow_scale_up=False,
                pad_val=dict(img=114))
        ],
        _scope_='mmyolo')
]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, _scope_='mmyolo'),
    dict(
        type='TestTimeAug',
        transforms=[[{
            'type':
            'Compose',
            'transforms': [{
                'type': 'YOLOv5KeepRatioResize',
                'scale': (640, 640)
            }, {
                'type': 'LetterResize',
                'scale': (640, 640),
                'allow_scale_up': False,
                'pad_val': {
                    'img': 114
                }
            }]
        }, {
            'type':
            'Compose',
            'transforms': [{
                'type': 'YOLOv5KeepRatioResize',
                'scale': (320, 320)
            }, {
                'type': 'LetterResize',
                'scale': (320, 320),
                'allow_scale_up': False,
                'pad_val': {
                    'img': 114
                }
            }]
        }, {
            'type':
            'Compose',
            'transforms': [{
                'type': 'YOLOv5KeepRatioResize',
                'scale': (960, 960)
            }, {
                'type': 'LetterResize',
                'scale': (960, 960),
                'allow_scale_up': False,
                'pad_val': {
                    'img': 114
                }
            }]
        }],
                    [{
                        'type': 'mmdet.RandomFlip',
                        'prob': 1.0
                    }, {
                        'type': 'mmdet.RandomFlip',
                        'prob': 0.0
                    }], [{
                        'type': 'mmdet.LoadAnnotations',
                        'with_bbox': True
                    }],
                    [{
                        'type':
                        'mmdet.PackDetInputs',
                        'meta_keys':
                        ('img_id', 'img_path', 'ori_shape', 'img_shape',
                         'scale_factor', 'pad_param', 'flip', 'flip_direction')
                    }]],
        _scope_='mmyolo')
]
data_root = '/data/coco/coco_minitrain_25k/images'
train_ann_file = '/data/annotations/instances_minitrain2017.json'
train_data_prefix = 'train2017/'
val_ann_file = '/data/annotations/instances_val2017.json'
val_data_prefix = 'val2017/'
num_classes = 80
train_batch_size_per_gpu = 8
train_num_workers = 10
persistent_workers = True
base_lr = 0.004
max_epochs = 300
num_epochs_stage2 = 20
model_test_cfg = dict(
    multi_label=True,
    nms_pre=30000,
    score_thr=0.001,
    nms=dict(type='nms', iou_threshold=0.65, _scope_='mmyolo'),
    max_per_img=300)
img_scale = (640, 640)
random_resize_ratio_range = (0.1, 2.0)
mosaic_max_cached_images = 40
mixup_max_cached_images = 20
dataset_type = 'YOLOv5CocoDataset'
val_batch_size_per_gpu = 32
val_num_workers = 10
batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=32,
    img_size=640,
    size_divisor=32,
    extra_pad_ratio=0.5,
    _scope_='mmyolo')
deepen_factor = 0.6666666666666666
widen_factor = 0.8
strides = [8, 16, 32]
norm_cfg = dict(type='BN', _scope_='mmyolo')
lr_start_factor = 1e-05
dsl_topk = 13
loss_cls_weight = 1.0
loss_bbox_weight = 2.0
qfl_beta = 2.0
weight_decay = 0.05
save_checkpoint_intervals = 10
val_interval_stage2 = 1
max_keep_ckpts = 3
model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False),
    backbone=dict(
        type='YOLOMS',
        arch='C3-K3579',
        deepen_factor=0.6666666666666666,
        widen_factor=0.8,
        in_expand_ratio=3,
        mid_expand_ratio=2,
        layers_num=3,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True),
        attention_cfg=dict(type='SE')),
    neck=dict(
        type='YOLOMSPAFPN',
        deepen_factor=0.6666666666666666,
        widen_factor=0.8,
        in_channels=[320, 640, 1280],
        mid_channels=[160, 320, 640],
        out_channels=240,
        in_expand_ratio=3,
        mid_expand_ratio=2,
        layers_num=3,
        kernel_sizes=[1, (3, 3), (3, 3)],
        in_down_ratio=2,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='RTMDetHead',
        head_module=dict(
            type='RTMDetSepBNHeadModule',
            num_classes=80,
            in_channels=240,
            stacked_convs=2,
            feat_channels=240,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='LeakyReLU', inplace=True),
            share_conv=True,
            pred_kernel_size=1,
            featmap_strides=[8, 16, 32],
            widen_factor=0.8),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.DIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        assigner=dict(
            type='BatchDynamicSoftLabelAssigner',
            num_classes=80,
            topk=13,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300),
    _scope_='mmyolo')
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, _scope_='mmyolo'),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmyolo'),
    dict(
        type='Mosaic',
        img_scale=(640, 640),
        use_cached=True,
        max_cached_images=40,
        pad_val=114.0,
        _scope_='mmyolo'),
    dict(
        type='mmdet.RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.1, 2.0),
        resize_type='mmdet.Resize',
        keep_ratio=True,
        _scope_='mmyolo'),
    dict(type='mmdet.RandomCrop', crop_size=(640, 640), _scope_='mmyolo'),
    dict(type='mmdet.YOLOXHSVRandomAug', _scope_='mmyolo'),
    dict(type='mmdet.RandomFlip', prob=0.5, _scope_='mmyolo'),
    dict(
        type='mmdet.Pad',
        size=(640, 640),
        pad_val=dict(img=(114, 114, 114)),
        _scope_='mmyolo'),
    dict(
        type='YOLOv5MixUp',
        use_cached=True,
        max_cached_images=20,
        _scope_='mmyolo'),
    dict(type='mmdet.PackDetInputs', _scope_='mmyolo')
]
train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=None, _scope_='mmyolo'),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmyolo'),
    dict(
        type='mmdet.RandomResize',
        scale=(640, 640),
        ratio_range=(0.1, 2.0),
        resize_type='mmdet.Resize',
        keep_ratio=True,
        _scope_='mmyolo'),
    dict(type='mmdet.RandomCrop', crop_size=(640, 640), _scope_='mmyolo'),
    dict(type='mmdet.YOLOXHSVRandomAug', _scope_='mmyolo'),
    dict(type='mmdet.RandomFlip', prob=0.5, _scope_='mmyolo'),
    dict(
        type='mmdet.Pad',
        size=(640, 640),
        pad_val=dict(img=(114, 114, 114)),
        _scope_='mmyolo'),
    dict(type='mmdet.PackDetInputs', _scope_='mmyolo')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, _scope_='mmyolo'),
    dict(type='YOLOv5KeepRatioResize', scale=(640, 640), _scope_='mmyolo'),
    dict(
        type='LetterResize',
        scale=(640, 640),
        allow_scale_up=False,
        pad_val=dict(img=114),
        _scope_='mmyolo'),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'),
        _scope_='mmyolo')
]
train_dataloader = dict(
    batch_size=8,
    num_workers=10,
    persistent_workers=True,
    pin_memory=True,
    collate_fn=dict(type='yolov5_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='/data/coco/coco_minitrain_25k/images',
        ann_file='/data/annotations/instances_minitrain2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Mosaic',
                img_scale=(640, 640),
                use_cached=True,
                max_cached_images=40,
                pad_val=114.0),
            dict(
                type='mmdet.RandomResize',
                scale=(1280, 1280),
                ratio_range=(0.1, 2.0),
                resize_type='mmdet.Resize',
                keep_ratio=True),
            dict(type='mmdet.RandomCrop', crop_size=(640, 640)),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(type='mmdet.RandomFlip', prob=0.5),
            dict(
                type='mmdet.Pad',
                size=(640, 640),
                pad_val=dict(img=(114, 114, 114))),
            dict(type='YOLOv5MixUp', use_cached=True, max_cached_images=20),
            dict(type='mmdet.PackDetInputs')
        ],
        _scope_='mmyolo'))
val_dataloader = dict(
    batch_size=32,
    num_workers=10,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='/data/coco/coco_minitrain_25k/images',
        ann_file='/data/annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        batch_shapes_cfg=dict(
            type='BatchShapePolicy',
            batch_size=32,
            img_size=640,
            size_divisor=32,
            extra_pad_ratio=0.5),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
            dict(
                type='LetterResize',
                scale=(640, 640),
                allow_scale_up=False,
                pad_val=dict(img=114)),
            dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor', 'pad_param'))
        ],
        _scope_='mmyolo'))
test_dataloader = dict(
    batch_size=32,
    num_workers=10,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='/data/coco/coco_minitrain_25k/images',
        ann_file='/data/annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        batch_shapes_cfg=dict(
            type='BatchShapePolicy',
            batch_size=32,
            img_size=640,
            size_divisor=32,
            extra_pad_ratio=0.5),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
            dict(
                type='LetterResize',
                scale=(640, 640),
                allow_scale_up=False,
                pad_val=dict(img=114)),
            dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor', 'pad_param'))
        ],
        _scope_='mmyolo'))
val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file='data/coco/annotations/instances_val2017.json',
    metric='bbox',
    _scope_='mmyolo')
test_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file='data/coco/annotations/instances_val2017.json',
    metric='bbox',
    _scope_='mmyolo')
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.004, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
    _scope_='mmyolo')
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-05,
        by_epoch=False,
        begin=0,
        end=1000,
        _scope_='mmyolo'),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0002,
        begin=150,
        end=300,
        T_max=150,
        by_epoch=True,
        convert_to_iter_based=True,
        _scope_='mmyolo')
]
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49,
        _scope_='mmyolo'),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=280,
        switch_pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='mmdet.RandomResize',
                scale=(640, 640),
                ratio_range=(0.1, 2.0),
                resize_type='mmdet.Resize',
                keep_ratio=True),
            dict(type='mmdet.RandomCrop', crop_size=(640, 640)),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(type='mmdet.RandomFlip', prob=0.5),
            dict(
                type='mmdet.Pad',
                size=(640, 640),
                pad_val=dict(img=(114, 114, 114))),
            dict(type='mmdet.PackDetInputs')
        ],
        _scope_='mmyolo')
]
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=300,
    val_interval=10,
    dynamic_intervals=[(280, 1)],
    _scope_='mmyolo')
val_cfg = dict(type='ValLoop', _scope_='mmyolo')
test_cfg = dict(type='TestLoop', _scope_='mmyolo')
layers_num = 3
in_channels = [320, 640, 1280]
mid_channels = [160, 320, 640]
out_channels = 240
in_expand_ratio = 3
mid_expand_ratio = 2
in_down_ratio = 2
act_cfg = dict(type='SiLU', inplace=True)
kernel_sizes = [1, (3, 3), (3, 3)]
auto_scale_lr = dict(enable=True, base_batch_size=256)
launcher = 'none'
work_dir = '/data/weights'
