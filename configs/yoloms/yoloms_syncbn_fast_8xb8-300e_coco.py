# Reference to
# https://github.com/open-mmlab/mmyolo/blob/main/configs/rtmdet/rtmdet_l_syncbn_fast_8xb32-300e_coco.py
_base_ = 'mmyolo::rtmdet/rtmdet_l_syncbn_fast_8xb32-300e_coco.py'

# ========================Frequently modified parameters======================
# -----data related-----
data_root = '/data/coco_minitrain_25k/images'
# Path of train annotation file
train_ann_file = '/data/annotations/instances_minitrain2017.json'
train_data_prefix = 'train2017/'  # Prefix of train image path
# Path of val annotation file
val_ann_file = '/data/annotations/instances_val2017.json'
val_data_prefix = 'val2017/'  # Prefix of val image path

# Number of classes for classification
class_name = ('airplane', 'ship', 'vehicle',)  # dataset category name
num_classes = len(class_name)  # dataset category number
# metainfo is a configuration that must be passed to the dataloader, otherwise it is invalid
# palette is a display color for category at visualization
# The palette length must be greater than or equal to the length of the classes
metainfo = dict(classes=class_name, palette=[(255, 0, 0)])

max_epochs = 300

# Batch size of a single GPU during training
train_batch_size_per_gpu = 32
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 8
# persistent_workers must be False if num_workers is 0.
persistent_workers = True

# ========================Possible modified parameters========================
# -----data related-----
# Batch size of a single GPU during validation
val_batch_size_per_gpu = 32
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 10

# Number of layer in MS-Block
layers_num = 3
# The scaling factor that controls the depth of the network structure
deepen_factor = 2 / 3
# The scaling factor that controls the width of the network structure
widen_factor = 0.8

# Input channels of PAFPN
in_channels = [320, 640, 1280]
# Middle channels of PAFPN
mid_channels = [160, 320, 640]
# Output channels of PAFPN
out_channels = 240

# Channel expand ratio for inputs of MS-Block
in_expand_ratio = 3
# Channel expand ratio for each branch in MS-Block
mid_expand_ratio = 2

# Channel down ratio for downsample conv layer in MS-Block
in_down_ratio = 2
# Normalization config
norm_cfg = dict(type='BN')
# Activation config
act_cfg = dict(type='SiLU', inplace=True)

# Kernel sizes of MS-Block in PAFPN
kernel_sizes = [1, (3, 3), (3, 3)]

loss_bbox_weight = 2.0

# =======================Unmodified in most cases==================
model = dict(backbone=dict(_delete_=True,
                           type='YOLOMS',
                           arch='C3-K3579',
                           deepen_factor=deepen_factor,
                           widen_factor=widen_factor,
                           in_expand_ratio=in_expand_ratio,
                           mid_expand_ratio=mid_expand_ratio,
                           layers_num=layers_num,
                           norm_cfg=norm_cfg,
                           act_cfg=act_cfg),
             neck=dict(_delete_=True,
                       type='YOLOMSPAFPN',
                       deepen_factor=deepen_factor,
                       widen_factor=widen_factor,
                       in_channels=in_channels,
                       mid_channels=mid_channels,
                       out_channels=out_channels,
                       in_expand_ratio=in_expand_ratio,
                       mid_expand_ratio=mid_expand_ratio,
                       layers_num=layers_num,
                       kernel_sizes=kernel_sizes,
                       in_down_ratio=in_down_ratio,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg),
             bbox_head=dict(head_module=dict(widen_factor=widen_factor,
                                             in_channels=out_channels,
                                             feat_channels=out_channels,
                                             num_classes=num_classes,
                                             act_cfg=dict(inplace=True,
                                                          type='LeakyReLU')),
                            loss_bbox=dict(type='mmdet.DIoULoss',
                                           loss_weight=loss_bbox_weight)),
             train_cfg=dict(assigner=dict(num_classes=num_classes)))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    collate_fn=dict(_delete_=True, type='yolov5_collate'),
    sampler=dict(_delete_=True, type='DefaultSampler', shuffle=True),
    dataset=dict(
        data_root=data_root,
        # metainfo=metainfo,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix)))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(_delete_=True, type='DefaultSampler', shuffle=False),
    dataset=dict(
        data_root=data_root,
        ann_file=val_ann_file,
        # metainfo=metainfo,
        data_prefix=dict(img=val_data_prefix),
        test_mode=True))

test_dataloader = val_dataloader

auto_scale_lr = dict(enable=True, base_batch_size=train_batch_size_per_gpu)

val_evaluator = dict(  # Validation evaluator config
    type='mmdet.CocoMetric',  # The coco metric used to evaluate AR, AP, and mAP for detection
    proposal_nums=(100, 1, 10),  # The number of proposal used to evaluate for detection
    ann_file=val_ann_file,  # Annotation file path
    metric='bbox',  # Metrics to be evaluated, `bbox` for detection
)
test_evaluator = val_evaluator  # Testing evaluator config

default_hooks = dict(
    # Save weights every 10 epochs and a maximum of three weights can be saved.
    # The best model is saved automatically during model evaluation
    checkpoint=dict(interval=10, max_keep_ckpts=3, save_best='auto'),

    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    # param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    # param_scheduler=dict(
    #     type='YOLOv5ParamSchedulerHook',
    #     scheduler_type='cosine',
    #     lr_factor=0.005,
    #     max_epochs=max_epochs),
    # The log printing interval is 5

    logger=dict(type='LoggerHook', interval=5))
# The evaluation interval is 10
train_cfg = dict(max_epochs=max_epochs, val_interval=10)

visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])
