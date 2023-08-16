_base_ = 'mmyolo::rtmdet/rtmdet_l_syncbn_fast_8xb32-300e_coco.py'

# ========================modified parameters======================
# Number of layer in MS-Block
layers_num=3
# The scaling factor that controls the depth of the network structure
deepen_factor = 2/3
# The scaling factor that controls the width of the network structure
widen_factor = 0.8

# Input channels of PAFPN
in_channels=[320, 640, 1280]
# Middle channels of PAFPN
mid_channels=[160, 320, 640]
# Output channels of PAFPN
out_channels = 240

# =======================Unmodified in most cases==================
# Batch size of a single GPU during training
train_batch_size_per_gpu = 8

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
kernel_sizes = [1,(3,3),(3,3)]

loss_bbox_weight = 2.0

model = dict(
    backbone=dict(
        _delete_=True,
        type='YOLOMS',
        arch='C3-K3579',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_expand_ratio=in_expand_ratio,
        mid_expand_ratio=mid_expand_ratio,
        layers_num=layers_num,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg),
    neck=dict(
        _delete_=True,
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
        act_cfg=act_cfg
        ),
    bbox_head=dict(
        head_module=dict(
            widen_factor=widen_factor,
            in_channels=out_channels,
            feat_channels=out_channels,
            act_cfg=dict(inplace=True, type='LeakyReLU')),
        loss_bbox=dict(type='mmdet.DIoULoss', loss_weight=loss_bbox_weight))
)

train_dataloader = dict(batch_size=train_batch_size_per_gpu)
auto_scale_lr = dict(enable=True, base_batch_size=32*8)