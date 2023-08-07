_base_ = './yoloms_yolov6_s_syncbn_fast_8xb32-300e_coco.py'

# ======================= Possible modified parameters =======================
layers_num = 1
widen_factor = 0.35


# ============================== Unmodified in most cases ===================
model = dict(
    backbone=dict(layers_num=layers_num, 
                  widen_factor=widen_factor),
    neck=dict(layers_num=layers_num,
              widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor),
                   loss_bbox=dict(iou_mode='siou'))
    )
