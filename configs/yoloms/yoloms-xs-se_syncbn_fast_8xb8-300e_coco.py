_base_ = './yoloms-xs_syncbn_fast_8xb8-300e_coco.py'
model = dict(backbone=dict(attention_cfg=dict(type="SE")))