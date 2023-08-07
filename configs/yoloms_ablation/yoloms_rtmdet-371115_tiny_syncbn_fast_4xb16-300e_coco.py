_base_ = '../yoloms/yoloms_rtmdet_tiny_syncbn_fast_8xb8-300e_coco.py'

model = dict(
    backbone=dict(
        arch='C3-K371115-80'),
)

train_batch_size_per_gpu = 16
train_dataloader = dict(batch_size=train_batch_size_per_gpu)