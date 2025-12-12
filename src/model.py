_base_ = '../yolov8/yolov8_l_syncbn_fast_8xb16-500e_coco.py'
data_root = '../ship_512_50x50_cleaned/'  # Root path of data
max_epochs = 150
train_batch_size_per_gpu = 8
train_num_workers = 8  # train_num_workers = nGPU x 4

save_epoch_intervals = 1
deepen_factor = 1.00
widen_factor = 1.25

work_dir = './work_dirs/ship_512_50x50_cleaned'

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals,
  )

class_name = ('ship',)
num_classes = len(class_name)

# base_lr_default * (your_bs / default_bs)
base_lr = 0.00001
metainfo = dict(classes=class_name, palette=[(255, 0, 0)])
# load_from = "./work_dirs/yolov8_ship_overlap/best_coco/bbox_mAP_epoch_1.pth"
load_from="./work_dirs/bbox_mAP_epoch_11.pth"
#load_from = "https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_x_syncbn_fast_8xb16-500e_coco/yolov8_x_syncbn_fast_8xb16-500e_coco_20230218_023338-5674673c.pth"

model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(num_classes=num_classes,widen_factor=widen_factor)),
    train_cfg=dict(assigner=dict(num_classes=num_classes)))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=_base_.dataset_type,
            data_root=data_root,
            metainfo=metainfo,
            ann_file='train.json',
            data_prefix=dict(img='train/'),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=_base_.train_pipeline)))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img='val/')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'val.json')
test_evaluator = val_evaluator

optim_wrapper = dict(optimizer=dict(lr=base_lr))


# optim_wrapper = dict(
#     type='OptimWrapper',
#     clip_grad=dict(max_norm=10.0),
#     optimizer=dict(
#         type='SGD',
#         lr=base_lr,
#         momentum=0.937,
#         weight_decay=0.0005,
#         nesterov=True,
#         batch_size_per_gpu=train_batch_size_per_gpu),
#     constructor='YOLOv5OptimizerConstructor')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals,
        max_keep_ckpts=2,
        save_best='auto'),
    param_scheduler=dict(max_epochs=max_epochs),
    logger=dict(type='LoggerHook', interval=10))
