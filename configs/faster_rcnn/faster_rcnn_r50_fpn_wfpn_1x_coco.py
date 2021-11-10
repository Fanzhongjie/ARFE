_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# resume_from='work_dirs/faster_rcnn_r50_fpn_wfpn_1x_coco/epoch_4.pth'