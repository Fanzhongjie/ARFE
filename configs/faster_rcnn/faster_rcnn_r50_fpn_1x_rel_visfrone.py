_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_relation_visdrone.py',
    '../_base_/datasets/visdrone_detection.py',
    # '../_base_/datasets/voc0712.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
