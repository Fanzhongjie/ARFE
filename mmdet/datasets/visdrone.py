from .coco import CocoDataset
from .builder import DATASETS


@DATASETS.register_module
class VisdroneDataset(CocoDataset):

    CLASSES = ('pedestrian', 'people', 'bicycle', 'car',
               'van', 'truck', 'tricycle', 'awning-tricycle',
               'bus', 'motor', 'others', 'ignored regions')
