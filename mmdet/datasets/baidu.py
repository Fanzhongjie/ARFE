from .coco import CocoDataset
from .builder import DATASETS


@DATASETS.register_module
class BaiduDataset(CocoDataset):

    CLASSES = ('rect_eye', 'sphere_eye')