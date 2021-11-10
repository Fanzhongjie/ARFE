import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
import torch
import numpy as np
from mmdet.ops import NonLocal2D
from ..builder import NECKS


@NECKS.register_module
class FPNRelation(nn.Module):
    """by weighting bottom-up and top-down process outputs
    (bottom-up  by multiplying )  current (top-down  by multiplying ) => 1*1 conv => relu => global avg pool => softmax
    """

    def __init__(self,
                 in_channels,
                 num_levels,
                 conv_cfg=None,
                 norm_cfg=None):
        super(FPNRelation, self).__init__()

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.en_convs = nn.ModuleList()
        self.com_convs = nn.ModuleList()
        for i in range(2):
            self.en_convs.append(ConvModule(
                in_channels,
                1,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                inplace=False))
            self.com_convs.append(ConvModule(
                in_channels,
                1,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                inplace=False))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == self.num_levels

        ###
        # compute pixel-wise same objectness
        ###
        b, c, h, w = inputs[2].size()
        input_ = F.adaptive_avg_pool2d(inputs[0], output_size=[h, w])
        object_map1 = F.relu(self.com_convs[0](input_)).view(b, -1, 1)
        object_map2 = F.relu(self.com_convs[1](input_)).view(b, 1, -1)
        object_map = object_map1.mul(object_map2).view(b, h*w, h*w)
        object_map = torch.sum(object_map, dim=-1) / (h*w)
        object_map = object_map.view(b, 1, h, w)
        
        ###
        # compute pixel-wise same classification
        ###
        class_map1 = F.relu(self.en_convs[0](inputs[2])).view(b, -1, 1)
        class_map2 = F.relu(self.en_convs[1](inputs[2])).view(b, 1, -1)
        class_map = class_map1.mul(class_map2).view(b, h*w, h*w)
        class_map = torch.sum(class_map, dim=-1) / (h*w)
        class_map = class_map.view(b, 1, h, w)
        
        outs = []
        for i in range(self.num_levels):
            outs.append(inputs[i] + \
                          F.interpolate(object_map, size=inputs[i].size()[2:]) + \
                          F.interpolate(class_map, size=inputs[i].size()[2:]))
        return tuple(outs)
