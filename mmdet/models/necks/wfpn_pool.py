import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
import torch
import numpy as np
from mmdet.ops import NonLocal2D
from ..builder import NECKS


@NECKS.register_module
class WFPNPool(nn.Module):
    """by weighting bottom-up and top-down process outputs
    (bottom-up  by multiplying )  current (top-down  by multiplying ) => 1*1 conv => relu => global avg pool => softmax
    """

    def __init__(self,
                 in_channels,
                 num_levels,
                 refine_level=2,
                 conv_cfg=None,
                 norm_cfg=None):
        super(WFPNPool, self).__init__()

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.refine_level = refine_level

        # reduce channels
        self.sta_convs = nn.ModuleList()
        self.end_convs = nn.ModuleList()
        
        self.reduce_convs1 = nn.ModuleList()
        self.reduce_convs2 = nn.ModuleList()
        
        for i in range(4):  # 1 2 3 6
            self.sta_convs.append(ConvModule(
                self.in_channels,
                self.in_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                inplace=False
            ))
            self.end_convs.append(ConvModule(
                self.in_channels,
                int(self.in_channels / 4),
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                inplace=False
            ))
        
        for i in range(num_levels):
            self.reduce_convs1.append(ConvModule(
                self.in_channels,
                1,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                inplace=False
            ))
            self.reduce_convs2.append(ConvModule(
                self.in_channels,
                1,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                inplace=False
            ))
            
        self.refine = ConvModule(
            self.in_channels * 2,
            self.in_channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            inplace=False
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == self.num_levels

        feats = []
        gather_size = inputs[self.refine_level].size()[2:]
        for i in range(self.num_levels):
            if i < self.refine_level:
                gathered = F.adaptive_max_pool2d(
                    inputs[i], output_size=gather_size)
            else:
                gathered = F.interpolate(
                    inputs[i], size=gather_size, mode='nearest')
            feats.append(gathered)

        ori_fe = sum(feats) / len(feats)
        
        pool_ = [i for i in range(4)]
        pool_size = [[1, 1], [2, 2], [3, 3], [6, 6]]
        for i in range(4):
            pool_[i] = F.relu(self.sta_convs[i](ori_fe))
            pool_[i] = F.adaptive_avg_pool2d(pool_[i], output_size=pool_size[i])
            pool_[i] = F.relu(self.end_convs[i](pool_[i]))
            pool_[i] = F.interpolate(pool_[i], size=ori_fe.size()[2:])

        pool_out = torch.cat([ori_fe, pool_[0], pool_[1], pool_[2], pool_[3]], dim=1)
        bsf = self.refine(pool_out)

        outs = []
        for i in range(self.num_levels):
            b, c, h, w = inputs[i].size()
            basic_map = torch.tanh(self.reduce_convs1[i](inputs[i]))     # b, 1, h, w
            com_map = torch.tanh(self.reduce_convs2[i](inputs[i]))
            attention_map = F.interpolate(bsf, size=[h, w]) * (basic_map + com_map)
            outs.append(inputs[i] + attention_map)
        
        return tuple(outs)
