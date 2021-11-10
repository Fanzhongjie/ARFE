import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
import torch
import numpy as np
from mmdet.ops import NonLocal2D
from ..builder import NECKS


@NECKS.register_module
class WFPNLargeKernel(nn.Module):
    """by weighting bottom-up and top-down process outputs
    (bottom-up  by multiplying )  current (top-down  by multiplying ) => 1*1 conv => relu => global avg pool => softmax
    """

    def __init__(self,
                 in_channels,
                 num_levels,
                 refine_level=2,
                 conv_cfg=None,
                 norm_cfg=None):
        super(WFPNLargeKernel, self).__init__()

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.refine_level = refine_level

        # reduce channels
        self.com_convs = nn.ModuleList()
        self.hor_convs = nn.ModuleList()
        self.ver_convs = nn.ModuleList()
        self.all_convs = nn.ModuleList()
        self.reduce_convs = nn.ModuleList()
        self.reduce_convs2 = nn.ModuleList()

        for i in range(num_levels):     
            self.com_convs.append(ConvModule(
                self.in_channels,
                1,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                inplace=False
            ))
            self.ver_convs.append(ConvModule(
                1,
                1,
                (1, 7),
                padding=(0, 3),
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                inplace=False
            ))
            self.hor_convs.append(ConvModule(
                1,
                1,
                (7, 1),
                padding=(3, 0),
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                inplace=False
            ))
            self.all_convs.append(ConvModule(
                1,
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
        
        pool_ = []
        pool_.append(F.adaptive_avg_pool2d(ori_fe, output_size=[1, 1]))
        pool_.append(F.adaptive_avg_pool2d(ori_fe, output_size=[2, 2]))
        pool_.append(F.adaptive_avg_pool2d(ori_fe, output_size=[3, 3]))
        pool_.append(F.adaptive_avg_pool2d(ori_fe, output_size=[6, 6]))
        for i in range(4):
            pool_[i] = F.relu(self.reduce_convs[i](pool_[i]))
            pool_[i] = F.interpolate(pool_[i], size=ori_fe.size()[2:])
        pool_out = torch.cat([ori_fe, pool_[0], pool_[1], pool_[2], pool_[3]], dim=1)
        
        bsf = self.refine(pool_out)

        outs = []
        for i in range(self.num_levels):
            h, w = inputs[i].size()[2:]
            input_ = F.relu(self.com_convs[i](inputs[i]))
            row_avg = F.adaptive_avg_pool2d(input_, output_size=[h, 1])
            col_avg = F.adaptive_avg_pool2d(input_, output_size=[1, w])
            row_avg = torch.tanh(self.hor_convs[i](row_avg))
            col_avg = torch.tanh(self.ver_convs[i](col_avg))
            
            att = F.relu(self.all_convs[i](row_avg + col_avg))

            outs.append(inputs[i] + F.interpolate(bsf, size=[h, w]) * att)
        return tuple(outs)
