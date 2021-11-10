import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
import torch
import numpy as np
from ..builder import NECKS


@NECKS.register_module
class WFPNPreTreat(nn.Module):
    def __init__(self,
                 in_channels,
                 num_levels,
                 conv_cfg=None,
                 norm_cfg=None):
        super(WFPNPreTreat, self).__init__()

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.fcs1 = nn.ModuleList()
        self.fcs2 = nn.ModuleList()
        self.fcs3 = nn.ModuleList()
        
        # self.out_convs = nn.ModuleList()
        
        for i in range(num_levels):
            self.fcs1.append(nn.Linear(self.in_channels, int(self.in_channels/16)))
            self.fcs2.append(nn.Linear(int(self.in_channels/16), self.in_channels))
            self.fcs3.append(nn.Linear(int(self.in_channels/16), 1))
            # self.out_convs.append(ConvModule(
            #     self.in_channels,
            #     self.in_channels,
            #     3,
            #     padding=1,
            #     conv_cfg=self.conv_cfg,
            #     norm_cfg=self.norm_cfg,
            #     inplace=False
            # ))
            
    def init_weights(self):
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         xavier_init(m, distribution='uniform')
        for module_list in [self.fcs1, self.fcs2, self.fcs3]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        assert len(inputs) == self.num_levels

        outs = []
        for i in range(self.num_levels):
            c_out = F.adaptive_avg_pool2d(inputs[i], output_size=[1, 1]).view(inputs[i].size(0), -1)
            c_fc1 = self.fcs1[i](c_out)
            c_fc2 = self.fcs2[i](c_fc1)
            c_fc3 = self.fcs3[i](c_fc1)
            out_ = inputs[i] + inputs[i] * c_fc2.view(inputs[i].size(0), -1, 1, 1) + c_fc3.view(inputs[i].size(0), -1, 1, 1)
            outs.append(out_)
            # x_ = F.relu(self.channel_convs[i](inputs[i]))
            # x_ = F.adaptive_avg_pool2d(x_, output_size=[1, 1])
            # x_ = F.relu(self.spatial_convs[i](inputs[i] * x_))
            # outs.append(inputs[i] + inputs[i] * x_)
        
        return tuple(outs)
