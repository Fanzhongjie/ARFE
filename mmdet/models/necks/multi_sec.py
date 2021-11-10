import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init

from ..builder import NECKS


@NECKS.register_module()
class MULTISEC(nn.Module):
    def __init__(self,
                 in_channels,
                 num_levels,
                 conv_cfg=None,
                 act_cfg=None,
                 norm_cfg=None):
        super(MULTISEC, self).__init__()

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.fir_convs = nn.ModuleList()
        self.sec_convs = nn.ModuleList()

        for i in range(self.num_levels):
            fir_conv = ConvModule(
                in_channels * (3 if 0 < i < self.num_levels-1 else 2),
                in_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            sec_conv = ConvModule(
                in_channels * (3 if 0 < i < self.num_levels - 1 else 2),
                in_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.fir_convs.append(fir_conv)
            self.sec_convs.append(sec_conv)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == self.num_levels

        fir_outs = [i for i in range(self.num_levels)]
        for i in range(self.num_levels):
            tmp_layer = inputs[i]
            adp_size = inputs[i].shape[2:]
            if i > 0:
                tmp_layer = torch.cat((F.interpolate(inputs[i-1], size=adp_size), tmp_layer), dim=1)
            if i < self.num_levels-1:
                tmp_layer = torch.cat((tmp_layer, F.interpolate(inputs[i + 1], size=adp_size)), dim=1)
            fir_outs[i] = F.relu(self.fir_convs[i](tmp_layer))

        sec_outs = [i for i in range(self.num_levels)]
        for i in range(self.num_levels):
            tmp_layer = fir_outs[i]
            adp_size = fir_outs[i].shape[2:]
            if i > 0:
                tmp_layer = torch.cat((F.interpolate(fir_outs[i - 1], size=adp_size), tmp_layer), dim=1)
            if i < self.num_levels - 1:
                tmp_layer = torch.cat((tmp_layer, F.interpolate(fir_outs[i + 1], size=adp_size)), dim=1)
            sec_outs[i] = F.relu(self.sec_convs[i](tmp_layer)) + (
                    F.adaptive_avg_pool2d(inputs[i], output_size=[1, 1]) +
                    F.adaptive_max_pool2d(inputs[i], output_size=[1, 1])) * 0.5
        
        return tuple(sec_outs)
