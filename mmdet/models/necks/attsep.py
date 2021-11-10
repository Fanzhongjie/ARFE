import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
import torch
from ..builder import NECKS
from mmdet.ops import NonLocal2D


@NECKS.register_module
class ATTSEP(nn.Module):
    """Separate fusion & attention injection"""

    def __init__(self,
                 in_channels,
                 num_levels,
                 conv_cfg=None,
                 norm_cfg=None):
        super(ATTSEP, self).__init__()

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # self.refine = NonLocal2D(
        #     self.in_channels,
        #     reduction=1,
        #     use_scale=False,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg)
        self.com_convs = nn.ModuleList()

        for i in range(num_levels):
            c_conv = ConvModule(
                in_channels,
                1,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                inplace=False)
            self.com_convs.append(c_conv)
        

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == self.num_levels

        # step 1: gather multi-level features by resize and average
        # 5 layers
        high_feat = inputs[2]
        high_feat = high_feat + F.interpolate(inputs[3], size=inputs[2].size()[2:])
        high_feat = high_feat + F.interpolate(inputs[4], size=inputs[2].size()[2:])
        # high_feat = self.refine(high_feat)
        # high_feat = self.att_self(high_feat, self.high_com_conv, self.high_com_conv2)

        low_feat = F.adaptive_max_pool2d(inputs[1], output_size=inputs[2].size()[2:])
        low_feat = low_feat + F.adaptive_max_pool2d(inputs[0], output_size=inputs[2].size()[2:])
        # low_feat = self.refine(low_feat)
        # low_feat = self.att_self(low_feat, self.low_com_conv, self.low_com_conv2)

        outs = []
        for i in range(self.num_levels):
            if i < int(self.num_levels/2):
                outs.append(inputs[i] + (torch.tanh(self.com_convs[i](inputs[i]))) * F.interpolate(
                                            high_feat, size=inputs[i].size()[2:]))
                # outs.append(inputs[i] + F.interpolate(high_feat, size=inputs[i].size()[2:]))
            else:
                outs.append(inputs[i] + (torch.tanh(self.com_convs[i](inputs[i]))) * F.adaptive_max_pool2d(
                    low_feat, output_size=inputs[i].size()[2:]))
                # outs.append(inputs[i] + F.adaptive_max_pool2d(low_feat, output_size=inputs[i].size()[2:]))

        return tuple(outs)
