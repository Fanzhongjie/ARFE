import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, ConvModule
import torch
import numpy as np

from mmdet.core import auto_fp16
from ..builder import NECKS


@NECKS.register_module
class ATTFF2(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(ATTFF2, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        
        # # seperate method after anticipation
        # self.sep_convs = nn.ModuleList()
        # # anticipate the original feature layer
        # self.pre_convs = nn.ModuleList()
        # for i in range(num_outs):
        #     self.pre_convs.append(ConvModule(
        #         self.in_channels[i] if i<len(in_channels) else self.in_channels[-1],
        #         self.out_channels,
        #         3,
        #         padding=1,
        #         norm_cfg=norm_cfg,
        #         act_cfg=self.activation,
        #         inplace=False
        #     ))
        #
        #     self.sep_convs.append(ConvModule(
        #         # self.in_channels[i] if i<len(in_channels) else self.in_channels[-1],
        #         self.out_channels,
        #         1,
        #         1,
        #         norm_cfg=norm_cfg,
        #         act_cfg=self.activation,
        #         # inplace=False
        #     ))
        
        # self.con_conv = ConvModule(
        #     self.num_outs,
        #     self.num_outs,
        #     3,
        #     padding=1,
        #     norm_cfg=norm_cfg,
        #     act_cfg=self.activation,
        #     # inplace=False
        # )

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(num_outs):
            l_conv = ConvModule(
                self.in_channels[i] if i<len(in_channels) else self.in_channels[-1],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=self.activation,
                # inplace=False
            )
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=self.activation,
                # inplace=False
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        used_backbone_levels = len(inputs)
        att_map = [inputs[i] for i in range(used_backbone_levels)]

        # extend to all layers
        if self.num_outs > used_backbone_levels:
            for i in range(self.num_outs - used_backbone_levels):
                att_map.append(F.max_pool2d(inputs[-1], 1, stride=2))

        # build laterals
        att_map = [
            lateral_conv(att_map[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # # [b, 1, h, w]
        # pre_map = [
        #     pre_conv(tmp_layers[i])
        #     for i, pre_conv in enumerate(self.pre_convs)
        # ]
        # # attention map's initialization
        # att_map = [
        #     sep_conv(pre_map[i])
        #     for i, sep_conv in enumerate(self.sep_convs)
        # ]

        b = att_map[0].size(0)
        att_maps = []
        for j in range(self.num_outs):
            h, w = att_map[j].size()[2:]
            
            att_tmp = torch.sum(att_map[j], dim=1, keepdim=True)
            for i in range(b):
                # normalization                
                tmp_ = att_tmp[i, ...].view(1, 1, h, w)
                max_ = torch.max(tmp_)
                min_ = torch.min(tmp_)
                tmp_ = (tmp_ - min_) / (max_ - min_ + 1.0)
                if i == 0:
                    att_ = tmp_
                else:
                    att_ = torch.cat([att_, tmp_], dim=0)
            att_maps.append((att_))
        
        laterals = [i for i in range(self.num_outs)]
        # outs = [i for i in range(self.num_outs)]
        for i in range(self.num_outs-1, -1, -1):
            out_size = att_map[i].size()[2:]
            if i != self.num_outs-1:
                # use cosin to compute the distance map
                big_distance_map = torch.cos((att_maps[i] - F.interpolate(att_maps[i+1], size=out_size, mode='nearest')) * np.pi / 2)
            if i == self.num_outs-1:
                # the top layer do not need to take distance map into consider,
                # just original plus the channel-wise value.( Just like FPN + GMP )
                laterals[i] = att_map[i] + \
                              F.adaptive_max_pool2d(att_map[i], output_size=[1, 1])
            else:
                # distance map is used as attentioned weights.
                laterals[i] = att_map[i] + \
                              F.interpolate(att_map[i+1], size=out_size, mode='nearest') * (-big_distance_map + 1.0) + \
                              F.adaptive_max_pool2d(att_map[i], output_size=[1, 1])
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(self.num_outs)
        ]

        return tuple(outs)
