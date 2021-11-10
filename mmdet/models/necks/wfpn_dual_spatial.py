import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
import torch
import numpy as np
from mmdet.ops import NonLocal2D
from ..builder import NECKS


@NECKS.register_module
class WFPNDualSpatial(nn.Module):
    """by weighting bottom-up and top-down process outputs
    (bottom-up  by multiplying )  current (top-down  by multiplying ) => 1*1 conv => relu => global avg pool => softmax
    """

    def __init__(self,
                 in_channels,
                 num_levels,
                 refine_level=2,
                 conv_cfg=None,
                 norm_cfg=None):
        super(WFPNDualSpatial, self).__init__()

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.refine_level = refine_level

        # reduce channels
        self.reduce_convs = nn.ModuleList()
        self.reduce_convs2 = nn.ModuleList()
        
        # self.reduce_convs3 = nn.ModuleList()
        # self.reduce_convs4 = nn.ModuleList()
        
        for i in range(num_levels):
            self.reduce_convs.append(ConvModule(
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
            
            '''
            if i < num_levels-1:
                self.reduce_convs3.append(ConvModule(
                    self.in_channels,
                    1,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=False
                ))
            '''
            # self.reduce_convs4.append(ConvModule(
            #     self.in_channels,
            #     1,
            #     3,
            #     padding=1,
            #     conv_cfg=self.conv_cfg,
            #     norm_cfg=self.norm_cfg,
            #     inplace=False
            # ))
        self.refine = NonLocal2D(
            self.in_channels,
            reduction=1,
            use_scale=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)
        # self.refine = ConvModule(
        #     self.in_channels,
        #     self.in_channels,
        #     3,
        #     padding=1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     inplace=False
        # )

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
        # bsf = ori_fe
        bsf = self.refine(ori_fe)
        
        outs = []
        for i in range(self.num_levels):        
            b, c, h, w = inputs[i].size()
            basic_map = torch.tanh(self.reduce_convs[i](inputs[i]))     # b, 1, h, w
            com_map = torch.tanh(self.reduce_convs2[i](inputs[i]))
            # tr_map = torch.tanh(self.reduce_convs3[i](inputs[i]))
            # fu_map = torch.tanh(self.reduce_convs4[i](inputs[i]))
            # attention_map = F.interpolate(ori_fe, size=[h, w]) * (basic_map + com_map)
            attention_map = F.interpolate(bsf, size=[h, w]) * (basic_map + com_map)
            
            '''
            if i < self.num_levels-1:
                dif_map = torch.abs(self.reduce_convs3[i](inputs[i])-F.interpolate(inputs[i+1], size=[h, w]))
                attention_map = attention_map + F.interpolate(bsf, size=[h, w]) * dif_map
            '''
            
            # attention_map = F.interpolate(bsf, size=[h, w]) * (basic_map + com_map + tr_map + fu_map)
            # outs.append(F.relu(self.out_convs[i](inputs[i] + attention_map)))
            outs.append(inputs[i] + attention_map)
        
        return tuple(outs)
