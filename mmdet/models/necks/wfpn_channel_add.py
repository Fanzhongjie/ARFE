import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
import torch
import numpy as np
from mmdet.ops import NonLocal2D
from ..builder import NECKS


@NECKS.register_module
class WFPNChannelADD(nn.Module):
    """by weighting bottom-up and top-down process outputs
    (bottom-up  by multiplying )  current (top-down  by multiplying ) => 1*1 conv => relu => global avg pool => softmax
    """

    def __init__(self,
                 in_channels,
                 num_levels,
                 refine_level=3,
                 conv_cfg=None,
                 norm_cfg=None):
        super(WFPNChannelADD, self).__init__()

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.refine_level = refine_level

        # reduce channels
        self.reduce_convs = nn.ModuleList()
        self.self_bn_convs = nn.ModuleList()
        self.self_update_convs = nn.ModuleList()
        self.final_convs = nn.ModuleList()
        self.fcs1 = nn.ModuleList()
        self.fcs2 = nn.ModuleList()
        self.fcs3 = nn.ModuleList()
        
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
            self.final_convs.append(ConvModule(
                self.in_channels,
                self.in_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                inplace=False
            ))
            self.fcs1.append(nn.Linear(self.in_channels, int(self.in_channels/16)))
            self.fcs2.append(nn.Linear(int(self.in_channels/16), self.in_channels))
            self.fcs3.append(nn.Linear(int(self.in_channels/16), 1))
            
        self.refine = NonLocal2D(
                self.in_channels,
                reduction=1,
                use_scale=False,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)

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
        bsf = self.refine(ori_fe)

        outs = []
        for i in range(self.num_levels):
            b, c, h, w = inputs[i].size()
            basic_map = self.reduce_convs[i](inputs[i])     # b, 1, h, w
            basic_map = F.relu(basic_map, inplace=False)
            # row_avg = F.adaptive_avg_pool2d(inputs[i], output_size=[h, 1])
            # col_avg = F.adaptive_avg_pool2d(inputs[i], output_size=[1, w])
            row_avg = F.adaptive_avg_pool2d(inputs[i], output_size=[h, 1]).view(b, c, h)
            col_avg = F.adaptive_avg_pool2d(inputs[i], output_size=[1, w]).view(b, c, w)
            cha_att = F.adaptive_avg_pool2d(inputs[i], output_size=[1, 1])
            cha_att = torch.softmax(cha_att, dim=1)
            
            tmp_map = row_avg[..., None] * col_avg[:, :, None, :]
            tmp_map = torch.sum(tmp_map * cha_att, dim=1).view(b, 1, h, w)
            max_value = torch.max(torch.max(tmp_map, dim=2).values, dim=2).values.view(b, 1, 1, 1)
            min_value = torch.min(torch.min(tmp_map, dim=2).values, dim=2).values.view(b, 1, 1, 1)
            avg_map = (tmp_map - min_value) / (max_value-min_value + 0.0000001)
            
            max_b = torch.max(torch.max(basic_map, dim=2).values, dim=2).values.view(b, 1, 1, 1)
            min_b = torch.min(torch.min(basic_map, dim=2).values, dim=2).values.view(b, 1, 1, 1)
            basic_reg_map = (basic_map - min_b) / (max_b - min_b + 0.0000001)

            # take the distance between avg_map and basic_amp as the distance_map
            # or take the distance map multiplying ori_fe as the attention map
            distance_map = torch.cos((avg_map - basic_reg_map) * np.pi / 2).view(b, 1, h, w)
            # attention_map = distance_map * F.interpolate(bsf, size=[h, w], mode='nearest')
            attention_map = F.interpolate(bsf, size=[h, w]) * distance_map
            out_ = F.relu(self.final_convs[i](inputs[i] + attention_map))
            c_out = F.adaptive_avg_pool2d(out_, output_size=[1, 1]).view(out_.size(0), -1)
            c_fc1 = self.fcs1[i](c_out)
            c_fc2 = self.fcs2[i](c_fc1)
            c_fc3 = self.fcs3[i](c_fc1)
            outs.append(out_ + out_ * c_fc2.view(out_.size(0), -1, 1, 1) * c_fc3.view(out_.size(0), -1, 1, 1))
        
        return tuple(outs)
