import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
import torch
import numpy as np
from mmdet.ops import NonLocal2D
from ..builder import NECKS


@NECKS.register_module
class WFPN(nn.Module):
    """by weighting bottom-up and top-down process outputs
    (bottom-up  by multiplying )  current (top-down  by multiplying ) => 1*1 conv => relu => global avg pool => softmax
    """

    def __init__(self,
                 in_channels,
                 num_levels,
                 refine_level=2,
                 conv_cfg=None,
                 norm_cfg=None):
        super(WFPN, self).__init__()

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.refine_level = refine_level

        # reduce channels
        self.reduce_convs = nn.ModuleList()
        
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
            basic_map = F.relu(self.reduce_convs[i](inputs[i]))     # b, 1, h, w
            row_avg = F.adaptive_avg_pool2d(inputs[i], output_size=[h, 1])
            col_avg = F.adaptive_avg_pool2d(inputs[i], output_size=[1, w])
            
            #row_avg = F.adaptive_avg_pool2d(inputs[i], output_size=[h, 1]).view(b, c, h)
            #col_avg = F.adaptive_avg_pool2d(inputs[i], output_size=[1, w]).view(b, c, w)
            #cha_att = F.adaptive_avg_pool2d(inputs[i], output_size=[1, 1])
            #cha_att = torch.softmax(cha_att, dim=1)
            
            # cha_att = cha_att.view(b, c)
            # pos = torch.argmax(cha_att, dim=1).view(b)   # pos scale: [b], fecth the value pos[i]
            
            #tmp_map = row_avg[..., None] * col_avg[:, :, None, :]
            #tmp_map = torch.sum(tmp_map * cha_att, dim=1).view(b, 1, h, w)
            # tmp_map = (torch.sum(tmp_map * cha_att, dim=1) / torch.sum(cha_att, dim=1)).view(b, 1, h, w)
            
            ## tmp_map = (torch.sum(tmp_map, dim=1)/c).view(b, 1, h, w)
            #max_value = torch.max(torch.max(tmp_map, dim=2).values, dim=2).values.view(b, 1, 1, 1)
            #min_value = torch.min(torch.min(tmp_map, dim=2).values, dim=2).values.view(b, 1, 1, 1)
            #avg_map = (tmp_map - min_value) / (max_value-min_value + 0.0000001)
            
            #max_b = torch.max(torch.max(basic_map, dim=2).values, dim=2).values.view(b, 1, 1, 1)
            #min_b = torch.min(torch.min(basic_map, dim=2).values, dim=2).values.view(b, 1, 1, 1)
            #basic_reg_map = (basic_map - min_b) / (max_b - min_b + 0.0000001)
            
            for j in range(b):
                tmp_map = row_avg[j, 0, :, :].mm(col_avg[j, 0, :, :]).view(1, 1, h, w)
                max_value = torch.max(tmp_map)
                min_value = torch.min(tmp_map)
                tmp_map = (tmp_map-min_value) / (max_value-min_value + 0.0000001)
                
                max_b = torch.max(basic_map[j, ...])
                min_b = torch.min(basic_map[j, ...])
                tmp_basic_map = (basic_map[j, ...] - min_b) / (max_b - min_b + 0.0000001)
                
                if j==0:
                    avg_map = tmp_map
                    basic_reg_map = tmp_basic_map.view(1, 1, h, w)
                else:
                    avg_map = torch.cat([avg_map, tmp_map], dim=0)
                    basic_reg_map = torch.cat([basic_reg_map, tmp_basic_map.view(1, 1, h, w)], dim=0)
            
            # take the distance between avg_map and basic_amp as the distance_map
            # or take the distance map multiplying ori_fe as the attention map
            distance_map = torch.cos((avg_map - basic_reg_map) * np.pi / 2).view(b, 1, h, w)
            # attention_map = distance_map * F.interpolate(bsf, size=[h, w], mode='nearest')
            attention_map = F.relu(F.interpolate(bsf, size=[h, w]) * distance_map)
            outs.append(inputs[i] + attention_map)
        return tuple(outs)
