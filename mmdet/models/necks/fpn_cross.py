import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
import torch
import numpy as np
from ..builder import NECKS


@NECKS.register_module
class FPNCROSS(nn.Module):
    """by weighting bottom-up and top-down process outputs
    (bottom-up  by multiplying )  current (top-down  by multiplying ) => 1*1 conv => relu => global avg pool => softmax
    """

    def __init__(self,
                 in_channels,
                 num_levels,
                 conv_cfg=None,
                 norm_cfg=None):
        super(FPNCROSS, self).__init__()

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.high_basic_conv = ConvModule(
            self.in_channels,
            self.num_levels - int(self.num_levels * 0.5),
            #self.num_levels,
            3, 
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            inplace=False
        )
        
        self.low_basic_conv = ConvModule(
            self.in_channels,
            int(self.num_levels * 0.5),
            3, 
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            inplace=False
        )
        

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == self.num_levels

        mid = int(self.num_levels * 0.5)
        low_level = int((mid-1) * 0.5)
        high_level = int((self.num_levels+mid) * 0.5)
        high_size = inputs[high_level].shape[2:]
        low_size = inputs[low_level].shape[2:]
        # high_size = inputs[mid].shape[2:]
        
        high_feats = []
        for i in range(mid, self.num_levels):
            if i < mid:
                gathered = F.adaptive_max_pool2d(
                    inputs[i], output_size=high_size)
            else:
                gathered = F.interpolate(inputs[i], size=high_size)
            high_feats.append(gathered)
        high_feats = sum(high_feats) / (self.num_levels - mid)
        b, c, h, w = high_feats.size()
        high_basic_map = self.high_basic_conv(high_feats)
        basic_max = torch.max(torch.max(high_basic_map, dim=2).values, dim=2).values.view(b, high_basic_map.size(1), -1, 1)
        basic_min = torch.min(torch.min(high_basic_map, dim=2).values, dim=2).values.view(b, high_basic_map.size(1), -1, 1)
        high_basic_map = (high_basic_map - basic_min) / (basic_max - basic_min + 0.0000001)
        
        high_avg_map = (torch.sum(high_feats, dim=1) / c).view(b, 1, h, w)
        la_avg = F.adaptive_avg_pool2d(high_avg_map, output_size=[1, w]).view(b, 1, w)
        ho_avg = F.adaptive_avg_pool2d(high_avg_map, output_size=[h, 1]).view(b, 1, h)
        high_avg_map = ho_avg[..., None] * la_avg[:, :, None, :]
        avg_max = torch.max(torch.max(high_avg_map, dim=2).values, dim=2).values.view(b, high_avg_map.size(1), -1, 1)
        avg_min = torch.min(torch.min(high_avg_map, dim=2).values, dim=2).values.view(b, high_avg_map.size(1), -1, 1)
        high_distance_map = torch.cos((high_basic_map - high_avg_map) * np.pi / 2)
        # print(high_distance_map.size())
        
        low_feats = []
        for i in range(mid):
            if i < low_level:
                gathered = F.adaptive_max_pool2d(
                    inputs[i], output_size=low_size)
            else:
                gathered = F.interpolate(
                    inputs[i], size=low_size, mode='nearest')
            low_feats.append(gathered)
        low_feats = sum(low_feats) / mid
        _, _, sh, sw = low_feats.size()
        b, c, h, w = low_feats.size()
        low_basic_map = self.low_basic_conv(low_feats)
        basic_max = torch.max(torch.max(low_basic_map, dim=2).values, dim=2).values.view(b, low_basic_map.size(1), -1, 1)
        basic_min = torch.min(torch.min(low_basic_map, dim=2).values, dim=2).values.view(b, low_basic_map.size(1), -1, 1)
        low_basic_map = (low_basic_map - basic_min) / (basic_max - basic_min + 0.0000001)
        
        low_avg_map = (torch.sum(low_feats, dim=1) / c).view(b, 1, h, w)
        la_avg = F.adaptive_avg_pool2d(low_avg_map, output_size=[1, w]).view(b, 1, w)
        ho_avg = F.adaptive_avg_pool2d(low_avg_map, output_size=[h, 1]).view(b, 1, h)
        low_avg_map = ho_avg[..., None] * la_avg[:, :, None, :]
        avg_max = torch.max(torch.max(low_avg_map, dim=2).values, dim=2).values.view(b, low_avg_map.size(1), -1, 1)
        avg_min = torch.min(torch.min(low_avg_map, dim=2).values, dim=2).values.view(b, low_avg_map.size(1), -1, 1)
        low_distance_map = torch.cos((low_basic_map - low_avg_map) * np.pi / 2)
        # print(low_distance_map.size())
        
        outs = []
        for i in range(self.num_levels):
            out_size = inputs[i].size()[2:]
            if i >= mid:
                residual = high_feats + high_feats * high_distance_map[:, i-mid, :, :].view(b, 1, high_feats.size(2), high_feats.size(3))
                if i < high_level:
                    residual = F.interpolate(residual, size=out_size)
                else:
                    residual = F.adaptive_max_pool2d(residual, output_size=out_size)
            else:
                residual = low_feats + low_feats * low_distance_map[:, i, :, :].view(b, 1, low_feats.size(2), low_feats.size(3))
                if i < low_level:
                    residual = F.interpolate(residual, size=out_size)
                else:
                    residual = F.adaptive_max_pool2d(residual, output_size=out_size)
            #residual = high_feats + high_feats * high_distance_map[:, i-mid, :, :].view(b, 1, high_feats.size(2), high_feats.size(3))
            #if i < mid:
            #    residual = F.interpolate(residual, size=out_size)
            #else:
            #    residual = F.adaptive_max_pool2d(residual, output_size=out_size)
            outs.append(inputs[i] + residual)

        return tuple(outs)
