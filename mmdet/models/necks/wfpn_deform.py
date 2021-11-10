import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init, normal_init
import torch
import numpy as np
# from mmdet.ops import NonLocal2D
from ..builder import NECKS
from mmdet.ops import DeformConv


@NECKS.register_module
class WFPNDeform(nn.Module):
    """by weighting bottom-up and top-down process outputs
    (bottom-up  by multiplying )  current (top-down  by multiplying ) => 1*1 conv => relu => global avg pool => softmax
    """

    def __init__(self,
                 in_channels,
                 num_levels,
                 refine_level=2,
                 num_points=9,
                 gradient_mul=0.1,
                 conv_cfg=None,
                 norm_cfg=None):
        super(WFPNDeform, self).__init__()

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.refine_level = refine_level
        self.num_points = num_points
        self.gradient_mul = gradient_mul
        
        # reduce channels
        self.reduce_convs = nn.ModuleList()
        self.reduce_convs2 = nn.ModuleList()
        
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
        
        self.dcn_kernel = int(np.sqrt(num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == num_points, \
            'The points number should be a square number.'
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd square number.'
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
        
        pts_out_dim = 2 * self.num_points
        self.refine = DeformConv(self.in_channels,
                                 self.in_channels,
                                 self.dcn_kernel, 1, self.dcn_pad)
        self.reppoints_pts_init_conv = nn.Conv2d(self.in_channels,
                                                 self.in_channels, 3,
                                                 1, 1)
        self.reppoints_pts_init_out = nn.Conv2d(self.in_channels,
                                                pts_out_dim, 1, 1, 0)
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.refine, std=0.01)
        normal_init(self.reppoints_pts_init_conv, std=0.01)
        normal_init(self.reppoints_pts_init_out, std=0.01)

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
        
        dcn_base_offset = self.dcn_base_offset.type_as(ori_fe)
        pts_out_init = self.reppoints_pts_init_out(
            F.relu(self.reppoints_pts_init_conv(ori_fe)))
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach() + self.gradient_mul * pts_out_init
        dcn_offset = pts_out_init_grad_mul - dcn_base_offset
        
        bsf = F.relu(self.refine(ori_fe, dcn_offset))

        outs = []
        for i in range(self.num_levels):
            b, c, h, w = inputs[i].size()
            basic_map = torch.tanh(self.reduce_convs[i](inputs[i]))
            com_map = torch.tanh(self.reduce_convs2[i](inputs[i]))
            attention_map = F.interpolate(bsf, size=[h, w]) * (basic_map + com_map)
            outs.append(inputs[i] + attention_map)
        return tuple(outs)
