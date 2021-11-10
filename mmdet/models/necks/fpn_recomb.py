import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init

from ..builder import NECKS


@NECKS.register_module
class FPNRECOMB(nn.Module):
    def __init__(self,
                 in_channels,
                 num_levels,
                 num_convs=1,
                 refine_level=2,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super(FPNRECOMB, self).__init__()

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.num_convs = num_convs
        self.refine_level = refine_level

        self.mid_convs = nn.ModuleList()
        for i in range(self.num_convs):
            self.mid_convs.append(ConvModule(
                in_channels,
                in_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False
            ))

        self.compress_conv = ConvModule(
            in_channels,
            self.num_levels * self.num_levels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == self.num_levels

        # step 1: gather multi-level features by resize and average
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

        bsf = sum(feats) / len(feats)
        for i in range(self.num_convs):
            bsf = self.mid_convs[i](bsf)
        bsf = F.relu(self.compress_conv(bsf))
        bsf = F.adaptive_avg_pool2d(bsf, output_size=[1, 1])

        bsf = bsf.view(-1, self.num_levels, self.num_levels)
        ws = F.softmax(bsf, dim=1)
        outs = []
        for i in range(self.num_levels):
            out_size = inputs[i].shape[2:]
            tmp = inputs[i]
            for j in range(self.num_levels):
                if j != i:
                    tmp = tmp + F.interpolate(inputs[j], size=out_size) * ws[:, j, i].view(-1, 1, 1, 1).contiguous()
            outs.append(tmp)

        return tuple(outs)
