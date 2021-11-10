import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from .bbox_head import BBoxHead


@HEADS.register_module()
class MultiBBoxHeadConvs(BBoxHead):
    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 num_ws_convs=2,
                 num_ws_fcs=2,
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super(MultiBBoxHeadConvs, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.num_ws_convs = num_ws_convs
        self.num_ws_fcs = num_ws_fcs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        
        self.first_convs = nn.ModuleList()
        self.second_convs = nn.ModuleList()
        
        self.com_conv = ConvModule(
            self.in_channels * 2,
            self.in_channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            inplace=False
        )
        
        self.first_convs.append(ConvModule(
            self.in_channels,
            self.in_channels,
            (1, 3),
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            inplace=False
        ))
        self.first_convs.append(ConvModule(
            self.in_channels,
            self.in_channels,
            (3, 1),
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            inplace=False
        ))
        self.second_convs.append(ConvModule(
            self.in_channels,
            self.in_channels,
            (3, 1),
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            inplace=False
        ))
        self.second_convs.append(ConvModule(
            self.in_channels,
            self.in_channels,
            (1, 3),
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            inplace=False
        ))
        
        self.final_conv = ConvModule(
            self.in_channels,
            self.in_channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            inplace=False
        )
        

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes + 1)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                                                             self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(MultiBBoxHeadConvs, self).init_weights()
        # conv layers are already initialized by ConvModule
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        ori_rois = x[:, :self.conv_out_channels, :, :]
        add_rois = x[:, self.conv_out_channels:, :, :]
        
        com_ = F.relu(self.com_conv(add_rois))
        
        fir_ = self.first_convs[0](com_)
        fir_ = F.relu(self.first_convs[1](fir_))
        sec_ = self.second_convs[0](com_)
        sec_ = F.relu(self.second_convs[1](sec_))
        
        x_out = ori_rois + F.interpolate(fir_ + sec_, size=[ori_rois.size(2), ori_rois.size(3)])
        
        x_out = F.relu(self.final_conv(x_out))
        
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x_out = conv(x_out)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x_out = self.avg_pool(x_out)

            x_out = x_out.flatten(1)
            for fc in self.shared_fcs:
                x_out = self.relu(fc(x_out))

        # separate branches
        x_cls = x_out
        x_reg = x_out

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred


@HEADS.register_module()
class MultiRoIsBBoxHeadConvs(MultiBBoxHeadConvs):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(MultiRoIsBBoxHeadConvs, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)