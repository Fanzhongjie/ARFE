import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from .bbox_head import BBoxHead

from mmdet.core import (auto_fp16, build_bbox_coder, force_fp32, multi_apply,
                        multiclass_nms)
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy

@HEADS.register_module()
class MultiClassesBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 loss_multi_cls=dict(
                     type='CrossEntropyLoss',
                     use_multi_cls=True,
                     loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super(MultiClassesBBoxHead, self).__init__(*args, **kwargs)
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
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.loss_multi_cls = build_loss(loss_multi_cls)

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
        # self.fc_multi_cls = nn.Linear(self.cls_last_dim, self.num_classes + 1)
        # multiple RoIs fusion
        # [512, 7, 7, 256] -> [512, 1] -> [1, 7, 7, 256] --fc--> [1, 81]
        #       |                  |
        #       +---------sc-------+
        self.spa_conv = ConvModule(
            self.in_channels,
            1,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            inplace=False)
        self.refine_conv = ConvModule(
            self.in_channels,
            self.in_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            inplace=False)
        self.pre_fc = nn.Linear(7*7*256, 256)
        self.multi_cls_reg = nn.Linear(256, (self.num_classes + 1) * 2)

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
        super(MultiClassesBBoxHead, self).init_weights()
        # conv layers are already initialized by ConvModule
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_multi_cls = x
        cha_att = self.relu(F.adaptive_avg_pool2d(x_multi_cls, output_size=[1, 1]))
        x_multi_cls = x + x*cha_att
        spa_att = self.relu(self.spa_conv(x_multi_cls))  # [512, 1, 7, 7]
        mean_mat = x + x*spa_att
        # mean_mat = x*spa_att + x*cha_att
        final_feat = self.relu(self.refine_conv(mean_mat))
        # spa_att = torch.softmax(F.adaptive_max_pool2d(spa_att, output_size=[1, 1]), dim=0)  # [512, 1, 1, 1]
        # mean_mat = x_multi_cls + self.relu(torch.sum(x_multi_cls * spa_att, dim=0).view(1, 256, 7, 7))  # [7, 7, 256]
        x = final_feat

        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

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

        # spa_att = self.relu(self.spa_conv(x_multi_cls)) # [512, 7, 7, 1]
        # spa_att = F.adaptive_max_pool2d(spa_att, output_size=[1, 1])    # [512, 1, 1, 1]
        # mean_mat = torch.mean(x_multi_cls + x_multi_cls * spa_att, dim=0).view(1, -1)   # [7, 7, 256]
        fc1 = self.relu(self.pre_fc(torch.mean(mean_mat, dim=0).view(1, -1)))
        multi_cls = self.multi_cls_reg(fc1).view(-1, 2)
        # cls_score = torch.cat((cls_score, torch.softmax(multi_cls, dim=-1)[:, 1].view(1, -1)), dim=0)

        return cls_score, bbox_pred, multi_cls

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'multi_cls'))
    def loss(self,
             cls_score,
             bbox_pred,
             multi_cls,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    multi_cls_pred=multi_cls,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, labels)
        if multi_cls is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if multi_cls.numel() > 0:
                losses['loss_multi_cls'] = self.loss_multi_cls(
                    multi_cls,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
        return losses


@HEADS.register_module()
class Shared2FCMultiClassesBBoxHead(MultiClassesBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCMultiClassesBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)