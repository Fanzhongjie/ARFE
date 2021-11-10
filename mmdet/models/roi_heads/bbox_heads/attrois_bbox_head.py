import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, xavier_init
import torch.nn.functional as F
from mmdet.models.builder import HEADS
from .bbox_head import BBoxHead
from mmdet.ops import NonLocal2D

@HEADS.register_module()
class AttBBoxHead(BBoxHead):
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
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super(AttBBoxHead, self).__init__(*args, **kwargs)
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

        # self.alpha = torch.nn.Parameter(torch.ones(1)).cuda()
        # self.alpha = torch.autograd.Variable(torch.FloatTensor([1])).cuda()
        self.channel_reduction = ConvModule(
            self.conv_out_channels,
            1,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            inplace=False
        )
        self.fc1 = nn.Linear(49, 49)
        """
        self.glb_extra_conv = ConvModule(
            self.conv_out_channels,
            self.conv_out_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            inplace=False
        )
        self.glb_reduction_conv = ConvModule(
            self.conv_out_channels,
            1,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            inplace=False
        )
        self.specific_attention_conv = ConvModule(
            self.conv_out_channels,
            1,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            inplace=False
        )
        self.final_refine_conv = NonLocal2D(
            self.conv_out_channels,
            reduction=1,
            use_scale=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)
        """
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
        super(AttBBoxHead, self).init_weights()
        # conv layers are already initialized by ConvModule
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        # torch.nn.init.constant_(self.alpha, 0.5)

    def forward(self, x):
        # N * N * 7 * 7 ==> N * N (the degree of relationship) & N * 7 * 7 (global general / specific)
        # first part: extract global info either N*7*7 or N*1/channel_num
        # alternative operations: global pooling, channel-wise mean, etc.
        # one: N * 7 * 7
        # by using channel mean

        """
        global refined instance feature map: 1 * c * 7 * 7
        attention maps : N * 1 * 7 * 7
        metric?
        """
        rdt_x = self.relu(self.channel_reduction(x)).view(x.size(0), -1)
        rtf_x = torch.softmax(self.fc1(rdt_x), dim=-1)
        att_x = rtf_x.mm(rdt_x.transpose(0, 1))
        att_x = torch.softmax(att_x, dim=-1)
        ref_x = att_x.mm(rdt_x)
        x = x + ref_x.view(x.size(0), 1, x.size(2), -1)
        
        # glb_feat = self.glb_extra_conv(x)
        # x_g = self.relu(self.glb_reduction_conv(x) ).view(x.size(0), -1) # [N, 7*7]
        # x_c = F.adaptive_avg_pool2d(glb_feat, output_size=[1, 1])       # N * c * 1 * 1
        # x_g = torch.softmax(torch.mean(x_g, dim=-1).view(-1), dim=0).view(x.size(0), 1, 1, 1)
        # glb_feat = torch.sum(glb_feat * x_c * x_g, dim=0).view(1, x.size(1), x.size(2), x.size(3))
        # glb_feat = self.final_refine_conv(glb_feat)
        # att_map = self.relu(self.specific_attention_conv(x))
        # print(att_map * glb_feat)
        # input()
        # print(self.alpha)
        # print(att_map * glb_feat)
        # x = x * 0.7 + (att_map * glb_feat) * 0.3# * self.alpha


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
        return cls_score, bbox_pred

@HEADS.register_module()
class AttRoIsBBoxHead(AttBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(AttRoIsBBoxHead, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
