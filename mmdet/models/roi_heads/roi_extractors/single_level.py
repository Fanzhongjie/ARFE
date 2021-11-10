import torch
import torch.nn as nn
import numpy as np

from mmdet import ops
from mmdet.core import force_fp32
from mmdet.models.builder import ROI_EXTRACTORS


@ROI_EXTRACTORS.register_module()
class SingleRoIExtractor(nn.Module):
    """Extract RoI features from a single level feature map.

    If there are mulitple input feature levels, each RoI is mapped to a level
    according to its scale.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0.
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56):
        super(SingleRoIExtractor, self).__init__()
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale
        self.fp16_enabled = False

    @property
    def num_inputs(self):
        """int: Input feature map levels."""
        return len(self.featmap_strides)

    def init_weights(self):
        pass

    def build_roi_layers(self, layer_cfg, featmap_strides):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        
        ### softer scale boundary
        ## ar = (rois[:, 3] - rois[:, 1]) / (rois[:, 4] - rois[:, 2])
        # lvl = scale / self.finest_scale / torch.pow(2, target_lvls)
        # target_lvls = torch.where(lvl < 1.1, target_lvls-1, target_lvls)
        # print(lvl.size())
        # for i in range(num_levels+1):
        #     iii = torch.where(target_lvls==torch.ones_like(lvl) * i, torch.ones_like(lvl), torch.zeros_like(lvl))
        #     pp0 = torch.sum(torch.where(lvl<1.1, iii, torch.zeros_like(lvl)))
        #     pp1 = torch.sum(torch.where(lvl<1.2, iii, torch.zeros_like(lvl)))
        #     pp2 = torch.sum(torch.where(lvl<1.3, iii, torch.zeros_like(lvl)))
        #     pp3 = torch.sum(torch.where(lvl<1.4, iii, torch.zeros_like(lvl)))
        #     pp4 = torch.sum(torch.where(lvl<1.5, iii, torch.zeros_like(lvl)))
        #     pp5 = torch.sum(torch.where(lvl<1.6, iii, torch.zeros_like(lvl)))
        #     pp6 = torch.sum(torch.where(lvl<1.7, iii, torch.zeros_like(lvl)))
        #     pp7 = torch.sum(torch.where(lvl<1.8, iii, torch.zeros_like(lvl)))
        #     pp8 = torch.sum(torch.where(lvl<1.9, iii, torch.zeros_like(lvl)))
        #     pp9 = torch.sum(torch.where(lvl>=1.9, iii, torch.zeros_like(lvl)))
            
        #     print("%d: %d %d %d %d %d %d %d %d %d %d" % (i, pp0, pp1, pp2, pp3, pp4, pp5, pp6, pp7, pp8, pp9))
        # exit(0)
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def roi_rescale(self, rois, scale_factor):
        cx = (rois[:, 1] + rois[:, 3]) * 0.5
        cy = (rois[:, 2] + rois[:, 4]) * 0.5
        w = rois[:, 3] - rois[:, 1]
        h = rois[:, 4] - rois[:, 2]
        new_w = w * scale_factor
        new_h = h * scale_factor
        x1 = cx - new_w * 0.5
        x2 = cx + new_w * 0.5
        y1 = cy - new_h * 0.5
        y2 = cy + new_h * 0.5
        new_rois = torch.stack((rois[:, 0], x1, y1, x2, y2), dim=-1)
        return new_rois

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None, lvl=None, replace_rois=None):
        ### additional parameters:
        # lvl:           taking the levels computed up or down by 1
        # replace_rois:  using other rois to compute the levels
        out_size = self.roi_layers[0].out_size
        num_levels = len(feats)
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *out_size)
        # print(num_levels)
        # input()
        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)
        
        if replace_rois is not None:
            target_lvls = self.map_roi_levels(replace_rois, num_levels)
        else:
            target_lvls = self.map_roi_levels(rois, num_levels)
        
        if lvl is not None:
            target_lvls = (target_lvls+lvl).clamp(min=0, max=num_levels-1).long()
        """
        f = open('./levels_before.txt')
        line = list(f.readline().strip().split(' '))
        with open('./levels_before.txt', 'w') as f:
            ws = [int(line[i]) + sum(target_lvls.view(-1) == i).cpu().numpy() for i in range(num_levels)]
            f.write(' '.join([str(i) for i in ws])) # 
            f.write('\n')
        f.close()
        """
        
        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)
        for i in range(num_levels):
            inds = target_lvls == i
            if inds.any():
                rois_ = rois[inds, :]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t
            else:
                roi_feats += sum(x.view(-1)[0] for x in self.parameters()) * 0.
        return roi_feats
