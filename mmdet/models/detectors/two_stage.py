import torch
import torch.nn as nn

# from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from .test_mixins import RPNTestMixin

import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import cv2
# import numpy as np
# import os

out_img_file = 'dif_imgs/'
def visualize_attetion(img_path, feats, pre_name, ratio=1, cmap='jet'):
    from PIL import Image
    import matplotlib
    matplotlib.use('AGG')
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    im_name = img_path.split('/')[-1].split('.')[0]
    
    img = Image.open(img_path, 'r')
    # img = img_path.view(img_path[1], img_path[2], img_path[3])
    # print(img.size)
    img_h, img_w = img.size[:]
    # print(img.size)
    for i in range(len(feats)):
        # img_w, img_h = feats[i].shape[:]
        # print(img_h, img_w)
        # plt.subplot(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))
        # img_h, img_w = int(img_h * ratio), int(img_w * ratio)
        # plt.subplot(1, 1, 1)
        # img = img.resize((img_h, img_w))
        if pre_name != 'dif':
            plt.imshow(img)
            plt.axis('off')    
        
        mask = feats[i].cpu().data.numpy()
        mask = cv2.resize(mask, (img_h, img_w))
        # mask = cv2.resize(feats[i].cpu().data.numpy().T, (img_h, img_w))
        # normed_mask = (mask-mask.min()) / (mask.max()-mask.min() + 1e-7)
        normed_mask = mask / mask.max()
        normed_mask = (normed_mask * 255).astype('uint8')
        # print(normed_mask.shape)
        # normed_maks = cv2.resize(normed_mask, (img_h, img_w))
        plt.imshow(normed_mask, alpha=0.5, interpolation='bilinear', cmap=cmap)
        
        plt.savefig(out_img_file + im_name + '_' + pre_name + '_' + str(i) + '.jpg')
        plt.close()
        
        '''
        mask = cv2.resize(feats[i].cpu().data.numpy(), (img_h, img_w))
        mask = mask / mask.max()
        heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap)
        cam = heatmap + np.float32(img)
        cam = (cam-cam.min()) / (np.max(cam)-cam.min())
        cv2.imwrite(out_img_file + im_name + '_' + pre_name + '_' + str(i) + '.jpg', np.uint8(255*cam))
        cv2.destroyAllWindows()
        '''
        

@DETECTORS.register_module()
class TwoStageDetector(BaseDetector, RPNTestMixin):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetector, self).__init__()
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_metas)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_bboxes(
                *rpn_outs, img_metas, cfg=proposal_cfg)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.async_test_rpn(x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        # print(type(img))
        # print(img_metas)
        # x1, x1_dif, x, x_dif = self.extract_feat(img)
        
        
        x = self.extract_feat(img)
        '''
        x_out = []
        for i in range(len(x)):
            tmp = torch.mean(torch.relu(x[i]), dim=1)
            tmp = torch.relu(x[i] - tmp)
            x_out.append(torch.mean(tmp, dim=1).view(x[i].shape[2], x[i].shape[3]))
        visualize_attetion(img_metas[0]['filename'], x_out, 'after')
        '''
        

        # x1, x = self.extract_feat(img)
        """
        x, x1 = self.extract_feat(img)
        
        x1_out, x1_dif_out, x_out, x_dif_out, dif = [], [], [], [], []
        # x1_out, x_out = [], []
        for i in range(len(x1)):
            tmp = torch.mean(torch.relu(x1[i]), dim=1)
            tmp = torch.relu(x1[i] - tmp)
            x1_out.append(torch.mean(tmp, dim=1).view(x1[i].shape[2], x1[i].shape[3]))
            # x1_out.append(torch.mean(torch.relu(x1[i]), dim=1).view(x1[i].shape[2], x1[i].shape[3]))
        for i in range(len(x)):
            tmp = torch.mean(torch.relu(x[i]), dim=1)
            tmp = torch.relu(x[i] - tmp)
            x_out.append(torch.mean(tmp, dim=1).view(x[i].shape[2], x[i].shape[3]))
            # x_out.append(torch.mean(torch.relu(x[i]), dim=1).view(x[i].shape[2], x[i].shape[3]))
            if 0 < i < len(x1):
                print(x_out[i].shape)
                x_dif_out.append(torch.relu(x_out[i] - x_out[i-1].resize_as_(x_out[i])))
                x1_dif_out.append(torch.relu(x1_out[i] - x1_out[i-1].resize_as_(x_out[i])))
                dif.append(x_out[i]-x1_out[i])
        """

        '''
        for i in range(len(x1_dif)):
            x1_dif_out.append(torch.mean(x1_dif[i], dim=1).view(x1_dif[i].shape[2], x1_dif[i].shape[3]))
            x_dif_out.append(torch.mean(x_dif[i], dim=1).view(x_dif[i].shape[2], x_dif[i].shape[3]))
            dif.append(x1_dif_out[-1]-x_dif_out[-1])
            # x1_dif_out[-1] = x1_dif_out[-1] - x_dif_out[-1]
        '''

        """
        visualize_attetion(img_metas[0]['filename'], x1_out, 'before')
        visualize_attetion(img_metas[0]['filename'], x1_dif_out, 'before_dif')
        visualize_attetion(img_metas[0]['filename'], x_out, 'after')
        visualize_attetion(img_metas[0]['filename'], x_dif_out, 'after_dif')
        visualize_attetion(img_metas[0]['filename'], dif, 'dif')
        """
        
        '''
        x_out = []
        for i in range(len(x)):
            x_out.append(torch.mean(torch.relu(x[i]), dim=1).view(x[i].shape[2], x[i].shape[3]))
        visualize_attetion(img_metas[0]['filename'], x_out, 'after')
        '''
        
        # levels = len(x)
        # for j in range(levels):
        #     b = x[j].size(0)
            
        #     x_ = torch.sum(x[j], dim=1) / x[j].size(1)
            # x_ = torch.softmax(x_, dim=0).view(b, x[j].size(2), -1)
        #     max_x = torch.max(torch.max(x_, dim=1).values, dim=1).values.view(b, -1, 1)
            # x_ = x_ / max_x
        #     min_x = torch.min(torch.min(x_, dim=1).values, dim=1).values.view(b, -1, 1)
        #     x_ = (x_ - min_x) / (max_x - min_x + 1e-6)
            # print(x_.size())
        #     x_ = x_.cpu().numpy()
            
        #     for i in range(b):
        #         x_ = x_[i, :, :]
        #         x_ = np.asarray(x_ * 255, dtype=np.uint8)
        #         dst_path = './features_img'
        #         x_ = cv2.applyColorMap(x_, cv2.COLORMAP_JET)
        #         dst_file = os.path.join(dst_path, str(j) + '_' + str(i) + '.png')
        #         cv2.imwrite(dst_file, x_)
        # exit(0)
        
        if proposals is None:
            proposal_list = self.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        x = self.extract_feats(imgs)
        proposal_list = self.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
