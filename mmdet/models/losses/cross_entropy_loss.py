import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


def cross_entropy(pred, label, multi_cls_pred=None, weight=None, reduction='mean', avg_factor=None):
    """
    :param pred: [N, 81]
    :param label: [N]
    :param weight: [N]
    :return:
    """
    # print('cross pred: ', pred.size())
    # element-wise losses [N]
    # soft_pred = torch.softmax(pred, dim=-1)
    if multi_cls_pred is not None:
        pred_all_cls = torch.softmax(multi_cls_pred, dim=-1)
        max_pred_idx = list(torch.argmax(pred, dim=-1).view(-1).cpu().numpy())
    loss = F.cross_entropy(pred, label, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float() + (0.0 if multi_cls_pred is None else 1.0 - (pred_all_cls[max_pred_idx]))
    elif multi_cls_pred is not None:
        weight = 1.0 - (pred_all_cls[max_pred_idx])
    # if weight is not None:
    #     weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss

def distribution_loss(pred, label, weight=None, reduction='mean', avg_factor=None):
    soft_pred = torch.softmax(pred, dim=-1)
    # min_ = torch.min(soft_pred, dim=-1).values.view(-1, 1)
    max_ = torch.max(soft_pred, dim=-1).values.view(-1, 1)  # [N, 1]
    # soft_pred = (soft_pred - min_) / (max_ - min_ + 1e-9)
    soft_pred = soft_pred / (max_ + 1e-9)
    max_pred = torch.max(soft_pred, dim=-1).values.view(-1, 1)
    soft_pred = torch.where(soft_pred == max_pred, torch.zeros_like(soft_pred), soft_pred)
    # print(max_pred)
    alpha, beta, gamma = 1.0, 0.0, 0.5
    # dis_ws = (-max_ + 1.0) ** gamma
    reg_pred = alpha * soft_pred + beta
    dis_loss = 1.0 - (torch.min((max_pred * 2.0 - soft_pred) * (1.0 - torch.tanh(reg_pred)), dim=-1).values * 0.5)
    # dis_loss = -(torch.min((max_pred * 2.0 - soft_pred) * torch.log2(reg_pred), dim=-1).values * 0.5)
    # print('dis_loss: ', torch.sum(dis_loss > 10.0))
    dis_loss = torch.mean(dis_loss)
    # dis_loss = weight_reduce_loss(
    #     dis_loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    loss = F.cross_entropy(pred, label, reduction='none')
    # apply weights and do the reduction
    # if weight is not None:
    #     weight = weight.float()
    # print('loss:, ', loss, ' dis_loss: ', dis_loss)
    # input()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss + dis_loss # * (1.0 + min(int(loss/(dis_loss + 0.01)), 5))

def multi_classes_loss(pred, label, weight=None, reduction='mean', avg_factor=None):
    import numpy as np
    # print(pred.size())
    # input()
    pred = torch.softmax(pred.view(-1, 2), dim=-1)
    labels = list(label.cpu().numpy())
    # pred_labels = list(torch.argmax(pred, dim=-1).view(-1).cpu().numpy())
    # pd = torch.where(pred[:, 1] > pred[:, 0], torch.ones_like(pred[:, 1]), torch.zeros_like(pred[:, 1]))
    # pd = torch.where(pred > 0.5, torch.ones_like(pred), torch.zeros_like(pred))

    # positive for cls possibility
    pd = pred[:, 1]
    ori = torch.zeros_like(pred[:, 1])
    # ori = torch.zeros(pred.size(-1))
    # pd = torch.zeros(pred.size(-1))
    ori[labels] = 1
    # pd[pred_labels] = 1
    # cls_entropy_loss = -torch.mean((ori * torch.log2(pd.clamp(min=0.1)) +
    #                                 (1.0 - ori) * torch.log2((1.0 - pd).clamp(min=0.1))))
    # cls_entropy_loss = 1.0 - (torch.sum(pd) / (torch.sum(ori) + 1)).clamp(min=0, max=1)

    # pos_cls_loss = torch.sum(torch.where(ori == 1, (torch.tanh(1-pd) + (pd < 0.5))**2, ori)) / (torch.sum(ori==1)).clamp(min=1)
    # neg_cls_loss = torch.sum(torch.where(ori == 0, (torch.tanh(pd) + (pd > 0.5)), torch.zeros_like(ori))) / (torch.sum(ori==0)).clamp(min=1)
    pos_cls_loss = torch.where(ori == 1, (torch.tanh(1-pd) + (pd < 0.5)), ori)
    pos_cls_loss = torch.max(pos_cls_loss)
    neg_cls_loss = torch.where(ori == 0, (torch.tanh(pd) + (pd > 0.5)), torch.zeros_like(ori))
    neg_cls_loss = torch.max(neg_cls_loss)

    # cls_entropy_loss = torch.mean(torch.where(ori > 0, (2.0 - pd) ** 2, (1.0 + pd) ** 2)) - 1.0
    # cls_loss = torch.sum(abs(ori - pd)) / (torch.sum(ori) + 1)
    # input()
    return (pos_cls_loss + neg_cls_loss) * 0.5

def _expand_binary_labels(labels, label_weights, label_channels):
    # Caution: this function should only be used in RPN
    # in other files such as in ghm_loss, the _expand_binary_labels
    # is used for multi-class classification.
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1, as_tuple=False).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         multi_cls_pred=None):
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred, target, label, reduction='mean', avg_factor=None, multi_cls_pred=None):
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, reduction='mean')[None]


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 use_dis=False,
                 use_multi_cls=False,
                 reduction='mean',
                 loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.use_dis = use_dis
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.use_multi_cls = use_multi_cls

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        elif self.use_dis:
            self.cls_criterion = distribution_loss
        elif self.use_multi_cls:
            self.cls_criterion = multi_classes_loss
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                multi_cls_pred=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            # multi_cls_pred=multi_cls_pred,
            **kwargs)
        return loss_cls
