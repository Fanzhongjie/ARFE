import torch
import numpy as np

from ..builder import BBOX_SAMPLERS
from .base_sampler import BaseSampler


@BBOX_SAMPLERS.register_module()
class RandomSamplerPrior(BaseSampler):

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        from mmdet.core.bbox import demodata
        super(RandomSamplerPrior, self).__init__(num, pos_fraction, neg_pos_ub,
                                            add_gt_as_proposals)
        self.rng = demodata.ensure_rng(kwargs.get('rng', None))

    def random_choice(self, gallery, num):
        """Random select some elements from the gallery.

        If `gallery` is a Tensor, the returned indices will be a Tensor;
        If `gallery` is a ndarray or list, the returned indices will be a
        ndarray.

        Args:
            gallery (Tensor | ndarray | list): indices pool.
            num (int): expected sample num.

        Returns:
            Tensor or ndarray: sampled indices.
        """
        assert len(gallery) >= num

        is_tensor = isinstance(gallery, torch.Tensor)
        if not is_tensor:
            gallery = torch.tensor(
                gallery, dtype=torch.long, device=torch.cuda.current_device())
        perm = torch.randperm(gallery.numel(), device=gallery.device)[:num]
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Randomly sample some positive samples."""
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
        
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Randomly sample some negative samples."""
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            return self.random_choice(neg_inds, num_expected)

    def _sample_pos_prior(self, prior_, assign_result, num_expected, **kwargs):
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
        # print(pos_inds)
        # pos_inds_p = torch.nonzero(prior_ > 0.8, as_tuple=False)
        # pos_inds = torch.cat((pos_inds, pos_inds_p), dim=0)
        # input()
        
        if pos_inds.numel() != 0:
            # if pos_inds_p.numel() != 0:
            #     pos_inds = torch.cat((pos_inds, pos_inds_p), dim=0)
                # pos_inds = torch.tensor(list(set(pos_inds.squeeze(1).cpu().numpy().tolist()))).view(-1, 1)
                # tmp_pos = list(set(pos_inds.squeeze(1).cpu().numpy().tolist()) | set(pos_inds_p.squeeze(1).cpu().numpy().tolist()))
                # tmp_pos = torch.tensor(tmp_pos).view(-1, 1)
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)
    
    def _sample_neg_prior(self, prior_, assign_result, num_expected, **kwargs):
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        neg_inds_p = torch.nonzero(prior_ < 0.5, as_tuple=False)      
        
        neg_len = len(neg_inds)
        neg_inds_tmp = neg_inds
        # neg_p_len = len(neg_inds_p)
        if neg_inds.numel() != 0:
            # neg_out = []
            # cnt = 0
            # for i in range(neg_len):
            #     for j in range(cnt, neg_p_len):
            #         if neg_inds[i, 0] == neg_inds_p[j, 0]:
            #             neg_out.append(neg_inds[i, 0])
            #             cnt = cnt+1
            #             break
            #         if neg_inds[i, 0] < neg_inds_p[j, 0]:
            #             break
            # neg_inds_tmp = torch.tensor(neg_out).view(-1, 1)
            # if neg_inds_tmp.numel() != 0:
            #     neg_inds_tmp = neg_inds_tmp.squeeze(1)
            # else:
            #     neg_inds_tmp = neg_inds.squeeze(1)
            
            if neg_inds_p.numel() != 0:
                neg_inds_tmp = np.array(list(set(neg_inds_p.cpu()) - set(neg_inds.cpu())))
                neg_inds_tmp = torch.from_numpy(neg_inds_tmp).to(
                    assign_result.gt_inds.device).long()
            # print(neg_inds_tmp.size())
            # input()
            
        if len(neg_inds_tmp) <= num_expected:
            if neg_len <= num_expected:
                return neg_inds
            num_extra = num_expected - len(neg_inds_tmp)
            extra_inds = np.array(
                list(set(neg_inds.cpu()) - set(neg_inds_tmp.cpu())))
            if len(extra_inds) > num_extra:
                extra_inds = self.random_choice(extra_inds, num_extra)
            extra_inds = torch.from_numpy(extra_inds).to(
                assign_result.gt_inds.device).long()
            return torch.cat([neg_inds_tmp, extra_inds])
        else:
            return self.random_choice(neg_inds_tmp, num_expected)

