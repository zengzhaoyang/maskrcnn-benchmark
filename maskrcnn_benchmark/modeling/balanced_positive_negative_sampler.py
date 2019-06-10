# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import numpy as np

class BalancedPositiveNegativeSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentace of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        """
        Arguments:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image in matched_idxs:
            positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
            negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)

            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)

            # randomly select positive and negative examples
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx


class IOUBalancedPositiveNegativeSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image, positive_fraction, hard_thr=0.0, hard_fraction=1.0, num_intervals=3):
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentace of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

        self.hard_thr = hard_thr
        self.hard_fraction = hard_fraction
        self.num_intervals = num_intervals

    def __call__(self, matched_idxs, max_overlaps):
        """
        Arguments:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image, max_ovlap_per_image in zip(matched_idxs, max_overlaps):
            positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
            negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)

            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)

            # pos sampling
            unique_gt_inds = matched_idxs_per_image[positive].unique()
            num_gts = len(unique_gt_inds)
            num_per_gt = int(round(num_pos / float(num_gts)) + 1)

            pos_idx_per_image = [] 
            for i in unique_gt_inds:
                inds = torch.nonzero(matched_idxs_per_image == i.item())
                if inds.numel() != 0:
                    inds = inds.squeeze(1)
                else:
                    continue
                if len(inds) > num_per_gt:
                    perm = torch.randperm(inds.numel(), device=inds.device)[:num_per_gt]
                    inds = inds[perm]
                pos_idx_per_image.append(inds)
            pos_idx_per_image = torch.cat(pos_idx_per_image)

            if len(pos_idx_per_image) < num_pos:
                num_extra = num_pos - len(pos_idx_per_image)
                extra_inds = np.array(
                    list(set(positive.cpu()) - set(pos_idx_per_image.cpu())))
                extra_inds = torch.from_numpy(extra_inds).to(positive.device).long()
                if len(extra_inds) > num_extra:
                    perm = torch.randperm(extra_inds.numel(), device=extra_inds.device)[:num_extra]
                    extra_inds = extra_inds[perm]
                pos_idx_per_image = torch.cat([pos_idx_per_image, extra_inds])
            elif len(pos_idx_per_image) > num_pos:
                perm = torch.randperm(pos_idx_per_image.numel(), device=pos_idx_per_image.device)[:num_pos]
                pos_idx_per_image = pos_idx_per_image[perm]
           
            # neg sampling
            max_ovlap_per_image = max_ovlap_per_image.cpu().numpy()
            neg_set = set(negative.cpu().numpy())
            easy_set = set(
                np.where(
                    np.logical_and(max_ovlap_per_image >= 0,
                                   max_ovlap_per_image < self.hard_thr))[0])
            hard_set = set(np.where(max_ovlap_per_image >= self.hard_thr)[0])
            easy_neg_inds = list(easy_set & neg_set)
            hard_neg_inds = hard_set & neg_set

            neg_idx_per_image = []

            max_iou = max_ovlap_per_image.max()
            iou_interval = (max_iou - self.hard_thr) / self.num_intervals
            per_num_neg = int(num_neg / self.num_intervals)
            for i in range(self.num_intervals):
                start_iou = self.hard_thr + i * iou_interval
                end_iou = self.hard_thr + (i + 1) * iou_interval
                tmp_set = set(
                    np.where(
                        np.logical_and(max_ovlap_per_image >= start_iou,
                                       max_ovlap_per_image < end_iou))[0])
                tmp_inds = list(tmp_set & hard_neg_inds)
                tmp_inds = np.array(tmp_inds)
                tmp_inds = torch.from_numpy(tmp_inds).to(positive.device).long()
                if len(tmp_inds) > per_num_neg:
                    perm = torch.randperm(tmp_inds.numel(), device=tmp_inds.device)[:per_num_neg]
                    tmp_inds = tmp_inds[perm]
                neg_idx_per_image.append(tmp_inds)
            neg_idx_per_image = torch.cat(neg_idx_per_image)

            if len(neg_idx_per_image) < num_neg:
                num_extra = num_neg - len(neg_idx_per_image)
                extra_inds = np.array(
                    list(set(negative.cpu()) - set(neg_idx_per_image.cpu())))
                extra_inds = torch.from_numpy(extra_inds).to(positive.device).long()
                if len(extra_inds) > num_extra:
                    perm = torch.randperm(extra_inds.numel(), device=extra_inds.device)[:num_extra]
                    extra_inds = extra_inds[perm]
                neg_idx_per_image = torch.cat([neg_idx_per_image, extra_inds])

            num_easy = num_neg - len(neg_idx_per_image)
            easy_neg_inds = np.array(easy_neg_inds)
            easy_neg_inds = torch.from_numpy(easy_neg_inds).to(positive.device).long()
            if len(easy_neg_inds) > num_easy:
                perm = torch.randperm(easy_neg_inds.numel(), device=easy_neg_inds.device)[:num_easy]
                easy_neg_inds = easy_neg_inds[perm]
            neg_idx_per_image = torch.cat([neg_idx_per_image, easy_neg_inds])

            if len(neg_idx_per_image) < num_neg:
                num_extra = num_neg - len(neg_idx_per_image)
                extra_inds = np.array(list(neg_set - set(neg_idx_per_image)))
                extra_inds = torch.from_numpy(extra_inds).to(positive.device).long()
                if len(extra_inds) > num_extra:
                    perm = randperm(extra_inds.numel(), device=extra_inds.device)[:num_extra]
                    extra_inds = extra_inds[perm]
                neg_idx_per_image = torch.cat([neg_idx_per_image, extra_inds]) 


            # randomly select positive and negative examples
            #perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            #perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            #pos_idx_per_image = positive[perm1]
            #neg_idx_per_image = negative[perm2]

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx
