# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.bounding_box import BoxList

import numpy as np

class IOULossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
        self, 
        proposal_matcher, 
        fg_bg_sampler, 
        box_coder
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder

        self.jitter_precomputed = np.load('datasets/iou.npy', allow_pickle=True)
        self.jitter_num = [self.jitter_precomputed[i].shape[0] for i in range(10)]

    def match_targets_to_proposals_withiou(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields("labels")
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)

        matched_iou, _ = match_quality_matrix.max(dim=0)
        matched_targets.add_field("matched_iou", matched_iou)
        return matched_targets


    def prepare_targets(self, proposals, targets):
        ious = []
        labels = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals_withiou(
                proposals_per_image, targets_per_image
            )
            matched_iou = matched_targets.get_field("matched_iou")
            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            ious.append(matched_iou)
            labels.append(labels_per_image)
        return ious, labels

    def jitter_gt(self, targets):

        proposals = []

        for target_per_image in targets:
            tot = target_per_image.bbox.shape[0]
            device = target_per_image.bbox.device

            ids = (torch.rand(500) * tot).to(device=device, dtype=torch.int64)

            jitter_delta = []
            for i in range(10):
                per_iou_ids = np.random.choice(self.jitter_num[i], 50, replace=False)
                jitter_delta_per_iou = self.jitter_precomputed[i][per_iou_ids]
                jitter_delta_per_iou = torch.from_numpy(jitter_delta_per_iou).to(device=device, dtype=torch.float32)
                jitter_delta.append(jitter_delta_per_iou)

            jitter_delta = torch.cat(jitter_delta, dim=0)

            sample_box = target_per_image.bbox[ids]
            sample_box_w = sample_box[:, 2] - sample_box[:, 0] + 1
            sample_box_h = sample_box[:, 3] - sample_box[:, 1] + 1

            jitter_delta[:, 0] *= sample_box_w
            jitter_delta[:, 1] *= sample_box_h
            jitter_delta[:, 2] *= sample_box_w
            jitter_delta[:, 3] *= sample_box_h
            
            sample_box += jitter_delta 

            sample_boxlist = BoxList(sample_box, target_per_image.size, mode="xyxy")
            proposals.append(sample_boxlist)

        return proposals


    def subsample(self, targets):

        proposals = self.jitter_gt(targets)

        ious, labels = self.prepare_targets(proposals, targets)

        # add corresponding label and regression_targets information to the bounding boxes
        for ious_per_image, labels_per_image, proposals_per_image in zip(ious, labels, proposals):
            proposals_per_image.add_field(
                "labels", labels_per_image
            )
            proposals_per_image.add_field(
                "ious", ious_per_image
            )

        self._proposals = proposals
        return proposals

    def __call__(self, iou_preds):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        iou_preds = cat(iou_preds, dim=0)
        device = iou_preds.device

        proposals = self._proposals

        labels = cat(
            [proposal.get_field("labels") for proposal in proposals], dim=0
        )
        iou_targets = cat(
            [proposal.get_field("ious") for proposal in proposals], dim=0
        )

        sids = torch.nonzero(labels > -1).squeeze(1)
        iou_preds = iou_preds[sids, labels]

        # normalize iou_targets
        iou_targets = (iou_targets - 0.5) * 4 - 1
        
        iou_loss = smooth_l1_loss(
            iou_preds,
            iou_targets,
            size_average=True,
            beta=1,
        )

        return iou_loss


def make_roi_iou_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_IOU_HEAD.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_IOU_HEAD.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_IOU_HEAD.BATCH_SIZE_PER_IMAGE, 1.0
    )

    box_coder = BoxCoder(weights=(1., 1., 1., 1.))

    loss_evaluator = IOULossComputation(
        matcher, 
        fg_bg_sampler,
        box_coder
    )

    return loss_evaluator
