# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
This file contains specific functions for computing losses on the RPN
file
"""

import torch
from torch.nn import functional as F

from .utils import concat_box_prediction_layers, concat_locs, concat_shapes

from ..balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
from ..utils import cat

from maskrcnn_benchmark.layers import smooth_l1_loss, SigmoidFocalLoss, bounded_iou_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist, cat_boxlist_broad


class RPNLossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder,
                 generate_labels_func):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        # self.target_preparator = target_preparator
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.copied_fields = []
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['not_visibility', 'between_thresholds']

    def match_targets_to_anchors(self, anchor, target, copied_fields=[]):
        match_quality_matrix = boxlist_iou(target, anchor)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        target = target.copy_with_fields(copied_fields)
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image, self.copied_fields
            )

            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = self.generate_labels_func(matched_targets)
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # discard anchors that go out of the boundaries of the image
            if "not_visibility" in self.discard_cases:
                labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            if "between_thresholds" in self.discard_cases:
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets


    def __call__(self, anchors, objectness, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        labels, regression_targets = self.prepare_targets(anchors, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness, box_regression = \
                concat_box_prediction_layers(objectness, box_regression)

        objectness = objectness.squeeze()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1.0 / 9,
            size_average=False,
        ) / (sampled_inds.numel())

        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss

# This function should be overwritten in RetinaNet
def generate_rpn_labels(matched_targets):
    matched_idxs = matched_targets.get_field("matched_idxs")
    labels_per_image = matched_idxs >= 0
    return labels_per_image


def make_rpn_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.RPN.FG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION
    )

    loss_evaluator = RPNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        generate_rpn_labels
    )
    return loss_evaluator

def calc_region(bbox, ratio, featmap_size=None):
    x1 = torch.round((1 - ratio) * bbox[0] + ratio * bbox[2]).long()
    y1 = torch.round((1 - ratio) * bbox[1] + ratio * bbox[3]).long()
    x2 = torch.round(ratio * bbox[0] + (1 - ratio) * bbox[2]).long()
    y2 = torch.round(ratio * bbox[1] + (1 - ratio) * bbox[3]).long()
    if featmap_size is not None:
        x1 = x1.clamp(min=0, max=featmap_size[1] - 1)
        y1 = y1.clamp(min=0, max=featmap_size[0] - 1)
        x2 = x2.clamp(min=0, max=featmap_size[1] - 1)
        y2 = y2.clamp(min=0, max=featmap_size[0] - 1)
    return x1, y1, x2, y2


class GARPNLossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder, anchor_box_coder,
                 generate_labels_func, cfg):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        # self.target_preparator = target_preparator
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.anchor_box_coder = anchor_box_coder
        self.copied_fields = []
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['not_visibility', 'between_thresholds']

        self.cfg = cfg
        self.octave_base_scale = cfg.MODEL.RPN.GA.OCTAVE_BASE_SCALE
        self.anchor_strides = cfg.MODEL.RPN.GA.ANCHOR_STRIDES
        self.center_ratio = cfg.MODEL.RPN.GA.CENTER_RATIO
        self.ignore_ratio = cfg.MODEL.RPN.GA.IGNORE_RATIO

        self.num_approx_anchors_per_location = len(cfg.MODEL.RPN.GA.ASPECT_RATIOS) * cfg.MODEL.RPN.GA.SCALES_PER_OCTAVE

        self.loss_loc_fn = SigmoidFocalLoss(gamma=2.0, alpha=0.25)

    def match_targets_to_anchors(self, anchor, target, copied_fields=[]):
        match_quality_matrix = boxlist_iou(target, anchor)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        target = target.copy_with_fields(copied_fields)
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets


    def ga_loc_target(self, targets, featmap_sizes):

        r1 = (1 - self.center_ratio) / 2
        r2 = (1 - self.ignore_ratio) / 2

        all_loc_targets = []
        all_loc_weights = []
        all_ignore_map = []

        img_per_gpu = len(targets)
        num_lvls = len(featmap_sizes)
        for lvl_id in range(num_lvls):
            size = featmap_sizes[lvl_id]
            loc_targets = torch.zeros(img_per_gpu, 1, size[0], size[1], device=targets[0].bbox.device, dtype=torch.int32)
            loc_weights = torch.full_like(loc_targets, -1).float()
            ignore_map = torch.zeros_like(loc_targets)
            all_loc_targets.append(loc_targets)
            all_loc_weights.append(loc_weights)
            all_ignore_map.append(ignore_map)

        for img_id, targets_per_image, in enumerate(targets):

            gt_bboxes = targets_per_image.bbox
            scale = torch.sqrt((gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) *
                               (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1))

            min_anchor_size = scale.new_full(
                (1, ), float(self.octave_base_scale * self.anchor_strides[0]))
            target_lvls = torch.floor(
               torch.log2(scale) - torch.log2(min_anchor_size) + 0.5)
            target_lvls = target_lvls.clamp(min=0, max=num_lvls - 1).long()
            for gt_id in range(gt_bboxes.size(0)):
                lvl = target_lvls[gt_id].item()
                gt_ = gt_bboxes[gt_id, :4] / self.anchor_strides[lvl]
                ignore_x1, ignore_y1, ignore_x2, ignore_y2 = calc_region(gt_, r2, featmap_sizes[lvl])
                ctr_x1, ctr_y1, ctr_x2, ctr_y2 = calc_region(gt_, r1, featmap_sizes[lvl])
                all_loc_targets[lvl][img_id, 0, ctr_y1: ctr_y2 + 1, ctr_x1: ctr_x2 + 1] = 1
                all_loc_weights[lvl][img_id, 0, ignore_y1: ignore_y2 + 1, ignore_x1: ignore_x2 + 1] = 0
                all_loc_weights[lvl][img_id, 0, ctr_y1: ctr_y2 + 1, ctr_x1: ctr_x2 + 1] = 1

                if lvl > 0:
                    d_lvl = lvl - 1
                    gt_ = gt_bboxes[gt_id, :4] / self.anchor_strides[d_lvl]
                    ignore_x1, ignore_y1, ignore_x2, ignore_y2 = calc_region(gt_, r2, featmap_sizes[d_lvl])
                    all_ignore_map[d_lvl][img_id, 0, ignore_y1: ignore_y2 + 1, ignore_x1: ignore_x2 + 1] = 1
                if lvl < num_lvls - 1:
                    u_lvl = lvl + 1
                    gt_ = gt_bboxes[gt_id, :4] / self.anchor_strides[u_lvl]
                    ignore_x1, ignore_y1, ignore_x2, ignore_y2 = calc_region(gt_, r2, featmap_sizes[u_lvl])
                    all_ignore_map[u_lvl][img_id, 0, ignore_y1: ignore_y2 + 1, ignore_x1: ignore_x2 + 1] = 1
                   
        for lvl_id in range(num_lvls):
            all_loc_weights[lvl_id][(all_loc_weights[lvl_id] < 0)
                                    & (all_ignore_map[lvl_id] > 0)] = 0
            all_loc_weights[lvl_id][all_loc_weights[lvl_id] < 0] = 0.1

        loc_avg_factor = sum(
            [t.size(0) * t.size(-1) * t.size(-2) for t in all_loc_targets]) / 200

        return all_loc_targets, all_loc_weights, loc_avg_factor

    def ga_shape_target(self, square_anchors, approx_anchors, targets):

        shape_targets = []
        shape_weights = []

        for square_anchors_per_image, approx_anchors_per_image, targets_per_image in zip(square_anchors, approx_anchors, targets):
            match_quality_matrix = boxlist_iou(targets_per_image, approx_anchors_per_image)
            num_gt = targets_per_image.bbox.shape[0]
            match_quality_matrix = match_quality_matrix.view(num_gt, -1, self.num_approx_anchors_per_location)
            match_quality_matrix, _ = match_quality_matrix.max(dim=2)
            matched_idxs = self.proposal_matcher(match_quality_matrix)
            target = targets_per_image.copy_with_fields(self.copied_fields)
            matched_targets = target[matched_idxs.clamp(min=0)]
            matched_targets.add_field("matched_idxs", matched_idxs)

            weights = self.generate_labels_func(matched_targets)
            weights = weights.to(dtype=torch.float32)

            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            weights[bg_indices] = 0
            #if "not_visibility" in self.discard_cases:
            #    weights[~square_anchors_per_image.get_field("visibility")] = 0
            #if "between_thresholds" in self.discard_cases:
            #    inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            #    weights[inds_to_discard] = 0

            shape_targets_per_image = matched_targets.bbox

            shape_targets.append(shape_targets_per_image)
            shape_weights.append(weights)

        return shape_targets, shape_weights


    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image, self.copied_fields
            )

            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = self.generate_labels_func(matched_targets)
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # discard anchors that go out of the boundaries of the image
            if "not_visibility" in self.discard_cases:
                labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            if "between_thresholds" in self.discard_cases:
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets


    def __call__(self, square_anchors, guided_anchors, loc_masks, approx_anchors, objectness, box_regression, shapes, locs, targets):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """
        featmap_sizes = [feat.shape[2:] for feat in objectness]
        loc_targets, loc_weights, loc_avg_factors = self.ga_loc_target(
            targets,
            featmap_sizes
        )
        locs, loc_targets, loc_weights = concat_locs(locs, loc_targets, loc_weights) 
        loc_loss = self.loss_loc_fn.forward_weights(locs, loc_targets, loc_weights) / loc_avg_factors

        square_anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in square_anchors]
        approx_anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in approx_anchors]

        shape_targets, shape_weights = self.ga_shape_target(
            square_anchors,
            approx_anchors,
            targets
        )

        shapes = concat_shapes(shapes)

        shape_pos_inds, shape_neg_inds = self.fg_bg_sampler(shape_weights)
        shape_pos_inds = torch.nonzero(torch.cat(shape_pos_inds, dim=0)).squeeze(1)
        shape_neg_inds = torch.nonzero(torch.cat(shape_neg_inds, dim=0)).squeeze(1)
        anchor_total_num = shape_pos_inds.shape[0] + shape_neg_inds.shape[0]

        shape_targets = torch.cat(shape_targets, dim=0)
        square_anchors = cat_boxlist_broad(square_anchors)

        shapes = shapes[shape_pos_inds]
        shape_targets = shape_targets[shape_pos_inds]
        square_anchors = square_anchors[shape_pos_inds]

        shapes = self.anchor_box_coder.decode(
            shapes, square_anchors.bbox
        )
        shape_loss = bounded_iou_loss(
            shapes, 
            shape_targets,
            beta=0.2,
            size_average=False
        ) / anchor_total_num * 0.1


        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in guided_anchors]
        labels, regression_targets = self.prepare_targets(anchors, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness, box_regression = \
                concat_box_prediction_layers(objectness, box_regression)

        objectness = objectness.squeeze()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1.0 / 9,
            size_average=False,
        ) / (sampled_inds.numel())

        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss, shape_loss, loc_loss

# This function should be overwritten in RetinaNet
def generate_garpn_labels(matched_targets):
    matched_idxs = matched_targets.get_field("matched_idxs")
    labels_per_image = matched_idxs >= 0
    return labels_per_image


def make_ga_rpn_loss_evaluator(cfg, box_coder, anchor_box_coder):
    matcher = Matcher(
        cfg.MODEL.RPN.FG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION
    )

    loss_evaluator = GARPNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        anchor_box_coder,
        generate_rpn_labels,
        cfg
    )
    return loss_evaluator
