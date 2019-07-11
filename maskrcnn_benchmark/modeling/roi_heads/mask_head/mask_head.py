# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from torch.nn import functional as F
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.make_layers import make_conv1x1
from .roi_mask_feature_extractors import make_roi_mask_feature_extractor
from .roi_mask_predictors import make_roi_mask_predictor
from .inference import make_roi_mask_post_processor
from .loss import make_roi_mask_loss_evaluator
from maskrcnn_benchmark.modeling.poolers import Pooler


def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds


class ROIMaskHead(torch.nn.Module):
    def __init__(self, cfg, in_channels, stage=None):
        super(ROIMaskHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_mask_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_mask_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_mask_post_processor(cfg)
        self.loss_evaluator = make_roi_mask_loss_evaluator(cfg)

        self.stage = stage
        self.loss_weight = 1.
        if self.stage == 2:
            self.loss_stage = 0.5
        elif self.stage == 3:
            self.loss_weight = 0.25

        if cfg.MODEL.ROI_MASK_HEAD.INFORMATION_FLOW and self.stage > 1:
            use_gn = cfg.MODEL.ROI_MASK_HEAD.USE_GN
            layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS
            dilation = cfg.MODEL.ROI_MASK_HEAD.DILATION
            flow_channels = self.feature_extractor.out_channels
            layer_name = "mask_flow{}".format(self.stage)
            use_ws = cfg.MODEL.USE_WS
            self.conv_res = make_conv1x1(
                flow_channels, flow_channels,
                stride=1, use_gn=use_gn, use_ws=use_ws
            )

        if cfg.MODEL.SEMANTIC_ON:
            resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
            scales = cfg.MODEL.ROI_MASK_HEAD.SEMANTIC_POOLER_SCALES
            sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
            pooler = Pooler(
                output_size=(resolution, resolution),
                scales=scales,
                sampling_ratio=sampling_ratio,
                cfg=cfg
            )
            self.pooler = pooler

    def forward(self, features, proposals, targets=None, res_feat=None, semanticx=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # during training, only focus on positive boxes
            all_proposals = proposals
            proposals, positive_inds = keep_only_positive_boxes(proposals)
        if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            if self.cfg.MODEL.ROI_MASK_HEAD.INFORMATION_FLOW or self.cfg.MODEL.SEMANTIC_ON:
                x = self.feature_extractor.forward_pool(features, proposals)
                if res_feat is not None:
                    res_feat = F.relu(self.conv_res(res_feat), inplace=True)
                    x = x + res_feat
                if semanticx is not None:
                    semanticx = self.pooler([semanticx], proposals)
                    x = x + semanticx
                x = self.feature_extractor.forward_conv(x)
            else:
                x = self.feature_extractor(features, proposals)

        mask_logits = self.predictor(x)

        if not self.training:
            result = self.post_processor(mask_logits, proposals)
            return x, result, {}

        loss_mask = self.loss_evaluator(proposals, mask_logits, targets)

        if self.stage is None:
            return x, all_proposals, dict(loss_mask=loss_mask)
        else:
            loss_mask *= self.loss_weight
            return x, all_proposals, {'loss_mask%d'%self.stage: loss_mask}


def build_roi_mask_head(cfg, in_channels, stage=None):
    return ROIMaskHead(cfg, in_channels, stage)
