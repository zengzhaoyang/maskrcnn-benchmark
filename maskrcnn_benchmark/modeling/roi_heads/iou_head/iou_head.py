# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from ..box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_iou_predictors import make_roi_iou_predictor
from .loss import make_roi_iou_loss_evaluator

class ROIIouHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIIouHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_iou_predictor(
            cfg, self.feature_extractor.out_channels)
        self.loss_evaluator = make_roi_iou_loss_evaluator(cfg)
        self.cfg = cfg

    def forward(self, features, proposals=None, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(targets)

        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        iou_preds = self.predictor(x)

        if not self.training:
            return x, iou_preds, {}

        loss_iou = self.loss_evaluator(
            [iou_preds]
        )

        return (
            x,
            proposals,
            dict(loss_iou=loss_iou),
        )

def build_roi_iou_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIIouHead(cfg, in_channels)
