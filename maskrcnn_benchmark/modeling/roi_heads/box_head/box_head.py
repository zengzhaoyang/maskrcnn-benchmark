# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor, make_roi_vg_box_predictor
from .inference import make_roi_box_post_processor, make_roi_box_cascade_processor
from .loss import make_roi_box_loss_evaluator, make_roi_vg_box_loss_evaluator



class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels, stage=None):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg, stage)
        self.cascade_processor = make_roi_box_cascade_processor(cfg, stage)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg, stage)
        self.cfg = cfg
        self.stage = stage
        self.loss_weight = 1.
        if self.stage == 2:
            self.loss_weight = 0.5
        elif self.stage == 3:
            self.loss_weight = 0.25

    def forward(self, features, proposals, targets=None):
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
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        if self.cfg.MODEL.FPN.ADAPTIVE_POOLING:
            x = self.feature_extractor.forward_pool_fc6(features, proposals)
            x, _ = torch.stack(x).max(0)
            #x = torch.stack(x).mean(0) 
            x = self.feature_extractor.forward_fc7(x)
        elif self.cfg.MODEL.FPN.CONTIGUOUS_ON:
            x = self.feature_extractor.forward_pool_fc6_weights(features, proposals)
            x = torch.stack(x).sum(0)
            x = self.feature_extractor.forward_fc7(x)
        else:
            x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)

        if not self.training:
            if not self.cfg.MODEL.CASCADE_ON:
                result = self.post_processor((class_logits, box_regression), proposals)
                return x, result, {}
            else:
                result = self.cascade_processor((class_logits, box_regression), proposals)
                return class_logits, box_regression, result

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        #loss_classifier *= self.loss_weight
        #loss_box_reg *= self.loss_weight

        if self.stage is None:
            return (
                x,
                proposals,
                dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
            )
        else:
            result = self.cascade_processor((class_logits, box_regression), proposals)
            loss_classifier *= self.loss_weight
            loss_box_reg *= self.loss_weight
            return (
                x,
                result,
                {'loss_classifier%d'%self.stage: loss_classifier, 'loss_box_reg%d'%self.stage: loss_box_reg},
            )

    def forward_post(self, class_logits, box_regression, proposals):
        return self.post_processor((class_logits, box_regression), proposals)


def build_roi_box_head(cfg, in_channels, stage=None):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels, stage)


class ROIVGBoxHead(torch.nn.Module):

    def __init__(self, cfg, in_channels):
        super(ROIVGBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_vg_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg, stage=None)
        self.loss_evaluator = make_roi_vg_box_loss_evaluator(cfg)
        self.cfg = cfg

    def forward(self, features, proposals, targets=None):

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)

        # final classifier that converts the features into predictions
        class_logits, box_regression, attr_logits = self.predictor(x, proposals)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}

        loss_classifier, loss_box_reg, loss_attr = self.loss_evaluator(
            [class_logits], [box_regression], [attr_logits]
        )
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg, loss_attr=loss_attr),
        )


def build_vg_roi_box_head(cfg, in_channels):
    return ROIVGBoxHead(cfg, in_channels)
