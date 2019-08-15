# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .box_head.box_head import build_roi_box_head, build_vg_roi_box_head
from .mask_head.mask_head import build_roi_mask_head
from .keypoint_head.keypoint_head import build_roi_keypoint_head


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.keypoint.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        if self.cfg.MODEL.CASCADE_ON:
            if self.training:
                x, detections1, loss_box1 = self.box1(features, proposals, targets)
                x, detections2, loss_box2 = self.box2(features, detections1, targets)
                x, detections, loss_box3 = self.box3(features, detections2, targets)
                losses.update(loss_box1)
                losses.update(loss_box2)
                losses.update(loss_box3)
            else:
                cls1, bbox1, detections1 = self.box1(features, proposals)
                cls2, bbox2, detections2 = self.box2(features, detections1)
                cls3, bbox3, detections3 = self.box3(features, detections2)
                cls = (cls1 + cls2 + cls3) / 3
                detections = self.box3.forward_post(cls, bbox3, detections2)
                return None, detections, {}
        else:
            x, detections, loss_box = self.box(features, proposals, targets)
            losses.update(loss_box)
        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_mask = self.mask(mask_features, detections, targets)
            losses.update(loss_mask)

        if self.cfg.MODEL.KEYPOINT_ON:
            keypoint_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                keypoint_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_keypoint = self.keypoint(keypoint_features, detections, targets)
            losses.update(loss_keypoint)
        return x, detections, losses


def build_roi_heads(cfg, in_channels):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []

    if not cfg.MODEL.RPN_ONLY:
        if cfg.MODEL.CASCADE_ON:
            roi_heads.append(("box1", build_roi_box_head(cfg, in_channels, stage=1)))
            roi_heads.append(("box2", build_roi_box_head(cfg, in_channels, stage=2)))
            roi_heads.append(("box3", build_roi_box_head(cfg, in_channels, stage=3)))
        else:
            roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels)))
    if cfg.MODEL.KEYPOINT_ON:
        roi_heads.append(("keypoint", build_roi_keypoint_head(cfg, in_channels)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads

def build_vg_roi_heads(cfg, in_channels):
    roi_heads = [('box', build_vg_roi_box_head(cfg, in_channels))]
    roi_heads = CombinedROIHeads(cfg, roi_heads)
    return roi_heads
