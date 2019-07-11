# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .box_head.box_head import build_roi_box_head
from .mask_head.mask_head import build_roi_mask_head
from .keypoint_head.keypoint_head import build_roi_keypoint_head
from .semantic_head.semantic_head import build_semantic_head
from .iou_head.iou_head import build_roi_iou_head

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

        semanticx = None
        if self.cfg.MODEL.SEMANTIC_ON:
            semanticx, _, loss_semantic = self.semantic(features, targets)
            losses.update(loss_semantic)

        x = None

        if self.cfg.MODEL.CASCADE_ON:
            if self.training:
                x, detections1, loss_box1 = self.box1(features, proposals, targets, semanticx)
                x, detections2, loss_box2 = self.box2(features, detections1, targets, semanticx)
                x, detections, loss_box3 = self.box3(features, detections2, targets, semanticx)
                losses.update(loss_box1)
                losses.update(loss_box2)
                losses.update(loss_box3)
            else:
                cls1, bbox1, detections1 = self.box1(features, proposals, semanticx=semanticx)
                cls2, bbox2, detections2 = self.box2(features, detections1, semanticx=semanticx)
                cls3, bbox3, detections3 = self.box3(features, detections2, semanticx=semanticx)
                cls = (cls1 + cls2 + cls3) / 3
                detections = self.box3.forward_post(cls, bbox3, detections2)
        else:
            x, detections, loss_box = self.box(features, proposals, targets, semanticx)
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
            if self.cfg.MODEL.CASCADE_ON:
                if self.training:
                    if self.cfg.MODEL.ROI_MASK_HEAD.INTERLEAVED_EXECUTION:
                        if self.cfg.MODEL.ROI_MASK_HEAD.INFORMATION_FLOW:
                            # stage 1
                            x, _, loss_mask1 = self.mask1(mask_features, detections1, targets, semanticx=semanticx)

                            # stage 2
                            x, _, _ = self.mask1(mask_features, detections2, targets, semanticx=semanticx)
                            x, _, loss_mask2 = self.mask2(mask_features, detections2, targets, x, semanticx=semanticx)

                            # stage 3
                            x, _, _ = self.mask1(mask_features, detections, targets, semanticx=semanticx)
                            x, _, _ = self.mask2(mask_features, detections, targets, x, semanticx=semanticx)
                            x, _, loss_mask3 = self.mask3(mask_features, detections, targets, x, semanticx=semanticx)
                        else:
                            x, _, loss_mask1 = self.mask1(mask_features, detections1, targets, semanticx=semanticx)
                            x, _, loss_mask2 = self.mask2(mask_features, detections2, targets, semanticx=semanticx)
                            x, _, loss_mask3 = self.mask3(mask_features, detections, targets, semanticx=semanticx)
                    else:
                        x, _, loss_mask1 = self.mask1(mask_features, proposals, targets, semanticx=semanticx)
                        x, _, loss_mask2 = self.mask2(mask_features, detections1, targets, semanticx=semanticx)
                        x, _, loss_mask3 = self.mask3(mask_features, detections2, targets, semanticx=semanticx)
                    losses.update(loss_mask1)
                    losses.update(loss_mask2)
                    losses.update(loss_mask3)
                else:
                    if self.cfg.MODEL.ROI_MASK_HEAD.INTERLEAVED_EXECUTION:
                        x, segs1, _ = self.mask1(mask_features, detections, targets, semanticx=semanticx)
                        x, segs2, _ = self.mask2(mask_features, detections, targets, x, semanticx=semanticx)
                        x, segs3, _ = self.mask3(mask_features, detections, targets, x, semanticx=semanticx)
                    else:
                        x, segs1, _ = self.mask1(mask_features, detections, targets, semanticx=semanticx)
                        x, segs2, _ = self.mask2(mask_features, detections, targets, semanticx=semanticx)
                        x, segs3, _ = self.mask3(mask_features, detections, targets, semanticx=semanticx)
                    tot = len(segs1)
                    segs = [(segs1[i].get_field("mask") + segs2[i].get_field("mask") + segs3[i].get_field("mask")) / 3 for i in range(tot)]
                    for i in range(tot):
                        detections[i].add_field("mask", segs[i])
            else:
                x, detections, loss_mask = self.mask(mask_features, detections, targets, semanticx=semanticx)
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

        if self.cfg.MODEL.IOU_ON:
            if self.training:
                x, _, loss_iou = self.iou(features, targets=targets)
                losses.update(loss_iou)

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
        if cfg.MODEL.CASCADE_ON:
            roi_heads.append(("mask1", build_roi_mask_head(cfg, in_channels, stage=1)))
            roi_heads.append(("mask2", build_roi_mask_head(cfg, in_channels, stage=2)))
            roi_heads.append(("mask3", build_roi_mask_head(cfg, in_channels, stage=3)))
        else:
            roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels)))
    if cfg.MODEL.KEYPOINT_ON:
        roi_heads.append(("keypoint", build_roi_keypoint_head(cfg, in_channels)))

    if cfg.MODEL.SEMANTIC_ON:
        roi_heads.append(("semantic", build_semantic_head(cfg, in_channels)))

    if cfg.MODEL.IOU_ON:
        roi_heads.append(("iou", build_roi_iou_head(cfg, in_channels)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
