# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.rpn.retinanet.retinanet import build_retinanet
from .loss import make_ga_rpn_loss_evaluator
from .ga_anchor_generator import make_ga_anchor_generator
from .inference import make_rpn_postprocessor
from maskrcnn_benchmark.layers import DeformConv

class FeatureAdaption(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, deformable_groups=4):
        super(FeatureAdaption, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(2, deformable_groups * offset_channels, 1, bias=False)
        self.conv_adaption = DeformConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)

        torch.nn.init.normal_(self.conv_offset.weight, std=0.1)
        torch.nn.init.normal_(self.conv_adaption.weight, std=0.01)

    def forward(self, x, shape):
        offset = self.conv_offset(shape.detach())
        x = self.relu(self.conv_adaption(x, offset))
        return x

@registry.RPN_HEADS.register("SingleConvGARPNHead")
class GARPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(GARPNHead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )
        self.feature_adaption = FeatureAdaption(
            in_channels, in_channels, kernel_size=3, deformable_groups=cfg.MODEL.RPN.GA.DEFORMABLE_GROUPS)
        self.conv_loc = nn.Conv2d(in_channels, 1, 1)
        self.conv_shape = nn.Conv2d(in_channels, num_anchors * 2, 1)

        # TODO: mask conv
        # TODO: conv_loc init

        for l in [self.conv, self.cls_logits, self.bbox_pred, self.conv_loc, self.conv_shape]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)
        #

    def forward(self, x):
        logits = []
        bbox_reg = []
        locs = []
        shapes = []
        for feature in x:
            t = F.relu(self.conv(feature))
            locs.append(self.conv_loc(t))
            shape_pred = self.conv_shape(t)
            shapes.append(shape_pred) 
            #TODO: add mask
            t = self.feature_adaption(t, shape_pred)
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg, shapes, locs


class GARPNModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and RPN
    proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg, in_channels):
        super(GARPNModule, self).__init__()

        self.cfg = cfg.clone()

        anchor_generator = make_ga_anchor_generator(cfg)

        rpn_head = registry.RPN_HEADS[cfg.MODEL.RPN.RPN_HEAD]
        head = rpn_head(
            cfg, in_channels, 1
        )

        #rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        rpn_box_coder = BoxCoder(weights=cfg.MODEL.RPN.GA.TARGET_WEIGHTS)
        anchor_box_coder = BoxCoder(weights=cfg.MODEL.RPN.GA.ANCHOR_WEIGHTS)

        box_selector_train = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=True)
        box_selector_test = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=False)

        loss_evaluator = make_ga_rpn_loss_evaluator(cfg, rpn_box_coder, anchor_box_coder)

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        objectness, rpn_box_regression, shapes, locs = self.head(features)
        #anchors = self.anchor_generator(images, features, shapes, locs)
        square_anchors, guided_anchors, loc_masks = self.anchor_generator(images, features, shapes, locs)

        if self.training:
            approx_anchors = self.anchor_generator.get_sampled_approxs(images, features)
            return self._forward_train(square_anchors, guided_anchors, loc_masks, approx_anchors, objectness, rpn_box_regression, shapes, locs, targets)
        else:
            #tot = len(objectness)
            #for i in range(tot):
            #    locs_mask = locs[i] >= 0.01
            #    locs_mask = locs_mask.float()
            #    objectness[i] = objectness[i] - (1 - locs_mask) * 13
            return self._forward_test(guided_anchors, objectness, rpn_box_regression)

    def _forward_train(self, square_anchors, guided_anchors, loc_masks, approx_anchors, objectness, rpn_box_regression, shapes, locs, targets):

        with torch.no_grad():
            boxes = self.box_selector_train(
                guided_anchors, objectness, rpn_box_regression, targets
            )
        loss_objectness, loss_rpn_box_reg, loss_shapes, loss_locs = self.loss_evaluator(
            square_anchors, guided_anchors, loc_masks, approx_anchors, objectness, rpn_box_regression, shapes, locs, targets
        )
        losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
            "loss_shapes": loss_shapes,
            "loss_locs": loss_locs
        }
        return boxes, losses

    def _forward_test(self, anchors, objectness, rpn_box_regression):
        boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
        if self.cfg.MODEL.RPN_ONLY:
            # For end-to-end models, the RPN proposals are an intermediate state
            # and don't bother to sort them in decreasing score order. For RPN-only
            # models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            inds = [
                box.get_field("objectness").sort(descending=True)[1] for box in boxes
            ]
            boxes = [box[ind] for box, ind in zip(boxes, inds)]
        return boxes, {}


def build_ga_rpn(cfg, in_channels):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    return GARPNModule(cfg, in_channels)
