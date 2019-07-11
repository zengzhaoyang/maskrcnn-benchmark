# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from torch.nn import functional as F
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.make_layers import make_conv1x1, make_conv3x3
from .loss import make_semantic_loss_evaluator

class SemanticHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(SemanticHead, self).__init__()
        self.cfg = cfg.clone()
        self.loss_evaluator = make_semantic_loss_evaluator(cfg)

        num_ins = cfg.MODEL.SEMANTIC_HEAD.NUM_INS
        num_conv = cfg.MODEL.SEMANTIC_HEAD.NUM_STACKED_CONVS
        conv_channels = cfg.MODEL.SEMANTIC_HEAD.CONV_CHANNELS
        use_gn = cfg.MODEL.SEMANTIC_HEAD.USE_GN
        dilation = cfg.MODEL.SEMANTIC_HEAD.DILATION
        use_ws = cfg.MODEL.USE_WS

        self.later_convs = []
        for layer_idx in range(num_ins):
            layer_name = "later{}".format(layer_idx)
            module = make_conv1x1(
                in_channels, conv_channels,
                stride=1, use_gn=use_gn, use_ws=use_ws
            )
            self.add_module(layer_name, module)
            self.later_convs.append(layer_name)

        self.blocks = []
        for layer_idx in range(num_conv):
            layer_name = "semantic_fcn{}".format(layer_idx + 1)
            module = make_conv3x3(
                conv_channels, conv_channels,
                dilation=dilation, stride=1, use_gn=use_gn, use_ws=use_ws
            )
            self.add_module(layer_name, module)
            self.blocks.append(layer_name)

        self.conv_embedding = make_conv1x1(
            conv_channels, conv_channels,
            stride=1, use_gn=use_gn, use_ws=use_ws
        )
        self.conv_logits = make_conv1x1(
            conv_channels, cfg.MODEL.SEMANTIC_HEAD.NUM_CLASSES,
            stride=1
        )

        self.fusion_level = cfg.MODEL.SEMANTIC_HEAD.FUSION_LEVEL

    def forward(self, features, targets=None):
        
        x = getattr(self, self.later_convs[self.fusion_level])(features[self.fusion_level]) 
        fused_size = tuple(x.shape[-2:])

        for i, feat in enumerate(features):
            if i != self.fusion_level:
                feat = F.interpolate(
                    feat, size=fused_size, mode='bilinear', align_corners=True
                )
                x += F.relu(getattr(self, self.later_convs[i])(feat), inplace=True)

        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x), inplace=True)

        mask_pred = self.conv_logits(x)
        x = F.relu(self.conv_embedding(x))

        if not self.training:
            return x, mask_pred, {}

        loss_semantic = self.loss_evaluator(mask_pred, targets)

        return x, mask_pred, dict(loss_semantic=loss_semantic)

def build_semantic_head(cfg, in_channels):
    return SemanticHead(cfg, in_channels)
