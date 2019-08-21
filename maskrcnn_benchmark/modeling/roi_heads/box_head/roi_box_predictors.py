# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
from torch import nn
import torch
import torch.nn.functional as F

@registry.ROI_BOX_PREDICTOR.register("FastRCNNPredictor")
class FastRCNNPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(FastRCNNPredictor, self).__init__()
        assert in_channels is not None

        num_inputs = in_channels

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred


@registry.ROI_BOX_PREDICTOR.register("FPNPredictor")
class FPNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = in_channels

        self.cls_score = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


def make_roi_box_predictor(cfg, in_channels):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg, in_channels)

class FPNVGPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNVGPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = in_channels

        self.cls_score = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)

        self.cls_embedding = nn.Linear(num_classes, 256, bias=False)
        self.fc_attr =  nn.Linear(256 + representation_size, 512)
        self.attr_score = nn.Linear(512, 401)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.normal_(self.cls_embedding.weight, std=0.01)
        nn.init.normal_(self.fc_attr.weight, std=0.01)
        nn.init.normal_(self.attr_score.weight, std=0.01)
        for l in [self.cls_score, self.bbox_pred, self.fc_attr, self.attr_score]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x, proposals):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)


        cls_labels = torch.cat([proposal.get_field("labels") for proposal in proposals], dim=0).view((-1, 1))
        cls_labels = cls_labels.detach()
        cls_labels_one_hot = torch.zeros(cls_labels.shape[0], 1601).cuda().scatter_(1, cls_labels, 1)

        emb = self.cls_embedding(cls_labels_one_hot)
        concat = torch.cat((x, emb), dim=1)
        fc_attr = self.fc_attr(concat)
        fc_attr = F.relu(fc_attr, inplace=True)
        attr_score = self.attr_score(fc_attr)

        return scores, bbox_deltas, attr_score

def make_roi_vg_box_predictor(cfg, in_channels):
    return FPNVGPredictor(cfg, in_channels)
