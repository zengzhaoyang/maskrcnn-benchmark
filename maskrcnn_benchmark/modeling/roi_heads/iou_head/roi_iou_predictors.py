# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn

class IOUPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(IOUPredictor, self).__init__()
        representation_size = in_channels

        self.iou_pred = nn.Linear(representation_size, cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES)

        nn.init.normal_(self.iou_pred.weight, std=0.01)
        nn.init.constant_(self.iou_pred.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        ious = self.iou_pred(x)
        return ious


def make_roi_iou_predictor(cfg, in_channels):
    return IOUPredictor(cfg, in_channels)
