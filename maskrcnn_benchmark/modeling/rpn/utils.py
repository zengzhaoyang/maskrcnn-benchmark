# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Utility functions minipulating the prediction layers
"""

from ..utils import cat

import torch

def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(
        box_cls, box_regression
    ):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(
            box_cls_per_level, N, A, C, H, W
        )
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(
            box_regression_per_level, N, A, 4, H, W
        )
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).reshape(-1, C)
    box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression

def concat_locs(locs, locs_target, locs_weight):
    locs_flattened = []
    locs_target_flattened = []
    locs_weight_flattened = []

    for locs_per_level, locs_target_per_level, locs_weight_per_level in zip(locs, locs_target, locs_weight):
        N, C, H, W = locs_per_level.shape
        locs_per_level = permute_and_flatten(
            locs_per_level, N, 1, C, H, W
        )
        locs_target_per_level = permute_and_flatten(
            locs_target_per_level, N, 1, C, H, W
        )
        locs_weight_per_level = permute_and_flatten(
            locs_weight_per_level, N, 1, C, H, W
        )
        locs_flattened.append(locs_per_level)
        locs_target_flattened.append(locs_target_per_level)
        locs_weight_flattened.append(locs_weight_per_level)

    locs = cat(locs_flattened, dim=1).reshape(-1, C)
    locs_target = cat(locs_target_flattened, dim=1).reshape(-1, C)
    locs_weight = cat(locs_weight_flattened, dim=1).reshape(-1, C)
    return locs, locs_target, locs_weight

def concat_shapes(shapes):
    shapes_flattened = []
    for shapes_per_level in shapes:
        N, C, H, W = shapes_per_level.shape
        shapes_per_level = permute_and_flatten(
            shapes_per_level, N, 1, C, H, W
        )
        shapes_flattened.append(shapes_per_level)

    shapes = cat(shapes_flattened, dim=1).reshape(-1, C)
    shapes_0 = torch.zeros_like(shapes)
    shapes = torch.cat([shapes_0, shapes], dim=1)
    return shapes
