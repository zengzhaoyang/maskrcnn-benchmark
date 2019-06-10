# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import numpy as np
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.box_coder import BoxCoder

class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


class GAnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set
    of anchors
    """

    def __init__(
        self,
        aspect_ratios=(0.5, 1.0, 2.0),
        anchor_strides=(4, 8, 16, 32, 64),
        straddle_thresh=0,
        octave_base_scale=8,
        scales_per_octave=3,
        anchor_weights=(10., 10., 5., 5.),
        loc_filter_thr=0.01
    ):
        super(GAnchorGenerator, self).__init__()

        if len(anchor_strides) == 1:
            anchor_stride = anchor_strides[0]
            approx_sizes = [octave_base_scale * 2**(i * 1.0 / scales_per_octave) * anchor_stride for i in range(scales_per_octave)]
            approx_anchors = [
                generate_anchors(anchor_stride, approx_sizes, aspect_ratios).float()
            ]
            square_anchors = [
                generate_anchors(anchor_stride, [octave_base_scale * anchor_stride], [1.0]).float()
            ]
        else:
            approx_anchors = [
                 generate_anchors(
                     anchor_stride,
                     [octave_base_scale * 2**(i * 1.0 / scales_per_octave) * anchor_stride for i in range(scales_per_octave)],
                     aspect_ratios
                 ).float()
                 for anchor_stride in anchor_strides
            ]
            square_anchors = [
                 generate_anchors(
                     anchor_stride,
                     [octave_base_scale * anchor_stride],
                     [1.0],
                 ).float()
                 for anchor_stride in anchor_strides
            ]

        self.strides = anchor_strides
        self.approx_anchors = BufferList(approx_anchors)
        self.square_anchors = BufferList(square_anchors)
        self.straddle_thresh = straddle_thresh
        self.anchor_box_coder = BoxCoder(weights=anchor_weights)
        self.loc_filter_thr = loc_filter_thr

    def num_approx_anchors_per_location(self):
        return [len(approx_anchors) for approx_anchors in self.approx_anchors]

    def grid_square_anchors(self, grid_sizes):
        anchors = []
        for size, stride, base_anchors in zip(
            grid_sizes, self.strides, self.square_anchors
        ):
            grid_height, grid_width = size
            device = base_anchors.device
            shifts_x = torch.arange(
                0, grid_width * stride, step=stride, dtype=torch.float32, device=device
            )
            shifts_y = torch.arange(
                0, grid_height * stride, step=stride, dtype=torch.float32, device=device
            )
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors

    def grid_approx_anchors(self, grid_sizes):
        anchors = []
        for size, stride, base_anchors in zip(
            grid_sizes, self.strides, self.approx_anchors
        ):
            grid_height, grid_width = size
            device = base_anchors.device
            shifts_x = torch.arange(
                0, grid_width * stride, step=stride, dtype=torch.float32, device=device
            )
            shifts_y = torch.arange(
                0, grid_height * stride, step=stride, dtype=torch.float32, device=device
            )
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors


    def add_visibility_to(self, boxlist):
        image_width, image_height = boxlist.size
        anchors = boxlist.bbox
        if self.straddle_thresh >= 0:
            inds_inside = (
                (anchors[..., 0] >= -self.straddle_thresh)
                & (anchors[..., 1] >= -self.straddle_thresh)
                & (anchors[..., 2] < image_width + self.straddle_thresh)
                & (anchors[..., 3] < image_height + self.straddle_thresh)
            )
        else:
            device = anchors.device
            inds_inside = torch.ones(anchors.shape[0], dtype=torch.uint8, device=device)
        boxlist.add_field("visibility", inds_inside)

    def forward(self, image_list, feature_maps, shapes, locs):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]

        num_levels = len(feature_maps)

        square_anchors_over_all_feature_maps = self.grid_square_anchors(grid_sizes)

        square_anchors = []
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            square_anchors_in_image = []
            for square_anchors_per_feature_map in square_anchors_over_all_feature_maps:
                boxlist = BoxList(
                    square_anchors_per_feature_map, (image_width, image_height), mode="xyxy"
                )
                self.add_visibility_to(boxlist)
                square_anchors_in_image.append(boxlist)
            square_anchors.append(square_anchors_in_image)

        guided_anchors = []
        loc_masks = []
        for img_id, (image_height, image_width) in enumerate(image_list.image_sizes):
            guided_anchors_in_image = []
            loc_mask_in_image = []
            for i in range(num_levels):
                squares = square_anchors[img_id][i]
                shape_pred = shapes[i][img_id]
                loc_pred = locs[i][img_id]
                guide_anchors_single, loc_mask_single = self.get_guided_anchors(
                    squares,
                    shape_pred,
                    loc_pred,
                    use_loc_filter=not self.training
                )
                guide_anchors_single = BoxList(
                    guide_anchors_single, (image_width, image_height), mode="xyxy"
                )
                self.add_visibility_to(guide_anchors_single)
                guided_anchors_in_image.append(guide_anchors_single)
                loc_mask_in_image.append(loc_mask_single)
            guided_anchors.append(guided_anchors_in_image)
            loc_masks.append(loc_mask_in_image)

        return square_anchors, guided_anchors, loc_masks

    def get_guided_anchors(self, squares, shape_pred, loc_pred, use_loc_filter=False):
        loc_pred = loc_pred.sigmoid().detach()
        if use_loc_filter:
            loc_mask = loc_pred >= self.loc_filter_thr
        else:
            loc_mask = loc_pred >= 0
        mask = loc_mask.permute(1, 2, 0).expand(-1, -1, 1)
        mask = mask.contiguous().view(-1)

        squares = squares[mask]
        anchor_deltas = shape_pred.permute(1, 2, 0).contiguous().view(-1, 2).detach()[mask]
        bbox_deltas = anchor_deltas.new_full(squares.bbox.size(), 0)
        bbox_deltas[:, 2:] = anchor_deltas
        guided_anchors = self.anchor_box_coder.decode(
            bbox_deltas, squares.bbox
        )
        return guided_anchors, mask

    def get_sampled_approxs(self, image_list, feature_maps):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]

        approx_anchors_over_all_feature_maps = self.grid_approx_anchors(grid_sizes)
        approx_anchors = []
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            approx_anchors_in_image = []
            for approx_anchors_per_feature_map in approx_anchors_over_all_feature_maps:
                boxlist = BoxList(
                    approx_anchors_per_feature_map, (image_width, image_height), mode="xyxy"
                )
                self.add_visibility_to(boxlist)
                approx_anchors_in_image.append(boxlist)
            approx_anchors.append(approx_anchors_in_image) 

        return approx_anchors

def make_ga_anchor_generator(config):
    aspect_ratios = config.MODEL.RPN.GA.ASPECT_RATIOS
    anchor_stride = config.MODEL.RPN.GA.ANCHOR_STRIDE
    straddle_thresh = config.MODEL.RPN.STRADDLE_THRESH
    octave_base_scale = config.MODEL.RPN.GA.OCTAVE_BASE_SCALE
    scales_per_octave = config.MODEL.RPN.GA.SCALES_PER_OCTAVE
    anchor_weights = config.MODEL.RPN.GA.ANCHOR_WEIGHTS
    loc_filter_thr = config.MODEL.RPN.GA.LOC_FILTER_THR

    #if config.MODEL.RPN.USE_FPN:
    #    assert len(anchor_stride) == len(
    #        anchor_sizes
    #    ), "FPN should have len(ANCHOR_STRIDE) == len(ANCHOR_SIZES)"
    #else:
    #    assert len(anchor_stride) == 1, "Non-FPN should have a single ANCHOR_STRIDE"
    anchor_generator = GAnchorGenerator(
        aspect_ratios,
        anchor_stride,
        straddle_thresh,
        octave_base_scale,
        scales_per_octave,
        anchor_weights,
        loc_filter_thr
    )
    return anchor_generator

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------


# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#        [-175.,  -87.,  192.,  104.],
#        [-359., -183.,  376.,  200.],
#        [ -55.,  -55.,   72.,   72.],
#        [-119., -119.,  136.,  136.],
#        [-247., -247.,  264.,  264.],
#        [ -35.,  -79.,   52.,   96.],
#        [ -79., -167.,   96.,  184.],
#        [-167., -343.,  184.,  360.]])


def generate_anchors(
    stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)
):
    """Generates a matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    are centered on stride / 2, have (approximate) sqrt areas of the specified
    sizes, and aspect ratios as given.
    """
    return _generate_anchors(
        stride,
        np.array(sizes, dtype=np.float) / stride,
        np.array(aspect_ratios, dtype=np.float),
    )


def _generate_anchors(base_size, scales, aspect_ratios):
    """Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
    """
    anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
    anchors = _ratio_enum(anchor, aspect_ratios)
    anchors = np.vstack(
        [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
    )
    return torch.from_numpy(anchors)


def _whctrs(anchor):
    """Return width, height, x center, and y center for an anchor (window)."""
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
        )
    )
    return anchors


def _ratio_enum(anchor, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """Enumerate a set of anchors for each scale wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors
