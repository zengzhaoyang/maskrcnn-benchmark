# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

# TODO maybe push this to nn?
def iou_loss(input, target, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """

    N = input.shape[0]
    M = target.shape[0]

    area1 = (input[:, 2] - input[:, 0] + 1) * (input[:, 3] - input[:, 1] + 1)
    area2 = (target[:, 2] - target[:, 0] + 1) * (target[:, 3] - target[:, 1] + 1)

    lt = torch.max(input[:, :2], target[:, :2])
    rb = torch.min(input[:, 2:], target[:, 2:])

    wh = (rb - lt + 1).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    ious = inter / (area1 + area2 - inter)

    loss = -ious.log()

    #if size_average:
    #    return loss.mean()
    return loss.sum()

def bounded_iou_loss(input, target, beta=0.2, eps=1e-3, size_average=True):

    pred_ctrx = (input[:, 0] + input[:, 2]) * 0.5
    pred_ctry = (input[:, 1] + input[:, 3]) * 0.5
    pred_w = input[:, 2] - input[:, 0] + 1
    pred_h = input[:, 3] - input[:, 1] + 1

    with torch.no_grad():
        target_ctrx = (target[:, 0] + target[:, 2]) * 0.5
        target_ctry = (target[:, 1] + target[:, 3]) * 0.5
        target_w = target[:, 2] - target[:, 0] + 1
        target_h = target[:, 3] - target[:, 1] + 1

    dx = target_ctrx - pred_ctrx
    dy = target_ctry - pred_ctry

    loss_dx = 1 - torch.max(
        (target_w - 2 * dx.abs()) /
        (target_w + 2 * dx.abs() + eps), torch.zeros_like(dx))
    loss_dy = 1 - torch.max(
        (target_h - 2 * dy.abs()) /
        (target_h + 2 * dy.abs() + eps), torch.zeros_like(dy))
    loss_dw = 1 - torch.min(target_w / (pred_w + eps), pred_w / (target_w + eps))
    loss_dh = 1 - torch.min(target_h / (pred_h + eps), pred_h / (target_h + eps))
    loss_comb = torch.stack([loss_dx, loss_dy, loss_dw, loss_dh],
                            dim=-1).view(loss_dx.size(0), -1)
    loss = torch.where(loss_comb < beta, 0.5 * loss_comb * loss_comb / beta, loss_comb - 0.5 * beta)

    if size_average:
        return loss.mean()
    return loss.sum()# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

# TODO maybe push this to nn?
def iou_loss(input, target, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """

    N = input.shape[0]
    M = target.shape[0]

    area1 = (input[:, 2] - input[:, 0] + 1) * (input[:, 3] - input[:, 1] + 1)
    area2 = (target[:, 2] - target[:, 0] + 1) * (target[:, 3] - target[:, 1] + 1)

    lt = torch.max(input[:, :2], target[:, :2])
    rb = torch.min(input[:, 2:], target[:, 2:])

    wh = (rb - lt + 1).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    ious = inter / (area1 + area2 - inter)

    loss = -ious.log()

    #if size_average:
    #    return loss.mean()
    return loss.sum()

def bounded_iou_loss(input, target, beta=0.2, eps=1e-3, size_average=True):

    pred_ctrx = (input[:, 0] + input[:, 2]) * 0.5
    pred_ctry = (input[:, 1] + input[:, 3]) * 0.5
    pred_w = input[:, 2] - input[:, 0] + 1
    pred_h = input[:, 3] - input[:, 1] + 1

    with torch.no_grad():
        target_ctrx = (target[:, 0] + target[:, 2]) * 0.5
        target_ctry = (target[:, 1] + target[:, 3]) * 0.5
        target_w = target[:, 2] - target[:, 0] + 1
        target_h = target[:, 3] - target[:, 1] + 1

    dx = target_ctrx - pred_ctrx
    dy = target_ctry - pred_ctry

    loss_dx = 1 - torch.max(
        (target_w - 2 * dx.abs()) /
        (target_w + 2 * dx.abs() + eps), torch.zeros_like(dx))
    loss_dy = 1 - torch.max(
        (target_h - 2 * dy.abs()) /
        (target_h + 2 * dy.abs() + eps), torch.zeros_like(dy))
    loss_dw = 1 - torch.min(target_w / (pred_w + eps), pred_w / (target_w + eps))
    loss_dh = 1 - torch.min(target_h / (pred_h + eps), pred_h / (target_h + eps))
    loss_comb = torch.stack([loss_dx, loss_dy, loss_dw, loss_dh],
                            dim=-1).view(loss_dx.size(0), -1)
    loss = torch.where(loss_comb < beta, 0.5 * loss_comb * loss_comb / beta, loss_comb - 0.5 * beta)

    if size_average:
        return loss.mean()
    return loss.sum()
