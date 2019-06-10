# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import numpy as np

# TODO maybe push this to nn?
def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()


def balanced_l1_loss(input, target, alpha=0.5, gamma=1.5, beta=1. / 9, size_average=True):

    n = torch.abs(input - target)
    cond = n < beta
    b = np.e**(gamma / alpha) - 1
    loss = torch.where(cond, 
        alpha / b * (b * n + 1) * torch.log(b * n / beta + 1) -
        alpha * n, gamma * n + gamma / b - alpha * beta)

    if size_average:
        return loss.mean()
    return loss.sum()
