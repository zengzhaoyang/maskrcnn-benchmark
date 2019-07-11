# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F
from maskrcnn_benchmark.modeling.utils import cat

class SemanticLossComputation(object):
    def __init__(self):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

    def prepare_targets(self, size, device, targets):
        masks = torch.full((len(targets), size[0], size[1]), 255, dtype=torch.int64).to(device=device)
        for idx, targets_per_image in enumerate(targets):
            stuff = targets_per_image.get_field("stuff").to(dtype=torch.int64, device=device)
            shape = stuff.shape
            masks[idx, :shape[1], :shape[2]] = stuff

        return masks

    def __call__(self, mask_pred, targets):
        """
        Arguments:
            proposals (list[BoxList])
            mask_logits (Tensor)
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """
        size = mask_pred.shape[-2:]
        labels = self.prepare_targets(size, mask_pred.device, targets)

        #mask_pred = mask_pred.permute(0, 2, 3, 1)
        #labels = labels.permute(0, 2, 3, 1)
        #log_softmax = F.log_softmax(mask_pred, dim=-1)
        #semantic_loss = 0.2 * (-labels * log_softmax).sum(dim=3).mean(dim=2).mean(dim=1).mean(dim=0)
        semantic_loss = 0.2 * self.criterion(mask_pred, labels)

        return semantic_loss


def make_semantic_loss_evaluator(cfg):

    loss_evaluator = SemanticLossComputation()

    return loss_evaluator
