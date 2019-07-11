# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn
from maskrcnn_benchmark.layers import SyncBatchNorm2d


class NASFPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(
        self, out_channels, conv_block
    ):
        super(NASFPN, self).__init__()

        self.gp_p5_p3_rcb = nn.Sequential(
            nn.ReLU(inplace=True),
            conv_block(out_channels, out_channels, 3, 1),
            SyncBatchNorm2d(out_channels) 
        )
        
        self.sum1_rcb = nn.Sequential(
            nn.ReLU(inplace=True),
            conv_block(out_channels, out_channels, 3, 1),
            SyncBatchNorm2d(out_channels) 
        )

        self.sum2_rcb = nn.Sequential(
            nn.ReLU(inplace=True),
            conv_block(out_channels, out_channels, 3, 1),
            SyncBatchNorm2d(out_channels) 
        )

        self.sum3_rcb = nn.Sequential(
            nn.ReLU(inplace=True),
            conv_block(out_channels, out_channels, 3, 1),
            SyncBatchNorm2d(out_channels) 
        )

        self.sum4_rcb = nn.Sequential(
            nn.ReLU(inplace=True),
            conv_block(out_channels, out_channels, 3, 1),
            SyncBatchNorm2d(out_channels) 
        )

        self.sum5_rcb = nn.Sequential(
            nn.ReLU(inplace=True),
            conv_block(out_channels, out_channels, 3, 1),
            SyncBatchNorm2d(out_channels) 
        )

        self.sum4_rcb_gp1_rcb = nn.Sequential(
            nn.ReLU(inplace=True),
            conv_block(out_channels, out_channels, 3, 1),
            SyncBatchNorm2d(out_channels) 
        )


    def gp(self, fm1, fm2):
        size = tuple(fm2.shape[-2:])
        global_ctx = fm1.mean(dim=(2,3), keepdim=True)
        global_ctx = global_ctx.sigmoid()
        output = (global_ctx * fm2) + F.interpolate(fm1, size=size, mode='bilinear', align_corners=True)
        return output

    def sum_fm(self, fm1, fm2):
        size = tuple(fm2.shape[-2:])
        output = fm2 + F.interpolate(fm1, size=size, mode='bilinear', align_corners=True)
        return output
         

    def forward(self, x):

        P2 = x[0]
        P3 = x[1]
        P4 = x[2]
        P5 = x[3]
        P6 = x[4]

        GP_P5_P3 = self.gp(P5, P3)
        GP_P5_P3_RCB = self.gp_p5_p3_rcb(GP_P5_P3)
        SUM1 = self.sum_fm(GP_P5_P3_RCB, P3)
        SUM1_RCB = self.sum1_rcb(SUM1)
        SUM2 = self.sum_fm(SUM1_RCB, P2)
        SUM2_RCB = self.sum2_rcb(SUM2) # P2
        SUM3 = self.sum_fm(SUM2_RCB, SUM1_RCB)
        SUM3_RCB = self.sum3_rcb(SUM3) # P3
        SUM3_RCB_GP = self.gp(SUM2_RCB, SUM3_RCB)
        SUM4 = self.sum_fm(SUM3_RCB_GP, P4)
        SUM4_RCB = self.sum4_rcb(SUM4) # P4
        SUM4_RCB_GP = self.gp(SUM1_RCB, SUM4_RCB)
        SUM5 = self.sum_fm(SUM4_RCB_GP, P6)
        SUM5_RCB = self.sum5_rcb(SUM5) # P6
        size = tuple(P5.shape[-2:])
        SUM5_RCB_resize = F.interpolate(SUM5_RCB, size=size, mode="bilinear", align_corners=True)
        SUM4_RCB_GP1 = self.gp(SUM4_RCB, SUM5_RCB_resize)
        SUM4_RCB_GP1_RCB = self.sum4_rcb_gp1_rcb(SUM4_RCB_GP1) # P5

        #P2 = SUM2_RCB
        #P3 = SUM3_RCB
        #P4 = SUM4_RCB
        #P5 = SUM4_RCB_GP1_RCB
        #P6 = SUM5_RCB

        #return [P2, P3, P4, P5, P6]

        return [SUM2_RCB, SUM3_RCB, SUM4_RCB, SUM4_RCB_GP1_RCB, SUM5_RCB]
