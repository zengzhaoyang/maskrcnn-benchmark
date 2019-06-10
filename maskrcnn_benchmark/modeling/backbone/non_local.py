import torch
import torch.nn as nn
import torch.nn.functional as F
from maskrcnn_benchmark.layers import Conv2d

class NonLocalBlock2D(nn.Module):

    def __init__(self, in_channels, conv_block, reduction=2):
        super(NonLocalBlock2D, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = in_channels // reduction
        
        self.g = Conv2d(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=True)
        self.theta = Conv2d(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=True)
        self.phi = Conv2d(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=True)

        self.conv_mask = Conv2d(
            self.inter_channels,
            self.in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=True)

        nn.init.kaiming_uniform_(self.g.weight, a=1)
        nn.init.constant_(self.g.bias, 0)
        nn.init.kaiming_uniform_(self.theta.weight, a=1)
        nn.init.constant_(self.theta.bias, 0)
        nn.init.kaiming_uniform_(self.phi.weight, a=1)
        nn.init.constant_(self.phi.bias, 0)
        nn.init.constant_(self.conv_mask.weight, 0)
        nn.init.constant_(self.conv_mask.bias, 0) 

    def forward(self, x):
        batch_size = x.size(0)
        
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        map_t_p = torch.matmul(theta_x, phi_x)
        mask_t_p = F.softmax(map_t_p, dim=-1)

        map_ = torch.matmul(mask_t_p, g_x)
        map_ = map_.permute(0, 2, 1).contiguous()
        map_ = map_.view(batch_size, self.inter_channels, x.size(2), x.size(3))
        mask = self.conv_mask(map_)
        final = mask + x
        return final

