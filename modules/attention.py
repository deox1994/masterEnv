import torch
import torch.nn as nn
from torch.nn import init

__all__ = (
    "CBAM",
    "LSKA",
)

"""Convolutional Block Attention Module"""
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelAttention(nn.Module):
    def __init__(self, c1, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(c1, c1 // ratio),
            nn.ReLU(),
            nn.Linear(c1 // ratio, c1)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a = self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))).unsqueeze(2).unsqueeze(3)
        return a

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, bn=False, relu=False):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = self.conv1(torch.cat([avg_out, max_out], dim=1))
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)            
        return self.sigmoid(x)
    
class CBAM(nn.Module):
    def __init__(self, c1, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(c1, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out
    

class LSKA(nn.Module):
    def __init__(self, c1):
        super(LSKA, self).__init__()
        self.DW_conv_h = nn.Conv2d(c1, c1, kernel_size=(1, 3), stride=(1,1), padding=(0,(3-1)//2), groups=c1)
        self.DW_conv_v = nn.Conv2d(c1, c1, kernel_size=(3, 1), stride=(1,1), padding=((3-1)//2,0), groups=c1)
        self.DW_D_conv_h = nn.Conv2d(c1, c1, kernel_size=(1, 3), stride=(1,1), padding=(0,2), groups=c1, dilation=2)
        self.DW_D_conv_v = nn.Conv2d(c1, c1, kernel_size=(3, 1), stride=(1,1), padding=(2,0), groups=c1, dilation=2)
        self.conv1 = nn.Conv2d(c1, c1, 1)

    def forward(self, x):
        return x * self.conv1(self.DW_D_conv_v(self.DW_D_conv_h(self.DW_conv_v(self.DW_conv_h(x)))))