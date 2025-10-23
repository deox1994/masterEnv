import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import Bottleneck, C2f

from torch.nn.modules.pooling import MaxPool2d, AvgPool2d

from .conv import ScConv
from .pooling import TMaxAvgPool2d, RAPool2d, RWPool2d, SoftPool2d
from .attention import CBAM, LSKA


__all__ = (
    "SPPFNew",
    "DualBranchSPPF",
    "C2f_ScConv",
)

class SPPFNew(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super(SPPFNew, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.k = k

        """ Pooling Types """
        self.pool = MaxPool2d(kernel_size=self.k, stride=1, padding=self.k//2)
        #self.m = AvgPool2d(kernel_size=self.k, stride=1, padding=self.k//2)
        #self.m = TMaxAvgPool2d(kernel_size=self.k, stride=1, padding=self.k//2, k=3, T=0.9)
        #self.m = RAPool2d(kernel_size=self.k, stride=1, padding=self.k//2)
        #self.m = RWPool2d(kernel_size=self.k, stride=1, padding=self.k//2)
        #self.m = SoftPool2d(kernel_size=self.k, stride=1, padding=self.k//2)

        """ Attention Mechanism """
        #self.att = CBAM(c_ * 4)
        self.att = LSKA(c_ * 4)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.pool(y[-1]) for _ in range(3))
        #return self.cv2(torch.cat(y, 1))
        return self.cv2(self.att(torch.cat(y, 1)))
    
class DualBranchSPPF(nn.Module):
    r"""
    """
    def __init__(self, c1, c2, k=5):
        super(DualBranchSPPF, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv = Conv(c1, c_, 1, 1)
        self.k = k
        #self.m1 = MaxPool2d(kernel_size=self.k, stride=1, padding=self.k//2)
        self.m1 = TMaxAvgPool2d(kernel_size=self.k, stride=1, padding=self.k//2, k=3, T=0.9)
        self.cv1 = Conv(c_ * 4, c2//2, 1, 1)
        #self.m2 = AvgPool2d(kernel_size=self.k, stride=1, padding=self.k//2)
        self.m2 = RWPool2d(kernel_size=self.k, stride=1, padding=self.k//2)
        self.cv2 = Conv(c_ * 4, c2//2, 1, 1)

    def forward(self, x):
        x_aux = self.cv(x)
        y1 = [x_aux]
        y2 = [x_aux]
        y1.extend(self.m1(y1[-1]) for _ in range(3))
        y2.extend(self.m2(y2[-1]) for _ in range(3))
        return torch.cat([self.cv1(torch.cat(y1, 1)), self.cv2(torch.cat(y2, 1))], 1)
    
class Bottleneck_ScConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = ScConv(c2)

class C2f_ScConv(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_ScConv(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))