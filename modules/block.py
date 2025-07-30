import torch
import torch.nn as nn
import torch.functional as F

from ultralytics.nn.modules.conv import Conv

from torch.nn.modules.pooling import MaxPool2d, AvgPool2d

__all__ = (
    "SPPFNew",
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
        self.m = MaxPool2d(kernel_size=self.k, stride=1, padding=self.k//2)
        #self.m = AvgPool2d(kernel_size=self.k, stride=1, padding=self.k//2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))