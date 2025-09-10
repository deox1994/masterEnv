import torch
import torch.nn as nn
import torch.functional as F

from ultralytics.nn.modules.conv import Conv

from torch.nn.modules.pooling import MaxPool2d, AvgPool2d
from .pooling import TMaxAvgPool2d, RAPool2d, RWPool2d, SoftPool2d

__all__ = (
    "SPPFNew",
    "DualBranchSPPF",
    "SPPF_LSKA",
    "DualBranchSPPF_LSKA",
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
        #self.m = TMaxAvgPool2d(kernel_size=self.k, stride=1, padding=self.k//2, k=3, T=0.9)
        #self.m = RAPool2d(kernel_size=self.k, stride=1, padding=self.k//2)
        #self.m = RWPool2d(kernel_size=self.k, stride=1, padding=self.k//2)
        #self.m = SoftPool2d(kernel_size=self.k, stride=1, padding=self.k//2)
        print(self.m)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))
    
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
    
class SPPF_LSKA(nn.Module):
    r"""
    """
    def __init__(self, c1, c2, k=5):
        super(SPPF_LSKA, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.k = k
        self.m = MaxPool2d(kernel_size=self.k, stride=1, padding=self.k//2)
        #self.m = AvgPool2d(kernel_size=self.k, stride=1, padding=self.k//2)
        #self.m = TMaxAvgPool2d(kernel_size=self.k, stride=1, padding=self.k//2, k=3, T=0.9)
        #self.m = RAPool2d(kernel_size=self.k, stride=1, padding=self.k//2)
        #self.m = RWPool2d(kernel_size=self.k, stride=1, padding=self.k//2)
        #self.m = SoftPool2d(kernel_size=self.k, stride=1, padding=self.k//2)
        self.DW_conv_h = nn.Conv2d(c_ * 4, c_ * 4, kernel_size=(1, 3), stride=(1,1), padding=(0,(3-1)//2), groups=c_ * 4)
        self.DW_conv_v = nn.Conv2d(c_ * 4, c_ * 4, kernel_size=(3, 1), stride=(1,1), padding=((3-1)//2,0), groups=c_ * 4)
        self.DW_D_conv_h = nn.Conv2d(c_ * 4, c_ * 4, kernel_size=(1, 3), stride=(1,1), padding=(0,2), groups=c_ * 4, dilation=2)
        self.DW_D_conv_v = nn.Conv2d(c_ * 4, c_ * 4, kernel_size=(3, 1), stride=(1,1), padding=(2,0), groups=c_ * 4, dilation=2)
        self.conv1 = nn.Conv2d(c_ * 4, c_ * 4, 1)
        #print(self.m)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        y = torch.cat(y, 1)
        y_axu = self.DW_conv_h(y)
        y_axu = self.DW_conv_v(y_axu)
        y_axu = self.DW_D_conv_h(y_axu)
        y_axu = self.DW_D_conv_v(y_axu)
        y_axu = self.conv1(y_axu)
        return self.cv2(y * y_axu)
       
class DualBranchSPPF_LSKA(nn.Module):
    r"""
    """
    def __init__(self, c1, c2, k=5):
        super(DualBranchSPPF_LSKA, self).__init__()
        self.k = k
        c_ = c1 // 2  # hidden channels
        self.cv_sta = Conv(c1, c_, 1, 1)
        self.cv1 = Conv(c_ * 4, c2, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.cv_end = Conv(c_ * 4, c2, 1, 1)
        #self.m1 = MaxPool2d(kernel_size=self.k, stride=1, padding=self.k//2)
        #self.m2 = AvgPool2d(kernel_size=self.k, stride=1, padding=self.k//2)
        self.m1 = TMaxAvgPool2d(kernel_size=self.k, stride=1, padding=self.k//2, k=3, T=0.9)
        self.m2 = RWPool2d(kernel_size=self.k, stride=1, padding=self.k//2)
        self.DW_conv_h = nn.Conv2d(c_ * 4, c_ * 4, kernel_size=(1, 3), stride=(1,1), padding=(0,(3-1)//2), groups=c_ * 4)
        self.DW_conv_v = nn.Conv2d(c_ * 4, c_ * 4, kernel_size=(3, 1), stride=(1,1), padding=((3-1)//2,0), groups=c_ * 4)
        self.DW_D_conv_h = nn.Conv2d(c_ * 4, c_ * 4, kernel_size=(1, 3), stride=(1,1), padding=(0,2), groups=c_ * 4, dilation=2)
        self.DW_D_conv_v = nn.Conv2d(c_ * 4, c_ * 4, kernel_size=(3, 1), stride=(1,1), padding=(2,0), groups=c_ * 4, dilation=2)
        self.conv1 = nn.Conv2d(c_ * 4, c_ * 4, 1)

    def forward(self, x):
        x_aux = self.cv_sta(x)
        y1 = [x_aux]
        y2 = [x_aux]
        y1.extend(self.m1(y1[-1]) for _ in range(3))
        y2.extend(self.m2(y2[-1]) for _ in range(3))
        y = torch.cat([self.cv1(torch.cat(y1, 1)), self.cv2(torch.cat(y2, 1))], 1)
        y_axu = self.DW_conv_h(y)
        y_axu = self.DW_conv_v(y_axu)
        y_axu = self.DW_D_conv_h(y_axu)
        y_axu = self.DW_D_conv_v(y_axu)
        y_axu = self.conv1(y_axu)
        return self.cv_end(y * y_axu)
