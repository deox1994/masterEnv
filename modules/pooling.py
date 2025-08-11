import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module

__all__ = (
    "TMaxAvgPool2d",
    "RAPool2d",
)

class TMaxAvgPool2d(Module):
    r""" Applies TMax Average Pooling
    """
    def __init__(self, kernel_size, stride=None, padding=0, k=3, T=0.5):
        super(TMaxAvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.k = k
        self.T = T
        
    def forward(self, x):
        # Agrega padding si es necesario
        device = x.device
        batch_size, channels, height, width = x.size()

        # Aplicar padding si es necesario
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)

        # Calcular las dimensiones de salida
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Usar unfold para extraer ventanas de cada región de pooling de manera eficiente
        x_unfolded = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        x_unfolded = x_unfolded.contiguous().view(batch_size, channels, out_height, out_width, -1)

        # Obtener los k valores máximos en cada ventana
        topK = torch.topk(x_unfolded, k=self.k, dim=-1).values  # Extrae directamente los valores top K

        # Crear máscaras para la condición Yi >= T y Yi < T
        mask = topK >= self.T

        # Calcular el max y avg condicionales
        topKMax = torch.max(topK, dim=-1).values
        topKAvg = torch.mean(topK, dim=-1)

        # Aplicar la función condicional usando la máscara booleana
        out = torch.where(mask.all(dim=-1), topKMax, topKAvg).to(device)

        return out
    
class RAPool2d(Module):
    r""" Applies Rank-based Average Pooling
    .. _link:
        https://sci-hub.se/https://doi.org/10.1016/j.neunet.2016.07.003

    """
    def __init__(self, kernel_size, stride=None, padding=0, t=None):
        super(RAPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.t = t if t is not None else kernel_size**2//2
        
    def forward(self, x):
        device = x.device
        batch_size, channels, height, width = x.size()

        # Add padding if requested
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)

        # Compute output tensor dimensions
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Extract windows from pooling regions efficiently
        x_unfolded = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        x_unfolded = x_unfolded.contiguous().view(batch_size, channels, out_height, out_width, -1)
        x_unfolded = x_unfolded.sort(dim=4)[0]
        out = x_unfolded[:,:,:,:,self.t:].mean(dim=4)

        return out