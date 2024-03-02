import copy
from typing import Optional
from torch import nn
import torch


class EMA(nn.Module):
    def __init__(self, module: nn.Module, decay: float = 0.99):
        super().__init__()
        self.ema_module = copy.deepcopy(module)
        self.ema_module.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, module: nn.Module, decay: Optional[float] = None):
        decay = self.decay if decay is None else decay
        for ema_param, param in zip(self.ema_module.parameters(), module.parameters()):
            ema_param.data.lerp_(param.data, 1.0 - decay)

    def forward(self, *args, **kwargs):
        return self.ema_module(*args, **kwargs)
