from typing import Optional, Sequence, Type
from torch import nn


class MLP(nn.Sequential):
    def __init__(
        self,
        in_size: int,
        hidden_sizes: Sequence[int],
        out_size: Optional[int] = None,
        norm: Type[nn.Module] = nn.LayerNorm,
        activation: Type[nn.Module] = nn.SiLU,
    ):
        layers = []
        for size in hidden_sizes:
            layers.append(nn.Linear(in_size, size))
            layers.append(norm(size))
            layers.append(activation())
            in_size = size
        if out_size is not None:
            layers.append(nn.Linear(in_size, out_size))
        super().__init__(*layers)
