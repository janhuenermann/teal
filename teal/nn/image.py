import math
from typing import Type
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels: int, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 1),
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        ).permute(0, 3, 1, 2)


def conv2d_norm_act(
    in_channels: int,
    out_channels: int,
    kernel: int,
    stride=1,
    transpose=False,
    preact=False,
    norm: Type[nn.Module] = LayerNorm2d,
    activation: Type[nn.Module] = nn.SiLU,
):
    padding = (kernel - 1) // 2
    conv = (
        nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding)
        if transpose
        else nn.Conv2d(in_channels, out_channels, kernel, stride, padding)
    )

    if preact:
        return nn.Sequential(norm(in_channels), activation(), conv)

    return nn.Sequential(conv, norm(out_channels), activation())


class ImageEncoderResnet(nn.Module):
    def __init__(self, in_channels: int, in_width: int, depth_scale=32, min_width=4):
        super().__init__()
        stages = int(math.log2(in_width) - math.log2(min_width))
        layers = []
        in_depth = in_channels
        depth = depth_scale
        for _ in range(stages):
            layers.append(conv2d_norm_act(in_depth, depth, 4, 2))
            in_depth = depth
            depth *= 2
        layers.append(nn.Flatten(-3, -1))
        self.layers = nn.Sequential(*layers)
        self.out_features = depth // 2 * min_width * min_width

    def forward(self, x: Tensor) -> Tensor:
        y = self.layers(x.flatten(0, -4))
        y = y.reshape(x.shape[:-3] + y.shape[-1:])
        return y


class ImageDecoderResnet(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_width: int,
        out_channels: int,
        depth_scale=32,
        min_width=4,
    ):
        super().__init__()
        stages = int(math.log2(out_width) - math.log2(min_width))
        depth = depth_scale * 2 ** (stages - 1)
        layers = [
            nn.Linear(in_features, min_width * min_width * depth),
            nn.Unflatten(-1, (depth, min_width, min_width)),
        ]
        for _ in range(stages - 1):
            out_depth = depth // 2
            layers.append(conv2d_norm_act(depth, out_depth, 4, 2, transpose=True))
            depth = out_depth

        layers.append(
            conv2d_norm_act(
                depth,
                out_channels,
                4,
                2,
                transpose=True,
                activation=nn.Identity,
                norm=nn.Identity,
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x.flatten(0, -2))
        y = y.reshape(x.shape[:-1] + y.shape[-3:])
        return y + 0.5
