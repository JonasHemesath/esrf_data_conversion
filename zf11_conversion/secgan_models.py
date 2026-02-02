# secgan_models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3DValid(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=0, bias=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=0, bias=True)

    @staticmethod
    def _center_crop(x: torch.Tensor, target_shape: Tuple[int, int, int]) -> torch.Tensor:
        _, _, d, h, w = x.shape
        td, th, tw = target_shape
        sd = (d - td) // 2
        sh = (h - th) // 2
        sw = (w - tw) // 2
        return x[:, :, sd:sd + td, sh:sh + th, sw:sw + tw]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.conv1(x), inplace=True)
        y = self.conv2(y)
        skip = self._center_crop(x, y.shape[-3:])
        return skip + y


class Generator3D(nn.Module):
    """
    VALID conv generator (paper-like). Output is tanh in [-1,1].
    """
    def __init__(self, in_ch: int = 1, base_ch: int = 32, num_blocks: int = 8):
        super().__init__()
        self.in_conv = nn.Conv3d(in_ch, base_ch, kernel_size=3, padding=0, bias=True)
        self.blocks = nn.ModuleList([ResBlock3DValid(base_ch) for _ in range(num_blocks)])
        self.out_conv = nn.Conv3d(base_ch, 1, kernel_size=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.in_conv(x), inplace=True)
        for b in self.blocks:
            y = b(y)
        return torch.tanh(self.out_conv(y))


class PatchDiscriminator3D(nn.Module):
    def __init__(self, in_ch: int = 1, base_ch: int = 32, n_layers: int = 4):
        super().__init__()
        layers = []
        ch = base_ch
        layers += [nn.Conv3d(in_ch, ch, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
        for _ in range(n_layers - 1):
            ch2 = min(ch * 2, 256)
            layers += [
                nn.Conv3d(ch, ch2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm3d(ch2, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = ch2
        layers += [nn.Conv3d(ch, 1, kernel_size=3, stride=1, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class LossWeights:
    lambda_cyc: float = 2.5
    lambda_seg: float = 1.0
    lambda_adv_x: float = 1.0
    lambda_adv_y: float = 1.0


def to_gan(x01: torch.Tensor) -> torch.Tensor:
    # [0,1] -> [-1,1]
    return x01 * 2.0 - 1.0


def from_gan(x: torch.Tensor) -> torch.Tensor:
    # [-1,1] -> [0,1]
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def lsgan_mse(pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return F.mse_loss(pred, target)


def cycle_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(a, b)