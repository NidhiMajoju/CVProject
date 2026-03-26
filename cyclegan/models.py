"""
models.py — CycleGAN Generator (ResNet-based) and PatchGAN Discriminator.

Architecture follows the original paper:
    Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent
    Adversarial Networks", ICCV 2017.

Generator
---------
  Encoder  → 3 down-sampling conv blocks
  Bottleneck → N residual blocks (9 for 256 px, 6 for 128 px)
  Decoder  → 3 up-sampling conv blocks
  Output   → Tanh activation  (images in [-1, 1])

Discriminator
-------------
  70 × 70 PatchGAN — classifies overlapping patches rather than the full image,
  which produces sharper high-frequency detail.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Utility: normalisation layer factory
# ---------------------------------------------------------------------------

def _norm_layer(norm_type: str = "instance") -> type[nn.Module]:
    """Return an (affine=False) InstanceNorm2d or BatchNorm2d class."""
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True)
    raise ValueError(f"Unsupported norm_type: {norm_type!r}")


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    conv → IN → ReLU → conv → IN  (+  skip connection)

    Reflection padding avoids border artifacts.
    """

    def __init__(self, channels: int, norm_layer: type[nn.Module]) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            norm_layer(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            norm_layer(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """
    ResNet-based generator.

    Args:
        in_channels:    Number of input image channels (3 for RGB).
        out_channels:   Number of output image channels (3 for RGB).
        ngf:            Base number of filters (64 by default).
        n_residual:     Number of residual blocks (9 for ≥256 px, 6 for 128 px).
        norm_type:      ``"instance"`` (default) or ``"batch"``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        ngf: int = 64,
        n_residual: int = 9,
        norm_type: str = "instance",
    ) -> None:
        super().__init__()
        norm = _norm_layer(norm_type)

        # ── Encoder ──────────────────────────────────────────────────────────
        encoder: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, bias=False),
            norm(ngf),
            nn.ReLU(inplace=True),
        ]
        # Two down-sampling blocks: H,W → H/2,W/2 → H/4,W/4
        for mult in [1, 2]:
            encoder += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                norm(ngf * mult * 2),
                nn.ReLU(inplace=True),
            ]

        # ── Bottleneck ────────────────────────────────────────────────────────
        bottleneck_channels = ngf * 4  # 256 with ngf=64
        residuals = [ResidualBlock(bottleneck_channels, norm) for _ in range(n_residual)]

        # ── Decoder ───────────────────────────────────────────────────────────
        decoder: list[nn.Module] = []
        for mult in [2, 1]:
            decoder += [
                nn.ConvTranspose2d(
                    ngf * mult * 2,
                    ngf * mult,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                ),
                norm(ngf * mult),
                nn.ReLU(inplace=True),
            ]
        decoder += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*encoder, *residuals, *decoder)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# PatchGAN Discriminator
# ---------------------------------------------------------------------------

class PatchDiscriminator(nn.Module):
    """
    70 × 70 PatchGAN discriminator.

    Produces a spatial grid of real/fake predictions rather than a single
    scalar, encouraging local texture realism.

    Args:
        in_channels:  Number of input channels (3 for RGB).
        ndf:          Base number of discriminator filters (64).
        n_layers:     Depth of the network (3 → 70 × 70 receptive field).
        norm_type:    ``"instance"`` (default) or ``"batch"``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        norm_type: str = "instance",
    ) -> None:
        super().__init__()
        norm = _norm_layer(norm_type)

        # First layer — no normalisation
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        mult = 1
        for n in range(1, n_layers):
            prev_mult = mult
            mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * prev_mult, ndf * mult, kernel_size=4, stride=2, padding=1, bias=False),
                norm(ndf * mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # Stride-1 layer
        prev_mult = mult
        mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * prev_mult, ndf * mult, kernel_size=4, stride=1, padding=1, bias=False),
            norm(ndf * mult),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Output layer — single-channel patch map
        layers += [
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=1),
        ]

        self.model = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------

def init_weights(net: nn.Module, init_type: str = "normal", gain: float = 0.02) -> None:
    """
    Initialise network weights in-place.

    Args:
        net:        The network to initialise.
        init_type:  ``"normal"`` | ``"xavier"`` | ``"kaiming"`` | ``"orthogonal"``.
        gain:       Standard deviation / gain for the chosen initialiser.
    """

    def _init_func(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if hasattr(m, "weight") and ("Conv" in classname or "Linear" in classname):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise ValueError(f"Unknown init_type: {init_type!r}")
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif "BatchNorm2d" in classname:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(_init_func)


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(n_residual=9).to(device)
    D = PatchDiscriminator().to(device)
    init_weights(G)
    init_weights(D)

    dummy = torch.randn(1, 3, 256, 256, device=device)
    fake = G(dummy)
    patch = D(fake)

    print(f"Generator   input:  {dummy.shape}  →  output: {fake.shape}")
    print(f"Discriminator input: {fake.shape}  →  output: {patch.shape}")
    assert fake.shape == dummy.shape, "Generator must preserve spatial dimensions"