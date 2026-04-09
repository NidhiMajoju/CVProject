"""
models.py — ResNet-50 encoder U-Net for semantic segmentation.

Architecture follows the classic U-Net skip-connection principle with a
pretrained ResNet-50 backbone.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


NUM_CLASSES  = 19
IGNORE_INDEX = 255


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _conv_bn_relu(in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DecoderBlock(nn.Module):
    """Bilinear 2× upsample → concat skip → two conv-BN-ReLU layers."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.skip_proj = (
            nn.Conv2d(skip_ch, out_ch, kernel_size=1, bias=False)
            if skip_ch != out_ch else nn.Identity()
        )
        self.conv1 = _conv_bn_relu(in_ch + out_ch, out_ch)
        self.conv2 = _conv_bn_relu(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x    = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        skip = self.skip_proj(skip)
        # Align spatial dims in case of rounding differences
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class SegmentationModel(nn.Module):
    """
    ResNet-50 encoder + U-Net decoder for 19-class semantic segmentation.

    Encoder feature map strides:
        stem    → stride 4,  64 ch   (after maxpool)
        layer1  → stride 4,  256 ch
        layer2  → stride 8,  512 ch
        layer3  → stride 16, 1024 ch
        layer4  → stride 32, 2048 ch

    Decoder path:
        bridge  : 2048 → 512
        decoder4: stride 32 → 16, skip = layer3 (1024 ch)
        decoder3: stride 16 → 8,  skip = layer2 (512 ch)
        decoder2: stride 8  → 4,  skip = layer1 (256 ch)
        decoder1: stride 4  → 2,  skip = stem   (64 ch)
        final_up: stride 2  → 1   (full resolution)
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────
        weights  = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = tvm.resnet50(weights=weights)

        # Stem produces 64-ch feature maps at stride 4
        # We split it so we can grab the pre-maxpool output (stride 2) as a skip
        self.stem_conv = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # stride 2, 64ch
        self.stem_pool = backbone.maxpool                                              # stride 4

        self.encoder1 = backbone.layer1   # stride 4,  256 ch
        self.encoder2 = backbone.layer2   # stride 8,  512 ch
        self.encoder3 = backbone.layer3   # stride 16, 1024 ch
        self.encoder4 = backbone.layer4   # stride 32, 2048 ch

        if freeze_encoder:
            for p in self.parameters():
                p.requires_grad = False

        # ── Bridge ───────────────────────────────────────────────────────
        self.bridge = _conv_bn_relu(2048, 512)

        # ── Decoder ──────────────────────────────────────────────────────
        self.decoder4 = DecoderBlock(in_ch=512,  skip_ch=1024, out_ch=256)
        self.decoder3 = DecoderBlock(in_ch=256,  skip_ch=512,  out_ch=128)
        self.decoder2 = DecoderBlock(in_ch=128,  skip_ch=256,  out_ch=64)
        # Skip for decoder1 comes from stem_conv (stride 2, 64 ch)
        self.decoder1 = DecoderBlock(in_ch=64,   skip_ch=64,   out_ch=64)

        # Final upsample stride 2 → 1 (full input resolution)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _conv_bn_relu(64, 64),
        )

        # ── Head ─────────────────────────────────────────────────────────
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

        self._init_decoder_weights()

    def _init_decoder_weights(self) -> None:
        modules = [
            self.bridge, self.decoder4, self.decoder3,
            self.decoder2, self.decoder1, self.final_up, self.head,
        ]
        for m in modules:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1.0)
                    nn.init.constant_(layer.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_hw = x.shape[2:]

        # ── Encoder ──────────────────────────────────────────────────────
        s_pre = self.stem_conv(x)        # stride 2,  64 ch  ← skip for decoder1
        s0    = self.stem_pool(s_pre)    # stride 4
        s1    = self.encoder1(s0)        # stride 4,  256 ch ← skip for decoder2
        s2    = self.encoder2(s1)        # stride 8,  512 ch ← skip for decoder3
        s3    = self.encoder3(s2)        # stride 16, 1024 ch← skip for decoder4
        s4    = self.encoder4(s3)        # stride 32, 2048 ch

        # ── Bridge ───────────────────────────────────────────────────────
        x = self.bridge(s4)              # 512 ch

        # ── Decoder ──────────────────────────────────────────────────────
        x = self.decoder4(x,     s3)     # 32 → 16
        x = self.decoder3(x,     s2)     # 16 → 8
        x = self.decoder2(x,     s1)     # 8  → 4
        x = self.decoder1(x,  s_pre)     # 4  → 2  (skip = pre-pool stem, stride 2)

        # ── Final upsample + head ─────────────────────────────────────────
        x = self.final_up(x)             # 2  → 1
        x = F.interpolate(x, size=input_hw, mode="bilinear", align_corners=False)
        return self.head(x)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class SegmentationLoss(nn.Module):
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = IGNORE_INDEX,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model  = SegmentationModel(pretrained=False).to(device)
    dummy  = torch.randn(2, 3, 512, 512, device=device)
    out    = model(dummy)
    assert out.shape == (2, NUM_CLASSES, 512, 512), f"Unexpected output shape: {out.shape}"
    print(f"Input: {dummy.shape} → Output: {out.shape}  ✔")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")