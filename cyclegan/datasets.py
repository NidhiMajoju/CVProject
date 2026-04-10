"""
datasets.py — Unpaired image dataset for CycleGAN training.

Loads images from two independent directories (domain A and domain B)
with no pairing requirement. Each epoch randomly samples from both.
"""

import os
import random
from pathlib import Path
from typing import Callable, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Default transforms
# ---------------------------------------------------------------------------

def build_transforms(
    image_size: int = 512,
    augment: bool = True,
) -> Callable:
    """Standard CycleGAN pre-processing pipeline."""
    ops = []
    if augment:
        ops += [
            T.Resize(int(image_size * 1.12), T.InterpolationMode.BICUBIC),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(),
        ]
    else:
        ops += [
            T.Resize((image_size, image_size), T.InterpolationMode.BICUBIC),
        ]
    ops += [
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # → [-1, 1]
    ]
    return T.Compose(ops)


# ---------------------------------------------------------------------------
# Image-buffer helper (replay buffer for discriminator stability)
# ---------------------------------------------------------------------------

class ImageBuffer:
    """
    Replay buffer that returns a mix of current and historical fake images
    to stabilise discriminator training (Shrivastava et al., 2017).

    Args:
        max_size: Maximum number of images stored. 0 disables buffering.
    """

    def __init__(self, max_size: int = 50) -> None:
        self.max_size = max_size
        self.data: list[torch.Tensor] = []

    def push_and_pop(self, images: torch.Tensor) -> torch.Tensor:
        if self.max_size == 0:
            return images

        result = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(img)
                result.append(img)
            else:
                if random.random() < 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    stored = self.data[idx].clone()
                    self.data[idx] = img
                    result.append(stored)
                else:
                    result.append(img)
        return torch.cat(result, dim=0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _collect_images(directory: str | Path) -> list[Path]:
    """Recursively collect all image paths under *directory*."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    paths = sorted(
        p for p in root.rglob("*") if p.suffix.lower() in _IMG_EXTENSIONS
    )
    if not paths:
        raise RuntimeError(f"No images found in {root}")
    return paths


class UnpairedImageDataset(Dataset):
    """
    Unpaired (unaligned) dataset for CycleGAN.

    Args:
        root_A:     Path to domain-A images.
        root_B:     Path to domain-B images.
        transform:  Callable applied to each PIL image.
        unaligned:  If True, domain-B index is randomly shuffled.
        max_size:   If set, cap both domains to this many images.
    """

    def __init__(
        self,
        root_A: str | Path,
        root_B: str | Path,
        transform: Optional[Callable] = None,
        unaligned: bool = True,
        max_size: int | None = None,
    ) -> None:
        super().__init__()
        self.files_A = _collect_images(root_A)
        self.files_B = _collect_images(root_B)
        if max_size is not None:
            self.files_A = self.files_A[:max_size]
            self.files_B = self.files_B[:max_size]
        self.transform = transform if transform is not None else build_transforms()
        self.unaligned = unaligned

    def __len__(self) -> int:
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path_A = self.files_A[index % len(self.files_A)]

        if self.unaligned:
            path_B = self.files_B[random.randint(0, len(self.files_B) - 1)]
        else:
            path_B = self.files_B[index % len(self.files_B)]

        img_A = Image.open(path_A).convert("RGB")
        img_B = Image.open(path_B).convert("RGB")

        return self.transform(img_A), self.transform(img_B)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_dataloader(
    root_A: str | Path,
    root_B: str | Path,
    image_size: int = 256,
    batch_size: int = 1,
    num_workers: int = 4,
    augment: bool = True,
    unaligned: bool = True,
    max_size: int | None = None,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """Build a ready-to-use DataLoader for CycleGAN training."""
    transform = build_transforms(image_size=image_size, augment=augment)
    dataset = UnpairedImageDataset(
        root_A=root_A,
        root_B=root_B,
        transform=transform,
        unaligned=unaligned,
        max_size=max_size,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )