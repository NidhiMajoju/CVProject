"""
datasets.py — BDD100K segmentation dataset and transforms.

Supports two training modes:

    baseline  : daytime images only
    augmented : daytime + synthetic adverse-weather images (night / rain)

BDD100K uses the Cityscapes 19-class label mapping. Masks are stored as
*_train_id.png files where each pixel value is a class index in [0, 18],
with 255 reserved for ignore / unlabelled regions.

Directory layout expected:

    data/bdd100k/
    ├── images/{train,val}/{day,night,rain}/    ← RGB images (.jpg)
    └── seg_maps/{train,val}/{day,night,rain}/  ← masks (*_train_id.png)

    data/synthetic/
    ├── night/   ← CycleGAN-generated night images (same filenames as day)
    └── rain/    ← CycleGAN-generated rain images
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# BDD100K label constants
# ---------------------------------------------------------------------------

NUM_CLASSES = 19   # Cityscapes train-id classes
IGNORE_INDEX = 255  # pixels to exclude from loss / metrics

BDD100K_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class SegmentationTransform:
    """
    Joint image + mask transform for semantic segmentation.

    Applies identical spatial augmentations to both the RGB image and its
    segmentation mask, and normalises the image to ImageNet statistics.

    Args:
        image_size:  Final crop / resize size (height == width).
        augment:     If True, apply random crop, flip, and colour jitter.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    def __init__(self, image_size: int = 512, augment: bool = True) -> None:
        self.image_size = image_size
        self.augment = augment
        self._to_tensor = T.ToTensor()
        self._normalise  = T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        self._colour_jitter = T.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
        )

    # ------------------------------------------------------------------
    def __call__(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # 1. Resize both to a fixed base size before cropping
        resize_size = int(self.image_size * 1.2) if self.augment else self.image_size
        image = TF.resize(image, (resize_size, resize_size), interpolation=TF.InterpolationMode.BILINEAR)
        mask  = TF.resize(mask,  (resize_size, resize_size), interpolation=TF.InterpolationMode.NEAREST)

        if self.augment:
            # 2. Random crop
            i, j, h, w = T.RandomCrop.get_params(image, (self.image_size, self.image_size))
            image = TF.crop(image, i, j, h, w)
            mask  = TF.crop(mask,  i, j, h, w)

            # 3. Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)

            # 4. Colour jitter (image only)
            image = self._colour_jitter(image)

        # 5. Image → normalised float tensor
        img_t = self._normalise(self._to_tensor(image))

        # 6. Mask → long tensor (class indices); keep 255 as ignore
        mask_np = np.array(mask, dtype=np.int64)
        mask_t  = torch.from_numpy(mask_np)

        return img_t, mask_t


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

_IMG_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".bmp"}
_MASK_SUFFIX     = "_train_id.png"


def _collect_image_mask_pairs(
    images_dir: Path, masks_dir: Path
) -> List[Tuple[Path, Path]]:
    """
    Collect (image_path, mask_path) pairs by matching filenames.

    Image stem: e.g. ``0000f77c-6257be58``
    Mask  stem: e.g. ``0000f77c-6257be58_train_id``
    """
    pairs: List[Tuple[Path, Path]] = []
    for img_path in sorted(images_dir.rglob("*")):
        if img_path.suffix.lower() not in _IMG_EXTENSIONS:
            continue
        mask_path = masks_dir / (img_path.stem + _MASK_SUFFIX)
        if mask_path.exists():
            pairs.append((img_path, mask_path))
        # If no mask exists we silently skip (can happen for synthetic images
        # if the mask directory wasn't set up; callers should ensure alignment).
    return pairs


def _collect_synthetic_pairs(
    synthetic_images_dir: Path, real_masks_dir: Path
) -> List[Tuple[Path, Path]]:
    """
    Match synthetic (CycleGAN-translated) images to their original masks.

    Synthetic images share the same stem as the original daytime images, so
    masks are looked up from the real day mask directory.
    """
    pairs: List[Tuple[Path, Path]] = []
    for img_path in sorted(synthetic_images_dir.rglob("*")):
        if img_path.suffix.lower() not in _IMG_EXTENSIONS:
            continue
        mask_path = real_masks_dir / (img_path.stem + _MASK_SUFFIX)
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    return pairs


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BDD100KSegDataset(Dataset):
    """
    BDD100K semantic segmentation dataset.

    Args:
        image_mask_pairs:  List of (image_path, mask_path) tuples.
        transform:         Joint image+mask callable (see SegmentationTransform).
    """

    def __init__(
        self,
        image_mask_pairs: List[Tuple[Path, Path]],
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        if not image_mask_pairs:
            raise ValueError("No (image, mask) pairs provided.")
        self.pairs = image_mask_pairs
        self.transform = transform

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path)   # single-channel, uint8

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            image = T.ToTensor()(image)
            mask  = torch.from_numpy(np.array(mask, dtype=np.int64))

        return image, mask


# ---------------------------------------------------------------------------
# High-level dataset builders
# ---------------------------------------------------------------------------

def build_baseline_dataset(
    bdd_root: str | Path,
    split: str,
    image_size: int = 512,
    augment: bool = True,
) -> "BDD100KSegDataset":
    root = Path(bdd_root)
    condition = "day" if split == "train" else "all"   # <-- KEY FIX
    images_dir = root / "images"   / split / condition
    masks_dir  = root / "seg_maps" / split / condition
    pairs = _collect_image_mask_pairs(images_dir, masks_dir)
    if not pairs:
        raise RuntimeError(f"No image-mask pairs found under {images_dir}")
    transform = SegmentationTransform(image_size=image_size, augment=augment)
    return BDD100KSegDataset(pairs, transform=transform)

def build_augmented_dataset(
    bdd_root: str | Path,
    synthetic_root: str | Path,
    split: str,
    image_size: int = 512,
    augment: bool = True,
    synthetic_modes: Tuple[str, ...] = ("night",),
) -> Dataset:
    """
    Daytime + synthetic adverse-weather dataset for the augmented model.

    Synthetic images are combined with their original daytime masks (valid
    because CycleGAN preserves scene geometry).

    Args:
        bdd_root:         Path to ``data/bdd100k/``.
        synthetic_root:   Path to ``data/synthetic/``.
        split:            ``"train"`` or ``"val"``.
        image_size:       Spatial size fed to the model.
        augment:          Enable data augmentation.
        synthetic_modes:  Tuple of synthetic domains to include, e.g.
                          ``("night",)`` or ``("night", "rain")``.
    """
    root      = Path(bdd_root)
    syn_root  = Path(synthetic_root)
    transform = SegmentationTransform(image_size=image_size, augment=augment)

    # Real daytime data
    day_images = root / "images"   / split / "day"
    day_masks  = root / "seg_maps" / split / "day"
    day_pairs  = _collect_image_mask_pairs(day_images, day_masks)
    if not day_pairs:
        raise RuntimeError(f"No daytime pairs found under {day_images}")

    datasets: List[Dataset] = [BDD100KSegDataset(day_pairs, transform=transform)]

    # Synthetic data for each requested mode
    for mode in synthetic_modes:
        syn_images = syn_root / mode
        if not syn_images.is_dir():
            raise FileNotFoundError(
                f"Synthetic image directory not found: {syn_images}. "
                "Run cyclegan/generate_synthetic.py first."
            )
        syn_pairs = _collect_synthetic_pairs(syn_images, day_masks)
        if syn_pairs:
            datasets.append(BDD100KSegDataset(syn_pairs, transform=transform))
            print(f"  Synthetic {mode}: {len(syn_pairs)} pairs")
        else:
            print(f"  WARNING: No synthetic {mode} pairs matched to day masks.")

    combined = ConcatDataset(datasets)
    return combined


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_seg_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = torch.cuda.is_available(),
) -> DataLoader:
    """Wrap a segmentation dataset in a DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=shuffle,   # drop last incomplete batch only during training
    )