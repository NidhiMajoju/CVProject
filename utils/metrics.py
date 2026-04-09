"""
metrics.py — Shared segmentation metric utilities.

Provides a GPU-friendly confusion-matrix accumulator and derives the
standard semantic segmentation metrics from it:

    mIoU             — primary metric (mean Intersection over Union)
    pixel_accuracy   — fraction of correctly classified pixels
    mean_class_accuracy — mean per-class recall

Usage
-----
    cm = ConfusionMatrix(num_classes=19, ignore_index=255)

    for images, masks in loader:
        preds = model(images).argmax(dim=1)
        cm.update(preds, masks)

    metrics = cm.compute()
    print(metrics["mIoU"], metrics["pixel_accuracy"])

    cm.reset()   # start fresh for next epoch / condition
"""

from __future__ import annotations

from typing import Dict, List

import torch

from segmentation.datasets import BDD100K_CLASSES, NUM_CLASSES


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

class ConfusionMatrix:
    """
    Accumulates per-class prediction counts in a (num_classes × num_classes)
    confusion matrix, then derives standard segmentation metrics.

    Args:
        num_classes:   Number of foreground classes (ignoring *ignore_index*).
        ignore_index:  Label value excluded from all statistics (default 255).
        device:        Torch device for the internal matrix.  If None, the
                       device is inferred from the first call to :meth:`update`.
    """

    def __init__(
        self,
        num_classes:  int = NUM_CLASSES,
        ignore_index: int = 255,
        device: torch.device | None = None,
    ) -> None:
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self._device      = device
        self._matrix      = self._zeros()

    # ------------------------------------------------------------------
    def _zeros(self) -> torch.Tensor:
        dev = self._device or torch.device("cpu")
        return torch.zeros(self.num_classes, self.num_classes,
                           dtype=torch.int64, device=dev)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear the accumulated confusion matrix."""
        self._matrix = self._zeros()

    # ------------------------------------------------------------------
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Accumulate predictions into the confusion matrix.

        Args:
            preds:    (B, H, W) or (H, W)  integer class predictions.
            targets:  (B, H, W) or (H, W)  ground-truth labels.
        """
        # Lazy device assignment
        if self._device is None:
            self._device = preds.device
            self._matrix = self._zeros()

        preds   = preds.flatten().long()
        targets = targets.flatten().long()

        # Mask out ignore pixels
        valid   = targets != self.ignore_index
        preds   = preds[valid]
        targets = targets[valid]

        # Clamp out-of-range predictions (e.g. from raw network output)
        preds   = preds.clamp(0, self.num_classes - 1)
        targets = targets.clamp(0, self.num_classes - 1)

        # Linearise indices and accumulate
        indices = targets * self.num_classes + preds
        counts  = torch.bincount(indices, minlength=self.num_classes ** 2)
        self._matrix += counts.reshape(self.num_classes, self.num_classes) \
                               .to(self._matrix.device)

    # ------------------------------------------------------------------
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics from the accumulated confusion matrix.

        Returns a flat dict with keys:
            mIoU, pixel_accuracy, mean_class_accuracy,
            iou_<class_name>  for each class.
        """
        mat = self._matrix.float()

        # True positives: diagonal
        tp = torch.diag(mat)

        # Per-class IoU = TP / (TP + FP + FN)
        fp   = mat.sum(dim=0) - tp    # column sum − TP
        fn   = mat.sum(dim=1) - tp    # row sum    − TP
        denom = tp + fp + fn
        iou_per_class = torch.where(
            denom > 0,
            tp / denom,
            torch.zeros_like(tp),
        )

        # Mean IoU over classes that appear at least once in GT
        valid_classes  = denom > 0
        if valid_classes.sum() == 0:
            miou = torch.tensor(0.0)
        else:
            miou = iou_per_class[valid_classes].mean()

        # Pixel accuracy
        total_pixels   = mat.sum()
        correct_pixels = tp.sum()
        pixel_acc = (correct_pixels / total_pixels) if total_pixels > 0 else torch.tensor(0.0)

        # Mean class accuracy (recall per class)
        row_sums = mat.sum(dim=1)
        cls_acc  = torch.where(row_sums > 0, tp / row_sums, torch.zeros_like(tp))
        mean_cls_acc = cls_acc[valid_classes].mean() if valid_classes.sum() > 0 else torch.tensor(0.0)

        # Assemble dict
        metrics: Dict[str, float] = {
            "mIoU":                 miou.item(),
            "pixel_accuracy":       pixel_acc.item(),
            "mean_class_accuracy":  mean_cls_acc.item(),
        }
        for i, cls_name in enumerate(BDD100K_CLASSES):
            key = f"iou_{cls_name.replace(' ', '_')}"
            metrics[key] = iou_per_class[i].item()

        return metrics

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"ConfusionMatrix(num_classes={self.num_classes}, "
            f"ignore_index={self.ignore_index}, "
            f"total_pixels={self._matrix.sum().item():,})"
        )