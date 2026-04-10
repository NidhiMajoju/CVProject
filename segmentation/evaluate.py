"""
evaluate.py — Evaluate and compare segmentation models on adverse-weather images.

Loads the baseline and augmented checkpoints and evaluates both on real
night and rain test images, then prints a side-by-side per-class IoU table
and saves comparison plots.

Usage
-----
    python segmentation/evaluate.py

    # Evaluate only one model on a specific condition
    python segmentation/evaluate.py \\
        --checkpoint_baseline checkpoints/segmentation/baseline_best.pth \\
        --conditions night

    # Save qualitative prediction images
    python segmentation/evaluate.py --save_predictions

Output
------
    Printed: per-class IoU table, mIoU and pixel-accuracy summary
    Saved:   outputs/plots/evaluation/miou_comparison.png
             outputs/plots/evaluation/per_class_iou.png
             outputs/plots/evaluation/predictions/<condition>/<filename>.png  (if --save_predictions)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segmentation.datasets import (
    BDD100KSegDataset,
    SegmentationTransform,
    _collect_image_mask_pairs,
    BDD100K_CLASSES,
    NUM_CLASSES,
    IGNORE_INDEX,
)
from segmentation.models import SegmentationModel
from utils.metrics import ConfusionMatrix


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate segmentation models on adverse-weather data")

    p.add_argument("--checkpoint_baseline",
                   default="checkpoints/segmentation/baseline_best.pth",
                   help="Path to baseline model checkpoint")
    p.add_argument("--checkpoint_augmented",
                   default="checkpoints/segmentation/augmented_best.pth",
                   help="Path to augmented model checkpoint")
    p.add_argument("--bdd_root", default="data/bdd100k",
                   help="Root of BDD100K dataset")
    p.add_argument("--split",    default="val", choices=["train", "val"],
                   help="Which split to evaluate on")
    p.add_argument("--conditions", nargs="+",
                   default=["all"],
                   choices=["day", "night", "rain", "all"],
                   help="Conditions to evaluate")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", default="outputs/plots/evaluation")
    p.add_argument("--save_predictions", action="store_true",
                   help="Save colourised prediction images to disk")
    p.add_argument("--max_pred_images", type=int, default=20,
                   help="Max prediction images to save per condition")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str | Path, device: torch.device) -> SegmentationModel:
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = SegmentationModel(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Per-condition evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model:      SegmentationModel,
    bdd_root:   str | Path,
    split:      str,
    condition:  str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device:     torch.device,
) -> Dict[str, float]:
    """
    Evaluate *model* on one condition (day / night / rain).

    Returns a metrics dict:
        mIoU, pixel_accuracy, mean_class_accuracy,
        iou_<classname> for all 19 classes.
    """
    root       = Path(bdd_root)
    actual_condition = condition
    if split == "val":
        actual_condition = "all"

    images_dir = root / "images"   / split / actual_condition
    masks_dir  = root / "seg_maps" / split / actual_condition

    pairs = _collect_image_mask_pairs(images_dir, masks_dir)
    if not pairs:
        print(f"  WARNING: No pairs found for condition '{condition}' in {images_dir}")
        return {}

    transform = SegmentationTransform(image_size=image_size, augment=False)
    dataset   = BDD100KSegDataset(pairs, transform=transform)
    loader    = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    cm = ConfusionMatrix(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)

    for images, masks in tqdm(loader, desc=f"    {condition}", leave=False):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)
        logits = model(images)
        preds  = logits.argmax(dim=1)
        cm.update(preds, masks)

    metrics = cm.compute()
    return metrics


# ---------------------------------------------------------------------------
# Prediction visualisation
# ---------------------------------------------------------------------------

# Cityscapes palette (19 train-id classes)
_PALETTE = np.array([
    [128,  64, 128],   # road
    [244,  35, 232],   # sidewalk
    [ 70,  70,  70],   # building
    [102, 102, 156],   # wall
    [190, 153, 153],   # fence
    [153, 153, 153],   # pole
    [250, 170,  30],   # traffic light
    [220, 220,   0],   # traffic sign
    [107, 142,  35],   # vegetation
    [152, 251, 152],   # terrain
    [ 70, 130, 180],   # sky
    [220,  20,  60],   # person
    [255,   0,   0],   # rider
    [  0,   0, 142],   # car
    [  0,   0,  70],   # truck
    [  0,  60, 100],   # bus
    [  0,  80, 100],   # train
    [  0,   0, 230],   # motorcycle
    [119,  11,  32],   # bicycle
], dtype=np.uint8)


def _colorise(pred: np.ndarray) -> np.ndarray:
    """Map (H, W) class-index array → (H, W, 3) RGB array using Cityscapes palette."""
    rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)
    for cls_id, colour in enumerate(_PALETTE):
        rgb[pred == cls_id] = colour
    return rgb


@torch.no_grad()
def save_prediction_images(
    model:       SegmentationModel,
    bdd_root:    str | Path,
    split:       str,
    condition:   str,
    image_size:  int,
    output_dir:  Path,
    max_images:  int,
    device:      torch.device,
) -> None:
    """Save side-by-side (original | colourised prediction) images."""
    root       = Path(bdd_root)
    actual_condition = condition
    if split == "val":
        actual_condition = "all"
    images_dir = root / "images"   / split / actual_condition
    masks_dir  = root / "seg_maps" / split / actual_condition
    pairs      = _collect_image_mask_pairs(images_dir, masks_dir)[:max_images]

    out_dir = output_dir / "predictions" / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = SegmentationTransform(image_size=image_size, augment=False)

    for img_path, _ in tqdm(pairs, desc=f"    Saving {condition} predictions", leave=False):
        image_pil = Image.open(img_path).convert("RGB")
        img_t, _  = transform(image_pil, Image.fromarray(np.zeros((image_size, image_size), dtype=np.uint8)))
        img_t     = img_t.unsqueeze(0).to(device)

        logits    = model(img_t)
        pred      = logits.argmax(dim=1).squeeze(0).cpu().numpy()   # (H, W)
        colour    = _colorise(pred)

        # Resize original to match prediction size for side-by-side display
        orig_resized = image_pil.resize((image_size, image_size), Image.BILINEAR)

        side_by_side = Image.new("RGB", (image_size * 2 + 4, image_size), color=(50, 50, 50))
        side_by_side.paste(orig_resized, (0, 0))
        side_by_side.paste(Image.fromarray(colour), (image_size + 4, 0))
        side_by_side.save(out_dir / f"{img_path.stem}_pred.png")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_table(results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    """
    Print a formatted per-class IoU comparison table.

    results[model_name][condition][metric] = value
    """
    model_names = list(results.keys())
    conditions  = list(next(iter(results.values())).keys())

    # ── Summary (mIoU + pixel accuracy) ─────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY  (mIoU | pixel accuracy)")
    print("=" * 80)
    header = f"{'Condition':<12}" + "".join(f"  {m:<22}" for m in model_names)
    print(header)
    print("-" * 80)
    for cond in conditions:
        row = f"{cond:<12}"
        for mn in model_names:
            m    = results[mn].get(cond, {})
            miou = m.get("mIoU", float("nan"))
            pa   = m.get("pixel_accuracy", float("nan"))
            row += f"  mIoU={miou:.3f}  PA={pa:.3f}    "
        print(row)

    # ── Per-class IoU ────────────────────────────────────────────────────────
    for cond in conditions:
        print(f"\n{'─'*80}")
        print(f"Per-class IoU  —  condition: {cond}")
        print(f"{'─'*80}")
        col_w = 12
        hdr   = f"{'Class':<18}" + "".join(f"{mn:>{col_w}}" for mn in model_names)
        if len(model_names) == 2:
            hdr += f"  {'Delta':>{col_w}}"
        print(hdr)
        print("─" * (18 + col_w * len(model_names) + (col_w + 2 if len(model_names) == 2 else 0)))

        for i, cls_name in enumerate(BDD100K_CLASSES):
            key  = f"iou_{cls_name.replace(' ', '_')}"
            vals = [results[mn].get(cond, {}).get(key, float("nan")) for mn in model_names]
            row  = f"{cls_name:<18}" + "".join(f"{v:>{col_w}.3f}" for v in vals)
            if len(model_names) == 2 and not any(np.isnan(vals)):
                delta = vals[1] - vals[0]
                sign  = "+" if delta >= 0 else ""
                row  += f"  {sign}{delta:>{col_w - 1}.3f}"
            print(row)
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save_plots(
    results:    Dict[str, Dict[str, Dict[str, float]]],
    output_dir: Path,
) -> None:
    """Save mIoU bar chart and per-class IoU grouped bar chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(results.keys())
    conditions  = list(next(iter(results.values())).keys())
    n_cond      = len(conditions)
    n_models    = len(model_names)
    x           = np.arange(n_cond)
    bar_w       = 0.35

    # ── mIoU bar chart ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, mn in enumerate(model_names):
        mious = [results[mn].get(c, {}).get("mIoU", 0.0) for c in conditions]
        offset = (i - (n_models - 1) / 2) * bar_w
        bars = ax.bar(x + offset, mious, bar_w, label=mn.capitalize())
        for bar, v in zip(bars, mious):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Condition")
    ax.set_ylabel("mIoU")
    ax.set_title("mIoU Comparison: Baseline vs Augmented")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in conditions])
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = output_dir / "miou_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✔ Saved mIoU comparison → {path}")

    # ── Per-class IoU (adverse conditions only) ───────────────────────────────
    adverse_conds = [c for c in conditions if c != "day"]
    for cond in adverse_conds:
        fig, ax = plt.subplots(figsize=(14, 5))
        x_cls = np.arange(NUM_CLASSES)
        for i, mn in enumerate(model_names):
            ious = []
            for cls_name in BDD100K_CLASSES:
                key = f"iou_{cls_name.replace(' ', '_')}"
                ious.append(results[mn].get(cond, {}).get(key, 0.0))
            offset = (i - (n_models - 1) / 2) * bar_w
            ax.bar(x_cls + offset, ious, bar_w, label=mn.capitalize())

        ax.set_xlabel("Class")
        ax.set_ylabel("IoU")
        ax.set_title(f"Per-class IoU — {cond.capitalize()}")
        ax.set_xticks(x_cls)
        ax.set_xticklabels(BDD100K_CLASSES, rotation=45, ha="right", fontsize=8)
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = output_dir / f"per_class_iou_{cond}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  ✔ Saved per-class IoU ({cond}) → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        ("mps"  if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Device:     {device}")
    print(f"Conditions: {args.conditions}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────────
    models_to_eval: Dict[str, Optional[SegmentationModel]] = {}

    for run_name, ckpt_path in [
        ("baseline",  args.checkpoint_baseline),
        ("augmented", args.checkpoint_augmented),
    ]:
        p = Path(ckpt_path)
        if p.exists():
            print(f"\nLoading {run_name} model from {p} …")
            models_to_eval[run_name] = load_model(p, device)
        else:
            print(f"  Checkpoint not found: {p}  — skipping {run_name}.")

    if not models_to_eval:
        print("ERROR: No valid checkpoints found. Train at least one model first.")
        return

    # ── Evaluate ─────────────────────────────────────────────────────────────
    # results[model_name][condition] = metrics dict
    results: Dict[str, Dict[str, Dict[str, float]]] = {mn: {} for mn in models_to_eval}

    for model_name, model in models_to_eval.items():
        print(f"\nEvaluating [{model_name}] …")
        for condition in args.conditions:
            print(f"  Condition: {condition}")
            metrics = evaluate_model(
                model       = model,
                bdd_root    = args.bdd_root,
                split       = args.split,
                condition   = condition,
                image_size  = args.image_size,
                batch_size  = args.batch_size,
                num_workers = args.num_workers,
                device      = device,
            )
            results[model_name][condition] = metrics

            if metrics:
                print(
                    f"    mIoU={metrics.get('mIoU', 0):.4f}  "
                    f"PA={metrics.get('pixel_accuracy', 0):.4f}  "
                    f"mAcc={metrics.get('mean_class_accuracy', 0):.4f}"
                )

        # Optionally save qualitative predictions for the augmented model
        if args.save_predictions:
            for condition in args.conditions:
                save_prediction_images(
                    model       = model,
                    bdd_root    = args.bdd_root,
                    split       = args.split,
                    condition   = condition,
                    image_size  = args.image_size,
                    output_dir  = output_dir,
                    max_images  = args.max_pred_images,
                    device      = device,
                )

    # ── Print comparison table ─────────────────────────────────────────────
    _print_table(results)

    # ── Save plots ────────────────────────────────────────────────────────
    _save_plots(results, output_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    main(args)