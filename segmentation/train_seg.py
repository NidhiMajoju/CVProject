"""
train_seg.py — Segmentation model training loop.

Usage
-----
    # Train baseline (daytime only)
    python segmentation/train_seg.py --run baseline

    # Train augmented (daytime + synthetic night)
    python segmentation/train_seg.py --run augmented

    # Resume a previous run
    python segmentation/train_seg.py --run augmented \\
        --resume checkpoints/segmentation/augmented_epoch_30.pth

    # Train with both synthetic night + rain
    python segmentation/train_seg.py --run augmented --synthetic_modes night rain

Two checkpoints are maintained throughout training:

    checkpoints/segmentation/<run>_best.pth     ← best val mIoU so far
    checkpoints/segmentation/<run>_epoch_N.pth  ← periodic snapshot (every --save_every epochs)

TensorBoard logs:  outputs/logs/segmentation/<run>/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segmentation.datasets import (
    build_baseline_dataset,
    build_augmented_dataset,
    make_seg_dataloader,
)
from segmentation.models import SegmentationModel, SegmentationLoss, NUM_CLASSES, IGNORE_INDEX
from utils.metrics import ConfusionMatrix


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train semantic segmentation model on BDD100K")

    # Run type
    p.add_argument("--run", required=True, choices=["baseline", "augmented"],
                   help="'baseline' = daytime only; 'augmented' = daytime + synthetic")

    # Data paths
    p.add_argument("--bdd_root",       default="data/bdd100k",
                   help="Root of BDD100K dataset (contains images/ and seg_maps/)")
    p.add_argument("--synthetic_root", default="data/synthetic",
                   help="Root of synthetic images (used only for 'augmented' run)")
    p.add_argument("--synthetic_modes", nargs="+", default=["night"],
                   choices=["night", "rain"],
                   help="Which synthetic domains to include in augmented training")

    # Model
    p.add_argument("--pretrained",      action="store_true", default=True,
                   help="Load ImageNet weights for ResNet-50 encoder")
    p.add_argument("--freeze_encoder",  action="store_true",
                   help="Freeze encoder weights (fast fine-tuning)")

    # Training
    p.add_argument("--epochs",          type=int,   default=80)
    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--image_size",      type=int,   default=512)
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--label_smoothing", type=float, default=0.1)

    # LR schedule
    p.add_argument("--lr_schedule",     default="cosine",
                   choices=["cosine", "poly", "step", "none"],
                   help="Learning-rate schedule")
    p.add_argument("--warmup_epochs",   type=int,   default=5,
                   help="Epochs of linear LR warm-up (0 to disable)")

    # I/O
    p.add_argument("--checkpoint_dir", default="checkpoints/segmentation")
    p.add_argument("--log_dir",        default="outputs/logs/segmentation")
    p.add_argument("--save_every",     type=int,   default=10,
                   help="Save a periodic checkpoint every N epochs")
    p.add_argument("--resume",         default=None,
                   help="Path to checkpoint .pth to resume from")
    p.add_argument("--val_every",      type=int,   default=1,
                   help="Run validation every N epochs")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path: Path, epoch: int, model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler,
                    best_miou: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict() if scheduler else None,
        "best_miou":  best_miou,
    }, path)
    print(f"  ✔ Checkpoint saved → {path}")


def load_checkpoint(path: str | Path, model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler,
                    device: torch.device) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    best_miou = ckpt.get("best_miou", 0.0)
    start_epoch = ckpt["epoch"] + 1
    print(f"  ✔ Resumed from {path}  (next epoch: {start_epoch}, best mIoU: {best_miou:.4f})")
    return start_epoch, best_miou


# ---------------------------------------------------------------------------
# LR Scheduling
# ---------------------------------------------------------------------------

def build_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    """Build a learning-rate scheduler with optional linear warm-up."""
    total_steps   = args.epochs
    warmup_epochs = args.warmup_epochs

    if args.lr_schedule == "cosine":
        base_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_epochs), eta_min=1e-6
        )
    elif args.lr_schedule == "poly":
        power = 0.9
        base_sched = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda ep: (1 - max(0, ep - warmup_epochs) / max(1, total_steps - warmup_epochs)) ** power,
        )
    elif args.lr_schedule == "step":
        base_sched = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, (total_steps - warmup_epochs) // 3), gamma=0.5
        )
    else:
        return None

    if warmup_epochs > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, base_sched],
            milestones=[warmup_epochs],
        )
    return base_sched


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def _fmt(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s   = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else (f"{m}m {s:02d}s" if m else f"{s}s")


# ---------------------------------------------------------------------------
# One epoch: training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    epoch:     int,
    writer:    SummaryWriter,
    global_step: int,
) -> tuple[float, int]:
    """Run one training epoch. Returns (mean_loss, updated_global_step)."""
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    for images, masks in tqdm(loader, desc=f"  Train E{epoch}", leave=False, unit="batch"):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)                            # (B, C, H, W)
        loss   = criterion(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss  += loss.item()
        global_step += 1

        if global_step % 50 == 0:
            writer.add_scalar("Loss/train", loss.item(), global_step)

    return total_loss / n_batches, global_step


# ---------------------------------------------------------------------------
# One epoch: validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    num_classes: int = NUM_CLASSES,
) -> Dict[str, float]:
    """Evaluate on the validation set. Returns dict of metrics."""
    model.eval()
    cm         = ConfusionMatrix(num_classes=num_classes, ignore_index=IGNORE_INDEX)
    total_loss = 0.0

    for images, masks in tqdm(loader, desc="  Val  ", leave=False, unit="batch"):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, masks)
        total_loss += loss.item()

        preds = logits.argmax(dim=1)    # (B, H, W)
        cm.update(preds, masks)

    metrics = cm.compute()
    metrics["val_loss"] = total_loss / len(loader)
    return metrics


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        ("mps"  if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(f"Run:    {args.run}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    print("\nBuilding datasets…")
    if args.run == "baseline":
        train_ds = build_baseline_dataset(
            args.bdd_root, split="train",
            image_size=args.image_size, augment=True,
        )
    else:
        train_ds = build_augmented_dataset(
            args.bdd_root, args.synthetic_root, split="train",
            image_size=args.image_size, augment=True,
            synthetic_modes=tuple(args.synthetic_modes),
        )

    val_ds = build_baseline_dataset(
        args.bdd_root, split="val",
        image_size=args.image_size, augment=False,
    )

    train_loader = make_seg_dataloader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
    )
    val_loader = make_seg_dataloader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )

    print(f"  Train samples: {len(train_ds):,}  ({len(train_loader)} batches)")
    print(f"  Val   samples: {len(val_ds):,}  ({len(val_loader)} batches)")

    # ── Model ────────────────────────────────────────────────────────────────
    model = SegmentationModel(
        num_classes=NUM_CLASSES,
        pretrained=args.pretrained,
        freeze_encoder=args.freeze_encoder,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ── Loss, optimiser, scheduler ───────────────────────────────────────────
    criterion = SegmentationLoss(label_smoothing=args.label_smoothing).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = build_scheduler(optimizer, args)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 1
    best_miou   = 0.0
    if args.resume:
        start_epoch, best_miou = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )

    # ── TensorBoard ──────────────────────────────────────────────────────────
    log_dir = Path(args.log_dir) / args.run
    writer  = SummaryWriter(log_dir=str(log_dir))

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Training {args.run}: epochs {start_epoch}–{args.epochs}")
    print(f"{'='*65}\n")

    global_step   = (start_epoch - 1) * len(train_loader)
    wall_start    = time.time()
    epoch_times: list[float] = []

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # ── Train ────────────────────────────────────────────────────────────
        mean_train_loss, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, writer, global_step,
        )
        if scheduler:
            scheduler.step()

        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        recent_avg = sum(epoch_times[-5:]) / len(epoch_times[-5:])
        epochs_left = args.epochs - epoch
        eta_str = _fmt(recent_avg * epochs_left) if epochs_left > 0 else "—"

        lr_now = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Loss/train_epoch", mean_train_loss, epoch)
        writer.add_scalar("LR",               lr_now,          epoch)

        # ── Validate ─────────────────────────────────────────────────────────
        metrics: Dict[str, float] = {}
        if epoch % args.val_every == 0:
            metrics = validate(model, val_loader, criterion, device)
            for k, v in metrics.items():
                writer.add_scalar(f"Val/{k}", v, epoch)

            miou       = metrics.get("mIoU", 0.0)
            pix_acc    = metrics.get("pixel_accuracy", 0.0)
            val_loss   = metrics.get("val_loss", 0.0)

            # Save best model
            if miou > best_miou:
                best_miou = miou
                save_checkpoint(
                    ckpt_dir / f"{args.run}_best.pth",
                    epoch, model, optimizer, scheduler, best_miou,
                )
                print(f"  ★ New best mIoU: {best_miou:.4f}")
        else:
            miou = pix_acc = val_loss = float("nan")

        # ── Progress line ─────────────────────────────────────────────────────
        print(
            f"Epoch [{epoch:>3}/{args.epochs}]  "
            f"time={_fmt(elapsed)}  ETA={eta_str}  lr={lr_now:.2e}  "
            f"train_loss={mean_train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  mIoU={miou:.4f}  pix_acc={pix_acc:.4f}"
        )

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if epoch % args.save_every == 0:
            save_checkpoint(
                ckpt_dir / f"{args.run}_epoch_{epoch:03d}.pth",
                epoch, model, optimizer, scheduler, best_miou,
            )

    # ── Save final checkpoint ────────────────────────────────────────────────
    save_checkpoint(
        ckpt_dir / f"{args.run}_final.pth",
        args.epochs, model, optimizer, scheduler, best_miou,
    )

    total_time = _fmt(time.time() - wall_start)
    print(f"\n{'='*65}")
    print(f"  Training complete.  Best mIoU: {best_miou:.4f}  |  Total time: {total_time}")
    print(f"{'='*65}")
    writer.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)