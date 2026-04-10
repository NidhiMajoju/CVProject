"""
train_cyclegan.py — CycleGAN training loop.

Usage
-----
    python cyclegan/train_cyclegan.py                          # default config
    python cyclegan/train_cyclegan.py --mode rain              # day ↔ rain
    python cyclegan/train_cyclegan.py --resume checkpoints/cyclegan/epoch_40.pth

The script trains two generators and two discriminators simultaneously:

    G_A2B : domain A (day) → domain B (night / rain)
    G_B2A : domain B        → domain A
    D_A   : distinguishes real A from fake A produced by G_B2A
    D_B   : distinguishes real B from fake B produced by G_A2B

Checkpoints
-----------
Saved every ``--save_every`` epochs to ``checkpoints/cyclegan/``.
Each checkpoint contains all four model state-dicts plus optimiser states so
training can be resumed exactly.

TensorBoard
-----------
Scalar losses and sample image grids are written to ``outputs/logs/cyclegan/``.
Launch with:  tensorboard --logdir outputs/logs
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from pathlib import Path
from unittest import loader

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import tqdm

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cyclegan.datasets import UnpairedImageDataset, ImageBuffer, build_transforms, make_dataloader
from cyclegan.models import Generator, PatchDiscriminator, init_weights
from cyclegan.losses import GANLoss, cycle_loss, identity_loss


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CycleGAN for weather augmentation")

    # Data
    p.add_argument("--data_root", default="data/bdd100k/images/train",
                   help="Root directory containing domain sub-folders")
    p.add_argument("--domain_A", default="day",
                   help="Sub-folder name for domain A (source)")
    p.add_argument("--mode", default="night", choices=["night", "rain"],
                   help="Target domain: 'night' or 'rain'")
    p.add_argument("--image_size", type=int, default=256)

    # Training
    p.add_argument("--epochs", type=int, default=200,
                   help="Total training epochs")
    p.add_argument("--decay_epoch", type=int, default=100,
                   help="Epoch at which LR begins linear decay to 0")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--num_workers", type=int, default=4)

    # Loss weights
    p.add_argument("--lambda_cycle", type=float, default=10.0,
                   help="Cycle-consistency loss weight")
    p.add_argument("--lambda_identity", type=float, default=5.0,
                   help="Identity loss weight (0 to disable)")
    p.add_argument("--gan_mode", default="lsgan", choices=["lsgan", "vanilla"])

    # Model
    p.add_argument("--ngf", type=int, default=64, help="Generator base filters")
    p.add_argument("--ndf", type=int, default=64, help="Discriminator base filters")
    p.add_argument("--n_residual", type=int, default=9,
                   help="Residual blocks in generator (9 for 256px)")

    # I/O
    p.add_argument("--checkpoint_dir", default="checkpoints/cyclegan")
    p.add_argument("--log_dir", default="outputs/logs/cyclegan")
    p.add_argument("--save_every", type=int, default=10,
                   help="Save checkpoint every N epochs")
    p.add_argument("--log_every", type=int, default=100,
                   help="Log TensorBoard scalars every N iterations")
    p.add_argument("--sample_every", type=int, default=500,
                   help="Write sample images to TensorBoard every N iterations")
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint .pth to resume from")
    p.add_argument("--max_dataset_size", type=int, default=None,
               help="Cap both domains to this many images")

    return p.parse_args()


# ---------------------------------------------------------------------------
# LR scheduler: linear decay from initial LR → 0 after decay_epoch
# ---------------------------------------------------------------------------

class LinearDecayLR:
    """Callable that returns the LR multiplier at a given epoch."""

    def __init__(self, n_epochs: int, decay_epoch: int) -> None:
        self.n_epochs = n_epochs
        self.decay_epoch = decay_epoch

    def __call__(self, epoch: int) -> float:
        if epoch < self.decay_epoch:
            return 1.0
        fraction = (epoch - self.decay_epoch) / max(1, self.n_epochs - self.decay_epoch)
        return max(0.0, 1.0 - fraction)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    epoch: int,
    G_A2B: nn.Module,
    G_B2A: nn.Module,
    D_A: nn.Module,
    D_B: nn.Module,
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "G_A2B": G_A2B.state_dict(),
            "G_B2A": G_B2A.state_dict(),
            "D_A": D_A.state_dict(),
            "D_B": D_B.state_dict(),
            "opt_G": opt_G.state_dict(),
            "opt_D": opt_D.state_dict(),
        },
        path,
    )
    print(f"  ✔ Checkpoint saved → {path}")


def load_checkpoint(
    path: str | Path,
    G_A2B: nn.Module,
    G_B2A: nn.Module,
    D_A: nn.Module,
    D_B: nn.Module,
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    """Load checkpoint in-place; returns the next epoch to train from."""
    ckpt = torch.load(path, map_location=device)
    G_A2B.load_state_dict(ckpt["G_A2B"])
    G_B2A.load_state_dict(ckpt["G_B2A"])
    D_A.load_state_dict(ckpt["D_A"])
    D_B.load_state_dict(ckpt["D_B"])
    opt_G.load_state_dict(ckpt["opt_G"])
    opt_D.load_state_dict(ckpt["opt_D"])
    start = ckpt["epoch"] + 1
    print(f"  ✔ Resumed from {path}  (next epoch: {start})")
    return start


# ---------------------------------------------------------------------------
# ETA tracker
# ---------------------------------------------------------------------------

class ETATracker:
    """Tracks elapsed time across epochs and estimates time remaining."""

    def __init__(self, total_epochs: int, start_epoch: int = 1) -> None:
        self.total_epochs = total_epochs
        self.start_epoch = start_epoch
        self.epoch_times: list[float] = []
        self._wall_start = time.time()

    def record(self, elapsed: float) -> None:
        self.epoch_times.append(elapsed)

    def eta_str(self, current_epoch: int) -> str:
        epochs_done = current_epoch - self.start_epoch + 1
        epochs_left = self.total_epochs - current_epoch
        if not self.epoch_times or epochs_left <= 0:
            return "—"
        # Use a rolling average of the last 5 epochs for stability
        recent = self.epoch_times[-5:]
        avg = sum(recent) / len(recent)
        eta_sec = avg * epochs_left
        return _fmt_duration(eta_sec)

    def elapsed_total_str(self) -> str:
        return _fmt_duration(time.time() - self._wall_start)


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as Xh Ym Zs."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device(
    "cuda" if torch.cuda.is_available() else
    ("mps"  if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    data_root = Path(args.data_root)
    root_A = data_root / args.domain_A
    root_B = data_root / args.mode

    loader = make_dataloader(
        root_A=root_A,
        root_B=root_B,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True,
        unaligned=True,
        max_size=args.max_dataset_size
        
    )
    n_batches = len(loader)
    print(f"Dataset  A ({args.domain_A}): {len(loader.dataset.files_A)} images")
    print(f"Dataset  B ({args.mode}):     {len(loader.dataset.files_B)} images")
    print(f"Batches per epoch: {n_batches}")

    # ── Models ───────────────────────────────────────────────────────────────
    G_A2B = Generator(ngf=args.ngf, n_residual=args.n_residual).to(device)
    G_B2A = Generator(ngf=args.ngf, n_residual=args.n_residual).to(device)
    D_A = PatchDiscriminator(ndf=args.ndf).to(device)
    D_B = PatchDiscriminator(ndf=args.ndf).to(device)

    init_weights(G_A2B)
    init_weights(G_B2A)
    init_weights(D_A)
    init_weights(D_B)

    # ── Losses ───────────────────────────────────────────────────────────────
    criterion_GAN = GANLoss(mode=args.gan_mode).to(device)

    # ── Optimisers ───────────────────────────────────────────────────────────
    # Generators share one optimiser; discriminators share another.
    opt_G = torch.optim.Adam(
        itertools.chain(G_A2B.parameters(), G_B2A.parameters()),
        lr=args.lr, betas=(args.beta1, args.beta2),
    )
    opt_D = torch.optim.Adam(
        itertools.chain(D_A.parameters(), D_B.parameters()),
        lr=args.lr, betas=(args.beta1, args.beta2),
    )

    # ── LR schedulers ────────────────────────────────────────────────────────
    lr_lambda = LinearDecayLR(args.epochs, args.decay_epoch)
    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lr_lambda)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda=lr_lambda)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(
            args.resume, G_A2B, G_B2A, D_A, D_B, opt_G, opt_D, device
        )
        # Fast-forward schedulers
        for _ in range(start_epoch - 1):
            sched_G.step()
            sched_D.step()

    # ── Image replay buffers ──────────────────────────────────────────────────
    buf_fake_A = ImageBuffer(max_size=50)
    buf_fake_B = ImageBuffer(max_size=50)

    # ── TensorBoard ──────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=args.log_dir)

    # ── ETA tracker ──────────────────────────────────────────────────────────
    eta = ETATracker(total_epochs=args.epochs, start_epoch=start_epoch)

    # ── Training loop ────────────────────────────────────────────────────────
    global_step = (start_epoch - 1) * n_batches

    print(f"\n{'='*70}")
    print(f"  Starting training: {start_epoch} → {args.epochs} epochs")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        G_A2B.train(); G_B2A.train(); D_A.train(); D_B.train()

        # Accumulators for detailed per-epoch loss breakdown
        acc = dict(
            G_total=0.0,
            G_adv_A2B=0.0,
            G_adv_B2A=0.0,
            G_cyc_A=0.0,
            G_cyc_B=0.0,
            G_idt=0.0,
            D_A=0.0,
            D_B=0.0,
        )

        t0 = time.time()

        from tqdm import tqdm

        for i, (real_A, real_B) in enumerate(tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch"), start=1):
            real_A = real_A.to(device, non_blocking=True)
            real_B = real_B.to(device, non_blocking=True)

            # ── 1. Train Generators ────────────────────────────────────────
            opt_G.zero_grad(set_to_none=True)

            # Forward pass through both generators
            fake_B = G_A2B(real_A)          # A → B
            fake_A = G_B2A(real_B)          # B → A
            rec_A  = G_B2A(fake_B)          # A → B → A  (cycle)
            rec_B  = G_A2B(fake_A)          # B → A → B  (cycle)

            # Adversarial losses (generators want D to say "real")
            loss_adv_A2B = criterion_GAN(D_B(fake_B), is_real=True)
            loss_adv_B2A = criterion_GAN(D_A(fake_A), is_real=True)

            # Cycle-consistency losses
            loss_cyc_A = cycle_loss(real_A, rec_A, weight=args.lambda_cycle)
            loss_cyc_B = cycle_loss(real_B, rec_B, weight=args.lambda_cycle)

            # Identity losses (optional)
            loss_idt = torch.tensor(0.0, device=device)
            if args.lambda_identity > 0:
                idt_B = G_A2B(real_B)   # G_{A→B}(real_B) should ≈ real_B
                idt_A = G_B2A(real_A)   # G_{B→A}(real_A) should ≈ real_A
                loss_idt = (
                    identity_loss(real_B, idt_B, weight=args.lambda_identity)
                    + identity_loss(real_A, idt_A, weight=args.lambda_identity)
                )

            loss_G = (
                loss_adv_A2B + loss_adv_B2A
                + loss_cyc_A + loss_cyc_B
                + loss_idt
            )
            loss_G.backward()
            opt_G.step()

            # ── 2. Train Discriminators ────────────────────────────────────
            opt_D.zero_grad(set_to_none=True)

            # D_B: distinguishes real B from fake B
            fake_B_buf = buf_fake_B.push_and_pop(fake_B.detach())
            loss_D_B_real = criterion_GAN(D_B(real_B), is_real=True)
            loss_D_B_fake = criterion_GAN(D_B(fake_B_buf), is_real=False)
            loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5

            # D_A: distinguishes real A from fake A
            fake_A_buf = buf_fake_A.push_and_pop(fake_A.detach())
            loss_D_A_real = criterion_GAN(D_A(real_A), is_real=True)
            loss_D_A_fake = criterion_GAN(D_A(fake_A_buf), is_real=False)
            loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5

            loss_D = loss_D_A + loss_D_B
            loss_D.backward()
            opt_D.step()

            # ── Accumulate losses ─────────────────────────────────────────
            acc["G_total"]   += loss_G.item()
            acc["G_adv_A2B"] += loss_adv_A2B.item()
            acc["G_adv_B2A"] += loss_adv_B2A.item()
            acc["G_cyc_A"]   += loss_cyc_A.item()
            acc["G_cyc_B"]   += loss_cyc_B.item()
            acc["G_idt"]     += loss_idt.item()
            acc["D_A"]       += loss_D_A.item()
            acc["D_B"]       += loss_D_B.item()

            global_step += 1

            if global_step % args.log_every == 0:
                writer.add_scalar("Loss/G_total", loss_G.item(), global_step)
                writer.add_scalar("Loss/G_adv", (loss_adv_A2B + loss_adv_B2A).item(), global_step)
                writer.add_scalar("Loss/G_cycle", (loss_cyc_A + loss_cyc_B).item(), global_step)
                writer.add_scalar("Loss/G_identity", loss_idt.item(), global_step)
                writer.add_scalar("Loss/D_A", loss_D_A.item(), global_step)
                writer.add_scalar("Loss/D_B", loss_D_B.item(), global_step)
                writer.add_scalar("LR/G", opt_G.param_groups[0]["lr"], global_step)

            if global_step % args.sample_every == 0:
                _log_images(writer, real_A, fake_B, rec_A, real_B, fake_A, rec_B, global_step)

        # ── End-of-epoch ─────────────────────────────────────────────────
        sched_G.step()
        sched_D.step()

        elapsed = time.time() - t0
        eta.record(elapsed)

        # Compute per-epoch averages
        avg = {k: v / n_batches for k, v in acc.items()}
        lr_now = opt_G.param_groups[0]["lr"]

        # ── Detailed epoch summary ────────────────────────────────────────
        print(
            f"\nEpoch [{epoch:>3}/{args.epochs}]  "
            f"time={_fmt_duration(elapsed)}  "
            f"ETA={eta.eta_str(epoch)}  "
            f"elapsed={eta.elapsed_total_str()}  "
            f"lr={lr_now:.2e}"
        )
        print(
            f"  Generator   │ total={avg['G_total']:.4f}  "
            f"adv(A→B)={avg['G_adv_A2B']:.4f}  "
            f"adv(B→A)={avg['G_adv_B2A']:.4f}  "
            f"cyc_A={avg['G_cyc_A']:.4f}  "
            f"cyc_B={avg['G_cyc_B']:.4f}  "
            f"idt={avg['G_idt']:.4f}"
        )
        print(
            f"  Discriminator │ D_A={avg['D_A']:.4f}  "
            f"D_B={avg['D_B']:.4f}  "
            f"D_total={avg['D_A'] + avg['D_B']:.4f}"
        )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = Path(args.checkpoint_dir) / f"epoch_{epoch:03d}.pth"
            save_checkpoint(ckpt_path, epoch, G_A2B, G_B2A, D_A, D_B, opt_G, opt_D)

    writer.close()
    print(f"\n{'='*70}")
    print(f"  Training complete.  Total time: {eta.elapsed_total_str()}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# TensorBoard image helper
# ---------------------------------------------------------------------------

def _denorm(t: torch.Tensor) -> torch.Tensor:
    """Map [-1, 1] → [0, 1] for visualisation."""
    return (t * 0.5 + 0.5).clamp(0, 1)


def _log_images(
    writer: SummaryWriter,
    real_A: torch.Tensor,
    fake_B: torch.Tensor,
    rec_A: torch.Tensor,
    real_B: torch.Tensor,
    fake_A: torch.Tensor,
    rec_B: torch.Tensor,
    step: int,
    n: int = 4,
) -> None:
    """Write a 2-row image grid to TensorBoard."""
    def _grid(a, b, c):
        imgs = torch.cat([_denorm(x[:n]) for x in (a, b, c)], dim=0)
        return vutils.make_grid(imgs, nrow=n, normalize=False, padding=2)

    writer.add_image("A_real | A→B_fake | A→B→A_rec", _grid(real_A, fake_B, rec_A), step)
    writer.add_image("B_real | B→A_fake | B→A→B_rec", _grid(real_B, fake_A, rec_B), step)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)