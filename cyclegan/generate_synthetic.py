"""
generate_synthetic.py — Inference: translate daytime images → synthetic weather.

Usage
-----
    # Generate synthetic night images from all day images in train split
    python cyclegan/generate_synthetic.py \\
        --checkpoint checkpoints/cyclegan/epoch_200.pth \\
        --mode night \\
        --split train

    # Generate synthetic rain images for validation split, keep originals too
    python cyclegan/generate_synthetic.py \\
        --checkpoint checkpoints/cyclegan/epoch_200.pth \\
        --mode rain \\
        --split val \\
        --save_pairs          # saves side-by-side comparison grids

Output
------
    data/synthetic/night/   (or /rain/)
        <original_filename>.png   ← translated image (same name as source)

    If --save_pairs:
        outputs/plots/synthetic_pairs/<mode>/
            <filename>_pair.png   ← [real_day | fake_<mode>] side-by-side
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cyclegan.models import Generator


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic weather images with CycleGAN")

    p.add_argument("--checkpoint", required=True,
                   help="Path to CycleGAN checkpoint (.pth)")
    p.add_argument("--mode", required=True, choices=["night", "rain"],
                   help="Target domain to generate")
    p.add_argument("--source_dir", default=None,
                   help="Override default source directory "
                        "(default: data/bdd100k/images/<split>/day)")
    p.add_argument("--output_dir", default=None,
                   help="Override default output directory "
                        "(default: data/synthetic/<mode>)")
    p.add_argument("--split", default="train", choices=["train", "val"],
                   help="Dataset split to process")
    p.add_argument("--image_size", type=int, default=256,
                   help="Resize images to this size before translation")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Number of images to translate simultaneously")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--n_residual", type=int, default=9)
    p.add_argument("--save_pairs", action="store_true",
                   help="Also save side-by-side comparison grids")
    p.add_argument("--pairs_dir", default="outputs/plots/synthetic_pairs",
                   help="Where to write pair images")
    p.add_argument("--keep_original_size", action="store_true",
                   help="Resize output back to original image dimensions")
    p.add_argument("--ext", default="png",
                   choices=["png", "jpg"],
                   help="Output image format")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _collect_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Source directory not found: {directory}")
    paths = sorted(p for p in directory.rglob("*") if p.suffix.lower() in _IMG_EXTENSIONS)
    if not paths:
        raise RuntimeError(f"No images found in {directory}")
    return paths


def _build_preprocess(image_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size), T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),   # → [-1, 1]
    ])


def _denorm(t: torch.Tensor) -> torch.Tensor:
    """Map [-1, 1] → [0, 1]."""
    return (t * 0.5 + 0.5).clamp(0, 1)


# ---------------------------------------------------------------------------
# Generator loader
# ---------------------------------------------------------------------------

def load_generator(checkpoint_path: str | Path, ngf: int, n_residual: int, device: torch.device) -> Generator:
    """Load G_A2B (day→target) from a training checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    G = Generator(ngf=ngf, n_residual=n_residual).to(device)
    if "G_A2B" not in ckpt:
        raise KeyError(
            f"Checkpoint at {checkpoint_path!r} does not contain key 'G_A2B'. "
            "Make sure you pass a full CycleGAN training checkpoint."
        )
    G.load_state_dict(ckpt["G_A2B"])
    G.eval()
    return G


# ---------------------------------------------------------------------------
# Per-image inference dataset (simple, no DataLoader overhead for inference)
# ---------------------------------------------------------------------------

class _InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, paths: list[Path], transform: T.Compose) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        original_size = img.size          # (W, H)
        tensor = self.transform(img)
        return tensor, str(path), original_size


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Paths ─────────────────────────────────────────────────────────────
    source_dir = Path(args.source_dir) if args.source_dir else \
        Path("data/bdd100k/images") / args.split / "day"
    output_dir = Path(args.output_dir) if args.output_dir else \
        Path("data/synthetic") / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs_dir = None
    if args.save_pairs:
        pairs_dir = Path(args.pairs_dir) / args.mode
        pairs_dir.mkdir(parents=True, exist_ok=True)

    # ── Generator ─────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    G = load_generator(args.checkpoint, ngf=args.ngf, n_residual=args.n_residual, device=device)

    # ── Dataset ───────────────────────────────────────────────────────────
    preprocess = _build_preprocess(args.image_size)
    image_paths = _collect_images(source_dir)
    print(f"Found {len(image_paths)} images in {source_dir}")

    dataset = _InferenceDataset(image_paths, preprocess)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── Inference ─────────────────────────────────────────────────────────
    n_saved = 0
    with torch.no_grad():
        for tensors, paths, original_sizes in tqdm(loader, desc=f"Generating {args.mode}"):
            tensors = tensors.to(device, non_blocking=True)
            fake = G(tensors)                        # (B, 3, H, W) in [-1, 1]
            fake_rgb = _denorm(fake).cpu()           # (B, 3, H, W) in [0, 1]
            real_rgb = _denorm(tensors).cpu()

            for j, (fake_img_t, real_img_t, src_path, orig_size) in enumerate(
                zip(fake_rgb, real_rgb, paths, zip(*original_sizes))
            ):
                orig_w, orig_h = int(orig_size[0]), int(orig_size[1])
                stem = Path(src_path).stem
                out_name = f"{stem}.{args.ext}"
                out_path = output_dir / out_name

                # Convert tensor → PIL
                fake_pil = TF.to_pil_image(fake_img_t)
                if args.keep_original_size:
                    fake_pil = fake_pil.resize((orig_w, orig_h), Image.BICUBIC)

                fake_pil.save(out_path)
                n_saved += 1

                # Optionally save side-by-side pair
                if pairs_dir is not None:
                    pair_path = pairs_dir / f"{stem}_pair.png"
                    pair_grid = vutils.make_grid(
                        torch.stack([real_img_t, fake_img_t]),
                        nrow=2, padding=4, pad_value=1.0,
                    )
                    TF.to_pil_image(pair_grid).save(pair_path)

    print(f"\n✔ Saved {n_saved} synthetic {args.mode} images → {output_dir}")
    if pairs_dir:
        print(f"✔ Saved comparison pairs → {pairs_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    generate(args)