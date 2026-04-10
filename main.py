import argparse
import subprocess
import sys
from pathlib import Path


# Raw downloaded data root:
#   data/bdd100k/
#     ├── 10k/
#     ├── 100k/
#     └── bdd100k_seg_maps/
RAW_ROOT = Path("data/bdd100k")

# Processed/filtered output root created by filter_bdd.py:
#   data/bdd100k/bdd100k/
#     ├── images/
#     └── seg_maps/
PROC_ROOT = RAW_ROOT / "bdd100k"

# CycleGAN training reads filtered train images from here
CYCLEGAN_TRAIN_ROOT = PROC_ROOT / "images" / "train"

# Source daytime images for synthetic generation
DAY_SOURCE_DIR = PROC_ROOT / "images" / "train" / "day"

# Synthetic outputs root
SYN_ROOT = Path("data/synthetic")

# Default checkpoints
CYCLEGAN_CKPT = Path("checkpoints/cyclegan/epoch_200.pth")
BASELINE_CKPT = Path("checkpoints/segmentation/baseline_best.pth")
AUGMENTED_CKPT = Path("checkpoints/segmentation/augmented_best.pth")


def run_cmd(cmd: list[str]) -> None:
    print("\n" + "=" * 80)
    print("RUNNING:", " ".join(map(str, cmd)))
    print("=" * 80)
    subprocess.run(cmd, check=True)


def stage_filter(dry_run: bool = False) -> None:
    cmd = [sys.executable, "utils/filter_bdd.py", "--data-root", str(RAW_ROOT)]
    if dry_run:
        cmd.append("--dry-run")
    run_cmd(cmd)


def stage_train_cyclegan(mode: str = "night", epochs: int | None = None) -> None:
    cmd = [
        sys.executable,
        "cyclegan/train_cyclegan.py",
        "--data_root", str(CYCLEGAN_TRAIN_ROOT),
        "--mode", mode,
    ]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    run_cmd(cmd)


def stage_generate_synthetic(mode: str = "night", checkpoint: Path | None = None) -> None:
    ckpt = checkpoint or CYCLEGAN_CKPT
    if not ckpt.exists():
        raise FileNotFoundError(f"CycleGAN checkpoint not found: {ckpt}")

    cmd = [
        sys.executable,
        "cyclegan/generate_synthetic.py",
        "--checkpoint", str(ckpt),
        "--mode", mode,
        "--source_dir", str(DAY_SOURCE_DIR),
        "--split", "train",
        "--save_pairs",
    ]
    run_cmd(cmd)


def stage_train_baseline(epochs: int | None = None) -> None:
    cmd = [
        sys.executable,
        "segmentation/train_seg.py",
        "--run", "baseline",
        "--bdd_root", str(PROC_ROOT),
    ]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    run_cmd(cmd)


def stage_train_augmented(modes: list[str], epochs: int | None = None) -> None:
    cmd = [
        sys.executable,
        "segmentation/train_seg.py",
        "--run", "augmented",
        "--bdd_root", str(PROC_ROOT),
        "--synthetic_root", str(SYN_ROOT),
        "--synthetic_modes", *modes,
    ]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    run_cmd(cmd)


def stage_evaluate(split: str, conditions: list[str]) -> None:
    if not BASELINE_CKPT.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {BASELINE_CKPT}")
    if not AUGMENTED_CKPT.exists():
        raise FileNotFoundError(f"Augmented checkpoint not found: {AUGMENTED_CKPT}")

    cmd = [
        sys.executable,
        "segmentation/evaluate.py",
        "--checkpoint_baseline", str(BASELINE_CKPT),
        "--checkpoint_augmented", str(AUGMENTED_CKPT),
        "--bdd_root", str(PROC_ROOT),
        "--split", split,
        "--conditions", *conditions,
    ]
    run_cmd(cmd)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the CV project pipeline")

    p.add_argument(
        "--stage",
        required=True,
        choices=[
            "filter",
            "train_cyclegan",
            "generate_synthetic",
            "train_baseline",
            "train_augmented",
            "evaluate",
            "all",
        ],
    )

    p.add_argument("--mode", choices=["night", "rain"], default="night")
    p.add_argument("--synthetic_modes", nargs="+", choices=["night", "rain"], default=["night"])
    p.add_argument("--conditions", nargs="+", choices=["day", "night", "rain", "all"], default=["all"])
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.stage == "filter":
        stage_filter(dry_run=args.dry_run)

    elif args.stage == "train_cyclegan":
        stage_train_cyclegan(mode=args.mode, epochs=args.epochs)

    elif args.stage == "generate_synthetic":
        ckpt = Path(args.checkpoint) if args.checkpoint else None
        stage_generate_synthetic(mode=args.mode, checkpoint=ckpt)

    elif args.stage == "train_baseline":
        stage_train_baseline(epochs=args.epochs)

    elif args.stage == "train_augmented":
        stage_train_augmented(modes=args.synthetic_modes, epochs=args.epochs)

    elif args.stage == "evaluate":
        stage_evaluate(split=args.split, conditions=args.conditions)

    elif args.stage == "all":
        stage_filter(dry_run=args.dry_run)
        stage_train_cyclegan(mode=args.mode, epochs=args.epochs)
        stage_generate_synthetic(
            mode=args.mode,
            checkpoint=Path(args.checkpoint) if args.checkpoint else None
        )
        stage_train_baseline(epochs=args.epochs)
        stage_train_augmented(modes=args.synthetic_modes, epochs=args.epochs)
        stage_evaluate(split=args.split, conditions=args.conditions)


if __name__ == "__main__":
    main()