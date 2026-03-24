"""
filter_bdd.py
=============
Reads per-image BDD100K JSON files and copies images + segmentation masks
into condition-based subfolders:

    data/bdd100k/images/{train,val}/{day,night,rain}/
    data/bdd100k/seg_maps/{train,val}/{day,night,rain}/

Actual layout expected:
    data/
    ├── 10k/{train,val,test}/*.jpg          ← images
    ├── 100k/{train,val,test}/*.json        ← per-image label JSONs
    └── bdd100k_seg_maps/labels/{train,val}/*_train_id.png  ← masks

Usage
-----
    python utils/filter_bdd.py [--data-root DATA_ROOT] [--symlink] [--dry-run]

Arguments
---------
--data-root   Parent directory containing 10k/, 100k/, bdd100k_seg_maps/
              (default: data)
--symlink     Create symlinks instead of copying (faster, saves disk)
--dry-run     Print what would happen without writing any files
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── condition logic ────────────────────────────────────────────────────────────

def get_conditions(attributes: dict) -> list[str]:
    """
    Map top-level attributes dict to condition bucket(s).

    Schema (from per-image JSON):
        attributes.timeofday : "daytime" | "night" | "dawn/dusk" | "undefined"
        attributes.weather   : "clear" | "partly cloudy" | "overcast" |
                               "rainy" | "snowy" | "foggy" | "undefined"

    Mapping:
        timeofday == "daytime"  → day
        timeofday == "night"    → night
        weather   == "rainy"    → rain  (any time-of-day)

    An image can belong to more than one bucket (e.g. rainy night → night + rain).
    """
    conditions = []
    timeofday = (attributes.get("timeofday") or "").lower().strip()
    weather   = (attributes.get("weather")   or "").lower().strip()

    if timeofday == "daytime":
        conditions.append("day")
    if timeofday == "night":
        conditions.append("night")
    if weather == "rainy":
        conditions.append("rain")

    return conditions


# ── file transfer ──────────────────────────────────────────────────────────────

def transfer(src: Path, dst: Path, symlink: bool, dry_run: bool) -> bool:
    """Copy or symlink src → dst. Returns True on success."""
    if not src.exists():
        log.warning("Source missing: %s", src)
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        log.debug("%s  %s  →  %s", "SYMLINK" if symlink else "COPY", src, dst)
        return True

    if dst.exists() or dst.is_symlink():
        return True  # already done

    if symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)
    return True


# ── per-split processing ───────────────────────────────────────────────────────

def process_split(
    split: str,
    labels_dir: Path,
    images_dir: Path,
    masks_dir: Path | None,
    out_images_dir: Path,
    out_masks_dir: Path | None,
    symlink: bool,
    dry_run: bool,
) -> dict[str, int]:

    json_files = sorted(labels_dir.glob("*.json"))
    if not json_files:
        log.error("No JSON files found in %s", labels_dir)
        return {}

    log.info("Found %d label files in %s", len(json_files), labels_dir)
    counts: dict[str, int] = {"day": 0, "night": 0, "rain": 0, "skipped": 0}

    for jf in json_files:
        with jf.open() as f:
            data = json.load(f)

        attributes = data.get("attributes", {})
        name       = data.get("name", jf.stem)   # e.g. "0000f77c-6257be58"
        conditions = get_conditions(attributes)

        if not conditions:
            counts["skipped"] += 1
            continue

        img_src  = images_dir / f"{name}.jpg"
        mask_src = (masks_dir / f"{name}_train_id.png") if masks_dir else None

        for cond in conditions:
            transfer(img_src, out_images_dir / cond / f"{name}.jpg", symlink, dry_run)

            if mask_src and out_masks_dir:
                transfer(mask_src, out_masks_dir / cond / f"{name}_train_id.png", symlink, dry_run)

            counts[cond] = counts.get(cond, 0) + 1

    return counts


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Filter BDD100K images by condition.")
    parser.add_argument(
        "--data-root", default="data",
        help="Directory containing 10k/, 100k/, bdd100k_seg_maps/ (default: data)",
    )
    parser.add_argument("--symlink", action="store_true",
                        help="Symlink instead of copy (faster, saves disk)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate without writing any files")
    args = parser.parse_args()

    root = Path(args.data_root)

    if args.dry_run:
        log.info("DRY RUN — no files will be written")

    SPLITS = {
        "train": {
            "labels_dir": root / "100k"  / "train",
            "images_dir": root / "10k"   / "train",
            "masks_dir":  root / "bdd100k_seg_maps" / "labels" / "train",
            "out_images": root / "bdd100k" / "images" / "train",
            "out_masks":  root / "bdd100k" / "seg_maps" / "train",
        },
        "val": {
            "labels_dir": root / "100k"  / "val",
            "images_dir": root / "10k"   / "val",
            "masks_dir":  root / "bdd100k_seg_maps" / "labels" / "val",
            "out_images": root / "bdd100k" / "images" / "val",
            "out_masks":  root / "bdd100k" / "seg_maps" / "val",
        },
    }

    summary = {}

    for split, cfg in SPLITS.items():
        log.info("=== Processing split: %s ===", split)

        masks_dir     = cfg["masks_dir"] if cfg["masks_dir"].exists() else None
        out_masks_dir = cfg["out_masks"] if masks_dir                 else None

        if not masks_dir:
            log.warning("Mask dir not found (%s) — skipping masks for %s", cfg["masks_dir"], split)

        counts = process_split(
            split          = split,
            labels_dir     = cfg["labels_dir"],
            images_dir     = cfg["images_dir"],
            masks_dir      = masks_dir,
            out_images_dir = cfg["out_images"],
            out_masks_dir  = out_masks_dir,
            symlink        = args.symlink,
            dry_run        = args.dry_run,
        )
        summary[split] = counts
        log.info(
            "  %-6s → day: %5d  night: %5d  rain: %5d  (skipped: %d)",
            split, counts.get("day", 0), counts.get("night", 0),
            counts.get("rain", 0), counts.get("skipped", 0),
        )

    # ── summary table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print(f"{'Split':<8} {'Day':>8} {'Night':>8} {'Rain':>8} {'Skipped':>9}")
    print("-" * 52)
    for split, counts in summary.items():
        print(
            f"{split:<8} {counts.get('day', 0):>8,} {counts.get('night', 0):>8,} "
            f"{counts.get('rain', 0):>8,} {counts.get('skipped', 0):>9,}"
        )
    print("=" * 52)


if __name__ == "__main__":
    main()