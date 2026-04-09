"""
filter_bdd.py
=============
Populates:

    data/bdd100k/images/{train,val}/{day,night,rain}/   ← filtered by condition
    data/bdd100k/seg_maps/{train,val}/{day,night,rain}/

TRAIN split: uses 100k JSON metadata to sort 10k/train images by condition.
VAL split  : the 10k val images have no matching JSON metadata, so all val
             images/masks are symlinked into a single 'all/' subfolder.
             Update datasets.py to use 'all' as the val condition.

Usage
-----
    python utils/filter_bdd.py [--data-root DATA_ROOT] [--symlink] [--dry-run]
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


def get_conditions(attributes: dict) -> list[str]:
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


def transfer(src: Path, dst: Path, symlink: bool, dry_run: bool) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log.debug("%s  %s  →  %s", "SYMLINK" if symlink else "COPY", src, dst)
        return True
    if dst.exists() or dst.is_symlink():
        return True
    if symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)
    return True


def process_train(
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
        name       = data.get("name", jf.stem)
        conditions = get_conditions(attributes)

        img_src = images_dir / f"{name}.jpg"
        if not img_src.exists():
            counts["skipped"] += 1
            continue

        if not conditions:
            counts["skipped"] += 1
            continue

        mask_src = (masks_dir / f"{name}_train_id.png") if masks_dir else None

        for cond in conditions:
            transfer(img_src, out_images_dir / cond / f"{name}.jpg", symlink, dry_run)
            if mask_src and out_masks_dir:
                transfer(mask_src, out_masks_dir / cond / f"{name}_train_id.png", symlink, dry_run)
            counts[cond] = counts.get(cond, 0) + 1

    return counts


def process_val(
    images_dir: Path,
    masks_dir: Path | None,
    out_images_dir: Path,
    out_masks_dir: Path | None,
    symlink: bool,
    dry_run: bool,
) -> dict[str, int]:
    """
    The 10k val images have no 100k-style JSON metadata so we cannot sort by
    condition. Symlink everything into an 'all/' subfolder instead.
    """
    counts = {"all": 0, "skipped": 0}

    img_files = sorted(images_dir.glob("*.jpg"))
    if not img_files:
        log.error("No .jpg files found in %s", images_dir)
        return counts

    log.info("Found %d val images in %s", len(img_files), images_dir)

    for img_path in img_files:
        stem     = img_path.stem
        mask_src = (masks_dir / f"{stem}_train_id.png") if masks_dir else None

        # Only include images that have a paired mask
        if mask_src and not mask_src.exists():
            counts["skipped"] += 1
            continue

        transfer(img_path, out_images_dir / "all" / f"{stem}.jpg", symlink, dry_run)
        if mask_src and out_masks_dir:
            transfer(mask_src, out_masks_dir / "all" / f"{stem}_train_id.png", symlink, dry_run)
        counts["all"] += 1

    return counts


def main():
    parser = argparse.ArgumentParser(description="Filter BDD100K images by condition.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--symlink", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.data_root)
    if args.dry_run:
        log.info("DRY RUN — no files will be written")

    # ── TRAIN ──────────────────────────────────────────────────────────────
    log.info("=== Processing split: train ===")
    train_masks_dir = root / "bdd100k_seg_maps" / "labels" / "train"
    train_masks_dir = train_masks_dir if train_masks_dir.exists() else None
    if not train_masks_dir:
        log.warning("Train mask dir not found — skipping masks for train")

    train_counts = process_train(
        labels_dir     = root / "100k" / "train",
        images_dir     = root / "10k"  / "train",
        masks_dir      = train_masks_dir,
        out_images_dir = root / "bdd100k" / "images"   / "train",
        out_masks_dir  = (root / "bdd100k" / "seg_maps" / "train") if train_masks_dir else None,
        symlink        = args.symlink,
        dry_run        = args.dry_run,
    )
    log.info(
        "  train → day: %5d  night: %5d  rain: %5d  (skipped: %d)",
        train_counts.get("day", 0), train_counts.get("night", 0),
        train_counts.get("rain", 0), train_counts.get("skipped", 0),
    )

    # ── VAL ────────────────────────────────────────────────────────────────
    log.info("=== Processing split: val ===")
    val_masks_dir = root / "bdd100k_seg_maps" / "labels" / "val"
    val_masks_dir = val_masks_dir if val_masks_dir.exists() else None
    if not val_masks_dir:
        log.warning("Val mask dir not found — skipping masks for val")

    val_counts = process_val(
        images_dir     = root / "10k"  / "val",
        masks_dir      = val_masks_dir,
        out_images_dir = root / "bdd100k" / "images"   / "val",
        out_masks_dir  = (root / "bdd100k" / "seg_maps" / "val") if val_masks_dir else None,
        symlink        = args.symlink,
        dry_run        = args.dry_run,
    )
    log.info("  val → all: %5d  (skipped: %d)",
             val_counts.get("all", 0), val_counts.get("skipped", 0))

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'Split':<8} {'Day':>8} {'Night':>8} {'Rain':>8} {'All':>8} {'Skipped':>9}")
    print("-" * 60)
    print(
        f"{'train':<8} {train_counts.get('day',0):>8,} {train_counts.get('night',0):>8,} "
        f"{train_counts.get('rain',0):>8,} {'—':>8} {train_counts.get('skipped',0):>9,}"
    )
    print(
        f"{'val':<8} {'—':>8} {'—':>8} {'—':>8} "
        f"{val_counts.get('all',0):>8,} {val_counts.get('skipped',0):>9,}"
    )
    print("=" * 60)
    print("\nNOTE: Val images → bdd100k/images/val/all/")
    print("      Update datasets.py: use 'all' as the val condition.")


if __name__ == "__main__":
    main()