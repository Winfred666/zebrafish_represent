#!/usr/bin/env python3
"""Disposable script: symlink LS-FIS TIFs into sample_sm train/val splits.

For each data/raw/LS-FIS/<n>dpf/Image_Shifted/Image_Shifted/ directory:
  - 100 randomly-selected TIFs → data/raw/sample_sm/train/
  - 10 randomly-selected TIFs  → data/raw/sample_sm/val/

Existing train/val files are removed first.  Run once, then delete this script.
"""
import os, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_GLOB = "data/raw/LS-FIS/*dpf/Image_Shifted/Image_Shifted"
TRAIN_DIR = ROOT / "data/raw/sample_sm/train"
VAL_DIR = ROOT / "data/raw/sample_sm/val"
N_TRAIN = 100
N_VAL = 10
SEED = 42

random.seed(SEED)

# Clear existing symlinks
for d in (TRAIN_DIR, VAL_DIR):
    if d.exists():
        for f in d.iterdir():
            if f.is_symlink():
                f.unlink()
    d.mkdir(parents=True, exist_ok=True)

src_dirs = sorted(ROOT.glob(SRC_GLOB))
if not src_dirs:
    print(f"ERROR: no source dirs matched {SRC_GLOB}", file=sys.stderr)
    sys.exit(1)

train_count = val_count = 0
for src_dir in src_dirs:
    dpf = src_dir.parent.parent.parent.name  # e.g. "3dpf"
    tifs = sorted(src_dir.glob("*.tif")) + sorted(src_dir.glob("*.tiff"))
    random.shuffle(tifs)

    train_files = tifs[:N_TRAIN]
    val_files = tifs[N_TRAIN:N_TRAIN + N_VAL]

    for tf in train_files:
        dst = TRAIN_DIR / f"{dpf}_{tf.name}"
        dst.symlink_to(tf)
        train_count += 1

    for tf in val_files:
        dst = VAL_DIR / f"{dpf}_{tf.name}"
        dst.symlink_to(tf)
        val_count += 1

    print(f"{dpf}: {len(train_files)} train + {len(val_files)} val symlinked")

print(f"\nDone: {train_count} train + {val_count} val symlinks total")
