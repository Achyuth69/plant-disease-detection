"""
split_data.py  —  Run this once to split your dataset.

Usage (from project root in YOUR terminal):
    python split_data.py

This will split data/raw/PlantVillage → data/train / data/val / data/test
using a stratified 80 / 10 / 10 split. Takes 2–5 minutes.
"""

import os, sys, shutil, random, stat
from pathlib import Path
from collections import defaultdict


def _force_remove(func, path, _):
    """Error handler for shutil.rmtree — removes read-only flag then retries."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# The dataset zip extracts to a nested structure — auto-find the deepest
# folder that contains the most class sub-directories.
def _best_source():
    candidates = [
        os.path.join(PROJECT_ROOT, "data", "raw", "PlantVillage", "PlantVillage"),
        os.path.join(PROJECT_ROOT, "data", "raw", "PlantVillage"),
        os.path.join(PROJECT_ROOT, "data", "raw"),
    ]
    best, best_count = None, 0
    for c in candidates:
        if os.path.isdir(c):
            subdirs = [x for x in os.listdir(c)
                       if os.path.isdir(os.path.join(c, x)) and not x.startswith(".")]
            if len(subdirs) > best_count:
                best, best_count = c, len(subdirs)
    return best

SOURCE_DIR   = _best_source() or os.path.join(PROJECT_ROOT, "data", "raw", "PlantVillage")
TRAIN_DIR    = os.path.join(PROJECT_ROOT, "data", "train")
VAL_DIR      = os.path.join(PROJECT_ROOT, "data", "val")
TEST_DIR     = os.path.join(PROJECT_ROOT, "data", "test")

TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
RANDOM_SEED  = 42
SUPPORTED    = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".JPG", ".JPEG", ".PNG"}

def find_source():
    """Auto-locate the PlantVillage folder if SOURCE_DIR doesn't exist."""
    if os.path.isdir(SOURCE_DIR):
        return Path(SOURCE_DIR)
    # search inside data/raw for any folder with 5+ subdirs
    raw = Path(PROJECT_ROOT) / "data" / "raw"
    if raw.exists():
        for d in raw.rglob("*"):
            if d.is_dir():
                subdirs = [x for x in d.iterdir() if x.is_dir()]
                if len(subdirs) >= 5:
                    print(f"[Auto] Found source at: {d}")
                    return d
    return None


def split():
    src = find_source()
    if src is None:
        print("ERROR: Cannot find PlantVillage folder.")
        print(f"Expected: {SOURCE_DIR}")
        sys.exit(1)

    class_dirs = sorted([d for d in src.iterdir() if d.is_dir()])
    print(f"Found {len(class_dirs)} class folders in: {src}")

    # Clean existing splits
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if os.path.exists(d):
            print(f"Removing existing {d} ...")
            shutil.rmtree(d, onerror=_force_remove)
        os.makedirs(d)

    random.seed(RANDOM_SEED)
    stats = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    total = 0

    for i, cls_dir in enumerate(class_dirs, 1):
        cls_name = cls_dir.name
        images = [f for f in cls_dir.iterdir()
                  if f.suffix in SUPPORTED or f.suffix.lower() in SUPPORTED]
        if not images:
            print(f"  [{i}/{len(class_dirs)}] SKIP {cls_name} — no images")
            continue

        random.shuffle(images)
        n       = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        splits = {
            "train": (TRAIN_DIR, images[:n_train]),
            "val":   (VAL_DIR,   images[n_train:n_train + n_val]),
            "test":  (TEST_DIR,  images[n_train + n_val:]),
        }

        for split_name, (dest_root, img_list) in splits.items():
            dest = os.path.join(dest_root, cls_name)
            os.makedirs(dest, exist_ok=True)
            for img in img_list:
                shutil.copy2(img, os.path.join(dest, img.name))
            stats[cls_name][split_name] = len(img_list)
            total += len(img_list)

        print(f"  [{i:>2}/{len(class_dirs)}] {cls_name:<50} "
              f"train={stats[cls_name]['train']:>4}  "
              f"val={stats[cls_name]['val']:>3}  "
              f"test={stats[cls_name]['test']:>3}")

    # Summary
    print(f"\n{'='*65}")
    print(f"  DONE — {total:,} images split into train / val / test")
    print(f"  train : {TRAIN_DIR}")
    print(f"  val   : {VAL_DIR}")
    print(f"  test  : {TEST_DIR}")
    print(f"{'='*65}")
    print("\nNext step — run training:")
    print("  python src/train.py")


if __name__ == "__main__":
    split()
