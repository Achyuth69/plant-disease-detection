"""
kaggle_download.py
------------------
Step 1 of the pipeline — downloads the PlantVillage dataset from Kaggle
and splits it into train / val / test directories.

Prerequisites
-------------
1. Install the Kaggle API:
       pip install kaggle

2. Create a Kaggle API token:
       kaggle.com → Account → API → "Create New API Token"
       Save the downloaded kaggle.json to:
           Windows : C:\\Users\\<YourName>\\.kaggle\\kaggle.json
           Linux   : ~/.kaggle/kaggle.json

3. Run this script from the project root:
       python src/kaggle_download.py

What it does
------------
  ① Downloads  emmarex/plantdisease  →  data/raw/plantdisease.zip
  ② Unzips it                        →  data/raw/PlantVillage/<class>/...
  ③ Splits images per class          →  data/train / data/val / data/test
     using a stratified 80/10/10 split (reproducible via RANDOM_SEED)
  ④ Prints a summary table of images per class per split
"""

import os
import sys
import zipfile
import shutil
import random
import stat
from pathlib import Path
from collections import defaultdict


def _force_remove(func, path, _):
    """Error handler for shutil.rmtree on Windows read-only files."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

# ── Local config ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    PROJECT_ROOT, RAW_DIR, DATA_DIR,
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    KAGGLE_DATASET, KAGGLE_ZIP_NAME, KAGGLE_INNER_FOLDER,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED,
)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Download from Kaggle
# ─────────────────────────────────────────────────────────────────────────────

def download_dataset() -> Path:
    """
    Uses the Kaggle CLI to download the dataset zip into RAW_DIR.
    Returns the path to the downloaded zip file.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    zip_path = Path(RAW_DIR) / KAGGLE_ZIP_NAME

    if zip_path.exists():
        print(f"[Download] Zip already exists — skipping download.\n"
              f"           {zip_path}")
        return zip_path

    print(f"[Download] Downloading '{KAGGLE_DATASET}' from Kaggle ...")
    print(f"           Target: {RAW_DIR}\n")

    # kaggle datasets download writes the zip to the current working dir
    # unless --path is specified.
    ret = os.system(
        f"kaggle datasets download -d {KAGGLE_DATASET} "
        f"--path \"{RAW_DIR}\" --unzip=False"
    )
    if ret != 0:
        print("\n[ERROR] Kaggle download failed. Common fixes:")
        print("  1. Make sure kaggle is installed:  pip install kaggle")
        print("  2. Place kaggle.json at  %USERPROFILE%\\.kaggle\\kaggle.json")
        print("  3. Check your internet connection.")
        sys.exit(1)

    # Kaggle may name the file differently — find it
    zips = list(Path(RAW_DIR).glob("*.zip"))
    if not zips:
        print("[ERROR] No zip file found after download.")
        sys.exit(1)

    downloaded_zip = zips[0]
    if downloaded_zip != zip_path:
        downloaded_zip.rename(zip_path)

    print(f"[Download] Saved to: {zip_path}")
    return zip_path


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Unzip
# ─────────────────────────────────────────────────────────────────────────────

def unzip_dataset(zip_path: Path) -> Path:
    """
    Extracts the zip into RAW_DIR.
    Returns the path to the inner folder containing class sub-directories.
    """
    extract_dir  = Path(RAW_DIR)
    inner_folder = extract_dir / KAGGLE_INNER_FOLDER

    if inner_folder.exists():
        print(f"[Unzip] Already extracted — skipping.\n"
              f"        {inner_folder}")
        return inner_folder

    print(f"[Unzip] Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print(f"[Unzip] Done → {extract_dir}")

    # The zip might produce a nested folder; find the one with class sub-dirs
    candidates = [p for p in extract_dir.iterdir()
                  if p.is_dir() and p.name != "__MACOSX"]
    if not candidates:
        print("[ERROR] No folder found after unzipping.")
        sys.exit(1)

    # Use the configured inner folder name, or fall back to the first found
    if inner_folder.exists():
        return inner_folder

    # rename / adopt whatever folder was extracted
    actual = candidates[0]
    actual.rename(inner_folder)
    print(f"[Unzip] Renamed '{actual.name}' → '{KAGGLE_INNER_FOLDER}'")
    return inner_folder


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Stratified train / val / test split
# ─────────────────────────────────────────────────────────────────────────────

def split_dataset(source_dir: Path) -> dict:
    """
    Reads all class folders from source_dir, shuffles images with a fixed seed,
    then copies them into data/train, data/val, data/test.

    Args:
        source_dir: the inner PlantVillage folder with one sub-dir per class.

    Returns:
        Dict of {class_name: {split: count}} for the summary table.
    """
    assert TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1.0, "Ratios must sum to 1.0"

    # Clean & create split directories
    for split_dir in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir, onerror=_force_remove)
        os.makedirs(split_dir)

    random.seed(RANDOM_SEED)
    stats = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})

    class_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        print(f"[ERROR] No class sub-folders found in {source_dir}")
        sys.exit(1)

    print(f"\n[Split] Found {len(class_dirs)} classes in '{source_dir.name}'")
    print(f"        Ratios — Train: {TRAIN_RATIO:.0%} | "
          f"Val: {VAL_RATIO:.0%} | Test: {TEST_RATIO:.0%}")
    print(f"        Random seed: {RANDOM_SEED}\n")

    total_copied = 0

    for cls_dir in class_dirs:
        cls_name = cls_dir.name

        # Gather all valid images
        images = [
            f for f in cls_dir.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTS
        ]
        if not images:
            print(f"  [WARN] No images in class '{cls_name}' — skipping.")
            continue

        random.shuffle(images)
        n       = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        # test gets the remainder to avoid rounding loss
        n_test  = n - n_train - n_val

        splits = {
            "train": images[:n_train],
            "val":   images[n_train : n_train + n_val],
            "test":  images[n_train + n_val :],
        }
        split_dirs = {
            "train": TRAIN_DIR,
            "val":   VAL_DIR,
            "test":  TEST_DIR,
        }

        for split_name, img_list in splits.items():
            dest_cls = os.path.join(split_dirs[split_name], cls_name)
            os.makedirs(dest_cls, exist_ok=True)
            for img_path in img_list:
                shutil.copy2(img_path, os.path.join(dest_cls, img_path.name))
            stats[cls_name][split_name] = len(img_list)
            total_copied += len(img_list)

    print(f"[Split] Copied {total_copied:,} images across all classes.\n")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(stats: dict) -> None:
    """Prints a formatted table: class | train | val | test | total"""
    col_w = max(len(k) for k in stats) + 2

    header = f"{'Class':<{col_w}} {'Train':>7} {'Val':>7} {'Test':>7} {'Total':>7}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    grand = {"train": 0, "val": 0, "test": 0}
    for cls, counts in sorted(stats.items()):
        total = sum(counts.values())
        print(
            f"{cls:<{col_w}} "
            f"{counts['train']:>7} "
            f"{counts['val']:>7} "
            f"{counts['test']:>7} "
            f"{total:>7}"
        )
        for s in grand:
            grand[s] += counts[s]

    print("=" * len(header))
    grand_total = sum(grand.values())
    print(
        f"{'TOTAL':<{col_w}} "
        f"{grand['train']:>7} "
        f"{grand['val']:>7} "
        f"{grand['test']:>7} "
        f"{grand_total:>7}"
    )
    print("=" * len(header))
    print(f"\nData directories:")
    print(f"  Train : {TRAIN_DIR}")
    print(f"  Val   : {VAL_DIR}")
    print(f"  Test  : {TEST_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def find_source_dir() -> Path:
    """
    Tries to locate the PlantVillage class folder automatically.
    Checks several common locations the manual download might produce.
    Returns the Path if found, else None.
    """
    candidates = [
        Path(RAW_DIR) / KAGGLE_INNER_FOLDER,           # data/raw/PlantVillage
        Path(RAW_DIR) / "plantvillage dataset" / "color",  # some zip variants
        Path(RAW_DIR) / "plant_disease" / "PlantVillage",
        Path(DATA_DIR) / KAGGLE_INNER_FOLDER,           # data/PlantVillage
        Path(DATA_DIR) / "PlantVillage",
    ]
    # Also check any single sub-folder directly inside raw/
    raw_path = Path(RAW_DIR)
    if raw_path.exists():
        for sub in raw_path.iterdir():
            if sub.is_dir() and sub not in candidates:
                candidates.append(sub)

    for c in candidates:
        if c.exists() and any(c.iterdir()):
            # Verify it actually contains class sub-directories with images
            subdirs = [d for d in c.iterdir() if d.is_dir()]
            if len(subdirs) >= 5:   # sanity check: at least 5 classes present
                return c
    return None


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Download (or use manually placed) PlantVillage dataset and split it."
    )
    parser.add_argument(
        "--source_dir", type=str, default=None,
        help=(
            "Path to the folder that contains the per-class sub-folders "
            "(e.g. data/raw/PlantVillage). "
            "If omitted the script searches common locations automatically."
        ),
    )
    parser.add_argument(
        "--skip_download", action="store_true",
        help="Skip the Kaggle download step (use when dataset is placed manually).",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Plant Disease Dataset — Setup & Split")
    print("=" * 60 + "\n")

    # ── If the user pointed us to a source dir, use it directly ─────────────
    if args.source_dir:
        source_dir = Path(args.source_dir)
        if not source_dir.exists():
            print(f"[ERROR] --source_dir not found: {source_dir}")
            sys.exit(1)
        print(f"[Source] Using provided source dir: {source_dir}")

    # ── Check if dataset was placed manually ────────────────────────────────
    elif args.skip_download or (auto := find_source_dir()) is not None:
        if args.skip_download:
            source_dir = find_source_dir()
            if source_dir is None:
                print("[ERROR] --skip_download was set but no class folders were found.")
                print(f"        Place your PlantVillage folder inside: {RAW_DIR}")
                print(f"        Expected: {RAW_DIR}\\PlantVillage\\<ClassName>\\*.jpg")
                sys.exit(1)
        else:
            source_dir = auto
        print(f"[Source] Found manually placed dataset: {source_dir}")

    # ── Otherwise try Kaggle download ────────────────────────────────────────
    else:
        zip_path   = download_dataset()
        source_dir = unzip_dataset(zip_path)

    # ③ Split
    stats = split_dataset(source_dir)

    # ④ Summary
    print_summary(stats)

    print("\n[Done] Dataset is ready. You can now run:")
    print("       python src/train.py\n")


if __name__ == "__main__":
    main()
