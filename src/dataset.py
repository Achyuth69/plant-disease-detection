"""
dataset.py
----------
Dataset loading and preprocessing pipeline for Plant Disease Detection.

All paths and hyperparameters are pulled from config.py so there is a
single source of truth for the entire project.

Expected directory layout (created by kaggle_download.py):
    data/
        train/ <ClassName>/ img.jpg ...
        val/   <ClassName>/ img.jpg ...
        test/  <ClassName>/ img.jpg ...
"""

import os
import sys
from typing import Tuple

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS,
    IMAGENET_MEAN, IMAGENET_STD,
    TRAIN_DIR, VAL_DIR, TEST_DIR,
)


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(split: str = "train") -> transforms.Compose:
    """
    Returns the torchvision transform pipeline for the requested split.

    Training uses heavy augmentation to improve generalisation.
    Val / Test use a deterministic center-crop only.

    Args:
        split: 'train', 'val', or 'test'

    Returns:
        torchvision.transforms.Compose
    """
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3,
                saturation=0.2, hue=0.1
            ),
            transforms.RandomRotation(degrees=30),
            transforms.RandomGrayscale(p=0.05),        # minor robustness boost
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloader(
    data_dir: str,
    split: str = "train",
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    shuffle: bool = None,
    balance_classes: bool = False,
) -> DataLoader:
    """
    Builds an ImageFolder dataset and wraps it in a DataLoader.

    Args:
        data_dir        : root directory containing train/val/test sub-folders.
        split           : 'train', 'val', or 'test'.
        batch_size      : images per batch.
        num_workers     : parallel data-loading workers.
        shuffle         : if None, True for train / False for val+test.
        balance_classes : if True (train only), uses WeightedRandomSampler to
                          oversample minority classes.

    Returns:
        A PyTorch DataLoader.
    """
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"Split directory not found: {split_dir}\n"
            f"Run  python src/kaggle_download.py  first."
        )

    dataset = datasets.ImageFolder(
        root=split_dir,
        transform=get_transforms(split),
    )

    if shuffle is None:
        shuffle = (split == "train")

    sampler = None
    if balance_classes and split == "train":
        # Compute per-sample weights inversely proportional to class frequency
        class_counts = np.bincount(
            [label for _, label in dataset.samples]
        )
        class_weights = 1.0 / class_counts
        sample_weights = torch.tensor(
            [class_weights[label] for _, label in dataset.samples],
            dtype=torch.float,
        )
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False   # mutually exclusive with sampler

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),  # avoid tiny last batch breaking BN
    )

    return loader


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_class_names(data_dir: str) -> list:
    """
    Returns sorted list of class names from the train sub-folder.

    Args:
        data_dir: root dataset directory.

    Returns:
        List of class-name strings.
    """
    train_dir = os.path.join(data_dir, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"train directory not found at {train_dir}.\n"
            f"Run  python src/kaggle_download.py  first."
        )
    dataset = datasets.ImageFolder(root=train_dir)
    return dataset.classes


def dataset_summary(data_dir: str) -> None:
    """
    Prints a formatted summary table:
        split | # images | # classes
    """
    print(f"\n{'Split':<8} {'Images':>8} {'Classes':>8}")
    print("-" * 28)
    for split in ("train", "val", "test"):
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            print(f"{'[' + split + ']':<8} {'NOT FOUND':>8}")
            continue
        ds = datasets.ImageFolder(root=split_dir)
        print(f"{split:<8} {len(ds):>8,} {len(ds.classes):>8}")
    print()


def get_split_stats(data_dir: str) -> dict:
    """
    Returns a nested dict:
        { split: { class_name: count } }
    Useful for plotting class distributions.
    """
    stats = {}
    for split in ("train", "val", "test"):
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            continue
        ds = datasets.ImageFolder(root=split_dir)
        counts = {}
        for cls in ds.classes:
            counts[cls] = 0
        for _, label in ds.samples:
            counts[ds.classes[label]] += 1
        stats[split] = counts
    return stats


def get_mean_std(data_dir: str, split: str = "train") -> Tuple[list, list]:
    """
    Computes the per-channel mean and std of a split on the fly.
    Useful if you want dataset-specific normalization instead of ImageNet stats.

    Returns:
        (mean, std)  — each a list of 3 floats for [R, G, B]
    """
    # Use no-normalize transform for this computation
    plain_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    split_dir = os.path.join(data_dir, split)
    ds     = datasets.ImageFolder(root=split_dir, transform=plain_tf)
    loader = DataLoader(ds, batch_size=64, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=False)

    mean   = torch.zeros(3)
    sq_sum = torch.zeros(3)
    n      = 0

    for imgs, _ in loader:
        # imgs: (B, 3, H, W)
        mean   += imgs.mean(dim=[0, 2, 3]) * imgs.size(0)
        sq_sum += (imgs ** 2).mean(dim=[0, 2, 3]) * imgs.size(0)
        n      += imgs.size(0)

    mean = mean / n
    std  = (sq_sum / n - mean ** 2).sqrt()
    return mean.tolist(), std.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from config import DATA_DIR
    print("Dataset summary:")
    dataset_summary(DATA_DIR)

    classes = get_class_names(DATA_DIR)
    print(f"Classes ({len(classes)}):")
    for i, c in enumerate(classes):
        print(f"  {i:>3}: {c}")

    print("\nPer-split per-class counts:")
    stats = get_split_stats(DATA_DIR)
    for split, counts in stats.items():
        total = sum(counts.values())
        print(f"  [{split}] total={total:,}")
