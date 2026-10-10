"""
visualize.py
------------
Results visualisation for Plant Disease Detection.

Generates:
  1. Training curves     — loss and accuracy over epochs
  2. Confusion matrix    — heatmap from saved confusion_matrix.npy
  3. Per-class F1 bar    — bar chart from classification_report.txt
  4. Sample predictions  — grid of sample images with predicted/true labels
  5. Class distribution  — bar chart of images per class in each split

Usage
-----
After training:
    python src/visualize.py --output_dir outputs --data_dir data

Individual plots can be disabled with flags (see --help).
"""

import os
import sys
import argparse
import json
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless / no GUI required
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, save_dir: str) -> None:
    """Plots train/val loss and accuracy curves side by side."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History — BNPlantDiseaseNet", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#e74c3c", linewidth=2)
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="#3498db", linewidth=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Accuracy
    train_acc_pct = [a * 100 for a in history["train_acc"]]
    val_acc_pct   = [a * 100 for a in history["val_acc"]]
    axes[1].plot(epochs, train_acc_pct, label="Train Acc", color="#e74c3c", linewidth=2)
    axes[1].plot(epochs, val_acc_pct,   label="Val Acc",   color="#3498db", linewidth=2)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    best_epoch = int(np.argmax(history["val_acc"])) + 1
    best_acc   = max(history["val_acc"]) * 100
    axes[1].axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7,
                    label=f"Best ({best_acc:.1f}% @ ep{best_epoch})")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Training curves saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Confusion matrix heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm_path:     str,
    class_names: list,
    save_dir:    str,
    normalize:   bool = True,
) -> None:
    """Plots the confusion matrix as a seaborn heatmap."""
    cm = np.load(cm_path)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1           # avoid divide-by-zero
        cm_plot   = cm.astype(float) / row_sums
        fmt, vmax = ".2f", 1.0
        title     = "Confusion Matrix (Normalised)"
    else:
        cm_plot, fmt, vmax = cm, "d", cm.max()
        title               = "Confusion Matrix (Counts)"

    n = len(class_names)
    fig_size = max(12, n // 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    sns.heatmap(
        cm_plot, annot=(n <= 40), fmt=fmt,
        xticklabels=class_names, yticklabels=class_names,
        cmap="Blues", vmin=0, vmax=vmax, ax=ax,
        linewidths=0.5 if n <= 20 else 0.0,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title(title,              fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=max(6, 10 - n//10))
    plt.yticks(rotation=0,             fontsize=max(6, 10 - n//10))

    plt.tight_layout()
    fname = "confusion_matrix_norm.png" if normalize else "confusion_matrix.png"
    path  = os.path.join(save_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Confusion matrix saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Per-class F1 bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_class_f1(report_path: str, save_dir: str) -> None:
    """
    Parses outputs/classification_report.txt and draws a horizontal bar chart
    of per-class F1 scores.
    """
    if not os.path.isfile(report_path):
        print(f"  [Skip] {report_path} not found — skipping F1 chart.")
        return

    class_names, f1_scores = [], []
    with open(report_path) as f:
        for line in f:
            # Lines like:  "Apple___Apple_scab    0.93  0.91  0.92   630"
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    f1 = float(parts[-2])   # f1-score column
                    support = int(parts[-1])
                    if support > 0 and parts[0] not in ("accuracy", "macro", "weighted"):
                        class_name = " ".join(parts[:-4]).replace("___", " / ")
                        class_names.append(class_name)
                        f1_scores.append(f1)
                except ValueError:
                    continue

    if not class_names:
        print("  [Skip] Could not parse classification report — skipping F1 chart.")
        return

    order   = np.argsort(f1_scores)
    names_s = [class_names[i] for i in order]
    f1_s    = [f1_scores[i]   for i in order]
    colours = ["#e74c3c" if v < 0.80 else "#f39c12" if v < 0.90 else "#27ae60"
               for v in f1_s]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names_s) * 0.35)))
    bars = ax.barh(names_s, f1_s, color=colours, edgecolor="white", height=0.7)
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.90, color="gray", linestyle="--", alpha=0.5, label="0.90 threshold")
    ax.set_xlabel("F1-Score", fontsize=12)
    ax.set_title("Per-Class F1 Scores", fontsize=14, fontweight="bold")
    ax.legend()

    for bar, val in zip(bars, f1_s):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, "per_class_f1.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Per-class F1 chart saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Class distribution bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(data_dir: str, save_dir: str) -> None:
    """
    Counts images per class for train/val/test and saves a grouped bar chart.
    """
    from torchvision.datasets import ImageFolder

    splits_data = {}
    all_classes = None

    for split in ("train", "val", "test"):
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            continue
        ds = ImageFolder(root=split_dir)
        if all_classes is None:
            all_classes = ds.classes
        counts = [0] * len(ds.classes)
        for _, label in ds.samples:
            counts[label] += 1
        splits_data[split] = counts

    if not splits_data or all_classes is None:
        print("  [Skip] No valid split found — skipping class distribution.")
        return

    x      = np.arange(len(all_classes))
    width  = 0.25
    colors = {"train": "#3498db", "val": "#e67e22", "test": "#2ecc71"}

    fig, ax = plt.subplots(figsize=(max(14, len(all_classes) * 0.6), 6))
    for i, (split, counts) in enumerate(splits_data.items()):
        offset = (i - len(splits_data) / 2 + 0.5) * width
        ax.bar(x + offset, counts, width, label=split.capitalize(),
               color=colors.get(split, "gray"), alpha=0.85)

    tick_labels = [c.replace("___", "\n").replace("_", " ") for c in all_classes]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right",
                       fontsize=max(5, 9 - len(all_classes) // 10))
    ax.set_ylabel("Number of Images")
    ax.set_title("Class Distribution per Split", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Class distribution saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Sample prediction grid
# ─────────────────────────────────────────────────────────────────────────────

def plot_sample_predictions(
    model_path:  str,
    data_dir:    str,
    save_dir:    str,
    split:       str = "test",
    n_samples:   int = 16,
    device_str:  str = "cpu",
) -> None:
    """
    Loads the best model and visualises a grid of sample images with
    true vs. predicted labels highlighted in green (correct) or red (wrong).
    """
    import torch
    from model import BNPlantDiseaseNet
    from dataset import get_dataloader

    if not os.path.isfile(model_path):
        print(f"  [Skip] Checkpoint not found: {model_path} — skipping sample grid.")
        return

    device = torch.device(device_str)
    ckpt   = torch.load(model_path, map_location=device)
    model  = BNPlantDiseaseNet(num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    class_names = ckpt["class_names"]

    loader = get_dataloader(data_dir, split=split, batch_size=n_samples,
                            num_workers=0, shuffle=True)
    images, labels = next(iter(loader))
    images, labels = images[:n_samples].to(device), labels[:n_samples]

    with torch.no_grad():
        _, preds = torch.max(model(images), dim=1)
    preds  = preds.cpu()
    images = images.cpu()

    # Un-normalise for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = (images * std + mean).clamp(0, 1)

    cols = 4
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten()
    fig.suptitle("Sample Predictions (Green=Correct, Red=Wrong)",
                 fontsize=13, fontweight="bold", y=1.01)

    for i in range(n_samples):
        ax  = axes[i]
        img = images[i].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis("off")

        true_lbl = class_names[labels[i].item()].replace("___", "\n").replace("_", " ")
        pred_lbl = class_names[preds[i].item()].replace("___", "\n").replace("_", " ")
        colour   = "green" if labels[i] == preds[i] else "red"
        ax.set_title(f"T: {true_lbl}\nP: {pred_lbl}", fontsize=7, color=colour)

    for j in range(n_samples, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    path = os.path.join(save_dir, "sample_predictions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Sample predictions saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualise training results")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--data_dir",   type=str, default="data")
    p.add_argument("--no_curves",  action="store_true")
    p.add_argument("--no_cm",      action="store_true")
    p.add_argument("--no_f1",      action="store_true")
    p.add_argument("--no_dist",    action="store_true")
    p.add_argument("--no_samples", action="store_true")
    p.add_argument("--cpu",        action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(project_root, args.output_dir)
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.join(project_root, args.data_dir)

    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Generating visualisations → {plots_dir}")
    print(f"{'='*55}\n")

    # 1. Training curves
    if not args.no_curves:
        history_path = os.path.join(args.output_dir, "training_history.json")
        if os.path.isfile(history_path):
            with open(history_path) as f:
                history = json.load(f)
            plot_training_curves(history, plots_dir)
        else:
            print(f"  [Skip] {history_path} not found — run training first.")

    # 2. Confusion matrix
    if not args.no_cm:
        cm_path = os.path.join(args.output_dir, "confusion_matrix.npy")
        if os.path.isfile(cm_path):
            # Load class names from checkpoint if available
            ckpt_path = os.path.join(args.output_dir, "checkpoints", "best_model.pth")
            import torch
            if os.path.isfile(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location="cpu")
                class_names = ckpt["class_names"]
            else:
                n = np.load(cm_path).shape[0]
                class_names = [str(i) for i in range(n)]
            plot_confusion_matrix(cm_path, class_names, plots_dir, normalize=True)
            plot_confusion_matrix(cm_path, class_names, plots_dir, normalize=False)
        else:
            print(f"  [Skip] {cm_path} not found — run training first.")

    # 3. Per-class F1
    if not args.no_f1:
        report_path = os.path.join(args.output_dir, "classification_report.txt")
        plot_per_class_f1(report_path, plots_dir)

    # 4. Class distribution
    if not args.no_dist:
        plot_class_distribution(args.data_dir, plots_dir)

    # 5. Sample predictions
    if not args.no_samples:
        ckpt_path   = os.path.join(args.output_dir, "checkpoints", "best_model.pth")
        device_str  = "cpu" if args.cpu else (
            "cuda" if __import__("torch").cuda.is_available() else "cpu"
        )
        plot_sample_predictions(
            ckpt_path, args.data_dir, plots_dir,
            split="test", n_samples=16, device_str=device_str
        )

    print(f"\n  All plots saved to: {plots_dir}\n")
