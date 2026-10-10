"""
predict.py
----------
Inference script for BNPlantDiseaseNet.

Supports:
  - Single image prediction with confidence scores
  - Batch prediction over a folder of images
  - Top-K predictions per image

Usage
-----
Single image:
    python src/predict.py --checkpoint outputs/checkpoints/best_model.pth \
                          --image path/to/leaf.jpg

Folder of images:
    python src/predict.py --checkpoint outputs/checkpoints/best_model.pth \
                          --image_dir path/to/images/ \
                          --output_csv outputs/predictions.csv

Show top-3 predictions:
    python src/predict.py --checkpoint outputs/checkpoints/best_model.pth \
                          --image path/to/leaf.jpg --topk 3
"""

import os
import sys
import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from model import BNPlantDiseaseNet
from config import BEST_CKPT, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple:
    """
    Loads the BNPlantDiseaseNet from a saved checkpoint.

    Args:
        checkpoint_path: path to .pth checkpoint saved by train.py.
        device         : target inference device.

    Returns:
        (model, class_names)
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    num_classes  = ckpt["num_classes"]
    class_names  = ckpt["class_names"]

    model = BNPlantDiseaseNet(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    print(f"[Predict] Loaded checkpoint: {checkpoint_path}")
    print(f"          Classes: {num_classes}  |  Val acc at save: "
          f"{ckpt.get('val_acc', 0)*100:.2f}%")
    return model, class_names


# ─────────────────────────────────────────────────────────────────────────────
# Core prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_image(
    model:        torch.nn.Module,
    image_path:   str,
    class_names:  list,
    device:       torch.device,
    topk:         int = 1,
) -> list:
    """
    Predicts the disease class for a single image.

    Args:
        model      : loaded BNPlantDiseaseNet in eval mode.
        image_path : path to the leaf image.
        class_names: list of class name strings.
        device     : inference device.
        topk       : how many top predictions to return.

    Returns:
        List of (class_name, probability) tuples, sorted by probability desc.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Cannot open image '{image_path}': {e}")

    tensor = INFERENCE_TRANSFORM(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    with torch.no_grad():
        logits = model(tensor)                             # (1, num_classes)
        probs  = F.softmax(logits, dim=1).squeeze(0)      # (num_classes,)

    topk = min(topk, len(class_names))
    top_probs, top_indices = torch.topk(probs, k=topk)

    results = [
        (class_names[idx.item()], top_probs[i].item())
        for i, idx in enumerate(top_indices)
    ]
    return results


def predict_folder(
    model:       torch.nn.Module,
    image_dir:   str,
    class_names: list,
    device:      torch.device,
    topk:        int = 1,
    output_csv:  str = None,
) -> list:
    """
    Runs prediction on all supported images in a directory.

    Args:
        model      : loaded model.
        image_dir  : folder containing leaf images.
        class_names: class names list.
        device     : inference device.
        topk       : top-k predictions per image.
        output_csv : optional path to save results as CSV.

    Returns:
        List of dicts with keys: filename, top1_class, top1_prob, ...
    """
    image_paths = [
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    ]
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {image_dir}")

    print(f"[Predict] Found {len(image_paths)} images in '{image_dir}'")

    all_results = []

    for img_path in sorted(image_paths):
        try:
            preds = predict_image(model, str(img_path), class_names, device, topk)
            row = {"filename": img_path.name}
            for rank, (cls, prob) in enumerate(preds, start=1):
                row[f"top{rank}_class"] = cls
                row[f"top{rank}_prob"]  = f"{prob*100:.2f}%"
            all_results.append(row)

            # Console output
            pred_str = "  |  ".join(
                f"{cls} ({prob*100:.1f}%)" for cls, prob in preds
            )
            print(f"  {img_path.name:<40} → {pred_str}")

        except ValueError as e:
            print(f"  [WARN] Skipping {img_path.name}: {e}")

    # ── Save CSV if requested ─────────────────────────────────────────────
    if output_csv and all_results:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        fieldnames = list(all_results[0].keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n[Predict] Results saved to: {output_csv}")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plant Disease Inference using BNPlantDiseaseNet"
    )
    p.add_argument("--checkpoint", type=str, default=BEST_CKPT,
                   help="Path to .pth checkpoint (default: outputs/checkpoints/best_model.pth)")
    p.add_argument("--image",      type=str, default=None,
                   help="Path to a single leaf image for prediction")
    p.add_argument("--image_dir",  type=str, default=None,
                   help="Path to a folder of leaf images for batch prediction")
    p.add_argument("--output_csv", type=str, default=None,
                   help="(batch mode) Save predictions to this CSV file")
    p.add_argument("--topk",       type=int, default=3,
                   help="Number of top predictions to show (default: 3)")
    p.add_argument("--cpu",        action="store_true",
                   help="Force CPU inference even if CUDA is available")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = torch.device(
        "cuda" if (not args.cpu and torch.cuda.is_available()) else "cpu"
    )
    print(f"[Predict] Device: {device}\n")

    model, class_names = load_model(args.checkpoint, device)

    if args.image:
        # ── Single image mode ──────────────────────────────────────────────
        preds = predict_image(model, args.image, class_names, device, topk=args.topk)
        print(f"\nImage: {args.image}")
        print(f"{'Rank':<6} {'Class':<45} {'Confidence':>10}")
        print("-" * 65)
        for rank, (cls, prob) in enumerate(preds, start=1):
            print(f"  {rank:<4} {cls:<45} {prob*100:>9.2f}%")

    elif args.image_dir:
        # ── Batch / folder mode ────────────────────────────────────────────
        predict_folder(
            model, args.image_dir, class_names, device,
            topk=args.topk, output_csv=args.output_csv
        )

    else:
        print("[Error] Provide either --image or --image_dir.")
        sys.exit(1)
