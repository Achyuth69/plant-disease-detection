---
title: PlantGuard AI - Plant Disease Detection
emoji: 🌿
colorFrom: green
colorTo: teal
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: mit
short_description: Detect plant diseases from leaf photos using a BN-CNN
---

# 🌿 PlantGuard AI — Plant Disease Detection

Detect diseases in plant leaves instantly using a **Batch-Normalized CNN** trained on the [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) dataset.

## Supported Crops & Classes (15 total)

| Crop | Conditions |
|------|-----------|
| 🌶️ Bell Pepper | Bacterial Spot · Healthy |
| 🥔 Potato | Early Blight · Late Blight · Healthy |
| 🍅 Tomato | Bacterial Spot · Early Blight · Late Blight · Leaf Mold · Septoria Leaf Spot · Spider Mites · Target Spot · Yellow Leaf Curl Virus · Mosaic Virus · Healthy |

## How to Use

1. Upload a clear, close-up photo of a **single plant leaf**
2. Tap **Analyze Leaf**
3. Get instant diagnosis with severity, cause, symptoms, and treatment

## Model Details

- Architecture: 4-block Batch-Normalized CNN (~3.2M parameters)
- Input: 224×224 RGB images (ImageNet normalization)
- Training: PlantVillage dataset, 15 classes
- Confidence threshold: 40% (below this, image is flagged as unrecognizable)

## Files in this Space

```
app.py           ← Gradio application (self-contained)
requirements.txt ← Python dependencies
best_model.pth   ← Trained model checkpoint (~13 MB)
```

## Project

Deep Learning PBL — Automated Plant Disease Detection using Batch-Normalized CNNs
