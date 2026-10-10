"""
run_all.py
----------
ONE command to complete the entire project pipeline:

    python run_all.py

What it does automatically:
  Step 1 — Split dataset  (data/raw/PlantVillage → train/val/test)
  Step 2 — Train model    (BNPlantDiseaseNet, auto GPU/CPU, ~15 epochs)
  Step 3 — Visualize      (plots saved to outputs/plots/)
  Step 4 — Launch app     (Gradio web app at http://127.0.0.1:7860)

Just run it and wait. Everything is handled.
"""

import os, sys, time, subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)


def banner(title: str, step: int, total: int = 4):
    print(f"\n{'='*58}")
    print(f"  STEP {step}/{total}  —  {title}")
    print(f"{'='*58}\n")


def check_data_split() -> bool:
    """Returns True if train/val/test already have class folders."""
    from config import TRAIN_DIR
    if not os.path.isdir(TRAIN_DIR):
        return False
    subdirs = [d for d in os.listdir(TRAIN_DIR)
               if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    return len(subdirs) >= 5


def find_raw_source() -> str:
    """Find the PlantVillage folder with the most class subdirs."""
    from config import PROJECT_ROOT
    candidates = [
        os.path.join(PROJECT_ROOT, "data", "raw", "PlantVillage", "PlantVillage"),
        os.path.join(PROJECT_ROOT, "data", "raw", "PlantVillage"),
        os.path.join(PROJECT_ROOT, "data", "raw"),
    ]
    best, best_n = None, 0
    for c in candidates:
        if os.path.isdir(c):
            n = len([x for x in os.listdir(c)
                     if os.path.isdir(os.path.join(c, x))])
            if n > best_n:
                best, best_n = c, n
    return best


def step1_split():
    banner("Splitting dataset  →  train / val / test", 1)

    if check_data_split():
        from config import TRAIN_DIR
        subdirs = [d for d in os.listdir(TRAIN_DIR)
                   if os.path.isdir(os.path.join(TRAIN_DIR, d))]
        print(f"  Data already split ({len(subdirs)} classes in train/).")
        print("  Skipping split step.\n")
        return

    src = find_raw_source()
    if src is None:
        print("  ERROR: No raw dataset found.")
        print("  Please place PlantVillage folder at:")
        print("    data/raw/PlantVillage/")
        sys.exit(1)

    print(f"  Source : {src}")

    # inline split (same logic as split_data.py)
    import shutil, random, stat
    from pathlib import Path
    from config import TRAIN_DIR, VAL_DIR, TEST_DIR, RANDOM_SEED

    SUPPORTED = {".jpg",".jpeg",".png",".bmp",".tiff",
                 ".JPG",".JPEG",".PNG",".BMP",".TIFF"}

    def force_rm(func, path, _):
        os.chmod(path, stat.S_IWRITE); func(path)

    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d, onerror=force_rm)
        os.makedirs(d)

    class_dirs = sorted([d for d in Path(src).iterdir() if d.is_dir()])
    print(f"  Found {len(class_dirs)} classes. Splitting 80/10/10 ...\n")
    random.seed(RANDOM_SEED)
    total = 0

    for i, cls_dir in enumerate(class_dirs, 1):
        imgs = [f for f in cls_dir.iterdir() if f.suffix in SUPPORTED]
        if not imgs:
            continue
        random.shuffle(imgs)
        n  = len(imgs)
        nt = int(n * 0.80)
        nv = int(n * 0.10)
        splits = {"train": (TRAIN_DIR, imgs[:nt]),
                  "val":   (VAL_DIR,   imgs[nt:nt+nv]),
                  "test":  (TEST_DIR,  imgs[nt+nv:])}
        for sname, (droot, slist) in splits.items():
            dest = os.path.join(droot, cls_dir.name)
            os.makedirs(dest, exist_ok=True)
            for img in slist:
                shutil.copy2(img, os.path.join(dest, img.name))
        total += n
        print(f"  [{i:>2}/{len(class_dirs)}] {cls_dir.name:<50} {n:>5} images")

    print(f"\n  Split complete — {total:,} total images.\n")


def step2_train():
    banner("Training BNPlantDiseaseNet", 2)
    from config import BEST_CKPT

    if os.path.isfile(BEST_CKPT):
        ans = input("  Trained model already exists. Re-train? [y/N]: ").strip().lower()
        if ans != "y":
            print("  Skipping training — using existing model.\n")
            return

    import train as tr
    tr.train()


def step3_visualize():
    banner("Generating result visualizations", 3)
    from config import HISTORY_JSON, BEST_CKPT, OUTPUT_DIR, PLOTS_DIR, DATA_DIR, CM_NPY, REPORT_TXT

    if not os.path.isfile(HISTORY_JSON):
        print("  No training history found — skipping visualizations.")
        print("  (Run training first)\n")
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # import visualize functions directly
    import json, numpy as np
    import matplotlib; matplotlib.use("Agg")

    # 1. Training curves
    try:
        import visualize as viz
        with open(HISTORY_JSON) as f:
            history = json.load(f)
        viz.plot_training_curves(history, PLOTS_DIR)

        if os.path.isfile(CM_NPY):
            import torch
            ckpt = torch.load(BEST_CKPT, map_location="cpu")
            viz.plot_confusion_matrix(CM_NPY, ckpt["class_names"], PLOTS_DIR, normalize=True)
            viz.plot_confusion_matrix(CM_NPY, ckpt["class_names"], PLOTS_DIR, normalize=False)

        if os.path.isfile(REPORT_TXT):
            viz.plot_per_class_f1(REPORT_TXT, PLOTS_DIR)

        viz.plot_class_distribution(DATA_DIR, PLOTS_DIR)

        if os.path.isfile(BEST_CKPT):
            device_str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
            viz.plot_sample_predictions(BEST_CKPT, DATA_DIR, PLOTS_DIR,
                                        split="test", n_samples=16,
                                        device_str=device_str)

        print(f"\n  All plots saved to: {PLOTS_DIR}\n")
    except Exception as e:
        print(f"  Visualization warning: {e}")
        print("  (Non-fatal — continuing)\n")


def step4_launch_app():
    banner("Launching Gradio web application", 4)
    from config import BEST_CKPT

    if not os.path.isfile(BEST_CKPT):
        print("  ERROR: No trained model found.")
        print(f"  Expected: {BEST_CKPT}")
        print("  Train the model first (Step 2).\n")
        return

    app_path = os.path.join(ROOT, "app.py")
    print("  Starting app...")
    print("  Open your browser at:  http://127.0.0.1:7860")
    print("  Press  Ctrl+C  to stop the app.\n")
    time.sleep(1)

    # Launch app.py in same process (blocks until Ctrl+C)
    import importlib.util
    spec = importlib.util.spec_from_file_location("app", app_path)
    app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_module)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    print("\n" + "="*58)
    print("  Plant Disease Detection — Full Pipeline")
    print("="*58)
    print("  This script runs all 4 steps automatically.")
    print("  Estimated time: 1–2 hrs on CPU, ~20 min on GPU")
    print("="*58)

    step1_split()
    step2_train()
    step3_visualize()
    step4_launch_app()
