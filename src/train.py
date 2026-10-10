"""
train.py
--------
Training pipeline for BNPlantDiseaseNet — Plant Disease Detection.

Run from project root:
    python src/train.py

Everything is automatic — GPU is used if available, otherwise CPU.
No arguments needed. All settings come from config.py.
"""

import os, sys, time, csv, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

sys.path.insert(0, os.path.dirname(__file__))
from dataset import get_dataloader, get_class_names, dataset_summary
from model   import build_model
from config  import (
    DATA_DIR, OUTPUT_DIR, CKPT_DIR,
    NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    NUM_WORKERS, PATIENCE, LABEL_SMOOTH, GRAD_CLIP,
    LR_T0, LR_TMULT, LR_MIN,
    BEST_CKPT, LAST_CKPT, LOG_CSV, HISTORY_JSON, REPORT_TXT, CM_NPY,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Auto-detect GPU. Falls back to CPU gracefully."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  Device     : GPU — {name}  ({mem:.1f} GB VRAM)")
    else:
        dev = torch.device("cpu")
        print(f"  Device     : CPU  (no CUDA GPU found)")
    return dev


def top1_acc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    _, preds = torch.max(logits, 1)
    return (preds == labels).float().mean().item()


class EarlyStopping:
    def __init__(self, patience: int = 5):
        self.patience   = patience
        self.best       = 0.0
        self.counter    = 0

    def __call__(self, val_acc: float) -> bool:
        if val_acc > self.best + 1e-4:
            self.best    = val_acc
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ─────────────────────────────────────────────────────────────────────────────
# One epoch
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, training=True):
    model.train() if training else model.eval()
    total_loss = total_acc = n = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for i, (images, labels) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss   = criterion(logits, labels)

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            total_loss += loss.item()
            total_acc  += top1_acc(logits, labels)
            n          += 1

            # Live progress every 50 batches
            if training and i % 50 == 0:
                print(f"    batch {i:>4}/{len(loader)}  "
                      f"loss={total_loss/n:.4f}  "
                      f"acc={total_acc/n*100:.1f}%", end="\r")

    if training:
        print(" " * 60, end="\r")   # clear the progress line

    return total_loss / n, total_acc / n


# ─────────────────────────────────────────────────────────────────────────────
# Final evaluation on test set
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, loader, criterion, device, class_names):
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        accuracy_score, precision_score, recall_score, f1_score,
    )
    import numpy as np

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            _, preds = torch.max(model(images), 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(y_true, y_pred,
                                   target_names=class_names, zero_division=0)
    cm     = confusion_matrix(y_true, y_pred)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(REPORT_TXT, "w") as f:
        f.write(report)
    np.save(CM_NPY, cm)

    return acc, prec, rec, f1, report


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train():
    # ── Banner ───────────────────────────────────────────────────────────────
    print("\n" + "="*58)
    print("   Plant Disease Detection — BNPlantDiseaseNet Training")
    print("="*58)

    device = get_device()

    # ── Data ─────────────────────────────────────────────────────────────────
    print(f"  Data dir   : {DATA_DIR}")
    dataset_summary(DATA_DIR)

    class_names  = get_class_names(DATA_DIR)
    num_classes  = len(class_names)
    print(f"  Classes    : {num_classes}")
    print(f"  Epochs     : {NUM_EPOCHS}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  LR         : {LEARNING_RATE}")
    print("="*58 + "\n")

    train_loader = get_dataloader(DATA_DIR, "train", BATCH_SIZE, NUM_WORKERS)
    val_loader   = get_dataloader(DATA_DIR, "val",   BATCH_SIZE, NUM_WORKERS)

    # ── Model ────────────────────────────────────────────────────────────────
    model     = build_model(num_classes, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = optim.AdamW(model.parameters(),
                             lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer,
                                            T_0=LR_T0, T_mult=LR_TMULT,
                                            eta_min=LR_MIN)

    # ── Output dirs ──────────────────────────────────────────────────────────
    os.makedirs(CKPT_DIR,   exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── CSV log ──────────────────────────────────────────────────────────────
    csv_f  = open(LOG_CSV, "w", newline="")
    csv_w  = csv.writer(csv_f)
    csv_w.writerow(["epoch","train_loss","train_acc","val_loss","val_acc","lr"])

    # ── Training loop ────────────────────────────────────────────────────────
    stopper      = EarlyStopping(patience=PATIENCE)
    best_val_acc = 0.0
    history      = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}

    print("  Starting training...\n")
    total_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                     optimizer, device, training=True)
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion,
                                     optimizer, device, training=False)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        # Epoch summary line
        saved = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "optim_state": optimizer.state_dict(),
                        "val_acc": vl_acc, "num_classes": num_classes,
                        "class_names": class_names}, BEST_CKPT)
            saved = "  ← best saved"

        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "val_acc": vl_acc, "num_classes": num_classes,
                    "class_names": class_names}, LAST_CKPT)

        print(f"  Epoch {epoch:>2}/{NUM_EPOCHS}  |  "
              f"Train  loss={tr_loss:.4f}  acc={tr_acc*100:5.1f}%  |  "
              f"Val  loss={vl_loss:.4f}  acc={vl_acc*100:5.1f}%  |  "
              f"LR={lr:.1e}  ({elapsed:.0f}s){saved}")

        # Log
        csv_w.writerow([epoch, tr_loss, tr_acc, vl_loss, vl_acc, lr])
        csv_f.flush()
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        if stopper(vl_acc):
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(best val acc = {best_val_acc*100:.1f}%)")
            break

    csv_f.close()
    total_time = (time.time() - total_start) / 60
    print(f"\n  Training done in {total_time:.1f} min  "
          f"|  Best val acc = {best_val_acc*100:.2f}%")

    # Save history JSON
    with open(HISTORY_JSON, "w") as f:
        json.dump(history, f, indent=2)

    # ── Final test evaluation ─────────────────────────────────────────────
    print("\n  Running final evaluation on test set...")
    ckpt = torch.load(BEST_CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_loader = get_dataloader(DATA_DIR, "test", BATCH_SIZE, NUM_WORKERS)
    acc, prec, rec, f1, report = evaluate(model, test_loader, criterion,
                                          device, class_names)

    print("\n" + "="*58)
    print("  FINAL TEST RESULTS")
    print("="*58)
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print("="*58)
    print(f"\n  Checkpoint : {BEST_CKPT}")
    print(f"  Report     : {REPORT_TXT}")
    print("\n  NEXT STEP  →  python src/visualize.py")
    print("  THEN       →  python app.py\n")


if __name__ == "__main__":
    # Windows multiprocessing guard (required for num_workers > 0)
    import multiprocessing
    multiprocessing.freeze_support()
    train()
