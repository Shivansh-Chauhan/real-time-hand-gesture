#!/usr/bin/env python3
"""
train_model.py
──────────────
Full training pipeline:
  1. Load & preprocess dataset
  2. Train neural network with early stopping
  3. Evaluate on held-out test set
  4. Save model + label encoder + scaler
  5. Plot training curves & confusion matrix

Usage
─────
  python train_model.py
  python train_model.py --augment --augment-factor 5
  python train_model.py --no-augment --epochs 50
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    MODEL_CFG, LOGS_DIR, MODEL_DIR,
    BEST_MODEL_PATH, HISTORY_PATH,
)
from src.dataset_manager import DatasetManager
from src.gesture_model   import GestureModelTrainer


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Hand Gesture Classifier")
    p.add_argument("--augment",          dest="augment",   action="store_true",  default=True)
    p.add_argument("--no-augment",       dest="augment",   action="store_false")
    p.add_argument("--augment-factor",   type=int, default=3)
    p.add_argument("--epochs",           type=int, default=MODEL_CFG["epochs"])
    p.add_argument("--batch-size",       type=int, default=MODEL_CFG["batch_size"])
    p.add_argument("--learning-rate",    type=float, default=MODEL_CFG["learning_rate"])
    p.add_argument("--plot",             action="store_true", default=True,
                   help="Save training plots (default: True)")
    return p.parse_args()


# ── plot helpers ──────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, save_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    # Accuracy
    ax = axes[0]
    ax.plot(history["accuracy"],     label="Train Acc",  linewidth=2)
    ax.plot(history["val_accuracy"], label="Val Acc",    linewidth=2, linestyle="--")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)

    # Loss
    ax = axes[1]
    ax.plot(history["loss"],     label="Train Loss", linewidth=2)
    ax.plot(history["val_loss"], label="Val Loss",   linewidth=2, linestyle="--")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = save_dir / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Training curves → {out}")


def plot_confusion_matrix(
    metrics: dict,
    save_dir: Path,
    top_n: int = 30,
) -> None:
    cm          = np.array(metrics["confusion_matrix"])
    class_names = metrics["class_names"]

    # If many classes, show only top_n most confused
    if len(class_names) > top_n:
        # Find classes with highest off-diagonal errors
        errors = cm.sum(axis=1) - np.diag(cm)
        top_idx = np.argsort(errors)[-top_n:]
        cm = cm[np.ix_(top_idx, top_idx)]
        class_names = [class_names[i] for i in top_idx]

    # Normalise
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm / row_sums, 0)

    fig_size = max(10, len(class_names) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f",
        xticklabels=class_names, yticklabels=class_names,
        cmap="Blues", linewidths=0.4,
        ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title(
        f"Confusion Matrix (normalised) — Accuracy: {metrics['accuracy']*100:.2f}%",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()

    out = save_dir / "confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Confusion matrix → {out}")


def save_metrics_json(metrics: dict, save_dir: Path) -> None:
    # confusion_matrix is already a list (JSON-serialisable)
    out = save_dir / "eval_metrics.json"
    with open(out, "w") as f:
        json.dump({
            "accuracy"         : metrics["accuracy"],
            "classification_report": metrics["classification_report"],
            "class_names"      : metrics["class_names"],
        }, f, indent=2)
    print(f"  📄 Metrics JSON → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 60)
    print("  🧠  Hand Gesture Classifier — Training Pipeline")
    print("═" * 60)

    # 1. Load data
    print("\n[1/5] Loading dataset …")
    dm = DatasetManager()
    X, y = dm.load_raw(augment=args.augment, augment_factor=args.augment_factor)
    dm.print_stats(y)

    # 2. Prepare
    print("[2/5] Preparing data …")
    trainer = GestureModelTrainer()
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(X, y)

    num_classes = len(trainer.label_encoder.classes_)
    print(f"  Classes: {num_classes}")

    # 3. Train
    print(f"\n[3/5] Training model (epochs={args.epochs}) …\n")
    MODEL_CFG["epochs"]        = args.epochs
    MODEL_CFG["batch_size"]    = args.batch_size
    MODEL_CFG["learning_rate"] = args.learning_rate

    history = trainer.train(X_train, y_train, X_val, y_val, num_classes)

    # 4. Evaluate
    print("\n[4/5] Evaluating on test set …")
    metrics = trainer.evaluate(X_test, y_test)

    # 5. Save
    print("\n[5/5] Saving artefacts …")
    trainer.save()
    save_metrics_json(metrics, LOGS_DIR)

    if args.plot:
        print("\n  Generating plots …")
        plot_training_curves(trainer.history, LOGS_DIR)
        plot_confusion_matrix(metrics, LOGS_DIR)

    best_acc = max(history.history.get("val_accuracy", [0]))
    print(f"\n{'═'*60}")
    print(f"  ✅  Training complete!")
    print(f"  Best val accuracy : {best_acc*100:.2f}%")
    print(f"  Test accuracy     : {metrics['accuracy']*100:.2f}%")
    print(f"  Model saved to    : {BEST_MODEL_PATH}")
    print(f"{'═'*60}\n")
    print("  Next step → run:  python run_recognition.py")
    print("           or      streamlit run app.py\n")


if __name__ == "__main__":
    main()
