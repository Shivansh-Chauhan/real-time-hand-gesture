#!/usr/bin/env python3
"""
quickstart.py
─────────────
One-command bootstrap: generates synthetic data, trains the model,
evaluates it, and prints a summary — all without a webcam.

Perfect for:
  • First-time setup in CI/CD
  • Demo environments without a camera
  • Sanity-checking the full pipeline

Usage
─────
  python quickstart.py                        # full run, 200 samples/class
  python quickstart.py --samples 100 --fast  # faster, lower accuracy
  python quickstart.py --samples 500          # higher accuracy, longer training
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _banner(msg: str, width: int = 60) -> None:
    print(f"\n{'═'*width}")
    print(f"  {msg}")
    print(f"{'═'*width}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gesture recognition quickstart")
    p.add_argument("--samples", type=int, default=200,
                   help="Synthetic samples per gesture class (default 200)")
    p.add_argument("--fast",    action="store_true",
                   help="Reduce epochs and samples for speed (overrides --samples to 80)")
    p.add_argument("--evaluate",action="store_true", default=True)
    p.add_argument("--no-eval", dest="evaluate", action="store_false")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    t0     = time.time()
    n_samp = 80 if args.fast else args.samples

    _banner("🤚  Gesture Recognition Quickstart")
    print(f"  Samples/class : {n_samp}")
    print(f"  Fast mode     : {args.fast}")
    print(f"  Seed          : {args.seed}")

    import numpy as np
    np.random.seed(args.seed)

    # ── Step 1: generate synthetic dataset ────────────────────────────────────
    _banner("Step 1/4 — Generating synthetic training data")
    from generate_synthetic_data import _build_pose_library, generate_samples
    from config import ALL_GESTURES, RAW_DIR
    import csv

    _build_pose_library()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    from tqdm import tqdm
    for gesture in tqdm(ALL_GESTURES, desc="  Synthesising", unit="class"):
        try:
            feats = generate_samples(gesture, n=n_samp)
        except KeyError:
            print(f"  ⚠️  No pose for '{gesture}', skipping")
            continue

        csv_path = RAW_DIR / f"{gesture}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            for feat in feats:
                writer.writerow([gesture] + feat.tolist())
        total += len(feats)

    print(f"\n  ✅ Generated {total:,} samples across {len(ALL_GESTURES)} classes")

    # ── Step 2: train model ───────────────────────────────────────────────────
    _banner("Step 2/4 — Training model")
    from config import MODEL_CFG, MODEL_DIR, LOGS_DIR
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if args.fast:
        MODEL_CFG["epochs"]       = 30
        MODEL_CFG["hidden_units"] = [256, 128, 64]
        MODEL_CFG["patience"]     = 8

    from src.dataset_manager import DatasetManager
    from src.gesture_model   import GestureModelTrainer

    dm = DatasetManager()
    X, y = dm.load_raw(augment=True, augment_factor=2)
    dm.print_stats(y)

    trainer = GestureModelTrainer()
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(X, y)
    num_classes = len(trainer.label_encoder.classes_)

    history = trainer.train(X_train, y_train, X_val, y_val, num_classes)
    metrics = trainer.evaluate(X_test, y_test)
    trainer.save()

    # ── Step 3: quick plot ────────────────────────────────────────────────────
    _banner("Step 3/4 — Saving training plots")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, key, val_key, title in [
            (axes[0], "accuracy", "val_accuracy", "Accuracy"),
            (axes[1], "loss",     "val_loss",     "Loss"),
        ]:
            h = history.history
            ax.plot(h[key],     label="Train", linewidth=2)
            ax.plot(h[val_key], label="Val",   linewidth=2, linestyle="--")
            ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)

        fig.suptitle("Training History — Quickstart Run", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plot_path = LOGS_DIR / "quickstart_training.png"
        plt.savefig(plot_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  ✅ Plot saved → {plot_path}")
    except Exception as e:
        print(f"  ⚠️  Plotting skipped: {e}")

    # ── Step 4: evaluate ──────────────────────────────────────────────────────
    if args.evaluate:
        _banner("Step 4/4 — Running evaluation benchmark")
        from src.gesture_model import GesturePredictor
        import time as _time

        predictor = GesturePredictor()

        # Speed test
        dummy = np.random.randn(88).astype(np.float32)
        for _ in range(5):
            predictor.predict(dummy)
        t_start = _time.perf_counter()
        for _ in range(200):
            predictor.predict(dummy)
        ips = 200 / (_time.perf_counter() - t_start)
        print(f"\n  ⚡ Inference speed : {ips:.0f} inf/s  ({1000/ips:.1f} ms)")

        # Per-category accuracy on test set
        categories = {
            "Alphabets": [c for c in trainer.label_encoder.classes_ if len(c) == 1 and c.isalpha()],
            "Numbers"  : [c for c in trainer.label_encoder.classes_ if c.isdigit()],
            "Commands" : [c for c in trainer.label_encoder.classes_ if len(c) > 1],
        }
        print()
        for cat_name, cat_classes in categories.items():
            if not cat_classes:
                continue
            idx  = [i for i, c in enumerate(trainer.label_encoder.classes_)
                    if c in cat_classes]
            mask = np.isin(y_test, idx)
            if mask.sum() == 0:
                continue
            y_p  = np.argmax(
                predictor.model.predict(
                    trainer.scaler.transform(X_test[mask]), verbose=0
                ), axis=1
            )
            acc  = (y_p == y_test[mask]).mean()
            print(f"  {cat_name:<12}: {acc*100:.1f}% accuracy")
    else:
        _banner("Step 4/4 — Skipped (--no-eval)")

    elapsed = time.time() - t0
    _banner("✅  Quickstart Complete")
    print(f"  Total time       : {elapsed:.0f}s")
    print(f"  Test accuracy    : {metrics['accuracy']*100:.2f}%")
    print(f"  Model saved to   : models/gesture_model_best.h5")
    print("""
  Next steps
  ──────────
  Real-time recognition  : python run_recognition.py
  Streamlit dashboard    : streamlit run app.py
  Deep evaluation        : python evaluate_model.py
  Mobile export          : python export_model.py
  Collect real data      : python collect_data.py --all
    """)


if __name__ == "__main__":
    main()
