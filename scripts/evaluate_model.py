#!/usr/bin/env python3
"""
evaluate_model.py
─────────────────
Deep evaluation of the trained model beyond the metrics produced during
training. Run this after `train_model.py` to get:

  1.  Per-class accuracy bar chart
  2.  Normalised confusion matrix heat-map
  3.  Most-confused gesture pairs
  4.  Confidence calibration curve (reliability diagram)
  5.  t-SNE embedding of the 88-D feature space
  6.  Per-category summary (alphabets / numbers / commands)
  7.  Speed benchmark (inferences/second)
  8.  Saved HTML + PNG report

Usage
─────
  python evaluate_model.py
  python evaluate_model.py --samples 1000   # generate 1000 test samples per class
  python evaluate_model.py --tsne           # include t-SNE plot (slow)
  python evaluate_model.py --output ./my_report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BEST_MODEL_PATH, LABEL_ENCODER_PATH, SCALER_PATH,
    LOGS_DIR, ALL_GESTURES, ALPHABET_GESTURES, NUMBER_GESTURES, COMMAND_GESTURES,
)
from generate_synthetic_data import _build_pose_library, generate_samples
from src.gesture_model import GesturePredictor


# ── plotting style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor" : "#0f172a",
    "axes.facecolor"   : "#1e293b",
    "axes.edgecolor"   : "#334155",
    "axes.labelcolor"  : "#94a3b8",
    "text.color"       : "#e2e8f0",
    "xtick.color"      : "#94a3b8",
    "ytick.color"      : "#94a3b8",
    "grid.color"       : "#334155",
    "grid.linestyle"   : "--",
    "grid.alpha"       : 0.4,
    "font.size"        : 9,
    "figure.dpi"       : 120,
})

CAT_COLORS = {"alphabets": "#38bdf8", "numbers": "#34d399", "commands": "#f472b6"}


# ═════════════════════════════════════════════════════════════════════════════
#  Data generation
# ═════════════════════════════════════════════════════════════════════════════

def _make_test_set(
    predictor: GesturePredictor,
    n_per_class: int = 300,
    noise_level: float = 1.2,    # slightly higher noise than training
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a held-out synthetic test set.

    Returns
    -------
    X_raw  : (N, 88) unscaled features
    X      : (N, 88) scaled features
    y_true : (N,)    integer labels (encoded)
    labels : list[str] in label-encoder order
    """
    _build_pose_library()
    classes   = predictor.label_encoder.classes_.tolist()
    X_list, y_list = [], []

    for cls in classes:
        try:
            feats = generate_samples(cls, n=n_per_class, noise_level=noise_level)
        except KeyError:
            feats = []

        for f in feats:
            X_list.append(f)
            y_list.append(cls)

    X_raw = np.array(X_list, dtype=np.float32)
    X     = predictor.scaler.transform(X_raw)
    y_enc = predictor.label_encoder.transform(y_list)

    return X_raw, X, y_enc, classes


# ═════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ═════════════════════════════════════════════════════════════════════════════

def _run_predictions(
    predictor: GesturePredictor,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_pred_int, confidence_array)."""
    probs   = predictor.model.predict(X, batch_size=256, verbose=0)
    y_pred  = np.argmax(probs, axis=1)
    confs   = probs.max(axis=1)
    return y_pred, confs


def _per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
) -> Dict[str, float]:
    result = {}
    for i, cls in enumerate(classes):
        mask = y_true == i
        if mask.sum() == 0:
            result[cls] = 0.0
        else:
            result[cls] = float((y_pred[mask] == i).mean())
    return result


def _category_of(label: str) -> str:
    if label in ALPHABET_GESTURES:
        return "alphabets"
    if label in NUMBER_GESTURES:
        return "numbers"
    return "commands"


def _confusion_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    top_n: int = 8,
) -> List[Tuple[str, str, int]]:
    """Return top-n most confused (true, predicted, count) pairs."""
    mistakes = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            mistakes.append((classes[y_true[i]], classes[y_pred[i]]))
    from collections import Counter
    return [(a, b, c) for (a, b), c in Counter(mistakes).most_common(top_n)]


def _speed_benchmark(
    predictor: GesturePredictor,
    n: int = 1000,
) -> float:
    """Return throughput in inferences/second."""
    feat = np.random.randn(88).astype(np.float32)
    # warm up
    for _ in range(10):
        predictor.predict(feat)
    start = time.perf_counter()
    for _ in range(n):
        predictor.predict(feat)
    elapsed = time.perf_counter() - start
    return n / elapsed


# ═════════════════════════════════════════════════════════════════════════════
#  Plots
# ═════════════════════════════════════════════════════════════════════════════

def _plot_per_class_accuracy(
    acc_dict: Dict[str, float],
    save_path: Path,
) -> None:
    classes = list(acc_dict.keys())
    accs    = [acc_dict[c] * 100 for c in classes]
    colors  = [CAT_COLORS[_category_of(c)] for c in classes]

    fig, ax = plt.subplots(figsize=(max(14, len(classes) * 0.35), 5))
    bars = ax.bar(classes, accs, color=colors, edgecolor="none", width=0.75)
    ax.axhline(np.mean(accs), color="#f472b6", linestyle="--", linewidth=1.2,
               label=f"Mean {np.mean(accs):.1f}%")
    ax.axhline(90, color="#64748b", linestyle=":", linewidth=0.8, alpha=0.6,
               label="90% threshold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-class Accuracy", fontsize=11, fontweight="bold", color="#e2e8f0")
    ax.tick_params(axis="x", rotation=60)
    ax.legend(fontsize=8)
    ax.grid(axis="y")

    # Colour legend
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=v, label=k.title()) for k, v in CAT_COLORS.items()]
    ax.legend(handles=legend_els + [
        plt.Line2D([0], [0], color="#f472b6", linestyle="--", label=f"Mean {np.mean(accs):.1f}%")
    ], fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  📊 Per-class accuracy → {save_path}")


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    save_path: Path,
) -> None:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm / row_sums, 0)

    n = len(classes)
    fs = max(8, n * 0.28)
    fig, ax = plt.subplots(figsize=(fs, fs * 0.9))

    sns.heatmap(
        cm_norm, ax=ax,
        xticklabels=classes, yticklabels=classes,
        cmap="Blues", vmin=0, vmax=1,
        linewidths=0.3, linecolor="#0f172a",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Confusion Matrix (row-normalised)", fontsize=10, color="#e2e8f0", pad=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right", fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  📊 Confusion matrix → {save_path}")


def _plot_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confs: np.ndarray,
    save_path: Path,
    n_bins: int = 10,
) -> None:
    bin_edges   = np.linspace(0, 1, n_bins + 1)
    bin_accs    = []
    bin_confs   = []
    bin_counts  = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            continue
        bin_accs.append((y_true[mask] == y_pred[mask]).mean())
        bin_confs.append(confs[mask].mean())
        bin_counts.append(mask.sum())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Reliability diagram
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.bar(bin_confs, bin_accs, width=0.07, alpha=0.7, color="#38bdf8",
           edgecolor="none", label="Model")
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Fraction correct")
    ax.set_title("Calibration (reliability diagram)", fontsize=10, color="#e2e8f0")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid()

    # Confidence distribution
    ax = axes[1]
    correct  = confs[y_true == y_pred]
    wrong    = confs[y_true != y_pred]
    ax.hist(correct, bins=25, alpha=0.7, color="#34d399", label="Correct",  density=True)
    ax.hist(wrong,   bins=25, alpha=0.7, color="#f87171", label="Incorrect", density=True)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Density")
    ax.set_title("Confidence Distribution", fontsize=10, color="#e2e8f0")
    ax.legend(fontsize=8)
    ax.grid()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  📊 Calibration curve → {save_path}")


def _plot_tsne(
    X_raw: np.ndarray,
    y_true: np.ndarray,
    classes: List[str],
    save_path: Path,
    max_samples: int = 3000,
) -> None:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    # Sub-sample if needed
    if len(X_raw) > max_samples:
        idx = np.random.choice(len(X_raw), max_samples, replace=False)
        X_raw = X_raw[idx]; y_true = y_true[idx]

    print(f"  Computing t-SNE on {len(X_raw)} samples…")
    # PCA first for speed
    X_pca = PCA(n_components=30, random_state=42).fit_transform(X_raw)
    tsne  = TSNE(n_components=2, perplexity=35, n_iter=750, random_state=42,
                 n_jobs=-1)
    emb   = tsne.fit_transform(X_pca)

    fig, ax = plt.subplots(figsize=(12, 10))
    palette = plt.cm.get_cmap("tab20", len(classes))
    for i, cls in enumerate(classes):
        mask = y_true == i
        if mask.sum() == 0:
            continue
        cat   = _category_of(cls)
        color = palette(i)
        ax.scatter(emb[mask, 0], emb[mask, 1], s=6, alpha=0.6,
                   color=color, label=cls)

    ax.set_title("t-SNE of 88-D feature space", fontsize=11, color="#e2e8f0")
    ax.axis("off")
    ax.legend(fontsize=5, ncol=4, loc="lower left",
              markerscale=2, framealpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 t-SNE plot → {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  HTML report
# ═════════════════════════════════════════════════════════════════════════════

def _write_html_report(
    metrics: Dict,
    save_path: Path,
    image_names: List[str],
) -> None:
    imgs_html = "\n".join(
        f'<img src="{n}" style="width:100%;border-radius:8px;margin:8px 0">'
        for n in image_names if (save_path.parent / n).exists()
    )
    confused_rows = "\n".join(
        f"<tr><td>{a}</td><td>{b}</td><td>{n}</td></tr>"
        for a, b, n in metrics.get("confused_pairs", [])
    )
    cat_rows = "\n".join(
        f"<tr><td>{k}</td><td>{v['n']}</td><td>{v['acc']*100:.1f}%</td>"
        f"<td>{v['mean_conf']*100:.1f}%</td></tr>"
        for k, v in metrics.get("category_summary", {}).items()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Gesture Model Evaluation Report</title>
<style>
  body{{background:#0f172a;color:#e2e8f0;font-family:system-ui,sans-serif;
        max-width:1100px;margin:0 auto;padding:2rem}}
  h1{{font-size:1.6rem;color:#7dd3fc;margin-bottom:0.25rem}}
  h2{{font-size:1.1rem;color:#94a3b8;font-weight:500;
      border-bottom:1px solid #334155;padding-bottom:0.4rem;margin-top:2rem}}
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:1rem 0}}
  .card{{background:#1e293b;border:1px solid #334155;border-radius:10px;
         padding:1rem;text-align:center}}
  .card .val{{font-size:1.8rem;font-weight:600;color:#38bdf8}}
  .card .lbl{{font-size:0.78rem;color:#64748b;margin-top:3px}}
  table{{width:100%;border-collapse:collapse;font-size:0.85rem}}
  th,td{{padding:6px 10px;border-bottom:1px solid #334155;text-align:left}}
  th{{color:#94a3b8;font-weight:500}}
  .img-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
</style>
</head>
<body>
<h1>🤚 Gesture Recognition — Evaluation Report</h1>
<p style="color:#64748b;font-size:0.85rem">Generated {time.strftime('%Y-%m-%d %H:%M:%S')}</p>

<div class="cards">
  <div class="card"><div class="val">{metrics['overall_accuracy']*100:.2f}%</div><div class="lbl">Overall Accuracy</div></div>
  <div class="card"><div class="val">{metrics['n_classes']}</div><div class="lbl">Gesture Classes</div></div>
  <div class="card"><div class="val">{metrics['n_samples']:,}</div><div class="lbl">Test Samples</div></div>
  <div class="card"><div class="val">{metrics['speed_ips']:.0f}/s</div><div class="lbl">Inferences/sec</div></div>
  <div class="card"><div class="val">{1000/metrics['speed_ips']:.1f}ms</div><div class="lbl">Latency</div></div>
  <div class="card"><div class="val">{metrics['mean_confidence']*100:.1f}%</div><div class="lbl">Mean Confidence</div></div>
</div>

<h2>Category Summary</h2>
<table><tr><th>Category</th><th>Classes</th><th>Accuracy</th><th>Mean Confidence</th></tr>
{cat_rows}</table>

<h2>Most Confused Pairs</h2>
<table><tr><th>True</th><th>Predicted as</th><th>Count</th></tr>
{confused_rows}</table>

<h2>Visualisations</h2>
<div class="img-grid">{imgs_html}</div>
</body></html>"""

    with open(save_path, "w") as f:
        f.write(html)
    print(f"  📄 HTML report → {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deep model evaluation & reporting")
    p.add_argument("--samples", type=int, default=300,
                   help="Test samples per class (default 300)")
    p.add_argument("--tsne",    action="store_true",
                   help="Include t-SNE visualisation (slow for large sets)")
    p.add_argument("--output",  type=str, default=str(LOGS_DIR))
    p.add_argument("--seed",    type=int, default=99)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 60)
    print("  📊  Gesture Model — Deep Evaluation Suite")
    print("═" * 60)

    # 1. Load predictor
    print("\n[1/6] Loading model …")
    predictor = GesturePredictor()

    # 2. Generate test set
    print(f"[2/6] Generating {args.samples} test samples per class …")
    X_raw, X, y_true, classes = _make_test_set(predictor, n_per_class=args.samples)
    print(f"      Total test samples: {len(X_raw):,}")

    # 3. Run predictions
    print("[3/6] Running inference …")
    y_pred, confs = _run_predictions(predictor, X)

    overall_acc = float((y_pred == y_true).mean())
    print(f"      Overall accuracy: {overall_acc*100:.2f}%")

    # Per-class accuracy
    per_class = _per_class_accuracy(y_true, y_pred, classes)

    # Category summary
    cat_summary: Dict[str, Dict] = {}
    for cat_name, cat_gestures in [
        ("alphabets", ALPHABET_GESTURES),
        ("numbers",   NUMBER_GESTURES),
        ("commands",  COMMAND_GESTURES),
    ]:
        cat_idx   = [classes.index(g) for g in cat_gestures if g in classes]
        mask      = np.isin(y_true, cat_idx)
        if mask.sum() == 0:
            continue
        cat_summary[cat_name] = {
            "n"         : int(len([g for g in cat_gestures if g in classes])),
            "acc"       : float((y_pred[mask] == y_true[mask]).mean()),
            "mean_conf" : float(confs[mask].mean()),
        }

    confused_pairs = _confusion_pairs(y_true, y_pred, classes)

    # Speed benchmark
    print("[4/6] Speed benchmark …")
    speed_ips = _speed_benchmark(predictor)
    print(f"      {speed_ips:.0f} inferences/sec  ({1000/speed_ips:.1f} ms latency)")

    # 4. Save metrics JSON
    metrics = {
        "overall_accuracy"  : overall_acc,
        "n_classes"         : len(classes),
        "n_samples"         : len(X_raw),
        "speed_ips"         : speed_ips,
        "mean_confidence"   : float(confs.mean()),
        "per_class_accuracy": per_class,
        "category_summary"  : cat_summary,
        "confused_pairs"    : confused_pairs,
    }
    with open(out_dir / "deep_eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 5. Generate plots
    print("[5/6] Generating visualisations …")
    _plot_per_class_accuracy(per_class, out_dir / "per_class_accuracy.png")
    _plot_confusion_matrix(y_true, y_pred, classes, out_dir / "confusion_matrix_deep.png")
    _plot_calibration(y_true, y_pred, confs, out_dir / "calibration.png")

    img_names = [
        "per_class_accuracy.png",
        "confusion_matrix_deep.png",
        "calibration.png",
    ]
    if args.tsne:
        _plot_tsne(X_raw, y_true, classes, out_dir / "tsne.png")
        img_names.append("tsne.png")

    # 6. HTML report
    print("[6/6] Writing HTML report …")
    _write_html_report(metrics, out_dir / "evaluation_report.html", img_names)

    print(f"\n{'═'*60}")
    print(f"  ✅  Evaluation complete!")
    print(f"  Overall accuracy : {overall_acc*100:.2f}%")
    print(f"  Speed            : {speed_ips:.0f} inf/s  ({1000/speed_ips:.1f} ms)")
    print(f"\n  Category breakdown:")
    for cat, info in cat_summary.items():
        print(f"    {cat:<12} {info['acc']*100:.1f}% acc  "
              f"{info['mean_conf']*100:.1f}% mean conf")
    print(f"\n  Most confused pairs:")
    for true, pred, cnt in confused_pairs[:5]:
        print(f"    '{true}' → '{pred}' ({cnt}x)")
    print(f"\n  Reports saved to: {out_dir}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
