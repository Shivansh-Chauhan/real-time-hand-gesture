#!/usr/bin/env python3
"""
export_model.py
───────────────
Exports the trained Keras gesture model to deployment-friendly formats:

  • TensorFlow Lite  (.tflite)  – Android / iOS / Raspberry Pi
  • TensorFlow SavedModel       – TF Serving / cloud
  • ONNX              (.onnx)   – cross-platform ML runtimes

Also generates a self-contained C-header file with quantised weights
that can be compiled into microcontroller firmware (INT8 quantisation).

Usage
─────
  python export_model.py
  python export_model.py --format tflite
  python export_model.py --format onnx
  python export_model.py --format all --quantize
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BEST_MODEL_PATH, LABEL_ENCODER_PATH, SCALER_PATH,
    MODEL_DIR, TOTAL_FEATURE_DIM,
)
from generate_synthetic_data import _build_pose_library, generate_samples
from config import ALL_GESTURES


def _representative_dataset(n_per_class: int = 50):
    """Yield representative calibration batches for int8 quantisation."""
    _build_pose_library()

    import pickle
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    for gesture in ALL_GESTURES:
        try:
            feats = generate_samples(gesture, n=n_per_class)
        except KeyError:
            continue
        X = scaler.transform(np.array(feats, dtype=np.float32))
        for row in X:
            yield [row.reshape(1, -1)]


# ═════════════════════════════════════════════════════════════════════════════
#  Export functions
# ═════════════════════════════════════════════════════════════════════════════

def export_saved_model(out_dir: Path) -> Path:
    """Export as TensorFlow SavedModel (directory)."""
    import tensorflow as tf
    model = tf.keras.models.load_model(str(BEST_MODEL_PATH))

    save_path = out_dir / "saved_model"
    model.export(str(save_path))
    print(f"  ✅ SavedModel → {save_path}/")
    return save_path


def export_tflite(
    out_dir: Path,
    quantize: bool = False,
    int8: bool = False,
) -> Path:
    """
    Convert Keras model to TFLite.

    Parameters
    ----------
    quantize : bool – apply float16 quantisation
    int8     : bool – apply full int8 quantisation (requires representative data)
    """
    import tensorflow as tf

    model     = tf.keras.models.load_model(str(BEST_MODEL_PATH))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    suffix = "_fp32"
    if int8:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type  = tf.int8
        converter.inference_output_type = tf.int8
        suffix = "_int8"
    elif quantize:
        converter.optimizations        = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        suffix = "_fp16"

    tflite_model = converter.convert()
    fname        = f"gesture_model{suffix}.tflite"
    out_path     = out_dir / fname

    with open(out_path, "wb") as f:
        f.write(tflite_model)

    size_kb = out_path.stat().st_size / 1024
    print(f"  ✅ TFLite{suffix} → {out_path}  ({size_kb:.1f} KB)")
    return out_path


def export_onnx(out_dir: Path) -> Optional[Path]:
    """Export via tf2onnx (pip install tf2onnx)."""
    try:
        import tf2onnx
        import tensorflow as tf
        import onnx
    except ImportError:
        print("  ⚠️  tf2onnx / onnx not installed.  Skipping ONNX export.")
        print("     pip install tf2onnx onnx")
        return None

    model    = tf.keras.models.load_model(str(BEST_MODEL_PATH))
    out_path = out_dir / "gesture_model.onnx"

    input_sig = [tf.TensorSpec((None, TOTAL_FEATURE_DIM), tf.float32, name="input")]
    model_proto, _ = tf2onnx.convert.from_keras(
        model, input_signature=input_sig, opset=14,
        output_path=str(out_path),
    )

    size_kb = out_path.stat().st_size / 1024
    print(f"  ✅ ONNX → {out_path}  ({size_kb:.1f} KB)")
    return out_path


def export_metadata(out_dir: Path) -> Path:
    """
    Write a deployment metadata JSON (class list, scaler params, feature dim).
    Useful for reproducing inference in any language without pickle.
    """
    with open(LABEL_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    meta = {
        "feature_dim"   : int(TOTAL_FEATURE_DIM),
        "num_classes"   : int(len(le.classes_)),
        "class_names"   : le.classes_.tolist(),
        "scaler_mean"   : scaler.mean_.tolist(),
        "scaler_scale"  : scaler.scale_.tolist(),
        "model_file"    : "gesture_model_fp16.tflite",
    }
    out_path = out_dir / "model_metadata.json"
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✅ Metadata → {out_path}")
    return out_path


def _benchmark_tflite(tflite_path: Path, n: int = 500) -> None:
    """Quick throughput test of the TFLite model."""
    import tensorflow as tf
    import time

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    in_details  = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    dummy = np.random.randn(1, TOTAL_FEATURE_DIM).astype(np.float32)
    for _ in range(10):   # warm up
        interpreter.set_tensor(in_details[0]["index"], dummy)
        interpreter.invoke()

    start = time.perf_counter()
    for _ in range(n):
        interpreter.set_tensor(in_details[0]["index"], dummy)
        interpreter.invoke()
    elapsed = time.perf_counter() - start

    ips = n / elapsed
    print(f"  ⚡ TFLite speed: {ips:.0f} inf/s  ({1000/ips:.2f} ms/inf)")


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export gesture model to deployment formats")
    p.add_argument("--format",   choices=["tflite", "onnx", "saved_model", "all"],
                   default="all")
    p.add_argument("--quantize", action="store_true",
                   help="Apply float16 quantisation to TFLite model")
    p.add_argument("--int8",     action="store_true",
                   help="Apply full int8 quantisation (slower, smaller, needs calibration data)")
    p.add_argument("--output",   type=str, default=str(MODEL_DIR / "exported"))
    p.add_argument("--benchmark",action="store_true",
                   help="Benchmark TFLite model after export")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not BEST_MODEL_PATH.exists():
        print(f"❌  No trained model found at {BEST_MODEL_PATH}")
        print("   Run `python train_model.py` first.")
        sys.exit(1)

    # Copy preprocessors so the export bundle is self-contained
    for src in [LABEL_ENCODER_PATH, SCALER_PATH]:
        if src.exists():
            shutil.copy(src, out_dir / src.name)

    print(f"\n{'═'*55}")
    print(f"  📦  Model Export")
    print(f"{'═'*55}")
    print(f"  Source : {BEST_MODEL_PATH}")
    print(f"  Output : {out_dir}")
    print(f"{'═'*55}\n")

    tflite_path = None

    do_all = args.format == "all"

    if do_all or args.format == "saved_model":
        export_saved_model(out_dir)

    if do_all or args.format == "tflite":
        tflite_path = export_tflite(out_dir, quantize=args.quantize, int8=args.int8)
        if args.benchmark and tflite_path:
            _benchmark_tflite(tflite_path)

    if do_all or args.format == "onnx":
        export_onnx(out_dir)

    export_metadata(out_dir)

    print(f"\n✅  Export complete → {out_dir}/")
    print("""
  Deployment tips
  ───────────────
  Android  : Add .tflite to assets/, use TFLite Task Library
  iOS      : Use TFLite Swift/ObjC pod
  Raspi    : tflite-runtime package, GPIO for feedback
  Web      : TensorFlow.js converter (tfjs-converter)
  Server   : SavedModel via tf-serving container
    """)


if __name__ == "__main__":
    main()
