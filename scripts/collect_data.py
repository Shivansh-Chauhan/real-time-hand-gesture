#!/usr/bin/env python3
"""
collect_data.py
───────────────
Interactive dataset collection using a webcam.

Usage
─────
  # Collect a specific gesture
  python collect_data.py --gesture A

  # Collect a category
  python collect_data.py --category alphabets
  python collect_data.py --category numbers
  python collect_data.py --category commands

  # Collect everything
  python collect_data.py --all

  # Override sample count per class
  python collect_data.py --all --samples 300

Keyboard shortcuts during collection
─────────────────────────────────────
  Q  →  quit current gesture (moves to next)
"""

import argparse
import sys
from pathlib import Path

# ── make project root importable ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    ALPHABET_GESTURES, NUMBER_GESTURES, COMMAND_GESTURES, ALL_GESTURES,
    RAW_DIR, SAMPLES_PER_CLASS,
)
from src.dataset_manager import DatasetCollector


CATEGORIES = {
    "alphabets" : ALPHABET_GESTURES,
    "numbers"   : NUMBER_GESTURES,
    "commands"  : COMMAND_GESTURES,
    "all"       : ALL_GESTURES,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hand Gesture Dataset Collector",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--gesture",  metavar="NAME",
                       help="Single gesture name (e.g. A, hello, thumbs_up)")
    group.add_argument("--category", choices=list(CATEGORIES.keys()),
                       help="Collect an entire category")
    group.add_argument("--all",      action="store_true",
                       help="Collect all gesture classes")

    p.add_argument("--samples",  type=int, default=SAMPLES_PER_CLASS,
                   help=f"Samples per class (default: {SAMPLES_PER_CLASS})")
    p.add_argument("--camera",   type=int, default=0,
                   help="Camera index (default: 0)")
    p.add_argument("--output",   type=str, default=str(RAW_DIR),
                   help="Output directory for CSV files")
    return p.parse_args()


def main():
    args = parse_args()

    collector = DatasetCollector(
        output_dir        = Path(args.output),
        samples_per_class = args.samples,
        camera_index      = args.camera,
    )

    print("\n" + "═" * 55)
    print("  🤚  Hand Gesture Dataset Collector")
    print("═" * 55)
    print(f"  Output dir : {args.output}")
    print(f"  Samples/class : {args.samples}")
    print(f"  Camera index  : {args.camera}")
    print("═" * 55 + "\n")

    if args.gesture:
        gestures = [args.gesture.strip()]
    elif args.category:
        gestures = CATEGORIES[args.category]
    else:
        gestures = ALL_GESTURES

    print(f"  Collecting {len(gestures)} gesture(s):\n"
          f"  {', '.join(gestures[:10])}"
          + ("..." if len(gestures) > 10 else "") + "\n")

    collector.collect_all(gestures)

    print("\n✅  Collection finished.")
    print(f"   Data saved in: {args.output}")
    print("   Next step → run:  python train_model.py\n")


if __name__ == "__main__":
    main()
