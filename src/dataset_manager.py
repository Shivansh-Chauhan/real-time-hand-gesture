"""
dataset_manager.py
──────────────────
Handles:
  • Creating and saving labelled gesture datasets from webcam frames
  • Loading / merging dataset files
  • On-the-fly data augmentation
"""

from __future__ import annotations

import csv
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    RAW_DIR, PROCESSED_DIR,
    SAMPLES_PER_CLASS, COUNTDOWN_SECONDS,
    ALL_GESTURES, COLOR_GREEN, COLOR_RED, COLOR_WHITE, COLOR_YELLOW,
)
from src.hand_tracker import HandTracker
from src.feature_extractor import extract_features, augment_landmarks


# ── raw-data collector ───────────────────────────────────────────────────────

class DatasetCollector:
    """
    Interactive webcam-based dataset collector.

    Usage
    -----
    collector = DatasetCollector()
    collector.collect_class("A")         # opens webcam, prompts user
    collector.collect_all(gestures)      # iterate over a list of gestures
    """

    def __init__(
        self,
        output_dir: Path = RAW_DIR,
        samples_per_class: int = SAMPLES_PER_CLASS,
        camera_index: int = 0,
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir       = output_dir
        self.samples_per_class = samples_per_class
        self.camera_index     = camera_index

    # ── public ───────────────────────────────────────────────────────────────

    def collect_class(self, gesture_name: str) -> int:
        """
        Open webcam and collect *samples_per_class* frames for *gesture_name*.

        Returns number of samples successfully saved.
        """
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")

        tracker   = HandTracker()
        save_path = self.output_dir / f"{gesture_name}.csv"
        saved     = 0

        # Read existing samples
        existing = self._count_existing(save_path)
        needed   = max(0, self.samples_per_class - existing)
        if needed == 0:
            print(f"  '{gesture_name}' already has {existing} samples. Skipping.")
            cap.release()
            tracker.close()
            return existing

        # ── countdown ────────────────────────────────────────────────────────
        start = time.time()
        while time.time() - start < COUNTDOWN_SECONDS:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            remaining = COUNTDOWN_SECONDS - int(time.time() - start)
            self._overlay_countdown(frame, gesture_name, remaining)
            cv2.imshow("Dataset Collector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                tracker.close()
                cv2.destroyAllWindows()
                return saved

        # ── capture loop ──────────────────────────────────────────────────────
        rows: List[List] = []
        while saved < needed:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            result = tracker.process(frame, draw=True)

            if result.primary is not None:
                lm = result.primary.landmarks
                feat = extract_features(lm)
                if feat is not None:
                    rows.append([gesture_name] + feat.tolist())
                    saved += 1

            progress = (saved / needed) * 100
            self._overlay_progress(frame, gesture_name, saved, needed, progress)
            cv2.imshow("Dataset Collector", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Write CSV (append mode)
        with open(save_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        cap.release()
        tracker.close()
        cv2.destroyWindow("Dataset Collector")
        print(f"  ✅ '{gesture_name}': saved {saved} samples → {save_path}")
        return saved + existing

    def collect_all(self, gestures: List[str] = ALL_GESTURES) -> None:
        """Collect data for every gesture in *gestures* sequentially."""
        for i, gesture in enumerate(gestures):
            print(f"\n[{i+1}/{len(gestures)}] Prepare: '{gesture}'. "
                  "Press SPACE to start, Q to quit.")
            input_ok = self._wait_for_space()
            if not input_ok:
                break
            self.collect_class(gesture)
        print("\nData collection complete.")

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _count_existing(csv_path: Path) -> int:
        if not csv_path.exists():
            return 0
        with open(csv_path, "r") as f:
            return sum(1 for _ in f)

    @staticmethod
    def _wait_for_space() -> bool:
        """Block until user presses ENTER in the terminal, return False on 'q'."""
        key = input("  [ENTER=start | q=quit] > ").strip().lower()
        return key != "q"

    @staticmethod
    def _overlay_countdown(frame, gesture: str, remaining: int):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"GET READY: '{gesture}'",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_YELLOW, 2)
        cv2.putText(frame, f"Starting in {remaining}...",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2)

    @staticmethod
    def _overlay_progress(frame, gesture: str, saved: int, needed: int, pct: float):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)
        cv2.putText(frame, f"Gesture: '{gesture}'",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GREEN, 2)
        cv2.putText(frame, f"Samples: {saved}/{needed}",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
        # Progress bar
        bar_w = int((pct / 100) * (w - 20))
        cv2.rectangle(frame, (10, 68), (w - 10, 82), (60, 60, 60), -1)
        cv2.rectangle(frame, (10, 68), (10 + bar_w, 82), COLOR_GREEN, -1)


# ── dataset loader ────────────────────────────────────────────────────────────

class DatasetManager:
    """
    Loads, merges and augments the collected CSV datasets.

    Each CSV file: gesture_name, feat0, feat1, ..., feat87
    """

    def __init__(self, raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR):
        self.raw_dir       = raw_dir
        self.processed_dir = processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ── loading ──────────────────────────────────────────────────────────────

    def load_raw(
        self,
        augment: bool = True,
        augment_factor: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read all CSVs in raw_dir, optionally augment, return (X, y).

        Parameters
        ----------
        augment        : bool – apply landmark augmentation
        augment_factor : int  – how many extra copies per real sample

        Returns
        -------
        X : (N, TOTAL_FEATURE_DIM) float32
        y : (N,) object (gesture name strings)
        """
        csv_files = sorted(self.raw_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {self.raw_dir}. "
                "Run collect_data.py first."
            )

        X_list, y_list = [], []
        class_counts: Dict[str, int] = defaultdict(int)

        for csv_file in csv_files:
            with open(csv_file, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 2:
                        continue
                    gesture = row[0]
                    try:
                        feat = np.array(row[1:], dtype=np.float32)
                    except ValueError:
                        continue
                    X_list.append(feat)
                    y_list.append(gesture)
                    class_counts[gesture] += 1

        if not X_list:
            raise ValueError("Dataset is empty.")

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=object)
        print(f"  Loaded {len(X)} samples across {len(class_counts)} classes.")

        if augment:
            X, y = self._augment(X, y, augment_factor)
            print(f"  After augmentation: {len(X)} samples.")

        return X, y

    def save_processed(self, X: np.ndarray, y: np.ndarray) -> None:
        """Save processed arrays as .npy files."""
        np.save(self.processed_dir / "X.npy", X)
        np.save(self.processed_dir / "y.npy", y)
        print(f"  Processed dataset saved to {self.processed_dir}/")

    def load_processed(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load pre-processed .npy files."""
        X = np.load(self.processed_dir / "X.npy", allow_pickle=False)
        y = np.load(self.processed_dir / "y.npy", allow_pickle=True)
        return X, y

    # ── augmentation ─────────────────────────────────────────────────────────

    @staticmethod
    def _augment(
        X: np.ndarray,
        y: np.ndarray,
        factor: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Because we stored feature vectors (not raw landmarks) in the CSV,
        we augment by adding small Gaussian noise in feature space.
        """
        X_aug  = [X]
        y_aug  = [y]
        for _ in range(factor):
            noise = np.random.normal(0, 0.01, X.shape).astype(np.float32)
            X_aug.append(X + noise)
            y_aug.append(y)
        return np.vstack(X_aug), np.concatenate(y_aug)

    # ── statistics ───────────────────────────────────────────────────────────

    def class_distribution(self, y: np.ndarray) -> Dict[str, int]:
        dist: Dict[str, int] = defaultdict(int)
        for label in y:
            dist[label] += 1
        return dict(sorted(dist.items()))

    def print_stats(self, y: np.ndarray) -> None:
        dist = self.class_distribution(y)
        print(f"\n{'Gesture':<18} {'Samples':>8}")
        print("-" * 28)
        total = 0
        for gesture, count in dist.items():
            print(f"  {gesture:<16} {count:>8}")
            total += count
        print("-" * 28)
        print(f"  {'TOTAL':<16} {total:>8}\n")
