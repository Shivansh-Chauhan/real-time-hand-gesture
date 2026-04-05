"""
utils.py
────────
Shared utilities:
  • PredictionSmoother  – majority-vote window over recent predictions
  • FPSCounter          – rolling average FPS display
  • OverlayRenderer     – rich OpenCV frame annotations
  • safe_resize         – resolution helper
"""

from __future__ import annotations

import time
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import (
    SMOOTHING_WINDOW, CONFIDENCE_THRESHOLD,
    COLOR_GREEN, COLOR_RED, COLOR_WHITE, COLOR_BLACK,
    COLOR_YELLOW, COLOR_CYAN, COLOR_BLUE,
    GESTURE_DISPLAY,
)


# ── prediction smoother ───────────────────────────────────────────────────────

class PredictionSmoother:
    """
    Majority-vote smoothing over the last N predictions.
    Returns the most frequent (label, average-confidence) pair only
    when the window is >50% filled with that label.
    """

    def __init__(self, window: int = SMOOTHING_WINDOW):
        self.window = window
        self._labels: deque[str]  = deque(maxlen=window)
        self._confs:  deque[float]= deque(maxlen=window)

    def update(
        self, label: str, confidence: float
    ) -> Tuple[Optional[str], float]:
        """
        Push a new prediction and return the smoothed result.

        Returns (None, 0.0) if confidence is below threshold or
        the window is inconclusive.
        """
        if confidence < CONFIDENCE_THRESHOLD:
            return None, 0.0

        self._labels.append(label)
        self._confs.append(confidence)

        # Majority vote
        from collections import Counter
        counts = Counter(self._labels)
        top_label, top_count = counts.most_common(1)[0]
        if top_count / len(self._labels) >= 0.55:
            avg_conf = float(np.mean([
                c for l, c in zip(self._labels, self._confs) if l == top_label
            ]))
            return top_label, avg_conf
        return None, 0.0

    def reset(self):
        self._labels.clear()
        self._confs.clear()


# ── FPS counter ───────────────────────────────────────────────────────────────

class FPSCounter:
    """Rolling-average FPS using a timestamp ring buffer."""

    def __init__(self, window: int = 30):
        self._times: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


# ── overlay renderer ─────────────────────────────────────────────────────────

class OverlayRenderer:
    """
    Draws all HUD elements on an OpenCV frame.

    Call in order:
        renderer.draw_background_panel(frame)
        renderer.draw_gesture_prediction(frame, label, conf)
        renderer.draw_fps(frame, fps)
        renderer.draw_sentence(frame, sentence)
        renderer.draw_top_k(frame, top_k)
    """

    PANEL_H        = 110
    PANEL_ALPHA    = 0.65
    CORNER_RADIUS  = 12

    # ── panel ────────────────────────────────────────────────────────────────

    @staticmethod
    def draw_background_panel(frame: np.ndarray) -> None:
        """Semi-transparent dark panel at the top of the frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, OverlayRenderer.PANEL_H),
                      (15, 15, 15), -1)
        cv2.addWeighted(overlay, OverlayRenderer.PANEL_ALPHA,
                        frame, 1 - OverlayRenderer.PANEL_ALPHA, 0, frame)

    # ── gesture prediction ───────────────────────────────────────────────────

    @staticmethod
    def draw_gesture_prediction(
        frame: np.ndarray,
        label: Optional[str],
        confidence: float,
    ) -> None:
        if label is None:
            cv2.putText(frame, "No gesture detected",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (120, 120, 120), 2, cv2.LINE_AA)
            return

        display_label = GESTURE_DISPLAY.get(label, label)
        color = _conf_colour(confidence)

        # Big label
        cv2.putText(frame, display_label,
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, color, 3, cv2.LINE_AA)

        # Confidence bar
        bar_x, bar_y = 10, 60
        bar_w, bar_h = 220, 14
        filled_w = int(bar_w * confidence)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + filled_w, bar_y + bar_h), color, -1)
        cv2.putText(frame, f"{confidence*100:.1f}%",
                    (bar_x + bar_w + 6, bar_y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1, cv2.LINE_AA)

    # ── FPS ──────────────────────────────────────────────────────────────────

    @staticmethod
    def draw_fps(frame: np.ndarray, fps: float) -> None:
        h, w = frame.shape[:2]
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (w - 110, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_CYAN, 2, cv2.LINE_AA)

    # ── sentence builder ─────────────────────────────────────────────────────

    @staticmethod
    def draw_sentence(frame: np.ndarray, sentence: str) -> None:
        h, w = frame.shape[:2]
        # Bottom bar
        bar_y = h - 50
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, bar_y), (w, h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        display = sentence[-55:] if len(sentence) > 55 else sentence
        cv2.putText(frame, f"Text: {display}",
                    (10, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_YELLOW, 2, cv2.LINE_AA)

    # ── top-k alternatives ───────────────────────────────────────────────────

    @staticmethod
    def draw_top_k(
        frame: np.ndarray,
        top_k: List[Tuple[str, float]],
    ) -> None:
        h, w = frame.shape[:2]
        x = w - 160
        cv2.putText(frame, "Alternatives:", (x, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        for i, (lbl, conf) in enumerate(top_k[1:3]):   # show 2nd and 3rd
            y = 118 + i * 18
            cv2.putText(frame, f"  {lbl}: {conf*100:.0f}%", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

    # ── mode / instructions ──────────────────────────────────────────────────

    @staticmethod
    def draw_mode(frame: np.ndarray, mode: str) -> None:
        cv2.putText(frame, f"Mode: {mode}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, COLOR_BLUE, 2, cv2.LINE_AA)

    @staticmethod
    def draw_help(frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        tips = [
            "Q : quit",
            "C : clear text",
            "SPACE : add space",
            "T : toggle TTS",
            "S : screenshot",
        ]
        for i, tip in enumerate(tips):
            cv2.putText(frame, tip,
                        (w - 160, 130 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 200, 150), 1)

    # ── hand count ───────────────────────────────────────────────────────────

    @staticmethod
    def draw_hand_count(frame: np.ndarray, count: int) -> None:
        h, w = frame.shape[:2]
        colour = COLOR_GREEN if count > 0 else COLOR_RED
        cv2.putText(frame, f"Hands: {count}",
                    (w - 110, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)


# ── helpers ──────────────────────────────────────────────────────────────────

def _conf_colour(conf: float) -> Tuple[int, int, int]:
    """Green → yellow → red as confidence drops."""
    if conf >= 0.85:
        return COLOR_GREEN
    if conf >= 0.70:
        return COLOR_YELLOW
    return COLOR_RED


def safe_resize(
    frame: np.ndarray,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def timestamp_label() -> str:
    return time.strftime("%Y%m%d_%H%M%S")
