"""
hand_tracker.py
───────────────
Wraps MediaPipe Hands to provide:
  • per-frame hand detection
  • 21-landmark extraction for up to 2 hands
  • annotated frame drawing
"""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from config import MEDIAPIPE_CFG, COLOR_GREEN, COLOR_WHITE


# ── data containers ──────────────────────────────────────────────────────────

@dataclass
class HandResult:
    """Stores landmarks and metadata for a single detected hand."""
    landmarks: np.ndarray          # shape (21, 3)  — normalised (x,y,z)
    handedness: str                # "Left" | "Right"
    score: float                   # detection confidence
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) in pixel space


@dataclass
class FrameResult:
    """All hands detected in one frame."""
    hands: List[HandResult] = field(default_factory=list)
    annotated_frame: Optional[np.ndarray] = None

    @property
    def num_hands(self) -> int:
        return len(self.hands)

    @property
    def primary(self) -> Optional[HandResult]:
        """Return the first (highest-confidence) hand."""
        return self.hands[0] if self.hands else None


# ── tracker ──────────────────────────────────────────────────────────────────

class HandTracker:
    """
    Thin wrapper around mediapipe.solutions.hands.

    Usage
    -----
    tracker = HandTracker()
    result  = tracker.process(bgr_frame)
    if result.primary:
        landmarks = result.primary.landmarks   # (21, 3)
    """

    # Landmark indices for skeleton drawing (pairs)
    _CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

    def __init__(self, **overrides):
        cfg = {**MEDIAPIPE_CFG, **overrides}
        self._mp_hands = mp.solutions.hands.Hands(**cfg)
        self._mp_draw  = mp.solutions.drawing_utils
        self._draw_spec_lm = mp.solutions.drawing_utils.DrawingSpec(
            color=COLOR_GREEN, thickness=2, circle_radius=3
        )
        self._draw_spec_cn = mp.solutions.drawing_utils.DrawingSpec(
            color=(200, 200, 200), thickness=1
        )

    # ── public API ───────────────────────────────────────────────────────────

    def process(
        self,
        bgr_frame: np.ndarray,
        draw: bool = True,
    ) -> FrameResult:
        """
        Detect hands in *bgr_frame*.

        Parameters
        ----------
        bgr_frame : np.ndarray  BGR image
        draw      : bool        if True, draw landmarks on a copy of the frame

        Returns
        -------
        FrameResult
        """
        frame_copy = bgr_frame.copy() if draw else bgr_frame
        rgb_frame  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        mp_result = self._mp_hands.process(rgb_frame)

        result = FrameResult(annotated_frame=frame_copy)

        if not mp_result.multi_hand_landmarks:
            return result

        h, w = bgr_frame.shape[:2]

        for mp_lms, mp_hand in zip(
            mp_result.multi_hand_landmarks,
            mp_result.multi_handedness,
        ):
            # Draw skeleton
            if draw:
                self._mp_draw.draw_landmarks(
                    frame_copy,
                    mp_lms,
                    self._CONNECTIONS,
                    self._draw_spec_lm,
                    self._draw_spec_cn,
                )

            landmarks, bbox = self._extract(mp_lms, h, w)
            hand_res = HandResult(
                landmarks   = landmarks,
                handedness  = mp_hand.classification[0].label,
                score       = mp_hand.classification[0].score,
                bbox        = bbox,
            )
            result.hands.append(hand_res)

        # Sort highest confidence first
        result.hands.sort(key=lambda x: x.score, reverse=True)
        return result

    def close(self):
        self._mp_hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _extract(
        mp_lms,
        frame_h: int,
        frame_w: int,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Convert MediaPipe landmark object → numpy array + bounding box.

        Returns
        -------
        landmarks : (21, 3) float32 — normalised coords in [0, 1]
        bbox      : (x, y, w, h) in pixels
        """
        pts = np.array(
            [[lm.x, lm.y, lm.z] for lm in mp_lms.landmark],
            dtype=np.float32,
        )

        # Bounding box in pixel space
        xs = pts[:, 0] * frame_w
        ys = pts[:, 1] * frame_h
        x1, y1 = int(xs.min()) - 15, int(ys.min()) - 15
        x2, y2 = int(xs.max()) + 15, int(ys.max()) + 15
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_w, x2), min(frame_h, y2)
        bbox = (x1, y1, x2 - x1, y2 - y1)

        return pts, bbox

    # ── utilities ────────────────────────────────────────────────────────────

    @staticmethod
    def draw_bbox(
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        label: str = "",
        color: Tuple[int, int, int] = COLOR_GREEN,
        label_color: Tuple[int, int, int] = COLOR_WHITE,
    ) -> None:
        """Draw a bounding box with optional label on *frame* (in-place)."""
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        if label:
            label_y = max(y - 10, 18)
            cv2.putText(
                frame, label, (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, label_color, 2,
                cv2.LINE_AA,
            )

    @staticmethod
    def landmark_names() -> List[str]:
        """Return the 21 MediaPipe landmark names in order."""
        return [lm.name for lm in mp.solutions.hands.HandLandmark]
