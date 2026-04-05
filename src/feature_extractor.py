"""
feature_extractor.py
────────────────────
Converts raw 21-landmark arrays into normalised, scale- and
translation-invariant feature vectors suitable for classification.

Feature vector layout  (88 total)
──────────────────────────────────
 [0:63]  – normalised x, y, z coords for all 21 landmarks  (63)
[63:78]  – 15 inter-joint angles (knuckle bend angles)       (15)
[78:88]  – 10 key inter-landmark distances                   (10)
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from config import (
    NUM_LANDMARKS, COORDS_PER_LM,
    ANGLE_FEATURES, DISTANCE_FEATURES,
    TOTAL_FEATURE_DIM,
)

# ── MediaPipe landmark indices ────────────────────────────────────────────────
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP       = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP       = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP  = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP          = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP      = 17, 18, 19, 20

# Finger joint triplets for angle calculation  (proximal, middle, distal)
_ANGLE_TRIPLETS = [
    (WRIST,      THUMB_CMC,   THUMB_MCP),
    (THUMB_CMC,  THUMB_MCP,   THUMB_IP),
    (THUMB_MCP,  THUMB_IP,    THUMB_TIP),
    (WRIST,      INDEX_MCP,   INDEX_PIP),
    (INDEX_MCP,  INDEX_PIP,   INDEX_DIP),
    (WRIST,      MIDDLE_MCP,  MIDDLE_PIP),
    (MIDDLE_MCP, MIDDLE_PIP,  MIDDLE_DIP),
    (WRIST,      RING_MCP,    RING_PIP),
    (RING_MCP,   RING_PIP,    RING_DIP),
    (WRIST,      PINKY_MCP,   PINKY_PIP),
    (PINKY_MCP,  PINKY_PIP,   PINKY_DIP),
    (THUMB_TIP,  INDEX_TIP,   MIDDLE_TIP),
    (INDEX_TIP,  MIDDLE_TIP,  RING_TIP),
    (MIDDLE_TIP, RING_TIP,    PINKY_TIP),
    (THUMB_CMC,  WRIST,       PINKY_MCP),
]

# Landmark pairs for distance calculation
_DISTANCE_PAIRS = [
    (THUMB_TIP,  INDEX_TIP),
    (THUMB_TIP,  MIDDLE_TIP),
    (THUMB_TIP,  RING_TIP),
    (THUMB_TIP,  PINKY_TIP),
    (INDEX_TIP,  MIDDLE_TIP),
    (INDEX_TIP,  RING_TIP),
    (INDEX_TIP,  PINKY_TIP),
    (MIDDLE_TIP, RING_TIP),
    (MIDDLE_TIP, PINKY_TIP),
    (RING_TIP,   PINKY_TIP),
]

assert len(_ANGLE_TRIPLETS) == ANGLE_FEATURES
assert len(_DISTANCE_PAIRS) == DISTANCE_FEATURES


# ── public API ───────────────────────────────────────────────────────────────

def extract_features(landmarks: np.ndarray) -> Optional[np.ndarray]:
    """
    Parameters
    ----------
    landmarks : (21, 3) float32  –  raw normalised coords from MediaPipe

    Returns
    -------
    feature_vector : (TOTAL_FEATURE_DIM,) float32  or  None on failure
    """
    if landmarks is None or landmarks.shape != (NUM_LANDMARKS, COORDS_PER_LM):
        return None

    try:
        coord_feats  = _normalise_coords(landmarks)          # (63,)
        angle_feats  = _compute_angles(landmarks)            # (15,)
        dist_feats   = _compute_distances(landmarks)         # (10,)
        vec = np.concatenate([coord_feats, angle_feats, dist_feats])
        assert vec.shape == (TOTAL_FEATURE_DIM,)
        return vec.astype(np.float32)
    except Exception:
        return None


# ── normalisation ────────────────────────────────────────────────────────────

def _normalise_coords(pts: np.ndarray) -> np.ndarray:
    """
    Make features translation- and scale-invariant:
      1. Subtract wrist position (translation invariance)
      2. Divide by hand span, i.e. distance wrist → middle-finger MCP
         (scale invariance)
    """
    pts = pts.copy()

    # 1. Translate to wrist origin
    pts -= pts[WRIST]

    # 2. Scale by wrist→middle-MCP distance
    span = np.linalg.norm(pts[MIDDLE_MCP])
    if span > 1e-6:
        pts /= span

    return pts.flatten()           # (63,)


# ── angles ───────────────────────────────────────────────────────────────────

def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine angle (radians) between two 3-D vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(cos))


def _compute_angles(pts: np.ndarray) -> np.ndarray:
    """Return angle at each middle joint for every triplet."""
    angles = []
    for a_idx, b_idx, c_idx in _ANGLE_TRIPLETS:
        v1 = pts[a_idx] - pts[b_idx]
        v2 = pts[c_idx] - pts[b_idx]
        angles.append(_angle_between(v1, v2))
    return np.array(angles, dtype=np.float32)


# ── distances ────────────────────────────────────────────────────────────────

def _compute_distances(pts: np.ndarray) -> np.ndarray:
    """
    Euclidean distances between landmark pairs, normalised by wrist–middle MCP span.
    """
    span = np.linalg.norm(pts[MIDDLE_MCP] - pts[WRIST])
    if span < 1e-6:
        span = 1.0
    dists = []
    for a_idx, b_idx in _DISTANCE_PAIRS:
        d = np.linalg.norm(pts[a_idx] - pts[b_idx]) / span
        dists.append(d)
    return np.array(dists, dtype=np.float32)


# ── data-augmentation helper ─────────────────────────────────────────────────

def augment_landmarks(
    landmarks: np.ndarray,
    noise_std: float = 0.005,
    scale_range: tuple = (0.90, 1.10),
    rotation_deg: float = 15.0,
) -> np.ndarray:
    """
    Lightweight augmentation applied to raw landmarks before feature extraction.
    Returns a new (21, 3) array.
    """
    pts = landmarks.copy()

    # Additive Gaussian noise
    pts += np.random.normal(0, noise_std, pts.shape).astype(np.float32)

    # Random scale in x-y plane
    scale = np.random.uniform(*scale_range)
    pts[:, :2] *= scale

    # Random 2-D rotation around wrist
    angle  = np.deg2rad(np.random.uniform(-rotation_deg, rotation_deg))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    centre = pts[WRIST, :2]
    pts[:, :2] = (pts[:, :2] - centre) @ rot.T + centre

    return pts
