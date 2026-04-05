#!/usr/bin/env python3
"""
generate_synthetic_data.py
──────────────────────────
Generates a labelled dataset of synthetic hand-landmark feature vectors
WITHOUT requiring a physical webcam or camera.

How it works
────────────
Each gesture class is modelled as a canonical hand pose (21 × 3 landmark
coordinates) that encodes the anatomical signature of that gesture.
The canonical pose is then jittered with:
  • Random Gaussian noise on landmark positions
  • Random 2-D rotation around the wrist
  • Random uniform scale
  • Per-finger random micro-bend (realistic anatomical variation)
  • Random global translation (cancelled by normalisation)

Feature vectors are then extracted from the perturbed poses, identically
to the real pipeline (src/feature_extractor.py), so the synthetic data
is fully compatible with the rest of the training pipeline.

Usage
─────
  python generate_synthetic_data.py                    # all gestures
  python generate_synthetic_data.py --samples 500      # 500 per class
  python generate_synthetic_data.py --gesture A --samples 1000
  python generate_synthetic_data.py --category commands
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    RAW_DIR, ALL_GESTURES, ALPHABET_GESTURES,
    NUMBER_GESTURES, COMMAND_GESTURES, SAMPLES_PER_CLASS,
)
from src.feature_extractor import extract_features, augment_landmarks

# ── Landmark index aliases ────────────────────────────────────────────────────
WRIST = 0
T_CMC, T_MCP, T_IP, T_TIP         = 1, 2, 3, 4
I_MCP, I_PIP, I_DIP, I_TIP        = 5, 6, 7, 8
M_MCP, M_PIP, M_DIP, M_TIP        = 9, 10, 11, 12
R_MCP, R_PIP, R_DIP, R_TIP        = 13, 14, 15, 16
P_MCP, P_PIP, P_DIP, P_TIP        = 17, 18, 19, 20


# ═════════════════════════════════════════════════════════════════════════════
#  Canonical pose library
#  Each function returns a (21, 3) float32 array in normalised MediaPipe space.
#  Conventions:
#    • Wrist at origin after normalisation
#    • x+ = thumb side (right hand mirrored)
#    • y+ = finger-tip direction
#    • z ≈ 0 (depth), small perturbations added later
# ═════════════════════════════════════════════════════════════════════════════

def _base_hand(
    thumb_ext:  float = 1.0,   # 0=fully folded, 1=fully extended
    index_ext:  float = 1.0,
    middle_ext: float = 1.0,
    ring_ext:   float = 1.0,
    pinky_ext:  float = 1.0,
    thumb_spread: float = 0.1,  # lateral spread of thumb
) -> np.ndarray:
    """
    Parameterised canonical hand pose.

    Each finger is modelled as three segments of equal length radiating
    from their respective MCP joint. Extension ∈ [0,1] controls how
    straight (1) or folded into the palm (0) the finger is.
    """
    pts = np.zeros((21, 3), dtype=np.float32)

    # Wrist
    pts[WRIST] = [0, 0, 0]

    # ── Finger MCP base positions (fan layout) ───────────────────────────────
    # x offsets: thumb left, index-middle-ring-pinky spread right
    mcp_x = np.array([-0.15 - thumb_spread, -0.10, 0.00, 0.09, 0.17], np.float32)
    mcp_y = np.array([ 0.30,                 0.45,  0.48, 0.44, 0.38], np.float32)

    seg = 0.14  # finger segment length (normalised)

    # Thumb (segments shorter, angled)
    pts[T_CMC] = [mcp_x[0] - 0.04, mcp_y[0] - 0.12, 0]
    pts[T_MCP] = [mcp_x[0],        mcp_y[0],          0]
    for i, idx in enumerate([T_IP, T_TIP]):
        frac = (i + 1) * seg * thumb_ext + (1 - thumb_ext) * 0.04 * (i + 1)
        pts[idx] = [
            pts[T_MCP, 0] - frac * 0.5,
            pts[T_MCP, 1] + frac * 0.8,
            0
        ]

    # Index
    pts[I_MCP] = [mcp_x[1], mcp_y[1], 0]
    for i, idx in enumerate([I_PIP, I_DIP, I_TIP]):
        ext = index_ext
        dy = seg * ext
        fold_y = mcp_y[1] - 0.06 * (1 - ext) * (i + 1)
        pts[idx] = [mcp_x[1], mcp_y[1] + dy * (i + 1), 0]
        if ext < 1.0:
            # fold curls finger toward palm
            pts[idx, 1] = pts[I_MCP, 1] + dy * (i + 1) * ext + fold_y * (1 - ext) * 0.15

    # Middle
    pts[M_MCP] = [mcp_x[2], mcp_y[2], 0]
    for i, idx in enumerate([M_PIP, M_DIP, M_TIP]):
        ext = middle_ext
        dy = seg * ext
        pts[idx] = [mcp_x[2], mcp_y[2] + dy * (i + 1), 0]
        if ext < 1.0:
            pts[idx, 1] = pts[M_MCP, 1] + dy * (i + 1) * ext

    # Ring
    pts[R_MCP] = [mcp_x[3], mcp_y[3], 0]
    for i, idx in enumerate([R_PIP, R_DIP, R_TIP]):
        ext = ring_ext
        dy = seg * ext
        pts[idx] = [mcp_x[3], mcp_y[3] + dy * (i + 1), 0]
        if ext < 1.0:
            pts[idx, 1] = pts[R_MCP, 1] + dy * (i + 1) * ext

    # Pinky
    pts[P_MCP] = [mcp_x[4], mcp_y[4], 0]
    for i, idx in enumerate([P_PIP, P_DIP, P_TIP]):
        ext = pinky_ext
        dy = seg * 0.85 * ext   # pinky is shorter
        pts[idx] = [mcp_x[4], mcp_y[4] + dy * (i + 1), 0]
        if ext < 1.0:
            pts[idx, 1] = pts[P_MCP, 1] + dy * (i + 1) * ext

    return pts


def _folded() -> np.ndarray:
    """All fingers folded (fist)."""
    return _base_hand(0.3, 0.1, 0.1, 0.1, 0.1)


def _open() -> np.ndarray:
    """All fingers extended (open palm)."""
    return _base_hand(1.0, 1.0, 1.0, 1.0, 1.0)


# ── Alphabet canonical poses ──────────────────────────────────────────────────
# Approximations of ASL static fingerspelling poses

def _asl_poses() -> Dict[str, np.ndarray]:
    poses = {}

    # A – fist with thumb alongside
    poses["A"] = _base_hand(0.6, 0.1, 0.1, 0.1, 0.1, thumb_spread=0.0)

    # B – four fingers up, thumb folded
    poses["B"] = _base_hand(0.1, 1.0, 1.0, 1.0, 1.0, thumb_spread=0.0)

    # C – curved fingers (half-cup)
    poses["C"] = _base_hand(0.7, 0.6, 0.6, 0.6, 0.6, thumb_spread=0.3)

    # D – index up, others folded, thumb touches middle
    poses["D"] = _base_hand(0.3, 1.0, 0.2, 0.2, 0.2, thumb_spread=0.2)

    # E – all fingers bent at PIP, thumb tucked
    poses["E"] = _base_hand(0.2, 0.3, 0.3, 0.3, 0.3, thumb_spread=0.0)

    # F – index + thumb touch, others open
    p = _base_hand(0.5, 0.4, 1.0, 1.0, 1.0, thumb_spread=0.2)
    p[I_TIP] = p[T_TIP].copy()   # pinch
    poses["F"] = p

    # G – index points sideways, thumb parallel
    p = _base_hand(0.2, 0.8, 0.1, 0.1, 0.1, thumb_spread=0.1)
    p[I_TIP, 0] += 0.1   # rotate index sideways
    poses["G"] = p

    # H – index + middle extended sideways
    p = _base_hand(0.2, 0.9, 0.9, 0.1, 0.1, thumb_spread=0.1)
    poses["H"] = p

    # I – pinky only up
    poses["I"] = _base_hand(0.2, 0.1, 0.1, 0.1, 1.0, thumb_spread=0.1)

    # J – I + motion trace (static: same as I with slight wrist tilt)
    poses["J"] = _base_hand(0.2, 0.1, 0.1, 0.1, 1.0, thumb_spread=0.1)

    # K – index + middle up and spread, thumb between them
    p = _base_hand(0.5, 0.9, 0.8, 0.1, 0.1, thumb_spread=0.15)
    p[M_TIP, 0] += 0.04
    poses["K"] = p

    # L – index up, thumb out (L-shape)
    p = _base_hand(0.9, 1.0, 0.1, 0.1, 0.1, thumb_spread=0.4)
    poses["L"] = p

    # M – three fingers folded over thumb
    poses["M"] = _base_hand(0.25, 0.2, 0.2, 0.2, 0.1, thumb_spread=0.0)

    # N – index + middle folded over thumb
    poses["N"] = _base_hand(0.25, 0.2, 0.2, 0.1, 0.1, thumb_spread=0.0)

    # O – round shape (all fingers curl to touch thumb)
    p = _base_hand(0.6, 0.5, 0.5, 0.5, 0.5, thumb_spread=0.3)
    # bring tips together
    centre = np.mean([p[T_TIP], p[I_TIP], p[M_TIP]], axis=0)
    for tip in [T_TIP, I_TIP, M_TIP, R_TIP, P_TIP]:
        p[tip] = p[tip] * 0.6 + centre * 0.4
    poses["O"] = p

    # P – like K but pointing down
    p = _base_hand(0.5, 0.9, 0.8, 0.1, 0.1, thumb_spread=0.15)
    p[:, 1] *= -0.6   # flip/compress downward
    poses["P"] = p

    # Q – like G but pointing down
    p = _base_hand(0.4, 0.7, 0.1, 0.1, 0.1, thumb_spread=0.2)
    p[:, 1] *= -0.5
    poses["Q"] = p

    # R – index + middle crossed (approximated as both extended, middle offset)
    p = _base_hand(0.2, 1.0, 0.9, 0.1, 0.1, thumb_spread=0.05)
    p[M_TIP, 0] -= 0.05   # middle crosses index
    poses["R"] = p

    # S – fist with thumb over fingers
    poses["S"] = _base_hand(0.45, 0.1, 0.1, 0.1, 0.1, thumb_spread=0.0)

    # T – thumb between index and middle
    poses["T"] = _base_hand(0.35, 0.15, 0.1, 0.1, 0.1, thumb_spread=0.05)

    # U – index + middle together up
    poses["U"] = _base_hand(0.2, 1.0, 1.0, 0.1, 0.1, thumb_spread=0.05)

    # V – index + middle spread (peace/victory)
    p = _base_hand(0.2, 1.0, 1.0, 0.1, 0.1, thumb_spread=0.1)
    p[I_TIP, 0] -= 0.06
    p[M_TIP, 0] += 0.06
    poses["V"] = p

    # W – index + middle + ring spread
    p = _base_hand(0.2, 1.0, 1.0, 1.0, 0.1, thumb_spread=0.1)
    p[I_TIP, 0] -= 0.06
    p[R_TIP, 0] += 0.06
    poses["W"] = p

    # X – index hooked
    p = _base_hand(0.2, 0.55, 0.1, 0.1, 0.1, thumb_spread=0.1)
    p[I_TIP, 0] += 0.05; p[I_TIP, 1] -= 0.04
    poses["X"] = p

    # Y – thumb and pinky out
    poses["Y"] = _base_hand(0.9, 0.1, 0.1, 0.1, 1.0, thumb_spread=0.4)

    # Z – index traces Z (static: index pointing with others folded)
    poses["Z"] = _base_hand(0.2, 0.85, 0.1, 0.1, 0.1, thumb_spread=0.1)

    return poses


# ── Number canonical poses ────────────────────────────────────────────────────

def _number_poses() -> Dict[str, np.ndarray]:
    poses = {}

    # 0 – O shape
    poses["0"] = _asl_poses()["O"]

    # 1 – index finger up
    poses["1"] = _base_hand(0.2, 1.0, 0.1, 0.1, 0.1)

    # 2 – index + middle up (V)
    poses["2"] = _asl_poses()["V"]

    # 3 – index + middle + ring
    poses["3"] = _base_hand(0.2, 1.0, 1.0, 1.0, 0.1)

    # 4 – four fingers up, thumb folded
    poses["4"] = _base_hand(0.1, 1.0, 1.0, 1.0, 1.0)

    # 5 – open hand
    poses["5"] = _open()

    # 6 – pinky + thumb touch
    p = _base_hand(0.9, 1.0, 1.0, 1.0, 0.3, thumb_spread=0.3)
    p[P_TIP] = p[T_TIP].copy()
    poses["6"] = p

    # 7 – ring + thumb touch
    p = _base_hand(0.9, 1.0, 1.0, 0.3, 1.0, thumb_spread=0.3)
    p[R_TIP] = p[T_TIP].copy()
    poses["7"] = p

    # 8 – middle + thumb touch
    p = _base_hand(0.9, 1.0, 0.4, 1.0, 1.0, thumb_spread=0.3)
    p[M_TIP] = p[T_TIP].copy()
    poses["8"] = p

    # 9 – index + thumb touch (OK)
    p = _base_hand(0.9, 0.4, 1.0, 1.0, 1.0, thumb_spread=0.3)
    p[I_TIP] = p[T_TIP].copy()
    poses["9"] = p

    return poses


# ── Command canonical poses ───────────────────────────────────────────────────

def _command_poses() -> Dict[str, np.ndarray]:
    poses = {}

    # hello – open palm, all fingers extended
    poses["hello"] = _open()

    # yes – fist (nodding approximated statically)
    poses["yes"] = _folded()

    # no – index + middle extended, waving (static: two fingers up)
    poses["no"] = _base_hand(0.2, 1.0, 1.0, 0.1, 0.1)

    # stop – open flat palm facing out
    p = _open()
    p[:, 0] *= 0.8   # slightly compressed for flat-palm look
    poses["stop"] = p

    # thumbs_up
    poses["thumbs_up"] = _base_hand(1.0, 0.1, 0.1, 0.1, 0.1, thumb_spread=0.05)

    # thumbs_down
    p = _base_hand(1.0, 0.1, 0.1, 0.1, 0.1, thumb_spread=0.05)
    p[:, 1] *= -1   # flip thumb downward
    poses["thumbs_down"] = p

    # peace – index + middle spread (same as V)
    poses["peace"] = _asl_poses()["V"]

    # ok – index + thumb touch, others extended
    p = _base_hand(0.8, 0.4, 1.0, 1.0, 1.0, thumb_spread=0.25)
    p[I_TIP] = (p[I_TIP] + p[T_TIP]) / 2
    p[T_TIP] = p[I_TIP].copy()
    poses["ok"] = p

    # point – index pointing
    poses["point"] = _base_hand(0.2, 1.0, 0.1, 0.1, 0.1)

    # rock – index + pinky up, middle + ring folded
    poses["rock"] = _base_hand(0.2, 1.0, 0.1, 0.1, 1.0, thumb_spread=0.15)

    # call_me – thumb + pinky extended (phone gesture)
    p = _base_hand(0.85, 0.1, 0.1, 0.1, 0.95, thumb_spread=0.4)
    poses["call_me"] = p

    # i_love_you – ILY (index + pinky + thumb)
    poses["i_love_you"] = _base_hand(0.85, 1.0, 0.1, 0.1, 1.0, thumb_spread=0.35)

    return poses


# ═════════════════════════════════════════════════════════════════════════════
#  Sampler
# ═════════════════════════════════════════════════════════════════════════════

CANONICAL_POSES: Dict[str, np.ndarray] = {}


def _build_pose_library() -> None:
    global CANONICAL_POSES
    CANONICAL_POSES.update(_asl_poses())
    CANONICAL_POSES.update(_number_poses())
    CANONICAL_POSES.update(_command_poses())


def _perturb(
    pts: np.ndarray,
    noise_std:   float = 0.008,
    scale_range: tuple = (0.85, 1.15),
    rot_deg:     float = 25.0,
    z_noise_std: float = 0.005,
    finger_jitter: float = 0.015,
) -> np.ndarray:
    """
    Apply realistic anatomical perturbations to a canonical pose.

    Returns a new (21, 3) array.
    """
    p = pts.copy()

    # Per-landmark Gaussian noise
    p += np.random.normal(0, noise_std, p.shape).astype(np.float32)

    # Z-axis noise (depth variation)
    p[:, 2] += np.random.normal(0, z_noise_std, 21).astype(np.float32)

    # Per-finger micro-bend (realistic variation in how bent each finger is)
    for mcp_idx, pip_idx, dip_idx, tip_idx in [
        (I_MCP, I_PIP, I_DIP, I_TIP),
        (M_MCP, M_PIP, M_DIP, M_TIP),
        (R_MCP, R_PIP, R_DIP, R_TIP),
        (P_MCP, P_PIP, P_DIP, P_TIP),
    ]:
        jitter = np.random.uniform(-finger_jitter, finger_jitter)
        for idx in [pip_idx, dip_idx, tip_idx]:
            p[idx, 1] += jitter

    # Random 2-D rotation around wrist
    angle = np.deg2rad(np.random.uniform(-rot_deg, rot_deg))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], np.float32)
    origin = p[WRIST, :2].copy()
    p[:, :2] = (p[:, :2] - origin) @ rot.T + origin

    # Random scale
    scale = np.random.uniform(*scale_range)
    p[:, :2] = (p[:, :2] - p[WRIST, :2]) * scale + p[WRIST, :2]

    # Random global translation (removed during feature normalisation)
    p[:, :2] += np.random.uniform(-0.2, 0.2, (1, 2)).astype(np.float32)

    return p


def generate_samples(
    gesture: str,
    n: int = SAMPLES_PER_CLASS,
    noise_level: float = 1.0,
) -> List[np.ndarray]:
    """
    Generate *n* augmented feature vectors for *gesture*.

    Returns list of (TOTAL_FEATURE_DIM,) arrays.
    """
    if gesture not in CANONICAL_POSES:
        raise KeyError(f"No canonical pose for gesture '{gesture}'")

    canonical = CANONICAL_POSES[gesture]
    samples   = []

    for _ in range(n):
        pts  = _perturb(canonical, noise_std=0.008 * noise_level)
        feat = extract_features(pts)
        if feat is not None:
            samples.append(feat)

    return samples


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

CATEGORIES = {
    "alphabets": ALPHABET_GESTURES,
    "numbers"  : NUMBER_GESTURES,
    "commands" : COMMAND_GESTURES,
    "all"      : ALL_GESTURES,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic hand-gesture training data (no webcam needed)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument("--gesture",  metavar="NAME",
                       help="Single gesture (e.g. A, thumbs_up)")
    group.add_argument("--category", choices=list(CATEGORIES.keys()),
                       default="all")
    p.add_argument("--samples",      type=int, default=SAMPLES_PER_CLASS,
                   help=f"Samples per class (default {SAMPLES_PER_CLASS})")
    p.add_argument("--noise",        type=float, default=1.0,
                   help="Noise multiplier (0.5=low, 1.0=default, 2.0=high)")
    p.add_argument("--output",       type=str, default=str(RAW_DIR))
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    _build_pose_library()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    gestures = (
        [args.gesture] if args.gesture
        else CATEGORIES[args.category]
    )

    # Filter to gestures that have canonical poses
    available = [g for g in gestures if g in CANONICAL_POSES]
    missing   = [g for g in gestures if g not in CANONICAL_POSES]
    if missing:
        print(f"  ⚠️  No canonical pose for: {missing}  (skipping)")

    print(f"\n{'═'*55}")
    print(f"  🤖  Synthetic Dataset Generator")
    print(f"{'═'*55}")
    print(f"  Gestures   : {len(available)}")
    print(f"  Samples/cls: {args.samples}")
    print(f"  Noise level: {args.noise}")
    print(f"  Output dir : {output_dir}")
    print(f"{'═'*55}\n")

    total_written = 0
    for gesture in tqdm(available, desc="Generating", unit="class"):
        samples = generate_samples(gesture, n=args.samples, noise_level=args.noise)

        csv_path = output_dir / f"{gesture}.csv"
        mode = "a" if csv_path.exists() else "w"
        with open(csv_path, mode, newline="") as f:
            writer = csv.writer(f)
            for feat in samples:
                writer.writerow([gesture] + feat.tolist())

        total_written += len(samples)

    print(f"\n✅  Generated {total_written:,} samples across {len(available)} classes")
    print(f"   Saved to: {output_dir}")
    print(f"\n   Next step → python train_model.py\n")


if __name__ == "__main__":
    main()
