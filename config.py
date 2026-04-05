"""
Configuration module for Hand Gesture Recognition System.
Centralises all hyperparameters, paths, and gesture mappings.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent
DATA_DIR       = BASE_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
MODEL_DIR      = BASE_DIR / "models"
LOGS_DIR       = BASE_DIR / "logs"
ASSETS_DIR     = BASE_DIR / "assets"

BEST_MODEL_PATH     = MODEL_DIR / "gesture_model_best.h5"
LABEL_ENCODER_PATH  = MODEL_DIR / "label_encoder.pkl"
SCALER_PATH         = MODEL_DIR / "feature_scaler.pkl"
HISTORY_PATH        = LOGS_DIR  / "training_history.json"

# ─────────────────────────────────────────────
# Gesture Classes
# ─────────────────────────────────────────────
ALPHABET_GESTURES = [chr(c) for c in range(ord('A'), ord('Z') + 1)]   # A-Z
NUMBER_GESTURES   = [str(n) for n in range(10)]                        # 0-9
COMMAND_GESTURES  = [
    "hello", "yes", "no", "stop",
    "thumbs_up", "thumbs_down",
    "peace", "ok", "point", "rock",
    "call_me", "i_love_you",
]

ALL_GESTURES = ALPHABET_GESTURES + NUMBER_GESTURES + COMMAND_GESTURES
NUM_CLASSES  = len(ALL_GESTURES)

# Human-readable display labels for commands
GESTURE_DISPLAY = {
    "thumbs_up"   : "👍 Thumbs Up",
    "thumbs_down" : "👎 Thumbs Down",
    "peace"       : "✌️ Peace",
    "ok"          : "👌 OK",
    "point"       : "☝️ Point",
    "rock"        : "🤘 Rock",
    "call_me"     : "🤙 Call Me",
    "i_love_you"  : "🤟 I Love You",
    "hello"       : "👋 Hello",
    "yes"         : "✅ Yes",
    "no"          : "❌ No",
    "stop"        : "🛑 Stop",
}

# ─────────────────────────────────────────────
# MediaPipe Settings
# ─────────────────────────────────────────────
MEDIAPIPE_CFG = {
    "static_image_mode"        : False,
    "max_num_hands"            : 2,
    "model_complexity"         : 1,
    "min_detection_confidence" : 0.70,
    "min_tracking_confidence"  : 0.60,
}

NUM_LANDMARKS  = 21      # MediaPipe hand landmarks
COORDS_PER_LM  = 3       # x, y, z
RAW_FEATURE_DIM = NUM_LANDMARKS * COORDS_PER_LM   # 63

# Additional engineered features
ANGLE_FEATURES     = 15   # joint angles per finger pair
DISTANCE_FEATURES  = 10   # key inter-landmark distances
TOTAL_FEATURE_DIM  = RAW_FEATURE_DIM + ANGLE_FEATURES + DISTANCE_FEATURES  # 88

# ─────────────────────────────────────────────
# Data Collection
# ─────────────────────────────────────────────
SAMPLES_PER_CLASS   = 200   # frames captured per gesture class
COLLECTION_FPS      = 15    # target FPS during collection
COUNTDOWN_SECONDS   = 3     # countdown before capture starts

# ─────────────────────────────────────────────
# Model Hyperparameters
# ─────────────────────────────────────────────
MODEL_CFG = {
    "hidden_units"  : [512, 256, 128, 64],
    "dropout_rate"  : 0.4,
    "learning_rate" : 1e-3,
    "batch_size"    : 64,
    "epochs"        : 100,
    "patience"      : 15,        # early-stopping patience
    "val_split"     : 0.20,
    "test_split"    : 0.10,
    "activation"    : "relu",
}

# ─────────────────────────────────────────────
# Inference / Display
# ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.65   # min confidence to display prediction
SMOOTHING_WINDOW     = 7      # frames for majority-vote smoothing
TTS_COOLDOWN_SEC     = 2.0    # min seconds between TTS announcements

# Display colours  (BGR for OpenCV)
COLOR_GREEN  = (0, 220, 100)
COLOR_RED    = (0, 60, 220)
COLOR_BLUE   = (220, 120, 0)
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0, 0, 0)
COLOR_YELLOW = (0, 200, 220)
COLOR_CYAN   = (220, 220, 0)

FONT = "data/Roboto-Bold.ttf"   # optional; falls back to cv2 font if missing
