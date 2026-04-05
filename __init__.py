"""Gesture Recognition – core package. Lazy imports to avoid mediapipe at module level."""

from src.feature_extractor import extract_features, augment_landmarks
from src.gesture_model     import GesturePredictor, GestureModelTrainer, build_model
from src.dataset_manager   import DatasetCollector, DatasetManager
from src.tts_engine        import TTSEngine, NullTTS, make_tts
from src.utils             import (
    PredictionSmoother, FPSCounter, OverlayRenderer,
    safe_resize, timestamp_label,
)

def get_hand_tracker():
    from src.hand_tracker import HandTracker, FrameResult, HandResult
    return HandTracker, FrameResult, HandResult

__all__ = [
    "extract_features", "augment_landmarks",
    "GesturePredictor", "GestureModelTrainer", "build_model",
    "DatasetCollector", "DatasetManager",
    "TTSEngine", "NullTTS", "make_tts",
    "PredictionSmoother", "FPSCounter", "OverlayRenderer",
    "safe_resize", "timestamp_label",
    "get_hand_tracker",
]
