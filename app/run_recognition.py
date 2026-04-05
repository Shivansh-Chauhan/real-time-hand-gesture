#!/usr/bin/env python3
"""
run_recognition.py
──────────────────
Real-time hand gesture recognition using a webcam.

Features
────────
  • Live hand tracking via MediaPipe
  • Gesture classification with confidence bar
  • Majority-vote smoothing to reduce flicker
  • Sentence builder (append recognised chars)
  • Text-to-speech output
  • Multi-hand support (up to 2 hands)
  • Screenshot capture

Keyboard controls
─────────────────
  Q        – quit
  C        – clear sentence
  SPACE    – append space to sentence
  T        – toggle TTS
  S        – save screenshot
  BACKSPACE– delete last char
  R        – reset prediction smoother
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    CONFIDENCE_THRESHOLD, ASSETS_DIR,
    COLOR_GREEN, COLOR_WHITE, COLOR_YELLOW,
)
from src.hand_tracker      import HandTracker
from src.feature_extractor import extract_features
from src.gesture_model     import GesturePredictor
from src.tts_engine        import make_tts
from src.utils             import (
    PredictionSmoother, FPSCounter, OverlayRenderer, safe_resize, timestamp_label,
)


ASSETS_DIR.mkdir(parents=True, exist_ok=True)


# ── recogniser state ──────────────────────────────────────────────────────────

class GestureRecogniser:

    WINDOW_NAME = "✋  Hand Gesture Recognition"

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 960,
        height: int = 540,
        tts_enabled: bool = True,
    ):
        self.camera_index = camera_index
        self.width        = width
        self.height       = height

        print("🔧 Loading model …")
        self.predictor = GesturePredictor()

        self.tracker    = HandTracker()
        self.smoother   = PredictionSmoother()
        self.fps_ctr    = FPSCounter()
        self.renderer   = OverlayRenderer()
        self.tts        = make_tts(enabled=tts_enabled)

        self.sentence       : str   = ""
        self.tts_enabled    : bool  = tts_enabled
        self.last_label     : str   = ""
        self.last_add_time  : float = 0.0
        self.add_cooldown   : float = 1.2   # seconds between auto-appending chars

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, 30)

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, self.width, self.height)

        print(f"✅ Camera opened. Press Q to quit.\n")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)       # mirror
            fps   = self.fps_ctr.tick()

            # ── hand detection ────────────────────────────────────────────────
            result = self.tracker.process(frame, draw=True)

            smoothed_label, smoothed_conf = None, 0.0
            top_k = []

            for hand in result.hands:
                feat = extract_features(hand.landmarks)
                if feat is not None:
                    label, conf, _ = self.predictor.predict(feat)
                    top_k          = self.predictor.top_k(feat, k=3)

                    s_lbl, s_conf  = self.smoother.update(label, conf)
                    if s_lbl is not None:
                        smoothed_label = s_lbl
                        smoothed_conf  = s_conf

                # Bounding box
                HandTracker.draw_bbox(
                    result.annotated_frame,
                    hand.bbox,
                    label=f"{hand.handedness} {hand.score:.0%}",
                    color=COLOR_GREEN,
                )

            # ── sentence builder ──────────────────────────────────────────────
            now = time.time()
            if (smoothed_label is not None
                    and smoothed_label == self.last_label
                    and now - self.last_add_time >= self.add_cooldown):
                self._append_to_sentence(smoothed_label)
                self.last_add_time = now

            if smoothed_label != self.last_label:
                self.last_label = smoothed_label or ""

            # ── HUD ───────────────────────────────────────────────────────────
            f = result.annotated_frame
            self.renderer.draw_background_panel(f)
            self.renderer.draw_gesture_prediction(f, smoothed_label, smoothed_conf)
            self.renderer.draw_fps(f, fps)
            self.renderer.draw_hand_count(f, result.num_hands)
            self.renderer.draw_sentence(f, self.sentence)
            if top_k:
                self.renderer.draw_top_k(f, top_k)
            self.renderer.draw_help(f)

            # TTS indicator
            tts_str = "TTS: ON" if self.tts_enabled else "TTS: OFF"
            tts_col = COLOR_GREEN if self.tts_enabled else (80, 80, 80)
            cv2.putText(f, tts_str, (10, 108),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, tts_col, 1, cv2.LINE_AA)

            cv2.imshow(self.WINDOW_NAME, f)

            # ── keyboard ──────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                self.sentence = ""
            elif key == 32:   # SPACE
                self.sentence += " "
            elif key == ord("t"):
                self.tts_enabled = not self.tts_enabled
                print(f"  TTS {'enabled' if self.tts_enabled else 'disabled'}")
            elif key == ord("s"):
                self._screenshot(f)
            elif key == 8:    # BACKSPACE
                self.sentence = self.sentence[:-1]
            elif key == ord("r"):
                self.smoother.reset()

        cap.release()
        self.tracker.close()
        self.tts.stop()
        cv2.destroyAllWindows()
        print("\n👋 Bye!")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _append_to_sentence(self, label: str) -> None:
        """Append single-char label or word label to sentence."""
        if len(label) == 1:   # alphabet / digit
            self.sentence += label
        else:                  # command word
            self.sentence += f" [{label.upper()}] "

        if self.tts_enabled:
            speak_text = label.replace("_", " ")
            self.tts.speak(speak_text)

    def _screenshot(self, frame: np.ndarray) -> None:
        fname = ASSETS_DIR / f"screenshot_{timestamp_label()}.png"
        cv2.imwrite(str(fname), frame)
        print(f"  📸 Screenshot → {fname}")


# ── entry point ───────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Real-time Gesture Recognition")
    p.add_argument("--camera",   type=int,   default=0,     help="Camera index")
    p.add_argument("--width",    type=int,   default=960,   help="Display width")
    p.add_argument("--height",   type=int,   default=540,   help="Display height")
    p.add_argument("--no-tts",   action="store_true",       help="Disable TTS")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    recogniser = GestureRecogniser(
        camera_index = args.camera,
        width        = args.width,
        height       = args.height,
        tts_enabled  = not args.no_tts,
    )
    recogniser.run()
