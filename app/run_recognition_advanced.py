#!/usr/bin/env python3
"""
run_recognition_advanced.py
────────────────────────────
Enhanced real-time recognition with ALL advanced features:

  • Per-hand independent classification
  • Two-hand compound gesture detection
  • Temporal event detection (double-tap, long-hold)
  • Gesture sequence recorder → automatic word spelling
  • Rich HUD with live event log
  • Statistics panel (session accuracy tracker)
  • Screenshot + video recording

Keyboard controls
─────────────────
  Q        – quit
  C        – clear sentence
  SPACE    – append space
  T        – toggle TTS
  S        – screenshot
  V        – start/stop video recording
  R        – reset all state
  H        – toggle HUD
  BACKSPACE – delete last char
  1-5      – set confidence threshold (1=50%, 5=90%)
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    CONFIDENCE_THRESHOLD, ASSETS_DIR, SMOOTHING_WINDOW,
    COLOR_GREEN, COLOR_WHITE, COLOR_YELLOW, COLOR_RED, COLOR_CYAN, COLOR_BLUE,
    GESTURE_DISPLAY,
)
from src.hand_tracker       import HandTracker
from src.feature_extractor  import extract_features
from src.gesture_model      import GesturePredictor
from src.gesture_combinator import MultiHandSession, TWO_HAND_DISPLAY
from src.tts_engine         import make_tts
from src.utils              import (
    PredictionSmoother, FPSCounter, OverlayRenderer,
    timestamp_label,
)

ASSETS_DIR.mkdir(parents=True, exist_ok=True)


# ── session statistics tracker ────────────────────────────────────────────────

class SessionStats:
    """Tracks per-session recognition statistics."""

    def __init__(self):
        self.frames_processed  : int   = 0
        self.gestures_detected : int   = 0
        self.gesture_counts    : dict  = {}
        self.confidence_hist   : Deque = deque(maxlen=200)
        self.start_time        : float = time.time()

    def record(self, label: str, conf: float) -> None:
        self.gestures_detected += 1
        self.gesture_counts[label] = self.gesture_counts.get(label, 0) + 1
        self.confidence_hist.append(conf)

    @property
    def elapsed_sec(self) -> float:
        return time.time() - self.start_time

    @property
    def mean_confidence(self) -> float:
        if not self.confidence_hist:
            return 0.0
        return float(np.mean(list(self.confidence_hist)))

    @property
    def top_gesture(self) -> Optional[str]:
        if not self.gesture_counts:
            return None
        return max(self.gesture_counts, key=lambda k: self.gesture_counts[k])


# ── event log ─────────────────────────────────────────────────────────────────

class EventLog:
    """Rolling log of notable events displayed in the HUD."""

    def __init__(self, max_entries: int = 5):
        self._entries: Deque[Tuple[str, float]] = deque(maxlen=max_entries)

    def add(self, msg: str) -> None:
        self._entries.appendleft((msg, time.time()))

    def draw(self, frame: np.ndarray, x: int, y_start: int) -> None:
        now = time.time()
        for i, (msg, ts) in enumerate(self._entries):
            age   = now - ts
            alpha = max(0.0, 1.0 - age / 6.0)   # fade out over 6s
            color = tuple(int(c * alpha) for c in COLOR_YELLOW)
            cv2.putText(frame, f"▸ {msg}",
                        (x, y_start + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


# ── main application ──────────────────────────────────────────────────────────

class AdvancedRecogniser:

    WINDOW_NAME = "✋  Advanced Gesture Recognition"

    def __init__(
        self,
        camera_index   : int   = 0,
        width          : int   = 1280,
        height         : int   = 720,
        tts_enabled    : bool  = True,
        conf_threshold : float = CONFIDENCE_THRESHOLD,
    ):
        self.camera_index   = camera_index
        self.width          = width
        self.height         = height
        self.conf_threshold = conf_threshold

        print("🔧 Loading model …")
        self.predictor  = GesturePredictor()
        self.tracker    = HandTracker()
        self.smoothers  = [PredictionSmoother(SMOOTHING_WINDOW) for _ in range(2)]
        self.fps_ctr    = FPSCounter()
        self.renderer   = OverlayRenderer()
        self.tts        = make_tts(enabled=tts_enabled)
        self.session    = MultiHandSession()
        self.stats      = SessionStats()
        self.event_log  = EventLog()

        self.sentence      : str   = ""
        self.tts_enabled   : bool  = tts_enabled
        self.show_hud      : bool  = True
        self.recording     : bool  = False
        self.video_writer  : Optional[cv2.VideoWriter] = None
        self.last_add_time : float = 0.0
        self.add_cooldown  : float = 1.0

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
        print("✅ Camera opened. Press Q to quit, H to toggle HUD.\n")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            fps   = self.fps_ctr.tick()
            self.stats.frames_processed += 1

            result = self.tracker.process(frame, draw=True)
            f      = result.annotated_frame

            # ── per-hand classification ───────────────────────────────────────
            hand_labels: List[Optional[str]] = [None, None]
            hand_confs : List[float]         = [0.0, 0.0]
            top_k_list : List                = []

            for i, hand in enumerate(result.hands[:2]):
                feat = extract_features(hand.landmarks)
                if feat is None:
                    continue
                label, conf, _ = self.predictor.predict(feat)
                top_k          = self.predictor.top_k(feat, k=3)
                s_lbl, s_conf  = self.smoothers[i].update(label, conf)

                hand_labels[i] = s_lbl
                hand_confs[i]  = s_conf
                if i == 0:
                    top_k_list = top_k

                if s_lbl:
                    self.stats.record(s_lbl, s_conf)

                # bbox
                display = GESTURE_DISPLAY.get(s_lbl or "", s_lbl or "")
                HandTracker.draw_bbox(
                    f, hand.bbox,
                    label=f"{hand.handedness}: {display} ({s_conf*100:.0f}%)" if s_lbl else hand.handedness,
                    color=COLOR_GREEN if s_lbl else (100, 100, 100),
                )

            # ── multi-hand session ────────────────────────────────────────────
            primary = hand_labels[0]
            events  = self.session.update(primary, result.hands)
            for ev_type, ev_val in events:
                self._handle_event(ev_type, ev_val)

            # ── auto-append to sentence ───────────────────────────────────────
            now = time.time()
            if (primary and
                    primary == getattr(self, "_last_primary", None) and
                    now - self.last_add_time >= self.add_cooldown):
                self._append_char(primary)
                self.last_add_time = now
            self._last_primary = primary

            # ── HUD ───────────────────────────────────────────────────────────
            if self.show_hud:
                self._draw_hud(f, hand_labels[0], hand_confs[0],
                               top_k_list, fps, result.num_hands)

            # ── recording ─────────────────────────────────────────────────────
            if self.recording and self.video_writer:
                self.video_writer.write(f)

            cv2.imshow(self.WINDOW_NAME, f)

            # ── keyboard ──────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                self.sentence = ""
            elif key == 32:
                self.sentence += " "
            elif key == ord("t"):
                self.tts_enabled = not self.tts_enabled
                self.event_log.add(f"TTS {'on' if self.tts_enabled else 'off'}")
            elif key == ord("s"):
                self._screenshot(f)
            elif key == ord("v"):
                self._toggle_recording(f)
            elif key == ord("r"):
                self._reset()
            elif key == ord("h"):
                self.show_hud = not self.show_hud
            elif key == 8:   # BACKSPACE
                self.sentence = self.sentence[:-1]
            elif ord("1") <= key <= ord("5"):
                thresholds = {ord("1"): 0.5, ord("2"): 0.6, ord("3"): 0.7,
                              ord("4"): 0.8, ord("5"): 0.9}
                self.conf_threshold = thresholds[key]
                self.event_log.add(f"Threshold → {self.conf_threshold:.0%}")

        self._cleanup(cap)

    # ── HUD rendering ─────────────────────────────────────────────────────────

    def _draw_hud(
        self,
        frame: np.ndarray,
        label: Optional[str],
        conf : float,
        top_k: list,
        fps  : float,
        n_hands: int,
    ) -> None:
        h, w = frame.shape[:2]

        # Top panel
        self.renderer.draw_background_panel(frame)
        self.renderer.draw_gesture_prediction(frame, label, conf)
        self.renderer.draw_fps(frame, fps)
        self.renderer.draw_hand_count(frame, n_hands)

        if top_k:
            self.renderer.draw_top_k(frame, top_k)
        self.renderer.draw_help(frame)

        # Sentence bar
        self.renderer.draw_sentence(frame, self.sentence)

        # Stats panel (bottom-left)
        stats_y = h - 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, stats_y), (220, h - 52), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame,
                    f"Session: {self.stats.elapsed_sec:.0f}s  "
                    f"Gestures: {self.stats.gestures_detected}",
                    (6, stats_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
        cv2.putText(frame,
                    f"Mean conf: {self.stats.mean_confidence*100:.1f}%  "
                    f"Top: {self.stats.top_gesture or '--'}",
                    (6, stats_y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
        cv2.putText(frame,
                    f"Threshold: {self.conf_threshold:.0%}  "
                    f"FPS: {fps:.0f}",
                    (6, stats_y + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

        # TTS + recording indicators
        tts_str = "TTS:ON" if self.tts_enabled else "TTS:OFF"
        cv2.putText(frame, tts_str, (6, stats_y + 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    COLOR_GREEN if self.tts_enabled else (80, 80, 80), 1)
        if self.recording:
            cv2.circle(frame, (w - 18, 18), 8, (0, 0, 220), -1)
            cv2.putText(frame, "REC", (w - 50, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 2)

        # Event log
        self.event_log.draw(frame, 6, stats_y + 82)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _handle_event(self, ev_type: str, ev_val: str) -> None:
        msg = None
        if ev_type == "two_hand":
            display = TWO_HAND_DISPLAY.get(ev_val, ev_val)
            msg = f"2-hand: {display}"
            if self.tts_enabled:
                self.tts.speak(ev_val.replace("_", " "), force=True)
        elif ev_type == "double_tap":
            msg = f"Double-tap: {ev_val}"
        elif ev_type == "long_hold":
            msg = f"Hold: {ev_val}"
            self.sentence += " "   # add a space on long hold
        elif ev_type == "sequence_complete":
            msg = f"Spelled: {ev_val}"
            if self.tts_enabled:
                self.tts.speak(ev_val, force=True)
        elif ev_type == "gesture_committed":
            pass   # handled by auto-append

        if msg:
            self.event_log.add(msg)

    def _append_char(self, label: str) -> None:
        if len(label) == 1:
            self.sentence += label
        else:
            self.sentence += f" [{label.upper()}] "
        if self.tts_enabled:
            self.tts.speak(label.replace("_", " "))

    def _screenshot(self, frame: np.ndarray) -> None:
        fname = ASSETS_DIR / f"shot_{timestamp_label()}.png"
        cv2.imwrite(str(fname), frame)
        self.event_log.add(f"Screenshot: {fname.name}")
        print(f"  📸 {fname}")

    def _toggle_recording(self, frame: np.ndarray) -> None:
        if self.recording:
            if self.video_writer:
                self.video_writer.release()
            self.recording  = False
            self.event_log.add("Recording stopped")
            print("  ⏹  Recording stopped")
        else:
            h, w = frame.shape[:2]
            fname = ASSETS_DIR / f"rec_{timestamp_label()}.avi"
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.video_writer = cv2.VideoWriter(str(fname), fourcc, 20, (w, h))
            self.recording = True
            self.event_log.add("Recording started")
            print(f"  ⏺  Recording → {fname}")

    def _reset(self) -> None:
        self.sentence = ""
        for s in self.smoothers:
            s.reset()
        self.session.reset()
        self.stats = SessionStats()
        self.event_log.add("Session reset")

    def _cleanup(self, cap: cv2.VideoCapture) -> None:
        if self.recording and self.video_writer:
            self.video_writer.release()
        cap.release()
        self.tracker.close()
        self.tts.stop()
        cv2.destroyAllWindows()
        print("\n👋 Session ended.")
        print(f"   Duration      : {self.stats.elapsed_sec:.0f}s")
        print(f"   Frames        : {self.stats.frames_processed}")
        print(f"   Gestures seen : {self.stats.gestures_detected}")
        if self.stats.top_gesture:
            print(f"   Most common   : {self.stats.top_gesture}")


# ── entry point ───────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Advanced Real-time Gesture Recognition")
    p.add_argument("--camera",    type=int,   default=0)
    p.add_argument("--width",     type=int,   default=1280)
    p.add_argument("--height",    type=int,   default=720)
    p.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    p.add_argument("--no-tts",    action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app  = AdvancedRecogniser(
        camera_index   = args.camera,
        width          = args.width,
        height         = args.height,
        tts_enabled    = not args.no_tts,
        conf_threshold = args.threshold,
    )
    app.run()
