"""
tts_engine.py
─────────────
Text-to-speech output for recognised gestures.

Uses pyttsx3 (offline, cross-platform) as the primary engine,
with gTTS (requires internet) as a fallback / higher quality option.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

from config import TTS_COOLDOWN_SEC


# ── base class ────────────────────────────────────────────────────────────────

class TTSEngine:
    """
    Thread-safe TTS wrapper with:
      • Cooldown enforcement (prevents rapid-fire speech)
      • Non-blocking background speech via daemon thread
      • Deduplication (won't repeat the same phrase immediately)
    """

    def __init__(
        self,
        cooldown_sec: float = TTS_COOLDOWN_SEC,
        engine: str = "auto",   # "pyttsx3" | "gtts" | "auto"
    ):
        self.cooldown_sec   = cooldown_sec
        self._last_spoken   = ""
        self._last_time     = 0.0
        self._lock          = threading.Lock()
        self._queue: deque  = deque(maxlen=1)   # only keep latest
        self._running       = True

        self._engine_name = engine
        self._engine      = self._init_engine(engine)

        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._worker.start()

    # ── public ───────────────────────────────────────────────────────────────

    def speak(self, text: str, force: bool = False) -> bool:
        """
        Enqueue *text* to be spoken.

        Returns True if the request was accepted, False if throttled/duplicate.
        """
        now = time.time()
        with self._lock:
            if not force:
                if text == self._last_spoken and (now - self._last_time) < self.cooldown_sec:
                    return False
                if (now - self._last_time) < self.cooldown_sec:
                    return False
            self._queue.append(text)
            return True

    def stop(self):
        """Cleanly stop the background worker."""
        self._running = False

    # ── internal ─────────────────────────────────────────────────────────────

    def _run_loop(self):
        while self._running:
            if self._queue:
                text = self._queue.popleft()
                with self._lock:
                    self._last_spoken = text
                    self._last_time   = time.time()
                self._say(text)
            time.sleep(0.05)

    def _say(self, text: str):
        try:
            if self._engine_name in ("pyttsx3", "auto") and self._engine:
                self._engine.say(text)
                self._engine.runAndWait()
            elif self._engine_name == "gtts":
                self._say_gtts(text)
        except Exception as e:
            print(f"[TTS] Error: {e}")

    @staticmethod
    def _say_gtts(text: str):
        """gTTS fallback — saves to temp file and plays with pygame."""
        try:
            import os, tempfile
            from gtts import gTTS
            import pygame

            tts = gTTS(text=text, lang="en", slow=False)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tts.save(tmp.name)
                tmp_path = tmp.name

            pygame.mixer.init()
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            os.unlink(tmp_path)
        except ImportError:
            print("[TTS] gTTS / pygame not installed.")

    @staticmethod
    def _init_engine(engine_name: str):
        """Try to initialise pyttsx3, return None on failure."""
        if engine_name == "gtts":
            return None
        try:
            import pyttsx3
            eng = pyttsx3.init()
            eng.setProperty("rate", 160)
            eng.setProperty("volume", 0.95)
            return eng
        except Exception:
            return None


# ── null engine ───────────────────────────────────────────────────────────────

class NullTTS:
    """Drop-in replacement when TTS is disabled."""
    def speak(self, text: str, force: bool = False) -> bool:
        return False
    def stop(self):
        pass


# ── factory ───────────────────────────────────────────────────────────────────

def make_tts(enabled: bool = True, **kwargs) -> TTSEngine | NullTTS:
    if not enabled:
        return NullTTS()
    return TTSEngine(**kwargs)
