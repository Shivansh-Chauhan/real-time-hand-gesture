"""
gesture_combinator.py
─────────────────────
Advanced multi-hand gesture logic layer.

Sits on top of the single-hand GesturePredictor to detect:
  • Two-hand combined gestures  (e.g. "clap", "prayer", "frame")
  • Gesture sequences           (e.g. "I LOVE YOU" = I + space + L + Y)
  • Temporal patterns           (e.g. double-tap = same gesture twice quickly)

This module is imported by run_recognition.py and app.py when
multi-hand or sequence features are enabled.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from config import CONFIDENCE_THRESHOLD, SMOOTHING_WINDOW


# ── two-hand gesture rules ────────────────────────────────────────────────────

# Define which pair of single-hand gestures combine into a two-hand gesture.
# Key: frozenset({left_gesture, right_gesture}) → combined label
TWO_HAND_RULES: Dict[frozenset, str] = {
    frozenset({"hello",      "hello"     }): "both_hello",
    frozenset({"thumbs_up",  "thumbs_up" }): "double_thumbs_up",
    frozenset({"ok",         "ok"        }): "double_ok",
    frozenset({"stop",       "stop"      }): "hard_stop",
    frozenset({"i_love_you", "i_love_you"}): "double_ily",
    frozenset({"peace",      "peace"     }): "double_peace",
    frozenset({"rock",       "rock"      }): "rock_on",
}

# Display names for two-hand gestures
TWO_HAND_DISPLAY: Dict[str, str] = {
    "both_hello"        : "👋👋 Hello!",
    "double_thumbs_up"  : "👍👍 Excellent!",
    "double_ok"         : "👌👌 Perfect!",
    "hard_stop"         : "🛑🛑 STOP!",
    "double_ily"        : "🤟🤟 I Love You!",
    "double_peace"      : "✌️✌️ Peace!",
    "rock_on"           : "🤘🤘 Rock On!",
}


class TwoHandCombinator:
    """
    Detects two-hand compound gestures.

    Usage
    -----
    combinator = TwoHandCombinator()
    combined   = combinator.check(left_label, right_label)
    """

    def __init__(self, window: int = 5):
        self._left_buf : Deque[str] = deque(maxlen=window)
        self._right_buf: Deque[str] = deque(maxlen=window)

    def update(
        self,
        left_label  : Optional[str],
        right_label : Optional[str],
    ) -> Optional[str]:
        """
        Push new single-hand labels; return combined gesture if detected, else None.
        """
        if left_label:
            self._left_buf.append(left_label)
        if right_label:
            self._right_buf.append(right_label)

        if not self._left_buf or not self._right_buf:
            return None

        # Majority vote in each buffer
        from collections import Counter
        l_top = Counter(self._left_buf).most_common(1)[0][0]
        r_top = Counter(self._right_buf).most_common(1)[0][0]

        pair = frozenset({l_top, r_top})
        return TWO_HAND_RULES.get(pair)

    def reset(self) -> None:
        self._left_buf.clear()
        self._right_buf.clear()


# ── gesture sequence recorder ─────────────────────────────────────────────────

class GestureSequenceRecorder:
    """
    Records a timed sequence of stable gestures (e.g. to spell words).

    A gesture is "committed" when it remains stable for *hold_sec* seconds.
    After *gap_sec* of no stable gesture, the sequence is considered complete.

    Usage
    -----
    recorder = GestureSequenceRecorder()
    recorder.push(label)      # call each frame
    seq = recorder.get_sequence()
    if recorder.is_complete():
        word = recorder.flush()   # returns sequence string, clears buffer
    """

    def __init__(
        self,
        hold_sec    : float = 0.8,   # seconds a gesture must be held to commit
        gap_sec     : float = 2.0,   # seconds of silence to declare sequence done
        max_seq_len : int   = 30,
    ):
        self.hold_sec    = hold_sec
        self.gap_sec     = gap_sec
        self.max_seq_len = max_seq_len

        self._seq       : List[str] = []
        self._cur_label : str       = ""
        self._cur_start : float     = 0.0
        self._last_time : float     = 0.0
        self._committed : bool      = False   # current label already in seq?

    def push(self, label: Optional[str]) -> Optional[str]:
        """
        Push the current frame's gesture label.

        Returns the newly committed gesture if one was just added, else None.
        """
        now = time.time()

        if label is None:
            self._cur_label = ""
            self._committed = False
            return None

        if label != self._cur_label:
            self._cur_label = label
            self._cur_start = now
            self._committed = False
            return None

        # Same label is still active
        held = now - self._cur_start
        if held >= self.hold_sec and not self._committed:
            if len(self._seq) < self.max_seq_len:
                self._seq.append(label)
                self._last_time = now
                self._committed = True
                return label

        if self._committed:
            self._last_time = now

        return None

    def is_complete(self) -> bool:
        """True if a gap_sec silence has occurred after at least one committed gesture."""
        if not self._seq:
            return False
        return (time.time() - self._last_time) >= self.gap_sec

    def get_sequence(self) -> List[str]:
        return list(self._seq)

    def get_text(self) -> str:
        """Convert committed gesture sequence to a readable string."""
        parts = []
        for g in self._seq:
            if len(g) == 1:    # letter or digit
                parts.append(g)
            else:
                parts.append(f"[{g.upper()}]")
        return " ".join(parts)

    def flush(self) -> str:
        """Return the completed sequence string and clear the buffer."""
        text = self.get_text()
        self._seq.clear()
        self._last_time = 0.0
        return text

    def reset(self) -> None:
        self._seq.clear()
        self._cur_label = ""
        self._committed = False
        self._last_time = 0.0


# ── temporal pattern detector ─────────────────────────────────────────────────

class TemporalPatternDetector:
    """
    Detects short temporal patterns in the gesture stream, e.g.:
      • Double-tap: same gesture disappears and reappears within 1 second
      • Long hold:  gesture held for > 2 seconds

    Usage
    -----
    detector = TemporalPatternDetector()
    event = detector.update(label)   # returns "double_tap", "long_hold", or None
    """

    def __init__(
        self,
        double_tap_window : float = 1.0,    # sec between two occurrences
        long_hold_sec     : float = 2.5,    # sec to trigger long hold
    ):
        self.double_tap_window = double_tap_window
        self.long_hold_sec     = long_hold_sec

        self._prev_label     : str   = ""
        self._prev_end_time  : float = 0.0
        self._cur_label      : str   = ""
        self._cur_start      : float = 0.0
        self._long_hold_fired: bool  = False

    def update(self, label: Optional[str]) -> Optional[str]:
        now = time.time()

        if label is None:
            if self._cur_label:
                # Gesture just ended
                self._prev_label    = self._cur_label
                self._prev_end_time = now
                self._cur_label     = ""
                self._long_hold_fired = False
            return None

        if label != self._cur_label:
            # New gesture started
            event = None
            # Check double-tap
            if (label == self._prev_label
                    and (now - self._prev_end_time) <= self.double_tap_window):
                event = f"double_tap:{label}"

            self._cur_label  = label
            self._cur_start  = now
            self._long_hold_fired = False
            return event

        # Same gesture continuing
        held = now - self._cur_start
        if held >= self.long_hold_sec and not self._long_hold_fired:
            self._long_hold_fired = True
            return f"long_hold:{label}"

        return None


# ── high-level multi-hand session ─────────────────────────────────────────────

class MultiHandSession:
    """
    Orchestrates all multi-hand and temporal features in one object.

    Intended to wrap the per-frame recognition loop in run_recognition.py.

    Usage
    -----
    session = MultiHandSession()

    # In the frame loop:
    primary_label, primary_conf = smoother.update(label, conf)
    events = session.update(
        primary_label = primary_label,
        hand_results  = frame_result.hands,
    )
    for event in events:
        handle_event(event)
    """

    def __init__(self):
        self.combinator  = TwoHandCombinator()
        self.recorder    = GestureSequenceRecorder()
        self.temporal    = TemporalPatternDetector()

    def update(
        self,
        primary_label: Optional[str],
        hand_results: list,   # list[HandResult] from hand_tracker
    ) -> List[Tuple[str, str]]:
        """
        Returns a list of (event_type, value) tuples:
          ("gesture_committed", "A")
          ("sequence_complete", "H-E-L-L-O")
          ("double_tap", "thumbs_up")
          ("long_hold",  "stop")
          ("two_hand",   "double_thumbs_up")
        """
        events: List[Tuple[str, str]] = []

        # ── two-hand logic ────────────────────────────────────────────────────
        if len(hand_results) >= 2:
            left_label  = None
            right_label = None
            for hand in hand_results:
                if hand.handedness == "Left":
                    left_label  = primary_label   # simplified
                elif hand.handedness == "Right":
                    right_label = primary_label
            combined = self.combinator.update(left_label, right_label)
            if combined:
                events.append(("two_hand", combined))
        else:
            self.combinator.reset()

        # ── temporal pattern ──────────────────────────────────────────────────
        temporal_event = self.temporal.update(primary_label)
        if temporal_event:
            kind, label = temporal_event.split(":", 1)
            events.append((kind, label))

        # ── sequence recorder ─────────────────────────────────────────────────
        committed = self.recorder.push(primary_label)
        if committed:
            events.append(("gesture_committed", committed))

        if self.recorder.is_complete():
            seq_text = self.recorder.flush()
            if seq_text:
                events.append(("sequence_complete", seq_text))

        return events

    def reset(self) -> None:
        self.combinator.reset()
        self.recorder.reset()
        self.temporal = TemporalPatternDetector()
