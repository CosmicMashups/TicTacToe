"""Threading utilities and shared state for the app."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import threading
import time
from typing import Deque, Optional

from emotion_game_ai.emotion_behavior import EmotionBehavior, behavior_for_emotion

EmotionLabel = str  # "Happy" | "Neutral"


class StoppableThread(threading.Thread):
    """Thread with a cooperative stop event."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stop_event = threading.Event()

    def stop(self) -> None:
        self.stop_event.set()


@dataclass
class RollingMajority:
    """Rolling majority vote for categorical labels."""

    window_size: int = 30
    _items: Deque[EmotionLabel] = field(default_factory=deque, init=False)

    def add(self, label: EmotionLabel) -> None:
        self._items.append(label)
        while len(self._items) > self.window_size:
            self._items.popleft()

    def majority(self, default: EmotionLabel = "Neutral") -> EmotionLabel:
        if not self._items:
            return default
        counts = Counter(self._items)
        return counts.most_common(1)[0][0]


@dataclass
class VisionSnapshot:
    """Latest vision outputs for UI consumption."""

    emotion_raw: EmotionLabel = "Neutral"
    emotion_smoothed: EmotionLabel = "Neutral"
    mouth_width: float = 0.0
    landmarks_detected: int = 0
    fps: float = 0.0
    camera_index: int = 0
    camera_ok: bool = False
    # RGB frame for UI preview (numpy uint8 HxWx3), optional
    preview_rgb: Optional[object] = None


@dataclass
class GameStats:
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    emotion_counts: dict[str, int] = field(default_factory=lambda: {"Happy": 0, "Neutral": 0})

    def record_emotion(self, emotion: EmotionLabel) -> None:
        self.emotion_counts[emotion] = self.emotion_counts.get(emotion, 0) + 1


@dataclass
class DifficultyTuning:
    """Cross-match tuning updated via sentiment."""

    assistive_mistake_prob: float = 0.22  # base for Neutral mode
    ai_thinking_delay_s: float = 1.2
    hint_frequency: float = 0.35
    ai_personality: str = "standard"
    ui_theme: str = "calm_blue"

    def adjust_for_sentiment(self, sentiment: str) -> None:
        if sentiment == "negative":
            self.assistive_mistake_prob = min(0.35, self.assistive_mistake_prob + 0.03)
        elif sentiment == "positive":
            self.assistive_mistake_prob = max(0.10, self.assistive_mistake_prob - 0.03)

    def apply_behavior(self, emotion: str) -> EmotionBehavior:
        """
        Apply a full gameplay response profile derived from an emotion label.
        Returns the resolved behavior profile.
        """
        b = behavior_for_emotion(emotion)
        self.ai_thinking_delay_s = float(b.ai_thinking_time)
        self.hint_frequency = float(b.hint_frequency)
        self.ai_personality = b.ai_personality
        self.ui_theme = b.ui_theme

        if b.ai_difficulty in {"significantly_reduced"}:
            self.assistive_mistake_prob = 0.33
        elif b.ai_difficulty in {"reduced"}:
            self.assistive_mistake_prob = 0.28
        elif b.ai_difficulty in {"balanced"}:
            self.assistive_mistake_prob = 0.22
        elif b.ai_difficulty in {"slightly_increased"}:
            self.assistive_mistake_prob = 0.18
        elif b.ai_difficulty in {"competitive", "maximum"}:
            self.assistive_mistake_prob = 0.12
        elif b.ai_difficulty in {"unpredictable"}:
            self.assistive_mistake_prob = 0.16
        return b


class SharedState:
    """Thread-safe state shared between vision thread and game/UI."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.vision = VisionSnapshot()
        self.stats = GameStats()
        self.tuning = DifficultyTuning()
        self.current_emotion: EmotionLabel = "Neutral"
        self.last_feedback_emotion: str = "neutral"
        self.last_behavior: EmotionBehavior = behavior_for_emotion("neutral")
        self.ai_mode: str = "Assistive"
        self.dialogue_message: str = ""
        self.dialogue_until_s: float = 0.0
        self.available_cameras: list[int] = [0]
        self._requested_camera_index: int = 0
        self._max_camera_index: int = 4

    def set_dialogue(self, message: str, ttl_s: float = 3.0) -> None:
        now = time.perf_counter()
        with self._lock:
            self.dialogue_message = message
            self.dialogue_until_s = now + ttl_s

    def get_dialogue(self) -> str:
        now = time.perf_counter()
        with self._lock:
            if self.dialogue_message and now <= self.dialogue_until_s:
                return self.dialogue_message
            return ""

    def update_vision(self, snap: VisionSnapshot) -> None:
        with self._lock:
            self.vision = snap
            self.current_emotion = snap.emotion_smoothed

    def get_snapshot(self) -> tuple[EmotionLabel, VisionSnapshot, GameStats, DifficultyTuning, str]:
        with self._lock:
            return (
                self.current_emotion,
                self.vision,
                self.stats,
                self.tuning,
                self.ai_mode,
            )

    def set_available_cameras(self, indices: list[int]) -> None:
        cleaned = sorted({int(i) for i in indices if i is not None and int(i) >= 0})
        if not cleaned:
            cleaned = [0]
        with self._lock:
            self.available_cameras = cleaned
            if self._requested_camera_index not in self.available_cameras:
                self._requested_camera_index = self.available_cameras[0]

    def request_camera(self, camera_index: int) -> None:
        with self._lock:
            self._requested_camera_index = int(camera_index)

    def request_next_camera(self) -> int:
        with self._lock:
            # Cycle through a fixed index range [0 .. _max_camera_index],
            # regardless of which indices were probed as available.
            self._requested_camera_index = (
                (self._requested_camera_index + 1) % (self._max_camera_index + 1)
            )
            return self._requested_camera_index

    def get_requested_camera(self) -> int:
        with self._lock:
            return int(self._requested_camera_index)

    def camera_status_summary(self, max_index: int = 4) -> str:
        """
        Human-friendly summary of camera indices and whether they were
        detected at startup. Uses available_cameras to mark indices as OK.
        """
        with self._lock:
            ok_set = set(int(i) for i in self.available_cameras)
        parts: list[str] = []
        for idx in range(0, int(max_index) + 1):
            label = "OK" if idx in ok_set else "N/A"
            parts.append(f"{idx}({label})")
        return ", ".join(parts)

    def set_feedback_emotion(self, emotion: str) -> EmotionBehavior:
        with self._lock:
            self.last_feedback_emotion = (emotion or "").strip().lower() or "neutral"
            self.last_behavior = self.tuning.apply_behavior(self.last_feedback_emotion)
            return self.last_behavior
