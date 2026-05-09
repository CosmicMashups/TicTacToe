"""Emotion → gameplay/UI behavior mapping for the adaptive game loop."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmotionBehavior:
    emotion: str
    ai_difficulty: str
    ai_personality: str
    ui_theme: str
    animation_intensity: float
    hint_frequency: float
    ai_thinking_time: float
    system_message: str
    ui_effect: str


_BEHAVIORS: dict[str, EmotionBehavior] = {
    "neutral": EmotionBehavior(
        emotion="neutral",
        ai_difficulty="balanced",
        ai_personality="standard",
        ui_theme="calm_blue",
        animation_intensity=0.35,
        hint_frequency=0.35,
        ai_thinking_time=1.1,
        system_message="Let's continue playing.",
        ui_effect="none",
    ),
    "love": EmotionBehavior(
        emotion="love",
        ai_difficulty="slightly_increased",
        ai_personality="friendly_competitive",
        ui_theme="warm_pink",
        animation_intensity=0.80,
        hint_frequency=0.15,
        ai_thinking_time=0.7,
        system_message="I'm glad you're enjoying the game!",
        ui_effect="hearts",
    ),
    "happiness": EmotionBehavior(
        emotion="happiness",
        ai_difficulty="competitive",
        ai_personality="playful",
        ui_theme="bright_glow",
        animation_intensity=0.95,
        hint_frequency=0.15,
        ai_thinking_time=0.65,
        system_message="You seem happy! Let's make the next round exciting.",
        ui_effect="board_glow",
    ),
    "sadness": EmotionBehavior(
        emotion="sadness",
        ai_difficulty="reduced",
        ai_personality="supportive",
        ui_theme="soft_blue",
        animation_intensity=0.20,
        hint_frequency=0.75,
        ai_thinking_time=1.4,
        system_message="Don't worry. Every move is a chance to improve.",
        ui_effect="calm_bg",
    ),
    "relief": EmotionBehavior(
        emotion="relief",
        ai_difficulty="balanced",
        ai_personality="calm",
        ui_theme="calm_blue",
        animation_intensity=0.25,
        hint_frequency=0.35,
        ai_thinking_time=1.0,
        system_message="Looks like you're feeling better.",
        ui_effect="slow_fade",
    ),
    "hate": EmotionBehavior(
        emotion="hate",
        ai_difficulty="significantly_reduced",
        ai_personality="apologetic_helpful",
        ui_theme="supportive",
        animation_intensity=0.15,
        hint_frequency=0.95,
        ai_thinking_time=1.6,
        system_message="I'm sorry the game felt frustrating. Let's try an easier round.",
        ui_effect="supportive_theme",
    ),
    "anger": EmotionBehavior(
        emotion="anger",
        ai_difficulty="reduced",
        ai_personality="deescalating",
        ui_theme="calm_blue",
        animation_intensity=0.20,
        hint_frequency=0.70,
        ai_thinking_time=1.5,
        system_message="Take a moment. Let's play at a calmer pace.",
        ui_effect="calming_overlay",
    ),
    "fun": EmotionBehavior(
        emotion="fun",
        ai_difficulty="slightly_increased",
        ai_personality="playful",
        ui_theme="party",
        animation_intensity=1.00,
        hint_frequency=0.20,
        ai_thinking_time=0.75,
        system_message="Looks like you're having fun!",
        ui_effect="particles",
    ),
    "enthusiasm": EmotionBehavior(
        emotion="enthusiasm",
        ai_difficulty="maximum",
        ai_personality="highly_competitive",
        ui_theme="dynamic",
        animation_intensity=1.00,
        hint_frequency=0.10,
        ai_thinking_time=0.55,
        system_message="You seem excited! Let's push your skills.",
        ui_effect="dynamic_lighting",
    ),
    "surprise": EmotionBehavior(
        emotion="surprise",
        ai_difficulty="balanced",
        ai_personality="curious",
        ui_theme="flash",
        animation_intensity=0.70,
        hint_frequency=0.30,
        ai_thinking_time=1.0,
        system_message="That was unexpected! Let's see what happens next.",
        ui_effect="flash",
    ),
    "empty": EmotionBehavior(
        emotion="empty",
        ai_difficulty="balanced",
        ai_personality="encouraging",
        ui_theme="subtle_pulse",
        animation_intensity=0.30,
        hint_frequency=0.45,
        ai_thinking_time=1.1,
        system_message="Let's keep the game engaging.",
        ui_effect="subtle_pulse",
    ),
    "worry": EmotionBehavior(
        emotion="worry",
        ai_difficulty="reduced",
        ai_personality="reassuring",
        ui_theme="soft_blue",
        animation_intensity=0.20,
        hint_frequency=0.80,
        ai_thinking_time=1.4,
        system_message="It's okay to take your time.",
        ui_effect="calm_bg",
    ),
    "boredom": EmotionBehavior(
        emotion="boredom",
        ai_difficulty="unpredictable",
        ai_personality="entertaining",
        ui_theme="dynamic",
        animation_intensity=1.00,
        hint_frequency=0.10,
        ai_thinking_time=0.8,
        system_message="Let's make things more interesting!",
        ui_effect="dynamic_board",
    ),
}


def behavior_for_emotion(emotion: str) -> EmotionBehavior:
    key = (emotion or "").strip().lower() or "neutral"
    return _BEHAVIORS.get(key, _BEHAVIORS["neutral"])

