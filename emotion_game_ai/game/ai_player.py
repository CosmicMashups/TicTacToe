"""Emotion- and sentiment-aware AI policy wrapper around Minimax."""

from __future__ import annotations

import random
from dataclasses import dataclass

from emotion_game_ai.game.board import Board
from emotion_game_ai.game.minimax import get_best_move, minimax, AI_MARK, HUMAN_MARK
from emotion_game_ai.utils.threading_utils import DifficultyTuning, EmotionLabel


@dataclass
class AIDecision:
    move: tuple[int, int]
    mode: str  # "Competitive" | "Assistive"
    thinking_delay_s: float
    message: str


def _suboptimal_move(board: Board) -> tuple[int, int]:
    """
    Choose a legal move that is not the best move when possible.
    Falls back to the best move if no alternative exists.
    """
    best = get_best_move(board)
    empties = board.empty_cells()
    if len(empties) <= 1:
        return best
    candidates = [m for m in empties if m != best]
    if not candidates:
        return best

    # Prefer "reasonable" suboptimal: pick the move with the next-best minimax value.
    scored: list[tuple[int, tuple[int, int]]] = []
    for r, c in candidates:
        b2 = board.copy()
        b2.place(r, c, AI_MARK)
        val = minimax(b2, 0, False)
        scored.append((val, (r, c)))
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[0][1]


def decide_move(
    board: Board,
    emotion: EmotionLabel,
    tuning: DifficultyTuning,
    rng: random.Random,
) -> AIDecision:
    # Vision emotion can still "nudge" competitive behavior,
    # but tuning is primarily driven by post-game NLP emotion behavior profiles.
    if emotion == "Happy" and tuning.assistive_mistake_prob <= 0.14:
        return AIDecision(
            move=get_best_move(board),
            mode="Competitive",
            thinking_delay_s=max(0.35, min(0.9, tuning.ai_thinking_delay_s)),
            message="You look confident. Let’s see if you can beat me.",
        )

    mistake_prob = max(0.05, min(0.45, tuning.assistive_mistake_prob))
    if rng.random() < mistake_prob:
        move = _suboptimal_move(board)
    else:
        move = get_best_move(board)

    personality = getattr(tuning, "ai_personality", "standard")
    if personality in {"supportive", "reassuring", "deescalating"}:
        msg = "Take your time. I'm here to help you improve."
    elif personality in {"friendly_competitive", "highly_competitive"}:
        msg = "Let’s make this round a challenge."
    elif personality in {"entertaining", "playful"}:
        msg = "Alright! Let’s keep it interesting."
    elif personality in {"apologetic_helpful"}:
        msg = "Sorry that felt rough. Let’s slow down and find a good move."
    else:
        msg = "Let's keep going."

    return AIDecision(
        move=move,
        mode="Assistive",
        thinking_delay_s=max(0.35, min(2.0, tuning.ai_thinking_delay_s)),
        message=msg,
    )

