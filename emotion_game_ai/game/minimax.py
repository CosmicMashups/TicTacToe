"""Minimax algorithm for optimal Tic-Tac-Toe AI."""

from __future__ import annotations

from emotion_game_ai.game.board import Board


AI_MARK = "O"
HUMAN_MARK = "X"


def evaluate(board: Board) -> int:
    winner = board.winner()
    if winner == AI_MARK:
        return 10
    if winner == HUMAN_MARK:
        return -10
    return 0


def minimax(board: Board, depth: int, is_maximizing: bool) -> int:
    score = evaluate(board)
    if score != 0:
        # Prefer faster wins / slower losses.
        return score - depth if score > 0 else score + depth
    if board.is_draw():
        return 0

    if is_maximizing:
        best = -10_000
        for r, c in board.empty_cells():
            b2 = board.copy()
            b2.place(r, c, AI_MARK)
            best = max(best, minimax(b2, depth + 1, False))
        return best

    best = 10_000
    for r, c in board.empty_cells():
        b2 = board.copy()
        b2.place(r, c, HUMAN_MARK)
        best = min(best, minimax(b2, depth + 1, True))
    return best


def get_best_move(board: Board) -> tuple[int, int]:
    best_val = -10_000
    best_move = (-1, -1)
    for r, c in board.empty_cells():
        b2 = board.copy()
        b2.place(r, c, AI_MARK)
        move_val = minimax(b2, 0, False)
        if move_val > best_val:
            best_val = move_val
            best_move = (r, c)
    return best_move

