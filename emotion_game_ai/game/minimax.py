"""Minimax and alpha-beta search for optimal Tic-Tac-Toe AI."""

from __future__ import annotations

import math
from typing import Any

from emotion_game_ai.game.board import Board
from emotion_game_ai.game.game_tree import GameTreeNode, board_to_state
from emotion_game_ai.game.search_stats import SearchStats


AI_MARK = "O"
HUMAN_MARK = "X"


def evaluate(board: Board) -> int:
    winner = board.winner()
    if winner == AI_MARK:
        return 10
    if winner == HUMAN_MARK:
        return -10
    return 0


def _terminal_score(board: Board, depth: int) -> int | None:
    score = evaluate(board)
    if score != 0:
        # Prefer faster wins / slower losses.
        return score - depth if score > 0 else score + depth
    if board.is_draw():
        return 0
    return None


def minimax(
    board: Board,
    depth: int,
    is_maximizing: bool,
    stats: SearchStats | None = None,
    tree_node: GameTreeNode | None = None,
) -> int:
    """Plain recursive Minimax, kept for comparison and compatibility."""

    if stats is not None:
        stats.visit(depth)

    terminal = _terminal_score(board, depth)
    if terminal is not None:
        if stats is not None:
            stats.record_leaf(depth)
        if tree_node is not None:
            tree_node.value = terminal
        return terminal

    if is_maximizing:
        best = -10_000
        for r, c in board.empty_cells():
            b2 = board.copy()
            b2.place(r, c, AI_MARK)
            child_node = _add_tree_child(tree_node, b2, (r, c), depth + 1, False)
            best = max(best, minimax(b2, depth + 1, False, stats, child_node))
        if tree_node is not None:
            tree_node.value = best
        return best

    best = 10_000
    for r, c in board.empty_cells():
        b2 = board.copy()
        b2.place(r, c, HUMAN_MARK)
        child_node = _add_tree_child(tree_node, b2, (r, c), depth + 1, True)
        best = min(best, minimax(b2, depth + 1, True, stats, child_node))
    if tree_node is not None:
        tree_node.value = best
    return best


def minimax_alpha_beta(
    board: Board,
    depth: int,
    is_maximizing: bool,
    alpha: float,
    beta: float,
    stats: SearchStats | None = None,
    tree_node: GameTreeNode | None = None,
) -> int:
    """Minimax with alpha-beta pruning and optional tree/stat tracing."""

    if stats is not None:
        stats.visit(depth)
    if tree_node is not None:
        tree_node.alpha = alpha
        tree_node.beta = beta

    terminal = _terminal_score(board, depth)
    if terminal is not None:
        if stats is not None:
            stats.record_leaf(depth)
        if tree_node is not None:
            tree_node.value = terminal
        return terminal

    moves = board.empty_cells()
    if is_maximizing:
        best = -10_000
        for idx, (r, c) in enumerate(moves):
            b2 = board.copy()
            b2.place(r, c, AI_MARK)
            child_node = _add_tree_child(tree_node, b2, (r, c), depth + 1, False)
            best = max(best, minimax_alpha_beta(b2, depth + 1, False, alpha, beta, stats, child_node))
            alpha = max(alpha, best)
            if tree_node is not None:
                tree_node.value = best
                tree_node.alpha = alpha
                tree_node.beta = beta
            if alpha >= beta:
                _record_pruned_siblings(tree_node, board, moves[idx + 1 :], AI_MARK, depth + 1, False, stats)
                break
        return best

    best = 10_000
    for idx, (r, c) in enumerate(moves):
        b2 = board.copy()
        b2.place(r, c, HUMAN_MARK)
        child_node = _add_tree_child(tree_node, b2, (r, c), depth + 1, True)
        best = min(best, minimax_alpha_beta(b2, depth + 1, True, alpha, beta, stats, child_node))
        beta = min(beta, best)
        if tree_node is not None:
            tree_node.value = best
            tree_node.alpha = alpha
            tree_node.beta = beta
        # Once alpha meets beta, the remaining siblings cannot affect ancestors.
        if alpha >= beta:
            _record_pruned_siblings(tree_node, board, moves[idx + 1 :], HUMAN_MARK, depth + 1, True, stats)
            break
    return best


def get_best_move(board: Board, stats: SearchStats | None = None, use_alpha_beta: bool = True) -> tuple[int, int]:
    """Return the best AI move, using alpha-beta by default."""

    move, _ = get_best_move_with_value(board, stats=stats, use_alpha_beta=use_alpha_beta)
    return move


def get_best_move_with_value(
    board: Board,
    stats: SearchStats | None = None,
    use_alpha_beta: bool = True,
) -> tuple[tuple[int, int], int]:
    """Return the best AI move and its value with stable row-major tie-breaking."""

    if stats is not None:
        stats.reset()
        stats.algorithm = "alpha-beta" if use_alpha_beta else "minimax"
        stats.start_timer()

    best_val = -10_000
    best_move = (-1, -1)
    for r, c in board.empty_cells():
        b2 = board.copy()
        b2.place(r, c, AI_MARK)
        if use_alpha_beta:
            move_val = minimax_alpha_beta(b2, 0, False, -math.inf, math.inf, stats)
        else:
            move_val = minimax(b2, 0, False, stats)
        if move_val > best_val:
            best_val = move_val
            best_move = (r, c)

    if stats is not None:
        stats.best_move = best_move
        stats.best_value = best_val
        stats.stop_timer()
    return best_move, best_val


def compare_search_algorithms(board: Board) -> dict[str, Any]:
    """Run plain Minimax and alpha-beta, then return a comparison summary."""

    plain_stats = SearchStats(algorithm="minimax")
    alpha_beta_stats = SearchStats(algorithm="alpha-beta")
    get_best_move_with_value(board, stats=plain_stats, use_alpha_beta=False)
    get_best_move_with_value(board, stats=alpha_beta_stats, use_alpha_beta=True)
    summary = SearchStats.comparison_summary(plain_stats, alpha_beta_stats)
    summary["minimax"] = plain_stats.summary()
    summary["alpha_beta"] = alpha_beta_stats.summary()
    return summary


def _add_tree_child(
    parent: GameTreeNode | None,
    board: Board,
    move: tuple[int, int],
    depth: int,
    is_maximizing: bool,
) -> GameTreeNode | None:
    if parent is None:
        return None
    return parent.add_child(
        GameTreeNode(
            board_state=board_to_state(board),
            move=move,
            depth=depth,
            is_maximizing=is_maximizing,
        )
    )


def _record_pruned_siblings(
    parent: GameTreeNode | None,
    board: Board,
    moves: list[tuple[int, int]],
    mark: str,
    depth: int,
    is_maximizing: bool,
    stats: SearchStats | None,
) -> None:
    if stats is not None:
        stats.record_pruning(len(moves))
    if parent is None:
        return
    for move in moves:
        b2 = board.copy()
        b2.place(move[0], move[1], mark)
        child = parent.add_child(
            GameTreeNode(
                board_state=board_to_state(b2),
                move=move,
                depth=depth,
                is_maximizing=is_maximizing,
                pruned=True,
            )
        )
        child.alpha = parent.alpha
        child.beta = parent.beta

