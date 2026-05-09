"""Plotting-neutral helpers for evaluated Tic-Tac-Toe game trees."""

from __future__ import annotations

import math
from typing import Iterable

from emotion_game_ai.game.board import Board
from emotion_game_ai.game.game_tree import GameTreeNode, build_game_tree, state_to_board
from emotion_game_ai.game.minimax import minimax_alpha_beta


def build_evaluated_gameplay_tree(
    board: Board,
    *,
    is_maximizing: bool = True,
    depth_limit: int = 3,
) -> GameTreeNode:
    """Build a depth-limited tree and propagate alpha-beta values."""

    root = build_game_tree(board, depth_limit=depth_limit, is_maximizing=is_maximizing)
    evaluate_game_tree(root, -math.inf, math.inf)
    return root


def build_evaluated_gameplay_tree_by_levels(
    board: Board,
    *,
    is_maximizing: bool = True,
    levels: int = 4,
) -> GameTreeNode:
    """Build a tree using displayed levels, including the root."""

    depth_limit = max(0, levels - 1)
    return build_evaluated_gameplay_tree(board, is_maximizing=is_maximizing, depth_limit=depth_limit)


def evaluate_game_tree(node: GameTreeNode, alpha: float, beta: float) -> int:
    """Propagate values through an explicit tree and mark pruned siblings."""

    node.alpha = alpha
    node.beta = beta
    if not node.children:
        value = minimax_alpha_beta(state_to_board(node.board_state), 0, node.is_maximizing, -math.inf, math.inf)
        node.value = value
        return value

    if node.is_maximizing:
        value = -math.inf
        for idx, child in enumerate(node.children):
            value = max(value, evaluate_game_tree(child, alpha, beta))
            alpha = max(alpha, value)
            node.alpha = alpha
            node.value = int(value)
            if alpha >= beta:
                _mark_pruned(node.children[idx + 1 :])
                break
        return int(value)

    value = math.inf
    for idx, child in enumerate(node.children):
        value = min(value, evaluate_game_tree(child, alpha, beta))
        beta = min(beta, value)
        node.beta = beta
        node.value = int(value)
        if alpha >= beta:
            _mark_pruned(node.children[idx + 1 :])
            break
    return int(value)


def _mark_pruned(nodes: Iterable[GameTreeNode]) -> None:
    for node in nodes:
        node.pruned = True
        _mark_pruned(node.children)
