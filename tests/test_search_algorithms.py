"""Regression checks for educational Tic-Tac-Toe search algorithms."""

from __future__ import annotations

import math
import unittest

from emotion_game_ai.game.board import Board
from emotion_game_ai.game.game_tree import GameTreeNode, generate_demo_tree
from emotion_game_ai.game.minimax import (
    compare_search_algorithms,
    get_best_move_with_value,
    minimax,
    minimax_alpha_beta,
)
from emotion_game_ai.game.search_stats import SearchStats


def board_from_rows(rows: list[list[str]]) -> Board:
    board = Board()
    board.grid = [row[:] for row in rows]
    return board


def walk(node: GameTreeNode) -> list[GameTreeNode]:
    nodes = [node]
    for child in node.children:
        nodes.extend(walk(child))
    return nodes


class SearchAlgorithmTests(unittest.TestCase):
    def test_alpha_beta_matches_plain_minimax_value(self) -> None:
        board = board_from_rows(
            [
                ["X", "O", "X"],
                ["", "O", ""],
                ["", "X", ""],
            ]
        )

        plain = minimax(board.copy(), 0, True)
        alpha_beta = minimax_alpha_beta(board.copy(), 0, True, -math.inf, math.inf)

        self.assertEqual(plain, alpha_beta)

    def test_best_move_matches_legacy_minimax_path(self) -> None:
        board = board_from_rows(
            [
                ["X", "", ""],
                ["", "O", ""],
                ["", "X", ""],
            ]
        )

        legacy_move, legacy_value = get_best_move_with_value(board, use_alpha_beta=False)
        alpha_beta_move, alpha_beta_value = get_best_move_with_value(board, use_alpha_beta=True)

        self.assertEqual(legacy_move, alpha_beta_move)
        self.assertEqual(legacy_value, alpha_beta_value)

    def test_compare_search_algorithms_reports_pruning(self) -> None:
        board = board_from_rows(
            [
                ["X", "", ""],
                ["", "O", ""],
                ["", "", "X"],
            ]
        )

        summary = compare_search_algorithms(board)

        self.assertTrue(summary["same_move"])
        self.assertTrue(summary["same_value"])
        self.assertLessEqual(summary["alpha_beta_nodes"], summary["minimax_nodes"])

    def test_demo_tree_has_four_levels_and_pruned_nodes(self) -> None:
        root = generate_demo_tree()
        nodes = walk(root)
        leaves = [node for node in nodes if not node.children]
        depths = {node.depth for node in nodes}
        terminal_values = {node.value for node in leaves}

        self.assertEqual(depths, {0, 1, 2, 3})
        self.assertGreaterEqual(len(leaves), 9)
        self.assertLessEqual(terminal_values, {10, 0, -10})
        self.assertTrue(any(node.pruned for node in nodes))
        self.assertIsNotNone(root.value)

    def test_stats_collect_tree_search_counts(self) -> None:
        board = board_from_rows(
            [
                ["X", "O", ""],
                ["", "O", ""],
                ["X", "", ""],
            ]
        )
        stats = SearchStats()

        get_best_move_with_value(board, stats=stats, use_alpha_beta=True)

        self.assertEqual(stats.algorithm, "alpha-beta")
        self.assertGreater(stats.nodes_visited, 0)
        self.assertGreater(stats.leaf_nodes, 0)
        self.assertGreaterEqual(stats.max_depth, 0)


if __name__ == "__main__":
    unittest.main()
