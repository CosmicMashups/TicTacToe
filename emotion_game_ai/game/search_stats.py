"""Search statistics for Minimax and alpha-beta diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any


@dataclass
class SearchStats:
    """Mutable accumulator for one search run."""

    nodes_visited: int = 0
    leaf_nodes: int = 0
    pruned_nodes: int = 0
    pruning_events: int = 0
    max_depth: int = 0
    execution_time_ms: float = 0.0
    algorithm: str = ""
    best_move: tuple[int, int] | None = None
    best_value: int | None = None
    _started_at: float | None = None

    def reset(self) -> None:
        """Clear counters so the same instance can be reused."""

        self.nodes_visited = 0
        self.leaf_nodes = 0
        self.pruned_nodes = 0
        self.pruning_events = 0
        self.max_depth = 0
        self.execution_time_ms = 0.0
        self.algorithm = ""
        self.best_move = None
        self.best_value = None
        self._started_at = None

    def start_timer(self) -> None:
        """Start measuring wall-clock time for the search."""

        self._started_at = time.perf_counter()

    def stop_timer(self) -> None:
        """Stop measuring wall-clock time if the timer is active."""

        if self._started_at is None:
            return
        self.execution_time_ms = (time.perf_counter() - self._started_at) * 1000.0
        self._started_at = None

    def visit(self, depth: int) -> None:
        """Record a node visit at the given search depth."""

        self.nodes_visited += 1
        self.max_depth = max(self.max_depth, depth)

    def record_leaf(self, depth: int) -> None:
        """Record that a terminal or depth-limited node was evaluated."""

        self.leaf_nodes += 1
        self.max_depth = max(self.max_depth, depth)

    def record_pruning(self, pruned_nodes: int) -> None:
        """Record one alpha-beta cutoff and the skipped sibling count."""

        self.pruning_events += 1
        self.pruned_nodes += max(0, pruned_nodes)

    def summary(self) -> dict[str, Any]:
        """Return a dictionary suitable for UI display or JSON output."""

        return {
            "algorithm": self.algorithm,
            "nodes_visited": self.nodes_visited,
            "leaf_nodes": self.leaf_nodes,
            "pruned_nodes": self.pruned_nodes,
            "pruning_events": self.pruning_events,
            "max_depth": self.max_depth,
            "execution_time_ms": round(self.execution_time_ms, 3),
            "best_move": self.best_move,
            "best_value": self.best_value,
        }

    @staticmethod
    def comparison_summary(minimax_stats: "SearchStats", alpha_beta_stats: "SearchStats") -> dict[str, Any]:
        """Compare plain Minimax with alpha-beta for one board state."""

        minimax_nodes = minimax_stats.nodes_visited
        alpha_beta_nodes = alpha_beta_stats.nodes_visited
        if minimax_nodes <= 0:
            reduction_pct = 0.0
        else:
            reduction_pct = 100.0 * (minimax_nodes - alpha_beta_nodes) / minimax_nodes

        return {
            "minimax_nodes": minimax_nodes,
            "alpha_beta_nodes": alpha_beta_nodes,
            "node_reduction_pct": round(reduction_pct, 2),
            "minimax_time_ms": round(minimax_stats.execution_time_ms, 3),
            "alpha_beta_time_ms": round(alpha_beta_stats.execution_time_ms, 3),
            "minimax_move": minimax_stats.best_move,
            "alpha_beta_move": alpha_beta_stats.best_move,
            "minimax_value": minimax_stats.best_value,
            "alpha_beta_value": alpha_beta_stats.best_value,
            "same_move": minimax_stats.best_move == alpha_beta_stats.best_move,
            "same_value": minimax_stats.best_value == alpha_beta_stats.best_value,
            "pruned_nodes": alpha_beta_stats.pruned_nodes,
            "pruning_events": alpha_beta_stats.pruning_events,
        }
