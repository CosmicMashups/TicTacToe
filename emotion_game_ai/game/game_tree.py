"""Explicit Tic-Tac-Toe game-tree structures for education and debugging."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from itertools import count
from typing import Any, Iterable

from emotion_game_ai.game.board import Board

BoardState = tuple[str, ...]
Move = tuple[int, int]

_NODE_IDS = count(1)


def board_to_state(board: Board) -> BoardState:
    """Convert a mutable board into an immutable row-major tuple."""

    return tuple(board.grid[r][c] for r in range(3) for c in range(3))


def state_to_board(state: BoardState) -> Board:
    """Convert an immutable row-major state into a Board copy."""

    board = Board()
    board.grid = [list(state[i : i + 3]) for i in range(0, 9, 3)]
    return board


def _empty_cells(state: BoardState) -> list[Move]:
    return [(i // 3, i % 3) for i, cell in enumerate(state) if cell == ""]


def _place(state: BoardState, move: Move, mark: str) -> BoardState:
    idx = move[0] * 3 + move[1]
    values = list(state)
    values[idx] = mark
    return tuple(values)


@dataclass
class GameTreeNode:
    """A fully inspectable node in a Tic-Tac-Toe search tree."""

    board_state: BoardState
    move: Move | None = None
    depth: int = 0
    is_maximizing: bool = True
    children: list["GameTreeNode"] = field(default_factory=list)
    value: int | None = None
    alpha: float = -math.inf
    beta: float = math.inf
    pruned: bool = False
    parent: "GameTreeNode | None" = field(default=None, repr=False, compare=False)
    node_id: int = field(default_factory=lambda: next(_NODE_IDS))

    def add_child(self, child: "GameTreeNode") -> "GameTreeNode":
        """Attach a child and return it for fluent tree construction."""

        child.parent = self
        self.children.append(child)
        return child

    def is_terminal(self) -> bool:
        """Return whether this node is terminal in the explicit tree."""

        return not self.children or state_to_board(self.board_state).game_over()

    def serialize(self) -> dict[str, Any]:
        """Serialize this node and descendants to plain Python data."""

        return {
            "node_id": self.node_id,
            "board_state": self.board_state,
            "move": self.move,
            "depth": self.depth,
            "role": "MAX" if self.is_maximizing else "MIN",
            "value": self.value,
            "alpha": _format_bound(self.alpha),
            "beta": _format_bound(self.beta),
            "pruned": self.pruned,
            "children": [child.serialize() for child in self.children],
        }

    def pretty_print(self) -> str:
        """Return a readable tree dump rooted at this node."""

        return print_tree_debug(self)


def build_game_tree(
    board: Board,
    depth_limit: int | None = None,
    is_maximizing: bool = True,
    current_depth: int = 0,
    parent: GameTreeNode | None = None,
    move: Move | None = None,
) -> GameTreeNode:
    """Build an explicit tree from a board, optionally limiting depth.

    `depth_limit` counts plies below the provided board. A value of `3`
    therefore produces four visual levels including the root.
    """

    root = GameTreeNode(
        board_state=board_to_state(board),
        move=move,
        depth=current_depth,
        is_maximizing=is_maximizing,
        parent=parent,
    )

    if board.game_over() or (depth_limit is not None and current_depth >= depth_limit):
        return root

    mark = "O" if is_maximizing else "X"
    for next_move in board.empty_cells():
        child_board = board.copy()
        child_board.place(next_move[0], next_move[1], mark)
        child = build_game_tree(
            child_board,
            depth_limit=depth_limit,
            is_maximizing=not is_maximizing,
            current_depth=current_depth + 1,
            parent=root,
            move=next_move,
        )
        root.add_child(child)
    return root


def generate_demo_tree() -> GameTreeNode:
    """Create a deterministic four-level tree with visible pruning cuts.

    The values are pedagogical leaf utilities, not a live board position.
    They satisfy the required MAX -> MIN -> MAX -> MIN four-level shape and
    use Tic-Tac-Toe terminal utility values: +10, 0, -10.
    """

    root = GameTreeNode(_demo_state(), depth=0, is_maximizing=True)
    leaf_groups = [
        [[10, 0, -10], [0, -10, 10], [10, 10, 0]],
        [[-10, -10, 0], [10, 0, -10], [0, 10, -10]],
        [[10, 10, 0], [-10, 0, 10], [0, -10, -10]],
    ]

    for min_idx, max_groups in enumerate(leaf_groups):
        min_node = root.add_child(
            GameTreeNode(_demo_state(), move=(0, min_idx), depth=1, is_maximizing=False)
        )
        for max_idx, leaf_values in enumerate(max_groups):
            max_node = min_node.add_child(
                GameTreeNode(_demo_state(), move=(1, max_idx), depth=2, is_maximizing=True)
            )
            for leaf_idx, value in enumerate(leaf_values):
                leaf = max_node.add_child(
                    GameTreeNode(_demo_state(), move=(2, leaf_idx), depth=3, is_maximizing=False)
                )
                leaf.value = value

    _evaluate_demo_alpha_beta(root, -math.inf, math.inf)
    return root


def export_tree_to_dict(root: GameTreeNode) -> dict[str, Any]:
    """Export a tree to a JSON-friendly dictionary."""

    return root.serialize()


def export_tree_to_json(root: GameTreeNode, *, indent: int = 2) -> str:
    """Export a tree as formatted JSON."""

    return json.dumps(export_tree_to_dict(root), indent=indent)


def print_tree_debug(root: GameTreeNode) -> str:
    """Return a text tree with role, depth, value, alpha/beta, and pruning."""

    lines: list[str] = []
    _append_debug_lines(root, lines, "", True)
    return "\n".join(lines)


def _append_debug_lines(node: GameTreeNode, lines: list[str], prefix: str, is_last: bool) -> None:
    connector = "" if node.parent is None else ("`-- " if is_last else "|-- ")
    role = "MAX" if node.is_maximizing else "MIN"
    move = f" move={node.move}" if node.move is not None else ""
    pruned = " PRUNED" if node.pruned else ""
    lines.append(
        f"{prefix}{connector}[{role} depth={node.depth}{move} value={node.value} "
        f"alpha={_format_bound(node.alpha)} beta={_format_bound(node.beta)}{pruned}]"
    )
    child_prefix = prefix if node.parent is None else prefix + ("    " if is_last else "|   ")
    for idx, child in enumerate(node.children):
        _append_debug_lines(child, lines, child_prefix, idx == len(node.children) - 1)


def _evaluate_demo_alpha_beta(node: GameTreeNode, alpha: float, beta: float) -> int:
    node.alpha = alpha
    node.beta = beta
    if not node.children:
        return int(node.value or 0)

    if node.is_maximizing:
        value = -math.inf
        for idx, child in enumerate(node.children):
            value = max(value, _evaluate_demo_alpha_beta(child, alpha, beta))
            alpha = max(alpha, value)
            node.alpha = alpha
            node.value = int(value)
            if alpha >= beta:
                _mark_pruned(node.children[idx + 1 :])
                break
        return int(value)

    value = math.inf
    for idx, child in enumerate(node.children):
        value = min(value, _evaluate_demo_alpha_beta(child, alpha, beta))
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


def _demo_state() -> BoardState:
    return ("", "", "", "", "", "", "", "", "")


def _format_bound(value: float) -> float | str:
    if value == math.inf:
        return "inf"
    if value == -math.inf:
        return "-inf"
    if float(value).is_integer():
        return int(value)
    return round(value, 3)
