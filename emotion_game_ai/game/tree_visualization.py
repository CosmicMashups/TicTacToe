"""Seaborn/Matplotlib visualization for Tic-Tac-Toe game trees."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from emotion_game_ai.game.board import Board
from emotion_game_ai.game.game_tree import GameTreeNode, build_game_tree, generate_demo_tree, state_to_board
from emotion_game_ai.game.minimax import minimax_alpha_beta


def build_evaluated_gameplay_tree(
    board: Board,
    *,
    is_maximizing: bool = True,
    depth_limit: int = 3,
) -> GameTreeNode:
    """Build a readable depth-limited tree and propagate Minimax values."""

    root = build_game_tree(board, depth_limit=depth_limit, is_maximizing=is_maximizing)
    _evaluate_display_tree(root, -math.inf, math.inf)
    return root


def save_demo_tree_graph(output_path: str | Path, *, title: str = "Four-Level Alpha-Beta Demo Tree") -> Path:
    """Save the deterministic four-level demo tree as a Seaborn-styled PNG."""

    return save_tree_graph(generate_demo_tree(), output_path, title=title)


def save_board_tree_graph(
    board: Board,
    output_path: str | Path,
    *,
    is_maximizing: bool = True,
    depth_limit: int = 3,
    title: str = "Current Board Game Tree",
) -> Path:
    """Build, evaluate, and save a current-board tree graph."""

    root = build_evaluated_gameplay_tree(board, is_maximizing=is_maximizing, depth_limit=depth_limit)
    return save_tree_graph(root, output_path, title=title)


def save_tree_graph(
    root: GameTreeNode,
    output_path: str | Path,
    *,
    title: str = "Tic-Tac-Toe Game Tree",
    figsize: tuple[float, float] = (15.0, 8.5),
    dpi: int = 160,
) -> Path:
    """Render a game tree to a PNG using Seaborn's rocket palette."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    nodes = list(_walk(root))
    positions = _layered_positions(root)
    palette = sns.color_palette("rocket", 8)

    sns.set_theme(style="darkgrid", palette="rocket")
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#17121f")
    ax.set_facecolor("#211827")

    for node in nodes:
        for child in node.children:
            x1, y1 = positions[node.node_id]
            x2, y2 = positions[child.node_id]
            edge_color = palette[1] if child.pruned else palette[4]
            alpha = 0.35 if child.pruned else 0.85
            ax.plot([x1, x2], [y1, y2], color=edge_color, linewidth=1.4, alpha=alpha, zorder=1)

    for node in nodes:
        x, y = positions[node.node_id]
        color = _node_color(node, palette)
        text_color = "#fff7fb" if not node.pruned else "#f2d7df"
        ax.text(
            x,
            y,
            _node_label(node),
            ha="center",
            va="center",
            fontsize=7.5,
            color=text_color,
            bbox={
                "boxstyle": "round,pad=0.38",
                "facecolor": color,
                "edgecolor": palette[6] if not node.pruned else palette[0],
                "linewidth": 1.2,
                "alpha": 0.96 if not node.pruned else 0.55,
            },
            zorder=2,
        )

    ax.set_title(title, color="#fff7fb", fontsize=16, pad=16)
    ax.set_axis_off()
    ax.margins(x=0.06, y=0.14)
    plt.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output


def _evaluate_display_tree(node: GameTreeNode, alpha: float, beta: float) -> int:
    node.alpha = alpha
    node.beta = beta
    if not node.children:
        value = minimax_alpha_beta(state_to_board(node.board_state), 0, node.is_maximizing, -math.inf, math.inf)
        node.value = value
        return value

    if node.is_maximizing:
        value = -math.inf
        for idx, child in enumerate(node.children):
            value = max(value, _evaluate_display_tree(child, alpha, beta))
            alpha = max(alpha, value)
            node.alpha = alpha
            node.value = int(value)
            if alpha >= beta:
                _mark_pruned(node.children[idx + 1 :])
                break
        return int(value)

    value = math.inf
    for idx, child in enumerate(node.children):
        value = min(value, _evaluate_display_tree(child, alpha, beta))
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


def _walk(root: GameTreeNode) -> Iterable[GameTreeNode]:
    yield root
    for child in root.children:
        yield from _walk(child)


def _layered_positions(root: GameTreeNode) -> dict[int, tuple[float, float]]:
    levels: dict[int, list[GameTreeNode]] = {}
    for node in _walk(root):
        levels.setdefault(node.depth, []).append(node)

    positions: dict[int, tuple[float, float]] = {}
    max_depth = max(levels) if levels else 0
    for depth, nodes in levels.items():
        width = max(1, len(nodes) - 1)
        y = float(max_depth - depth)
        for idx, node in enumerate(nodes):
            x = 0.5 if len(nodes) == 1 else idx / width
            positions[node.node_id] = (x, y)
    return positions


def _node_label(node: GameTreeNode) -> str:
    role = "MAX" if node.is_maximizing else "MIN"
    parts = [role, f"d={node.depth}"]
    if node.move is not None:
        parts.append(f"m={node.move}")
    parts.append(f"v={node.value}")
    parts.append(f"a={_format_bound(node.alpha)}")
    parts.append(f"b={_format_bound(node.beta)}")
    if node.pruned:
        parts.append("PRUNED")
    return "\n".join(parts)


def _node_color(node: GameTreeNode, palette: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    if node.pruned:
        return palette[1]
    if node.depth == 0:
        return palette[6]
    if node.is_maximizing:
        return palette[5]
    return palette[3]


def _format_bound(value: float) -> str:
    if value == math.inf:
        return "inf"
    if value == -math.inf:
        return "-inf"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}"
