"""Tic-Tac-Toe board representation and rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional


PlayerMark = str  # "X" | "O"


@dataclass
class Board:
    grid: list[list[str]] = field(default_factory=lambda: [["", "", ""], ["", "", ""], ["", "", ""]])

    def copy(self) -> "Board":
        b = Board()
        b.grid = [row[:] for row in self.grid]
        return b

    def empty_cells(self) -> list[tuple[int, int]]:
        cells: list[tuple[int, int]] = []
        for r in range(3):
            for c in range(3):
                if self.grid[r][c] == "":
                    cells.append((r, c))
        return cells

    def place(self, r: int, c: int, mark: PlayerMark) -> bool:
        if self.grid[r][c] != "":
            return False
        self.grid[r][c] = mark
        return True

    def winner(self) -> Optional[PlayerMark]:
        lines: list[list[tuple[int, int]]] = []
        lines.extend([[(r, 0), (r, 1), (r, 2)] for r in range(3)])
        lines.extend([[(0, c), (1, c), (2, c)] for c in range(3)])
        lines.append([(0, 0), (1, 1), (2, 2)])
        lines.append([(0, 2), (1, 1), (2, 0)])

        for line in lines:
            vals = [self.grid[r][c] for r, c in line]
            if vals[0] and vals[0] == vals[1] == vals[2]:
                return vals[0]
        return None

    def is_draw(self) -> bool:
        return self.winner() is None and all(self.grid[r][c] for r in range(3) for c in range(3))

    def game_over(self) -> bool:
        return self.winner() is not None or self.is_draw()

    @staticmethod
    def winning_line_coords(grid: list[list[str]]) -> Optional[list[tuple[int, int]]]:
        lines: list[list[tuple[int, int]]] = []
        lines.extend([[(r, 0), (r, 1), (r, 2)] for r in range(3)])
        lines.extend([[(0, c), (1, c), (2, c)] for c in range(3)])
        lines.append([(0, 0), (1, 1), (2, 2)])
        lines.append([(0, 2), (1, 1), (2, 0)])
        for line in lines:
            vals = [grid[r][c] for r, c in line]
            if vals[0] and vals[0] == vals[1] == vals[2]:
                return line
        return None

