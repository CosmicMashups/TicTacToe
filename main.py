"""
Emotion-Aware Tic-Tac-Toe Gaming System.

Run:
  python main.py
"""

from __future__ import annotations

import sys

from emotion_game_ai.runtime import run_app


def main() -> int:
    run_app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

