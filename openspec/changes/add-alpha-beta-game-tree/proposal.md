## Why
The game currently has correct plain Minimax gameplay, but it does not expose alpha-beta pruning, explicit game-tree state, or search diagnostics needed for academic AI-search demonstration.

## What Changes
- Add alpha-beta pruning while preserving plain Minimax for comparison.
- Add explicit inspectable game-tree nodes, pruning markers, intermediate values, and four-level demonstration tree generation.
- Add search statistics and lightweight Pygame diagnostics for educational/debug visibility.

## Impact
- Affected specs: tictactoe-game-ai, pygame-ui
- Affected code: `emotion_game_ai/game/minimax.py`, `emotion_game_ai/game/game_tree.py`, `emotion_game_ai/game/search_stats.py`, `emotion_game_ai/game/ai_player.py`, `emotion_game_ai/game/pygame_ui.py`
