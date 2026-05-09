## Why
Create a local, professional-looking Tic-Tac-Toe game that demonstrates a practical integration of computer vision, NLP, and game AI by adapting the opponent and UI in real time based on the player's facial expression and post-game text feedback.

## What Changes
- Add a modular Python application (`emotion_game_ai/`) runnable via `python main.py`.
- Add computer-vision emotion detection using OpenCV + MediaPipe Face Mesh with landmark overlay and background-thread processing.
- Add sentiment analysis training/inference pipeline using pandas + NLTK preprocessing + TF-IDF + Logistic Regression, persisted via pickle.
- Add optimal Tic-Tac-Toe AI via Minimax, plus emotion- and sentiment-aware difficulty modulation.
- Add a modern Pygame UI (1280x720) with panel layout (HUD/webcam/board/status), animations, theming, and sound hooks.
- Add thread-safe shared state (`current_emotion`, smoothed) and basic analytics (games played, win rate, emotion distribution).

## Impact
- Affected specs:
  - `openspec/changes/add-emotion-aware-tictactoe-system/specs/emotion-vision/spec.md`
  - `openspec/changes/add-emotion-aware-tictactoe-system/specs/sentiment-nlp/spec.md`
  - `openspec/changes/add-emotion-aware-tictactoe-system/specs/tictactoe-game-ai/spec.md`
  - `openspec/changes/add-emotion-aware-tictactoe-system/specs/pygame-ui/spec.md`
  - `openspec/changes/add-emotion-aware-tictactoe-system/specs/runtime-orchestration/spec.md`
- Affected code:
  - New package `emotion_game_ai/` including modules for vision, NLP, game logic, UI, and utilities.
  - New runtime assets under `assets/` and example data under `data/`.
