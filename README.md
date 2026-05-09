## Emotion-Aware Tic-Tac-Toe (Local Python)

This project is a local desktop Tic-Tac-Toe game where the AI opponent and UI adapt based on:

- Webcam facial expression (Happy / Neutral via a mouth-width heuristic)
- Post-game text feedback sentiment (positive / neutral / negative via TF-IDF and logistic regression)

The human plays **X**; the AI plays **O** and uses **alpha-beta Minimax** (`emotion_game_ai/game/minimax.py`) for optimal move evaluation, optionally softened by emotion-driven difficulty (`emotion_game_ai/game/ai_player.py`).

The project also includes educational AI-search tools:

- Explicit game-tree nodes (`emotion_game_ai/game/game_tree.py`)
- Plain Minimax vs alpha-beta comparison statistics
- Pruned branch tracking
- Intermediate node values and root-value debug output
- A deterministic four-level demonstration tree for academic inspection
- A lightweight in-game AI diagnostics overlay toggled with **Tab**
- Seaborn `rocket` palette game-tree graph exports and in-game graph display

### Documentation

For player instructions and educational game-tree examples, see:

[docs/USER_MANUAL.md](docs/USER_MANUAL.md)

For an architecture-focused description of current features (board rules, UI flow, multimodal tuning, minimax wiring, alpha-beta pruning, and explicit game-tree support), see:

[docs/FEATURES.md](docs/FEATURES.md)

### Setup

Create and activate a Python 3.10+ virtual environment, then:

    pip install -r requirements.txt

NLTK resources are downloaded automatically on first run (cached in your user profile).

### Run

    python main.py

### Gameplay controls

- Click an empty square to place **X**
- Press **Tab** to show/hide AI search diagnostics
- Press **G** to generate and display a Seaborn game-tree graph for the current board
- Press **C** to cycle camera index
- Press **Esc** to quit
- After a match, type feedback and press **Enter** to tune the next game

### Educational search examples

Print the four-level demonstration tree with root and intermediate values:

    python -c "from emotion_game_ai.game.game_tree import generate_demo_tree, print_tree_debug; print(print_tree_debug(generate_demo_tree()))"

Compare plain Minimax with alpha-beta pruning on a sample board:

    python -c "from emotion_game_ai.game.board import Board; from emotion_game_ai.game.minimax import compare_search_algorithms; b=Board(); b.grid=[['X','',''],['','O',''],['','','X']]; print(compare_search_algorithms(b))"

Save the four-level demo tree as a Seaborn `rocket` PNG:

    python -c "from emotion_game_ai.game.tree_visualization import save_demo_tree_graph; save_demo_tree_graph('data/demo_tree_graph.png')"

Save a current-board tree graph from a scripted board:

    python -c "from emotion_game_ai.game.board import Board; from emotion_game_ai.game.tree_visualization import save_board_tree_graph; b=Board(); b.grid=[['X','',''],['','O',''],['','','X']]; save_board_tree_graph(b, 'data/current_board_tree_graph.png', is_maximizing=True)"

### Optional: Train the sentiment model

On first run, if no saved model exists, the game will train a smaller sentiment model automatically on a sample of `assets/sentiment_data.csv`.

You can also train it manually:

    python -m emotion_game_ai.nlp.sentiment_model --train

Artifacts are saved under `data/models/` by default.

### Data files

- `data/face.jpg`: baseline face image for mouth-width calibration
- `assets/sentiment_data.csv`: sentiment dataset (referenced by training flows)

If the webcam is unavailable, the game still runs and falls back to **Neutral** for vision-driven emotion.
