# Game features and implementation

This document describes how the emotion-aware Tic-Tac-Toe application works today: game rules, UI flow, multimodal tuning, and the current AI search. It is intended to support planning **alpha-beta pruning** and a clearer **explicit game-tree** programming model.

---

## Technology stack

| Layer | Implementation |
|--------|----------------|
| Application entry | `main.py` delegates to `emotion_game_ai.runtime.run_app` |
| Game loop & UI | **Pygame** (`emotion_game_ai.game.pygame_ui.PygameApp`) |
| Board & rules | Pure Python (`emotion_game_ai.game.board.Board`) |
| Optimal-play search | **Minimax** (`emotion_game_ai.game.minimax`) — **no alpha-beta yet** |
| AI policy | Wrapper that mixes minimax with difficulty (`emotion_game_ai.game.ai_player`) |
| Vision | Background thread + MediaPipe/OpenCV (`emotion_game_ai.vision.webcam_emotion`) |
| Post-game NLP | TF-IDF + logistic regression sentiment (`emotion_game_ai.nlp.sentiment_model`) |
| Behavior profiles | Emotion name → tuning/theme (`emotion_game_ai.emotion_behavior`) |

The project is a **desktop Python** application, not a web React app; the conceptual ideas (state, rendering, AI) translate similarly across stacks.

---

## High-level architecture

```mermaid
flowchart LR
  subgraph threads
    VW[WebcamEmotionWorker]
  end
  subgraph main_thread
    UI[PygameApp]
    BOARD[Board]
    AI[decide_move]
    MX[minimax / get_best_move]
  end
  SS[(SharedState)]
  NLP[SentimentModel]
  VW --> SS
  UI --> BOARD
  UI --> AI
  AI --> MX
  UI reads SS
  NLP --> SS
```

- **Vision** writes smoothed Happy/Neutral (and diagnostics) into `SharedState`; the UI reads a snapshot each frame.
- After each match, **text feedback** is classified into an emotion label; `DifficultyTuning.apply_behavior` maps that to mistake probability, thinking delay, personality, and UI theme for the **next** match.

---

## Board model and rules

**File:** `emotion_game_ai/game/board.py`

- **Representation:** `3×3` grid of strings: `""` (empty), `"X"` (human), `"O"` (AI).
- **API highlights:**
  - `empty_cells()` — list of `(row, col)` for legal moves (unordered; iteration order affects move ordering in search — see Planning).
  - `place(r, c, mark)` — mutating; returns `False` if cell occupied.
  - `copy()` — immutable-style simulation for minimax branches.
  - `winner()` — returns `"X"`, `"O"`, or `None`.
  - `is_draw()`, `game_over()`, `winning_line_coords(...)` — used for outcome UI and drawing the win streak.

Terminal states are detected before expanding children in minimax (`evaluate` + draw check).

---

## Player roles and turn order

| Player | Mark | Implemented as |
|--------|------|----------------|
| Human | `X` | Mouse clicks when `scene == "play"` and `turn == "X"` |
| AI | `O` | Move chosen by `decide_move`; applied after an artificial delay |

Constants in minimax explicitly tie **maximizing** to the AI (`AI_MARK = "O"`) and **minimizing** to the human (`HUMAN_MARK = "X"`). Any future refactor (swap sides or variable first player) must keep these aligned with UI turn logic.

---

## Pygame UI: scenes and gameplay flow

**File:** `emotion_game_ai/game/pygame_ui.py`

### Scenes

- **`play`** — active match; board input, hints, AI thinking animation.
- **`postgame`** — overlay after win/loss/draw; collects free-text feelings; Enter submits,NLP adjusts tuning and starts `_start_new_match()`.

### Match lifecycle

1. New `Board()`; human moves first (`turn = "X"`).
2. On valid human click: `place(..., "X")`, `_post_move_updates`, then `_queue_ai_move()`.
3. `_queue_ai_move()` calls **`decide_move(board, emotion, tuning, rng)`**, stores pending coordinates and **`thinking_delay_s`** until `perf_counter()` passes.
4. When due, AI `place(..., "O")`, `_post_move_updates`, `turn = "X"`.
5. On terminal state: `_finish_match()` → **`postgame`** scene.

### Auxiliary UI features

- **Webcam panel** (`C` cycles camera index): preview + diagnostics (`camera_ok`, landmarks, smoothed emotion).
- **Hints:** If webcam emotion stays Neutral for several player turns (`neutral_player_turns`), a heuristic suggests center then corners (not tied to full game-tree search).
- **Themes / particles:** `renderer.py` themes, tweens, confetti keyed off emotion and post-game behavior.
- **Stats:** Wins/losses/draws; per-move emotion tally for HUD distribution.

---

## Current AI: plain minimax

**File:** `emotion_game_ai/game/minimax.py`

### Evaluation

```text
evaluate(board):
  AI wins    → +10
  Human wins → -10
  else       →  0 (non-terminal; score refined by recursion / draw)
```

On terminal win, the raw score is **depth-adjusted**:

- Maximizer (AI) prefers **shorter** wins: `score - depth`.
- Human win from minimizer side: `score + depth` (slow the loss).

Draws return `0` with no depth bonus (explicit `board.is_draw()` branch).

### Recursion shape

```text
minimax(board, depth, is_maximizing):
  if terminal_win: return adjusted_win_score
  if draw: return 0
  if maximizing:
    best = max over empty cells (AI places O, recurse with is_maximizing=False)
  else:
    best = min over empty cells (human places X, recurse with True)
```

`get_best_move(board)` evaluates every legal AI move once: copy board, place `O`, call `minimax(..., depth=0, is_maximizing=False)`, pick maximizing move value.

### What is **not** implemented yet

- **Alpha-beta pruning** — every child is explored; branching is small so performance is acceptable, but the call graph is full-width minimax.
- **Transposition table** — symmetric positions could repeat; unused.
- **Explicit game-tree object** — the tree exists only implicitly as recursion; no node list, edge list, or visualization structure.

---

## AI policy wrapper (beyond optimal minimax)

**File:** `emotion_game_ai/game/ai_player.py`

`decide_move` returns `AIDecision(move, mode, thinking_delay_s, message)`.

1. **Competitive cue:** Happy face + sufficiently low `assistive_mistake_prob` ⇒ always **`get_best_move`**, `"Competitive"` mode, tighter delay cap.
2. **Assistive default:** Random draw vs `mistake_prob` (clamped `[0.05, 0.45]`): with probability `(1 - mistake)`, optimal move via `get_best_move`; otherwise **`_suboptimal_move`**.
3. **`_suboptimal_move`:** Among non-best legal moves (if any), pick the move with **next-best minimax score** after AI plays (`minimax(b2, 0, False)` per candidate).

So difficulty is modeled as **stochastic deviations from optimal play**, not shallow search depth.

---

## Emotion and difficulty wiring

### Webcam emotion (during play)

**Types:** coarse `Happy` vs `Neutral` (see vision worker).

**Used for:** Competitive branch in `decide_move`; UI theme baseline; HUD; hint logic; cosmetics.

### Post-game NLP and behavior profiles

**Flow:** Post-game overlay → Enter → `SentimentModel.predict_emotion(feedback)` → stored as `last_feedback_emotion`; `DifficultyTuning.adjust_for_sentiment` (positive/negative nudge); `apply_behavior(emotion)` loads `emotion_behavior.behavior_for_emotion` fields (thinking time, mistake rate ranges, personality, `ui_theme`).

**Important for tree search:** Profiles change **mistake probability and timing**, not the minimax scorer itself; alpha-beta belongs in **`minimax.py`** and remains valid for the **evaluation of a deterministic child position**. The stochastic layer sits **above** that.

---

## Game-tree properties (for programming the full tree)

Classic 3×3 Tic-Tac-Toe:

- **Max depth:** 9 ply from empty board (fewer once terminal).
- **Branching:** up to **9 − k** empties after `k` moves; total nodes are on the order of **~550k** across the full exhaustive tree — small enough that building an explicit tree once (e.g. for analysis or teaching) is feasible in Python if desired.
- **Admissibility:** Current minimax with symmetric win/draw payoff is suitable for optimal play assuming both players maximize their outcome.

Move **ordering** for alpha-beta effectiveness (when added) depends on **`empty_cells()` iteration order**. Today that is row-major `(0,0)…(2,2)`; heuristic ordering (center first, forks, threats) tightens pruning in larger games — here gains are marginal but patterns still apply pedagogically.

---

## Planning alpha-beta pruning

### Where to integrate

- **Primary:** `emotion_game_ai/game/minimax.py` — extend `minimax` (or introduce `minimax_ab`) with `(alpha, beta)` parameters threaded through recursion.
- **Call sites:** `get_best_move`, `_suboptimal_move`, and `minimax` itself must agree on signatures; optionally keep legacy `minimax` as a thin wrapper for tests.

### Suggested semantics

```text
minimax(board, depth, is_maximizing, alpha, beta):
  terminal / draw handling unchanged
  if maximizing:
    value = -inf
    for each move:
      value = max(value, child minimax(..., alpha=alpha, beta=beta))
      if value >= beta: return value   # pruneMin
      alpha = max(alpha, value)
  else:
    symmetric with min / alpha prune
```

- Preserve **existing depth-adjusted leaf scores** so move preferences among wins of equal nominal score stay consistent unless you unify scoring in one place.

### Relation to `_suboptimal_move` and `get_best_move`

- **Alpha-beta preserves the same optimal move choices** given deterministic evaluation and full search; `_suboptimal_move` compares scores over candidate moves — each still requires correct leaf values under the **same** rules.
- If you refactor to avoid redundant full-board copies, factor `apply_undo` / `mutable board + undo stack` separately; correctness first.

---

## Planning an explicit game-tree representation

If the goal includes **education, visualization, or search debugging**, consider introducing (separate from Pygame surfaces):

| Concept | Suggested mapping |
|---------|-------------------|
| Node | Frozen board fingerprint (e.g. tuple of nine cells), current side to move, optional depth |
| Edge | `(from_node, move, resulting_child_hash)` |
| Value | Stored minimax value from node's perspective |

Implementation options:

1. **On-demand DAG** keyed by canonical board string (cheap for 3×3).
2. **Precomputed table** offline (all equivalences under symmetry optional).
3. **Debug-only tree build** tracing one `get_best_move` call depth-first with pruning — useful to verify alpha-beta against baseline minimax on random positions.

The **existing** codebase does not expose such structures; they would live next to `minimax.py` or under a new package (e.g. `emotion_game_ai.game.tree`).

---

## File index

| Topic | Primary files |
|--------|----------------|
| Entry & orchestration | `main.py`, `emotion_game_ai/runtime.py` |
| Board & rules | `emotion_game_ai/game/board.py` |
| Search | `emotion_game_ai/game/minimax.py` |
| Policy / difficulty | `emotion_game_ai/game/ai_player.py` |
| UI | `emotion_game_ai/game/pygame_ui.py`, `emotion_game_ai/game/renderer.py` |
| Shared state & tuning | `emotion_game_ai/utils/threading_utils.py` |
| Emotion profiles | `emotion_game_ai/emotion_behavior.py` |
| Vision worker | `emotion_game_ai/vision/webcam_emotion.py` |
| Sentiment model | `emotion_game_ai/nlp/sentiment_model.py` |

---

## Summary for downstream work

- **Rules engine** is clean and deterministic in `Board`.
- **Search** is full-depth minimax **without pruning** or explicit tree artifacts.
- **Alpha-beta** is a localized enhancement to recursion in `minimax.py` plus signature updates at callers.
- **Adaptive difficulty** deliberately **wraps** perfect play rather than weakening search internally; keep separation so optimal analysis and stochastic play stay testable independently.
