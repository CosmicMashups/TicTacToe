# Emotion-Aware Tic-Tac-Toe User Manual

This manual explains how to play the desktop game and how to use the educational game-tree tools to inspect Minimax and alpha-beta search.

## 1. Starting The Game

Install the project dependencies first:

```powershell
pip install -r requirements.txt
```

Run the game from the project root:

```powershell
python main.py
```

The game opens a Pygame desktop window. If the webcam is unavailable, the game still works and falls back to neutral emotion behavior.

## 2. How To Play

The human player is `X`. The AI player is `O`.

1. Click an empty square on the board to place `X`.
2. The AI thinks briefly, then places `O`.
3. The first player to complete a row, column, or diagonal wins.
4. If all cells are filled and nobody wins, the game is a draw.
5. After each match, type how you felt about the game and press `Enter`. The game uses this feedback to tune the next match.

Useful controls:

| Key | Action |
| --- | --- |
| `Esc` | Quit the game |
| `C` | Cycle camera index |
| `Tab` | Toggle AI search diagnostics |
| `G` | Generate and display a Seaborn game-tree graph |
| `Enter` | Submit post-game feedback |
| `Backspace` | Edit post-game feedback |

## 3. Emotion-Aware AI Behavior

The AI always uses correct search evaluation underneath. The emotion-aware layer changes how often the AI intentionally chooses a reasonable suboptimal move and how quickly it responds.

| Situation | Behavior |
| --- | --- |
| Competitive mode | Uses the best alpha-beta move |
| Assistive mode | Usually uses the best move, but may choose a next-best legal move |
| Positive feedback | Can make later games more competitive |
| Negative feedback | Can make later games more supportive |

The player remains `X`, the AI remains `O`, and the search still treats the AI as `MAX`.

## 4. AI Search Concepts

The project includes both plain Minimax and alpha-beta pruning.

| Concept | Meaning |
| --- | --- |
| `MAX` | AI turn, tries to maximize the score |
| `MIN` | Human turn, tries to minimize the score |
| `+10` | AI win |
| `0` | Draw |
| `-10` | Human win |
| `alpha` | Best value MAX can already guarantee |
| `beta` | Best value MIN can already guarantee |
| Pruned branch | A branch skipped because `alpha >= beta` |

The AI uses depth-adjusted scoring for wins:

```text
AI win    = 10 - depth
Human win = -10 + depth
Draw      = 0
```

This means the AI prefers faster wins and slower losses.

## 5. Initializing A Game Tree

Use `build_game_tree()` when you want an explicit tree for a real board state.

```python
from emotion_game_ai.game.board import Board
from emotion_game_ai.game.game_tree import build_game_tree, print_tree_debug

board = Board()
board.place(0, 0, "X")
board.place(1, 1, "O")

# depth_limit=3 creates four displayed levels including the root.
root = build_game_tree(board, depth_limit=3, is_maximizing=True)

print(print_tree_debug(root))
```

The root is the current board. Its children are legal moves. Each child stores:

- immutable board state
- move that produced the node
- depth
- `MAX` or `MIN` role
- child nodes
- computed value, when evaluated
- alpha and beta values, when alpha-beta search is traced
- pruning state

For academic demonstration, use `generate_demo_tree()`. It creates exactly four levels:

```text
Level 1: MAX root
Level 2: MIN nodes
Level 3: MAX nodes
Level 4: MIN terminal leaves
```

Example:

```python
from emotion_game_ai.game.game_tree import generate_demo_tree, print_tree_debug

root = generate_demo_tree()
print(print_tree_debug(root))
```

## 6. How Recursive Evaluation Works

Minimax evaluates from the leaves back to the root.

1. Terminal leaves receive values: `+10`, `0`, or `-10`.
2. A `MAX` node takes the maximum value of its children.
3. A `MIN` node takes the minimum value of its children.
4. The root value is the final best value for `MAX`.

Example tree:

```text
MAX root
|-- MIN A
|   |-- MAX A1
|   |   |-- leaf +10
|   |   |-- leaf 0
|   |   `-- leaf -10
|   `-- MAX A2
|       |-- leaf 0
|       |-- leaf -10
|       `-- leaf +10
`-- MIN B
    |-- MAX B1
    |   |-- leaf -10
    |   |-- leaf 0
    |   `-- leaf +10
    `-- MAX B2
        |-- leaf 0
        |-- leaf 0
        `-- leaf -10
```

Evaluation:

```text
MAX A1 = max(+10, 0, -10) = +10
MAX A2 = max(0, -10, +10) = +10
MIN A  = min(+10, +10) = +10

MAX B1 = max(-10, 0, +10) = +10
MAX B2 = max(0, 0, -10) = 0
MIN B  = min(+10, 0) = 0

ROOT MAX = max(MIN A, MIN B)
ROOT MAX = max(+10, 0)
ROOT MAX = +10
```

So the final value of the root is `+10`, which means the best line favors the AI.

## 7. Displaying Root And Intermediate Values

Run this command from the project root to print the built-in four-level demonstration tree:

```powershell
python -c "from emotion_game_ai.game.game_tree import generate_demo_tree, print_tree_debug; print(print_tree_debug(generate_demo_tree()))"
```

Example output excerpt:

```text
[MAX depth=0 value=10 alpha=10 beta=inf]
|-- [MIN depth=1 move=(0, 0) value=10 alpha=-inf beta=10]
|   |-- [MAX depth=2 move=(1, 0) value=10 alpha=10 beta=inf]
|   |   |-- [MIN depth=3 move=(2, 0) value=10 alpha=-inf beta=inf]
|   |   |-- [MIN depth=3 move=(2, 1) value=0 alpha=10 beta=inf]
|   |   `-- [MIN depth=3 move=(2, 2) value=-10 alpha=10 beta=inf]
|   |-- [MAX depth=2 move=(1, 1) value=10 alpha=10 beta=10]
|   `-- [MAX depth=2 move=(1, 2) value=10 alpha=10 beta=10]
|       |-- [MIN depth=3 move=(2, 0) value=10 alpha=-inf beta=10]
|       |-- [MIN depth=3 move=(2, 1) value=10 alpha=-inf beta=inf PRUNED]
|       `-- [MIN depth=3 move=(2, 2) value=0 alpha=-inf beta=inf PRUNED]
|-- [MIN depth=1 move=(0, 1) value=0 alpha=10 beta=0]
`-- [MIN depth=1 move=(0, 2) value=-10 alpha=10 beta=-10]
```

Important fields:

| Field | Meaning |
| --- | --- |
| `MAX` or `MIN` | Whose turn the node represents |
| `depth` | Tree depth from the root |
| `move` | Move that created the node |
| `value` | Minimax value propagated from descendants |
| `alpha` | Current best value available to MAX |
| `beta` | Current best value available to MIN |
| `PRUNED` | Branch skipped by alpha-beta pruning |

In the excerpt, the root value is:

```text
Root value = 10
```

The intermediate values include:

```text
Depth 1 MIN node move=(0, 0): value=10
Depth 1 MIN node move=(0, 1): value=0
Depth 1 MIN node move=(0, 2): value=-10
```

Because the root is a `MAX` node, it returns:

```text
max(10, 0, -10) = 10
```

## 8. Comparing Plain Minimax And Alpha-Beta

Run this command to compare both algorithms on a sample board:

```powershell
python -c "from emotion_game_ai.game.board import Board; from emotion_game_ai.game.minimax import compare_search_algorithms; b=Board(); b.grid=[['X','',''],['','O',''],['','','X']]; print(compare_search_algorithms(b))"
```

Typical summary:

```text
minimax_nodes: 1052
alpha_beta_nodes: 497
node_reduction_pct: 52.76
minimax_move: (0, 1)
alpha_beta_move: (0, 1)
minimax_value: 0
alpha_beta_value: 0
same_move: True
same_value: True
pruned_nodes: 129
pruning_events: 138
```

The important result is that both algorithms return the same move and value, while alpha-beta visits fewer nodes.

## 9. Exporting The Tree

Use dictionary or JSON export helpers when you want to inspect the tree in another tool.

```python
from emotion_game_ai.game.game_tree import (
    export_tree_to_dict,
    export_tree_to_json,
    generate_demo_tree,
)

root = generate_demo_tree()

tree_dict = export_tree_to_dict(root)
tree_json = export_tree_to_json(root)

print(tree_dict["value"])
print(tree_json)
```

This is useful for reports, notebooks, or visualization experiments.

## 10. Seaborn Game-Tree Graphs

The project can also render the game tree as a Seaborn/Matplotlib graph using Seaborn's `rocket` palette.

Save the deterministic four-level demo tree:

```powershell
python -c "from emotion_game_ai.game.tree_visualization import save_demo_tree_graph; save_demo_tree_graph('data/demo_tree_graph.png')"
```

Save a graph from a scripted board state:

```powershell
python -c "from emotion_game_ai.game.board import Board; from emotion_game_ai.game.tree_visualization import save_board_tree_graph; b=Board(); b.grid=[['X','',''],['','O',''],['','','X']]; save_board_tree_graph(b, 'data/current_board_tree_graph.png', is_maximizing=True)"
```

Use the Python API directly when writing reports or notebooks:

```python
from emotion_game_ai.game.board import Board
from emotion_game_ai.game.tree_visualization import (
    save_board_tree_graph,
    save_demo_tree_graph,
)

save_demo_tree_graph("data/demo_tree_graph.png")

board = Board()
board.grid = [
    ["X", "", ""],
    ["", "O", ""],
    ["", "", "X"],
]
save_board_tree_graph(
    board,
    "data/current_board_tree_graph.png",
    is_maximizing=True,
    depth_limit=3,
)
```

The graph labels show:

- role: `MAX` or `MIN`
- depth
- move that created the node
- propagated value
- alpha and beta values
- `PRUNED` marker for skipped branches

During gameplay, press `G` to generate a current-board graph and display it inside the diagnostics overlay. The file is saved to:

```text
data/tree_graph_current.png
```

The overlay uses the saved PNG, so the game does not redraw Matplotlib every frame.

## 11. In-Game Diagnostics

During gameplay, press `Tab` to show or hide AI diagnostics in the right-side status panel.

The panel can show:

- current search algorithm
- selected move and value
- visited nodes
- evaluated leaves
- pruned nodes
- pruning events
- maximum search depth
- execution time in milliseconds
- generated Seaborn game-tree graph after pressing `G`

These diagnostics describe the most recent AI decision. They do not change gameplay behavior.
