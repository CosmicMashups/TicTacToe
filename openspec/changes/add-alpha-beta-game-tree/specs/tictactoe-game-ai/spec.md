## ADDED Requirements

### Requirement: Alpha-beta optimal move selection
The system SHALL provide alpha-beta pruning for Tic-Tac-Toe move search while preserving the same terminal scoring, depth adjustment, and optimal move selection as plain Minimax.

#### Scenario: Alpha-beta matches Minimax
- **WHEN** plain Minimax and alpha-beta search evaluate the same board state
- **THEN** both algorithms return the same move value
- **AND** the selected best move remains unchanged for gameplay.

### Requirement: Explicit inspectable game tree
The system SHALL provide explicit game-tree node objects that store immutable board state, move, depth, maximizing/minimizing role, children, computed value, alpha, beta, pruning state, parent, and node identifier.

#### Scenario: Tree node exposes search state
- **WHEN** a search tree is built or traced
- **THEN** each node can be serialized or printed with its move, role, value, alpha, beta, and pruning state.

### Requirement: Search statistics comparison
The system SHALL track search statistics for plain Minimax and alpha-beta pruning, including visited nodes, leaf nodes, pruned nodes, pruning events, maximum depth, and execution time.

#### Scenario: Comparison summary is available
- **WHEN** the comparison utility runs on a board state
- **THEN** it reports Minimax and alpha-beta node counts, elapsed times, selected moves, values, and node-reduction percentage.

### Requirement: Four-level educational demo tree
The system SHALL generate a deterministic educational tree with exactly four levels, alternating MAX, MIN, MAX, MIN, and at least nine terminal leaf nodes.

#### Scenario: Demo tree supports academic inspection
- **WHEN** the demo tree is generated and evaluated
- **THEN** it exposes a root value, intermediate node values, terminal values from the set `+10`, `0`, `-10`, traversal counts, and pruned branches.
