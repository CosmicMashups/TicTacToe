## ADDED Requirements

### Requirement: Board representation and rules
The system SHALL represent the Tic-Tac-Toe board as a 3x3 matrix and enforce legal move placement, win detection, and draw detection.

#### Scenario: Legal move updates the board
- **WHEN** the human selects an empty cell
- **THEN** the board is updated with an `X` in that cell
- **AND** the turn advances

#### Scenario: Illegal move is rejected
- **WHEN** the human selects a non-empty cell
- **THEN** the move is rejected
- **AND** the board state is unchanged

### Requirement: Minimax optimal move selection
The system SHALL implement Minimax with `evaluate(board)`, `minimax(board, depth, is_maximizing)`, and `get_best_move(board)` such that the AI always chooses an optimal move in Competitive mode.

#### Scenario: Competitive AI plays optimally
- **WHEN** the AI is in Competitive mode
- **THEN** `get_best_move(board)` returns an optimal move for the current position
- **AND** the AI does not intentionally choose suboptimal moves

### Requirement: Emotion-aware AI mode selection
The system SHALL set AI behavior based on the smoothed `current_emotion`:
- `Happy` → Competitive mode with full-depth Minimax and faster decision pacing
- `Neutral` → Supportive mode with an intentional suboptimal move chance and slower pacing

#### Scenario: Happy emotion selects Competitive mode
- **WHEN** the active emotion is `Happy`
- **THEN** the AI mode is `Competitive`
- **AND** the AI uses full-depth Minimax for move selection

#### Scenario: Neutral emotion selects Supportive mode
- **WHEN** the active emotion is `Neutral`
- **THEN** the AI mode is `Assistive` (Supportive)
- **AND** the AI has a 20–30% chance to choose a suboptimal legal move

### Requirement: Emotion-based AI thinking time
The system SHALL apply an emotion-based AI thinking delay:
- `Happy` → approximately 0.5 seconds
- `Neutral` → approximately 1.5 seconds

#### Scenario: Thinking delay adapts by emotion
- **WHEN** the AI is about to move
- **THEN** the applied delay depends on the current emotion (0.5s Happy, 1.5s Neutral)

### Requirement: Neutral persistence hint system
The system SHALL track consecutive turns where emotion remains `Neutral` and offer a hint after 3 consecutive Neutral turns.

#### Scenario: Hint appears after sustained Neutral emotion
- **WHEN** the player remains in `Neutral` emotion for 3 consecutive player turns
- **THEN** a hint message is displayed
- **AND** a suggested square is highlighted in the UI

### Requirement: Sentiment-aware difficulty adjustment across matches
The system SHALL adjust difficulty parameters between matches based on post-game sentiment:
- `negative` → decrease difficulty slightly for the next match
- `positive` → increase difficulty slightly for the next match
- `neutral` → no change

#### Scenario: Negative sentiment decreases difficulty next match
- **WHEN** post-game sentiment is `negative`
- **THEN** the next match uses slightly more assistive parameters (e.g., higher suboptimal-move probability or reduced depth in non-competitive modes)

#### Scenario: Positive sentiment increases difficulty next match
- **WHEN** post-game sentiment is `positive`
- **THEN** the next match uses slightly more competitive parameters (e.g., lower suboptimal-move probability)
