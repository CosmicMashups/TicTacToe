## ADDED Requirements

### Requirement: Window and layout zones
The system SHALL present a Pygame window at 1280x720 and implement four UI zones: Top HUD panel, Left Webcam panel, Center Game Board, and Right AI Status panel.

#### Scenario: UI zones are visible and stable
- **WHEN** the game starts
- **THEN** the window size is 1280x720
- **AND** the HUD panel is rendered at the top
- **AND** webcam preview is rendered in the left panel (or a clear fallback message is shown)
- **AND** the Tic-Tac-Toe board is rendered in the center
- **AND** the AI status panel is rendered on the right

### Requirement: HUD displays emotion, AI mode, status, and statistics
The HUD SHALL display the current smoothed emotion, AI mode, game status, and basic player statistics (games played and wins at minimum).

#### Scenario: HUD updates with emotion and match state
- **WHEN** `current_emotion` changes or the match state changes
- **THEN** the HUD reflects the updated emotion label and AI mode
- **AND** the game status text updates (e.g., turn, win, draw)

### Requirement: Smooth board rendering and hover effects
The system SHALL render a smooth Tic-Tac-Toe grid, detect hover over empty cells, and display hover highlights using emotion-specific accent colors.

#### Scenario: Hover highlight appears on empty cell
- **WHEN** the mouse is over an empty cell on the board
- **THEN** a highlight appears on that cell
- **AND** the highlight color matches the active emotion theme

### Requirement: Animated move placement
The system SHALL animate piece placement:
- `X`: two animated strokes over ~300ms
- `O`: radial/arc draw animation over ~300ms

#### Scenario: X animates with two strokes
- **WHEN** the human places an `X`
- **THEN** stroke 1 is animated
- **AND** stroke 2 is animated
- **AND** the combined animation completes in approximately 300ms

#### Scenario: O animates radially
- **WHEN** the AI places an `O`
- **THEN** the circle is animated via a radial/arc draw
- **AND** the animation completes in approximately 300ms

### Requirement: Emotion-based theming with smooth transitions
The UI SHALL implement at least two themes:
- Happy: green accents, energetic animation pacing
- Neutral: blue accents, calmer pacing
Theme changes SHALL transition smoothly via color interpolation rather than abrupt switches.

#### Scenario: Theme transitions smoothly on emotion change
- **WHEN** the active emotion changes between `Neutral` and `Happy`
- **THEN** UI accent colors interpolate smoothly over time
- **AND** the board/webcam panels reflect the new theme without abrupt jumps

### Requirement: Win and draw animations
The system SHALL provide distinct end-of-match feedback:
- Win: highlight the winning line and show a confetti/particle effect
- Draw: a subtle board shake and a draw message

#### Scenario: Winning line and particles appear
- **WHEN** a win is detected
- **THEN** the winning row/column/diagonal is highlighted
- **AND** a particle/confetti effect is shown

#### Scenario: Draw animation appears
- **WHEN** a draw is detected
- **THEN** the board performs a subtle shake animation
- **AND** a draw message is displayed

### Requirement: Dialogue and hint message display
The system SHALL display short-lived dialogue messages in the HUD or status panel, including emotion-specific encouragement and hint text.

#### Scenario: Message is shown temporarily
- **WHEN** the system emits a dialogue or hint message
- **THEN** the message appears in the UI
- **AND** the message automatically clears after a few seconds

### Requirement: 60 FPS game UI loop
The UI loop SHALL target 60 FPS and use delta-time for animation updates.

#### Scenario: Animations are time-based
- **WHEN** the frame rate fluctuates
- **THEN** animations progress based on measured delta-time rather than fixed per-frame steps
