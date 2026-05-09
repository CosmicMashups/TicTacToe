## ADDED Requirements

### Requirement: Local runnable entrypoint
The system SHALL be runnable locally via `python main.py` and SHALL initialize all subsystems (vision thread, UI/game loop, NLP inference availability).

#### Scenario: Application starts successfully
- **WHEN** the user runs `python main.py`
- **THEN** the emotion detection subsystem is started (or a clear fallback is shown if unavailable)
- **AND** the Pygame UI launches

### Requirement: Shared emotion state and thread safety
The system SHALL maintain a thread-safe shared variable `current_emotion` that can be read continuously by the game/UI and written by the vision thread.

#### Scenario: Concurrent read/write does not corrupt state
- **WHEN** the vision thread updates the emotion while the UI reads it each frame
- **THEN** the program continues without race-condition crashes
- **AND** the UI reads a valid emotion value at all times

### Requirement: Optional webcam preview integration
The system SHALL display an in-game webcam preview panel with facial landmark overlay when a webcam is available, and SHALL show a friendly placeholder when it is not.

#### Scenario: Webcam panel degrades gracefully
- **WHEN** no webcam is available or permissions are denied
- **THEN** the webcam panel shows a clear "camera unavailable" message
- **AND** the rest of the game remains playable

### Requirement: Post-game feedback prompt and sentiment inference
After each match, the system SHALL prompt the player for textual feedback and compute sentiment (`positive`/`neutral`/`negative`) using the trained or loaded model.

#### Scenario: Feedback is requested after match ends
- **WHEN** a match ends (win or draw)
- **THEN** the player is prompted with a question (e.g., "How did you feel about this game?")
- **AND** the entered text is classified into one of the runtime sentiment labels

### Requirement: Basic analytics tracking and display
The system SHALL track at least:
- total games played
- wins/losses/draws
- emotion distribution across play time or turns
and SHALL display these statistics in a post-game screen.

#### Scenario: Analytics are updated each match
- **WHEN** a match ends
- **THEN** the analytics counters update (games played, outcomes)
- **AND** the post-game screen displays emotion distribution

### Requirement: Emotion-aware endgame messaging
The system SHALL display adaptive endgame messaging based on outcome and current emotion (e.g., Neutral + loss → offer tip; Happy + win → celebratory message).

#### Scenario: Neutral loss offers support
- **WHEN** the player loses and the active emotion is `Neutral`
- **THEN** the UI shows a supportive message offering a tip for next time

#### Scenario: Happy win celebrates
- **WHEN** the player wins and the active emotion is `Happy`
- **THEN** the UI shows a celebratory improvement-focused message
