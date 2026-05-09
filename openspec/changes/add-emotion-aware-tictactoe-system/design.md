## Context
This change introduces a single-user, local desktop game that integrates:
- Real-time computer vision inference (webcam → face landmarks → simple emotion rule)
- Post-game NLP sentiment inference (text → TF-IDF → Logistic Regression)
- Game AI (Minimax) and adaptive difficulty logic
- A modern Pygame UI with animations and theming driven by the current emotion

Constraints:
- Must run locally on Windows with Python 3.10+.
- Must keep the game loop responsive (target 60 FPS) and avoid blocking operations.
- Emotion detection must run concurrently and achieve at least 20 FPS (best-effort, hardware dependent).

## Goals / Non-Goals
- Goals:
  - A clean, modular architecture that keeps CV/NLP/game/UI concerns isolated.
  - A thread-safe shared state model for emotion and match/session metadata.
  - An extensible UI animation system based on delta-time and easing functions.
  - Deterministic “optimal” baseline gameplay (Minimax) with controlled degradation in assistive modes.
- Non-Goals:
  - General-purpose emotion recognition beyond the specified Happy/Neutral heuristic.
  - Cloud services, online play, or external telemetry.
  - Training a production-grade sentiment model beyond the specified baseline pipeline.

## Decisions
- Decision: Use a background thread for webcam capture and MediaPipe processing.
  - Rationale: Keeps Pygame’s main thread devoted to event handling and rendering, and avoids blocking calls.
  - Alternative: Multiprocessing for isolation; rejected for complexity and cross-platform packaging friction.

- Decision: Implement emotion smoothing via a fixed-size rolling window of recent predictions.
  - Rationale: Reduces flicker from frame-to-frame variance and gives stable UX transitions.
  - Alternative: Exponential moving average over probabilities; not applicable with a rule-based classifier.

- Decision: Keep Minimax “correct” and introduce assistive behavior as a policy wrapper.
  - Rationale: Preserves correctness, makes it easy to reason about intentional mistakes, and supports sentiment-based tuning.

- Decision: Maintain a single shared `GameState`/`SharedState` object with a lock.
  - Rationale: A single synchronization point is easier to audit for thread safety.

## Risks / Trade-offs
- Webcam performance variability:
  - Mitigation: Allow resolution downscaling and frame skipping in the CV thread; keep UI decoupled.
- Very large sentiment dataset (~839k rows):
  - Mitigation: Support training on the full dataset when available; allow configurable sampling for development; always support loading a pre-trained pickle for gameplay.
- Pygame audio availability:
  - Mitigation: Treat audio as optional (graceful degradation if assets missing or mixer init fails).

## Migration Plan
Not applicable (new project capability set).

## Open Questions
- None (scope is fully defined by this proposal).
