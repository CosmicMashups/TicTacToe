## ADDED Requirements

### Requirement: Webcam emotion detection thread
The system SHALL run facial landmark detection and emotion classification in a dedicated background thread to avoid blocking the game loop.

#### Scenario: Background processing does not block gameplay
- **WHEN** the game UI is running
- **THEN** webcam capture and face landmark processing occur off the main game/render thread
- **AND** the main loop remains responsive to input and rendering

### Requirement: MediaPipe Face Mesh landmark detection and overlay
The system SHALL use MediaPipe Face Mesh to detect 468 facial landmarks from webcam frames and render the landmark overlay on a preview image.

#### Scenario: Landmarks are detected and visualized
- **WHEN** a face is visible to the webcam
- **THEN** the system detects up to 468 facial landmarks
- **AND** draws the facial landmarks overlay on the preview
- **AND** displays the count of detected landmarks

### Requirement: Mouth-width feature extraction
The system SHALL extract the left and right mouth corner landmarks (indices 61 and 291) and compute a mouth width value based on their distance.

#### Scenario: Mouth width is computed from specified landmarks
- **WHEN** landmarks 61 and 291 are available in a processed frame
- **THEN** the system computes a numeric mouth width value from those landmarks
- **AND** displays the mouth width value in the preview/panel UI

### Requirement: Baseline calibration from reference image
The system SHALL load `data/face.jpg`, compute the baseline mouth width from that image, and derive a configurable Happy/Neutral threshold from the baseline.

#### Scenario: Threshold is derived from baseline
- **WHEN** the application starts
- **THEN** the baseline mouth width is computed from `data/face.jpg`
- **AND** the runtime threshold is derived from the baseline using a documented formula (e.g., baseline * factor)

### Requirement: Rule-based emotion classification
The system SHALL classify emotion as `Happy` when `mouth_width > threshold`, otherwise `Neutral`.

#### Scenario: Happy classification
- **WHEN** the mouth width is greater than the calibrated threshold
- **THEN** the detected emotion is `Happy`

#### Scenario: Neutral classification
- **WHEN** the mouth width is less than or equal to the calibrated threshold
- **THEN** the detected emotion is `Neutral`

### Requirement: Emotion smoothing via rolling window
The system SHALL smooth emotion output using a rolling window of the last 30 predictions and set the active emotion to the majority vote within the window.

#### Scenario: Majority vote smoothing prevents flicker
- **WHEN** per-frame emotion predictions fluctuate over time
- **THEN** the active emotion is determined by majority vote over the most recent 30 predictions
- **AND** rapid toggling between `Happy` and `Neutral` is reduced compared to raw per-frame output

### Requirement: Performance targets for emotion detection
The system SHALL process webcam emotion detection at a best-effort rate of at least 20 FPS on typical hardware and SHALL degrade gracefully if the target cannot be reached.

#### Scenario: Best-effort 20 FPS with graceful degradation
- **WHEN** the webcam thread is running
- **THEN** the system reports (or can display) the effective processing FPS
- **AND** if FPS drops below 20, the system continues running without blocking the game loop (e.g., via downscaling, skipping frames, or reduced overlay density)
