"""Application orchestration."""

from __future__ import annotations

import os
import time

from emotion_game_ai.utils.threading_utils import SharedState
from emotion_game_ai.vision.webcam_emotion import WebcamEmotionWorker
from emotion_game_ai.game.pygame_ui import PygameApp
from emotion_game_ai.nlp.sentiment_model import SentimentModel


def run_app() -> None:
    shared = SharedState()

    model_dir = os.path.join("data", "models")
    os.makedirs(model_dir, exist_ok=True)
    sentiment = SentimentModel(model_dir=model_dir)
    loaded = sentiment.try_load()
    if not loaded:
        # Best-effort lightweight training so post-game feedback affects behavior
        # even if the user has not run the training CLI explicitly.
        try:
            dataset_path = os.path.join("assets", "sentiment_data.csv")
            sentiment.train(dataset_path=dataset_path, sample_rows=20000)
            sentiment.save()
        except Exception:
            # If training fails (e.g., missing CSV), runtime will gracefully
            # fall back to neutral sentiment for predictions.
            pass

    vision_worker = WebcamEmotionWorker(
        shared=shared,
        face_image_path=os.path.join("data", "face.jpg"),
        camera_index=0,
        target_fps=20.0,
        smoothing_window=30,
    )
    vision_worker.start()

    try:
        app = PygameApp(shared=shared, sentiment=sentiment)
        app.run()
    finally:
        vision_worker.stop()
        vision_worker.join(timeout=2.0)
        time.sleep(0.05)

