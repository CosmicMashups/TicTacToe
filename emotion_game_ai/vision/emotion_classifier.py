"""Rule-based emotion classifier for Happy/Neutral using mouth width."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


LEFT_MOUTH_IDX = 61
RIGHT_MOUTH_IDX = 291


@dataclass(frozen=True)
class Calibration:
    baseline_mouth_width: float
    threshold: float


def mouth_width_px(
    landmarks_xy: np.ndarray, image_shape_hw: Tuple[int, int]
) -> float:
    """
    Compute mouth width in pixels using landmark indices 61 and 291.

    Parameters
    ----------
    landmarks_xy:
        Array of shape (N, 2) with normalized landmark coordinates (x,y) in [0,1].
    image_shape_hw:
        (height, width) of the image.
    """
    h, w = image_shape_hw
    p1 = landmarks_xy[LEFT_MOUTH_IDX]
    p2 = landmarks_xy[RIGHT_MOUTH_IDX]
    dx = (p1[0] - p2[0]) * w
    dy = (p1[1] - p2[1]) * h
    return float((dx * dx + dy * dy) ** 0.5)


def classify_emotion(mouth_width: float, threshold: float) -> str:
    """Happy if mouth_width > threshold else Neutral."""
    return "Happy" if mouth_width > threshold else "Neutral"

