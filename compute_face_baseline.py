from __future__ import annotations

import cv2
import numpy as np

from mediapipe.python.solutions import face_mesh as mp_face_mesh

from emotion_game_ai.vision.emotion_classifier import mouth_width_px


def main() -> int:
    img_bgr = cv2.imread("data/face.jpg")
    if img_bgr is None:
        print("face.jpg not found or unreadable")
        return 1

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as fm:
        res = fm.process(img_rgb)
        if not res.multi_face_landmarks:
            print("No face landmarks detected in face.jpg")
            return 1
        face = res.multi_face_landmarks[0]
        xy = np.array([(lm.x, lm.y) for lm in face.landmark], dtype=np.float32)
        baseline = mouth_width_px(xy, (h, w))
        threshold = baseline * 1.10
        print(f"Baseline mouth width: {baseline:.2f}")
        print(f"Calibrated threshold (baseline*1.10): {threshold:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

