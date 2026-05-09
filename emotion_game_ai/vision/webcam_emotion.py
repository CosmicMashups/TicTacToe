"""Webcam-based facial landmark emotion detection using MediaPipe Face Mesh."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional

import cv2
import numpy as np

from emotion_game_ai.utils.threading_utils import (
    RollingMajority,
    SharedState,
    StoppableThread,
    VisionSnapshot,
)
from emotion_game_ai.vision.emotion_classifier import (
    Calibration,
    classify_emotion,
    mouth_width_px,
)


@dataclass(frozen=True)
class VisionConfig:
    camera_index: int = 0
    target_fps: float = 20.0
    preview_width: int = 300
    calibration_factor: float = 1.10
    smoothing_window: int = 30
    max_faces: int = 1


class WebcamEmotionWorker(StoppableThread):
    """
    Background thread that captures webcam frames, runs MediaPipe Face Mesh,
    computes mouth width and a rule-based emotion, then publishes a smoothed
    emotion + preview frame to SharedState.
    """

    def __init__(
        self,
        shared: SharedState,
        face_image_path: str,
        camera_index: int = 0,
        target_fps: float = 20.0,
        smoothing_window: int = 30,
    ) -> None:
        super().__init__(daemon=True)
        self.shared = shared
        self.face_image_path = face_image_path
        self.cfg = VisionConfig(
            camera_index=camera_index,
            target_fps=target_fps,
            smoothing_window=smoothing_window,
        )
        self._calibration: Optional[Calibration] = None

        self._majority = RollingMajority(window_size=self.cfg.smoothing_window)
        self._mp_face_mesh = None
        self._mp_drawing = None
        self._mp_styles = None

    def _compute_calibration(self) -> Optional[Calibration]:
        img_bgr = cv2.imread(self.face_image_path)
        if img_bgr is None:
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        if self._mp_face_mesh is None:
            return None

        with self._mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as fm:
            res = fm.process(img_rgb)
            if not res.multi_face_landmarks:
                return None
            face = res.multi_face_landmarks[0]
            xy = np.array([(lm.x, lm.y) for lm in face.landmark], dtype=np.float32)
            baseline = mouth_width_px(xy, (h, w))
            threshold = baseline * self.cfg.calibration_factor
            return Calibration(baseline_mouth_width=float(baseline), threshold=float(threshold))

    @staticmethod
    def _resize_for_preview(rgb: np.ndarray, preview_width: int) -> np.ndarray:
        h, w = rgb.shape[:2]
        if w == 0 or h == 0:
            return rgb
        scale = preview_width / float(w)
        new_h = max(1, int(h * scale))
        return cv2.resize(rgb, (preview_width, new_h), interpolation=cv2.INTER_AREA)

    def run(self) -> None:
        # Probe available camera indices (best-effort).
        try:
            available = self._probe_cameras(max_index=4)
            self.shared.set_available_cameras(available)
        except Exception:
            self.shared.set_available_cameras([self.cfg.camera_index])

        mp_face_mesh = None
        mp_drawing = None
        mp_styles = None

        # Prefer direct solution imports to avoid pulling in mediapipe.tasks
        # (which drags TensorFlow and can break on protobuf mismatches).
        try:
            from mediapipe.python.solutions import face_mesh as _mp_face_mesh
            from mediapipe.python.solutions import drawing_utils as _mp_drawing
            from mediapipe.python.solutions import drawing_styles as _mp_styles

            mp_face_mesh = _mp_face_mesh
            mp_drawing = _mp_drawing
            mp_styles = _mp_styles
        except Exception:
            # Fallback to classic top-level import if direct solutions are unavailable.
            try:
                import mediapipe as mp  # may fail if TF/protobuf are broken
                mp_face_mesh = mp.solutions.face_mesh
                mp_drawing = mp.solutions.drawing_utils
                mp_styles = mp.solutions.drawing_styles
            except Exception:
                # MediaPipe is not usable; we will still run a plain camera loop so
                # the GUI can see and switch cameras, just without landmarks/emotion.
                mp_face_mesh = None
                mp_drawing = None
                mp_styles = None

        self._mp_face_mesh = mp_face_mesh
        self._mp_drawing = mp_drawing
        self._mp_styles = mp_styles

        # Temporarily ignore calibration from face.jpg and use a fixed threshold.
        # Happy: mouth_width > 65.0, Neutral otherwise.
        self._calibration = Calibration(baseline_mouth_width=0.0, threshold=65.0)

        cap: Optional[cv2.VideoCapture] = None
        active_index = None

        def open_camera(index: int) -> Optional[cv2.VideoCapture]:
            # Try multiple backends to maximize compatibility.
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]
            for backend in backends:
                try:
                    c = cv2.VideoCapture(int(index), backend) if isinstance(backend, int) else cv2.VideoCapture(int(index), backend)
                except Exception:
                    continue
                if c.isOpened():
                    return c
                try:
                    c.release()
                except Exception:
                    pass
            return None

        last_t = time.perf_counter()
        fps_window = []
        next_tick = time.perf_counter()
        tick_dt = 1.0 / max(1e-6, self.cfg.target_fps)

        # If MediaPipe is available, we use FaceMesh; otherwise we just stream frames.
        if self._mp_face_mesh is not None:
            with self._mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=self.cfg.max_faces,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as fm:
                while not self.stop_event.is_set():
                    now = time.perf_counter()
                    if now < next_tick:
                        time.sleep(min(0.005, next_tick - now))
                        continue
                    next_tick = now + tick_dt

                    desired = self.shared.get_requested_camera()
                    if cap is None or active_index != desired:
                        if cap is not None:
                            try:
                                cap.release()
                            except Exception:
                                pass
                            cap = None
                        cap = open_camera(desired)
                        active_index = desired if cap is not None else None

                    if cap is None:
                        snap = VisionSnapshot(
                            emotion_raw="Neutral",
                            emotion_smoothed="Neutral",
                            mouth_width=0.0,
                            landmarks_detected=0,
                            fps=0.0,
                            camera_index=int(desired),
                            camera_ok=False,
                            preview_rgb=None,
                        )
                        self.shared.update_vision(snap)
                        continue

                    ok, frame_bgr = cap.read()
                    if not ok or frame_bgr is None:
                        continue

                    frame_bgr = cv2.flip(frame_bgr, 1)
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    h, w = frame_rgb.shape[:2]

                    res = fm.process(frame_rgb)
                    emotion_raw = "Neutral"
                    emotion_smoothed = "Neutral"
                    mouth_w = 0.0
                    landmarks_count = 0

                    overlay_rgb = frame_rgb.copy()

                    if res.multi_face_landmarks:
                        face = res.multi_face_landmarks[0]
                        landmarks_count = len(face.landmark)
                        xy = np.array([(lm.x, lm.y) for lm in face.landmark], dtype=np.float32)
                        mouth_w = mouth_width_px(xy, (h, w))
                        emotion_raw = classify_emotion(mouth_w, self._calibration.threshold)
                        self._majority.add(emotion_raw)
                        emotion_smoothed = self._majority.majority(default="Neutral")

                        self._mp_drawing.draw_landmarks(
                            image=overlay_rgb,
                            landmark_list=face,
                            connections=self._mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self._mp_styles.get_default_face_mesh_tesselation_style(),
                        )

                    dt = max(1e-6, now - last_t)
                    last_t = now
                    fps_window.append(1.0 / dt)
                    if len(fps_window) > 30:
                        fps_window.pop(0)
                    fps = float(sum(fps_window) / len(fps_window))

                    preview = self._resize_for_preview(overlay_rgb, self.cfg.preview_width)

                    snap = VisionSnapshot(
                        emotion_raw=emotion_raw,
                        emotion_smoothed=emotion_smoothed,
                        mouth_width=float(mouth_w),
                        landmarks_detected=int(landmarks_count),
                        fps=fps,
                        camera_index=int(active_index if active_index is not None else desired),
                        camera_ok=True,
                        preview_rgb=preview,
                    )
                    self.shared.update_vision(snap)
        else:
            # Plain camera loop: no landmarks/emotion, but full camera-switching diagnostics.
            while not self.stop_event.is_set():
                now = time.perf_counter()
                if now < next_tick:
                    time.sleep(min(0.005, next_tick - now))
                    continue
                next_tick = now + tick_dt

                desired = self.shared.get_requested_camera()
                if cap is None or active_index != desired:
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = None
                    cap = open_camera(desired)
                    active_index = desired if cap is not None else None

                if cap is None:
                    snap = VisionSnapshot(
                        emotion_raw="Neutral",
                        emotion_smoothed="Neutral",
                        mouth_width=0.0,
                        landmarks_detected=0,
                        fps=0.0,
                        camera_index=int(desired),
                        camera_ok=False,
                        preview_rgb=None,
                    )
                    self.shared.update_vision(snap)
                    continue

                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    continue

                frame_bgr = cv2.flip(frame_bgr, 1)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                dt = max(1e-6, now - last_t)
                last_t = now
                fps_window.append(1.0 / dt)
                if len(fps_window) > 30:
                    fps_window.pop(0)
                fps = float(sum(fps_window) / len(fps_window))

                preview = self._resize_for_preview(frame_rgb, self.cfg.preview_width)

                snap = VisionSnapshot(
                    emotion_raw="Neutral",
                    emotion_smoothed="Neutral",
                    mouth_width=0.0,
                    landmarks_detected=0,
                    fps=fps,
                    camera_index=int(active_index if active_index is not None else desired),
                    camera_ok=True,
                    preview_rgb=preview,
                )
                self.shared.update_vision(snap)

        if cap is not None:
            cap.release()

    @staticmethod
    def _probe_cameras(max_index: int = 4) -> list[int]:
        """Best-effort probe of camera indices 0..max_index."""
        found: list[int] = []
        for idx in range(0, int(max_index) + 1):
            ok = False
            # Try a couple of backends.
            for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, 0):
                try:
                    cap = cv2.VideoCapture(idx, backend)
                except Exception:
                    continue
                if cap.isOpened():
                    ok = True
                    try:
                        cap.release()
                    except Exception:
                        pass
                    break
                try:
                    cap.release()
                except Exception:
                    pass
            if ok:
                found.append(idx)
        return found or [0]

