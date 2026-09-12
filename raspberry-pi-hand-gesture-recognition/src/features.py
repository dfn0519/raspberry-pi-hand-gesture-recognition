"""Shared MediaPipe hand-landmark extraction and normalization."""

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def normalize_landmarks(landmarks) -> np.ndarray:
    """Convert 21 landmarks into a translation- and scale-invariant vector."""
    coordinates = np.array(
        [[landmark.x, landmark.y, landmark.z] for landmark in landmarks],
        dtype=np.float32,
    )
    coordinates -= coordinates[0]
    scale = np.linalg.norm(coordinates, axis=1).max()
    if scale > 0:
        coordinates /= scale
    return coordinates.reshape(-1)


class HandLandmarkExtractor:
    """Small wrapper around the MediaPipe Tasks Hand Landmarker."""

    def __init__(self, model_path: Path, running_mode: str = "image") -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found: {model_path}. "
                "Run src/download_hand_landmarker.py first."
            )

        model_buffer = model_path.read_bytes()
        mode = (
            vision.RunningMode.VIDEO
            if running_mode.lower() == "video"
            else vision.RunningMode.IMAGE
        )
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=model_buffer),
            num_hands=1,
            running_mode=mode,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._mode = mode

    def detect(self, bgr_image: np.ndarray, timestamp_ms: int | None = None):
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        if self._mode == vision.RunningMode.VIDEO:
            if timestamp_ms is None:
                raise ValueError("timestamp_ms is required in video mode")
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self._landmarker.detect(mp_image)
        return result.hand_landmarks[0] if result.hand_landmarks else None

    def close(self) -> None:
        self._landmarker.close()


def draw_landmarks(frame: np.ndarray, landmarks) -> None:
    """Draw the detected hand skeleton in place."""
    height, width = frame.shape[:2]
    points = [(int(item.x * width), int(item.y * height)) for item in landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 220, 80), 2)
    for point in points:
        cv2.circle(frame, point, 4, (0, 230, 255), -1)
