"""Run real-time rock-paper-scissors inference from a camera."""

import argparse
import time
from pathlib import Path

import cv2
import joblib
import numpy as np

from features import HandLandmarkExtractor, draw_landmarks, normalize_landmarks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = {0: "Rock", 1: "Paper", 2: "Scissors"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "best_landmark_model.pkl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1")

    classifier = joblib.load(args.model)
    detector = HandLandmarkExtractor(
        PROJECT_ROOT / "models" / "hand_landmarker.task", running_mode="video"
    )
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        detector.close()
        raise RuntimeError(f"Could not open camera {args.camera}")

    started_at = time.perf_counter()
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            timestamp_ms = int((time.perf_counter() - started_at) * 1000)
            landmarks = detector.detect(frame, timestamp_ms)

            label = "No hand"
            if landmarks is not None:
                draw_landmarks(frame, landmarks)
                vector = normalize_landmarks(landmarks).reshape(1, -1)
                probabilities = classifier.predict_proba(vector)[0]
                class_index = int(np.argmax(probabilities))
                confidence = float(probabilities[class_index])
                label = (
                    f"{LABELS[class_index]} ({confidence:.1%})"
                    if confidence >= args.threshold
                    else f"Unknown ({confidence:.1%})"
                )

            cv2.rectangle(frame, (10, 10), (410, 60), (20, 20, 20), -1)
            cv2.putText(
                frame, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (50, 230, 100), 2, cv2.LINE_AA,
            )
            cv2.imshow("Raspberry Pi Hand Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
