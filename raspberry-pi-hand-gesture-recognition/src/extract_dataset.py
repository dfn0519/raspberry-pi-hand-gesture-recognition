"""Extract normalized MediaPipe landmarks from the image dataset."""

import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np

from features import HandLandmarkExtractor, normalize_landmarks


LABELS = {"rock": 0, "paper": 1, "scissors": 2}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_image(path: Path):
    """Read paths containing non-ASCII characters reliably on Windows."""
    encoded = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def extract_split(folder: Path, detector: HandLandmarkExtractor):
    features, labels = [], []
    for category, label in LABELS.items():
        category_folder = folder / category
        if not category_folder.exists():
            raise FileNotFoundError(f"Missing dataset folder: {category_folder}")

        accepted = 0
        for path in sorted(category_folder.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            image = read_image(path)
            if image is None:
                continue
            landmarks = detector.detect(image)
            if landmarks is None:
                continue
            features.append(normalize_landmarks(landmarks))
            labels.append(label)
            accepted += 1
        print(f"{folder.name}/{category}: {accepted} samples")
    return np.asarray(features), np.asarray(labels)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "landmark_features.joblib",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = HandLandmarkExtractor(PROJECT_ROOT / "models" / "hand_landmarker.task")
    try:
        x_train, y_train = extract_split(args.dataset / "train", detector)
        x_test, y_test = extract_split(args.dataset / "test", detector)
    finally:
        detector.close()

    if not len(x_train) or not len(x_test):
        raise RuntimeError("No usable landmarks were extracted from the dataset")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"X_train": x_train, "y_train": y_train, "X_test": x_test, "y_test": y_test},
        args.output,
    )
    print(f"Saved features to {args.output}")


if __name__ == "__main__":
    main()
