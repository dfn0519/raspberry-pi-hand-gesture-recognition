"""Download the MediaPipe hand-landmarker model used by the project."""

from pathlib import Path
from urllib.request import urlretrieve


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT_ROOT / "models" / "hand_landmarker.task"


def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    if DESTINATION.exists():
        print(f"Model already exists: {DESTINATION}")
        return
    print("Downloading MediaPipe Hand Landmarker...")
    urlretrieve(MODEL_URL, DESTINATION)
    print(f"Saved to {DESTINATION}")


if __name__ == "__main__":
    main()
