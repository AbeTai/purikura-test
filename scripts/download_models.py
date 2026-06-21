from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    PROJECT_ROOT / "models" / "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    ),
    PROJECT_ROOT / "models" / "selfie_multiclass_256x256.tflite": (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
    ),
}


def main() -> None:
    for model_path, model_url in MODELS.items():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if model_path.exists() and model_path.stat().st_size > 0:
            print(f"Already exists: {model_path}")
            continue
        print(f"Downloading {model_url}")
        urlretrieve(model_url, model_path)
        print(f"Saved: {model_path}")


if __name__ == "__main__":
    main()
