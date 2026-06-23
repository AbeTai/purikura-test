import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from purikura_test.api_models import CameraInfo
from purikura_test.app import create_app
from purikura_test.repository import CaptureRepository
from purikura_test.runtime import PurikuraRuntime


def png_bytes() -> bytes:
    image = Image.new("RGBA", (4, 4), (255, 0, 0, 128))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_app_core_api_flow(monkeypatch) -> None:
    repository = CaptureRepository("sqlite:///:memory:")
    runtime = PurikuraRuntime(repository)
    runtime._latest_processed = np.full((12, 16, 3), 120, dtype=np.uint8)
    app = create_app(runtime=runtime, start_camera=False)

    monkeypatch.setattr(
        "purikura_test.app.discover_cameras",
        lambda: [CameraInfo(id=0, name="Built-in camera", available=True)],
    )

    with TestClient(app) as client:
        cameras = client.get("/api/cameras")
        assert cameras.status_code == 200
        assert cameras.json()[0]["id"] == 0

        effects = client.put(
            "/api/effects",
            json={
                "processing_profile": "fast",
                "skin_smoothing": 0.2,
                "purikura_intensity": 0.7,
                "eye_enlarge": 0.12,
                "face_slim": 0.2,
                "doll_intensity": 0.9,
                "background_high_key": 0.6,
                "debug_overlay": "masks",
            },
        )
        assert effects.status_code == 200
        assert effects.json()["processing_profile"] == "fast"
        assert effects.json()["debug_overlay"] == "masks"
        assert effects.json()["doll_intensity"] == 0.9
        assert "lip_gloss" not in effects.json()

        performance = client.get("/api/performance")
        assert performance.status_code == 200
        assert performance.json()["profile"] in {"quality", "fast"}
        assert isinstance(performance.json()["dropped_frames"], int)
        assert isinstance(performance.json()["discarded_processed_frames"], int)
        assert isinstance(performance.json()["frame_age_ms"], float)
        assert isinstance(performance.json()["motion_factor"], float)
        assert isinstance(performance.json()["latest_raw_frame_id"], int)

        invalid_effects = client.put(
            "/api/effects",
            json={
                "processing_profile": "quality",
                "skin_smoothing": 2,
                "purikura_intensity": 0.7,
                "eye_enlarge": 0.12,
                "face_slim": 0.2,
                "doll_intensity": 0.7,
                "background_high_key": 0.6,
                "debug_overlay": "off",
            },
        )
        assert invalid_effects.status_code == 422

        removed_effects = client.put(
            "/api/effects",
            json={
                "processing_profile": "quality",
                "skin_smoothing": 0.2,
                "purikura_intensity": 0.7,
                "eye_enlarge": 0.12,
                "face_slim": 0.2,
                "doll_intensity": 0.7,
                "background_high_key": 0.6,
                "lip_gloss": 0.7,
                "debug_overlay": "off",
            },
        )
        assert removed_effects.status_code == 422

        invalid_profile = client.put(
            "/api/effects",
            json={
                "processing_profile": "turbo",
                "skin_smoothing": 0.2,
                "purikura_intensity": 0.7,
                "eye_enlarge": 0.12,
                "face_slim": 0.2,
                "doll_intensity": 0.7,
                "background_high_key": 0.6,
                "debug_overlay": "off",
            },
        )
        assert invalid_profile.status_code == 422

        upload = client.post(
            "/api/frames",
            files={"file": ("frame.png", png_bytes(), "image/png")},
        )
        assert upload.status_code == 200
        frame_id = upload.json()["id"]

        selected = client.put("/api/frame/current", json={"frame_id": frame_id})
        assert selected.status_code == 200

        captured = client.post("/api/captures")
        assert captured.status_code == 200
        capture_id = captured.json()["id"]

        captures = client.get("/api/captures")
        assert captures.status_code == 200
        assert captures.json()[0]["id"] == capture_id

        image = client.get(f"/api/captures/{capture_id}/image")
        assert image.status_code == 200
        assert image.headers["content-type"] == "image/jpeg"


def test_frame_upload_rejects_non_png() -> None:
    repository = CaptureRepository("sqlite:///:memory:")
    runtime = PurikuraRuntime(repository)
    app = create_app(runtime=runtime, start_camera=False)

    with TestClient(app) as client:
        response = client.post(
            "/api/frames",
            files={"file": ("frame.txt", b"not an image", "text/plain")},
        )

    assert response.status_code == 400
