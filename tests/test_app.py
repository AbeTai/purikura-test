import io
import time

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from purikura_test.api_models import CameraInfo, EffectSettings
from purikura_test.app import create_app
from purikura_test.repository import CaptureRecord, CaptureRepository
from purikura_test.runtime import FramePacket, PurikuraRuntime


def png_bytes() -> bytes:
    image = Image.new("RGBA", (4, 4), (255, 0, 0, 128))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class PassthroughPipeline:
    last_motion_ratio = 0.0

    def __init__(self) -> None:
        self.settings: list[EffectSettings] = []

    def apply(self, frame_bgr: np.ndarray, settings: EffectSettings, frame_asset=None) -> np.ndarray:
        self.settings.append(settings)
        return frame_bgr.copy()


class FakeAliveThread:
    def is_alive(self) -> bool:
        return True

    def join(self, timeout: float | None = None) -> None:
        return None


def test_app_core_api_flow(monkeypatch) -> None:
    repository = CaptureRepository("sqlite:///:memory:")
    pipeline = PassthroughPipeline()
    runtime = PurikuraRuntime(repository, pipeline=pipeline)
    runtime._camera_thread = FakeAliveThread()
    runtime._latest_raw_packet = FramePacket(
        id=1,
        captured_at=time.perf_counter(),
        frame=np.full((12, 16, 3), 120, dtype=np.uint8),
    )
    runtime._latest_raw_frame_id = 1
    app = create_app(runtime=runtime, start_camera=False)

    monkeypatch.setattr(
        "purikura_test.app.discover_cameras",
        lambda **kwargs: [CameraInfo(id=kwargs["active_camera_id"], name="Built-in camera", available=True)],
    )

    with TestClient(app) as client:
        cameras = client.get("/api/cameras")
        assert cameras.status_code == 200
        assert cameras.json()[0]["id"] == 0

        current_effects = client.get("/api/effects")
        assert current_effects.status_code == 200
        assert current_effects.json()["processing_profile"] == "fast"

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
        assert isinstance(performance.json()["landmark_age_ms"], float)
        assert isinstance(performance.json()["mask_age_ms"], float)
        assert isinstance(performance.json()["motion_factor"], float)
        assert isinstance(performance.json()["publish_interval_ms"], float)
        assert isinstance(performance.json()["publish_lag_frames"], int)
        assert isinstance(performance.json()["preview_stall_ms"], float)
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
        assert pipeline.settings[-1].processing_profile == "quality"

        captures = client.get("/api/captures")
        assert captures.status_code == 200
        assert captures.json()[0]["id"] == capture_id

        with repository.session() as session:
            record = session.get(CaptureRecord, capture_id)
            assert record is not None
            assert '"processing_profile":"quality"' in record.effect_settings_json

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
