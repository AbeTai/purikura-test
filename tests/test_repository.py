from pathlib import Path

from purikura_test.api_models import EffectSettings
from purikura_test.repository import CaptureRepository


def test_repository_persists_frames_and_captures(tmp_path: Path) -> None:
    repository = CaptureRepository(f"sqlite:///{tmp_path / 'purikura.sqlite3'}")
    repository.init_schema()

    frame = repository.add_frame(name="frame.png", image_blob=b"png-bytes")
    capture = repository.add_capture(
        camera_id=0,
        settings=EffectSettings(),
        frame_id=frame.id,
        image_blob=b"jpeg-bytes",
        image_mime="image/jpeg",
        width=640,
        height=480,
    )

    assert repository.list_frames()[0].id == frame.id
    assert repository.get_frame_blob(frame.id) == (b"png-bytes", "image/png")
    assert repository.list_captures()[0].id == capture.id
    assert repository.get_capture_image(capture.id) == (b"jpeg-bytes", "image/jpeg")
