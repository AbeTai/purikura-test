import numpy as np
import pytest
from pydantic import ValidationError

from purikura_test.api_models import EffectSettings
from purikura_test.effects import (
    DetectionBox,
    EffectPipeline,
    FaceDetections,
    FrameAsset,
    alpha_composite_bgra,
    build_face_skin_mask,
    draw_detection_debug,
    local_zoom,
)


class StaticDetector:
    def __init__(self, detections: FaceDetections) -> None:
        self.detections = detections

    def detect(self, frame_bgr: np.ndarray) -> FaceDetections:
        return self.detections


def test_alpha_composite_resizes_and_applies_alpha() -> None:
    base = np.zeros((4, 4, 3), dtype=np.uint8)
    overlay = np.zeros((2, 2, 4), dtype=np.uint8)
    overlay[:, :] = [0, 0, 255, 255]

    result = alpha_composite_bgra(base, overlay)

    assert result.shape == base.shape
    assert np.all(result[:, :, 2] == 255)
    assert np.all(result[:, :, :2] == 0)


def test_effect_pipeline_keeps_shape_with_frame_asset() -> None:
    base = np.full((8, 8, 3), 80, dtype=np.uint8)
    overlay = np.zeros((8, 8, 4), dtype=np.uint8)
    overlay[:, :, 1] = 255
    overlay[:, :, 3] = 128

    detector = StaticDetector(FaceDetections(faces=(), eyes=()))
    result = EffectPipeline(detector=detector).apply(
        base,
        EffectSettings(skin_smoothing=0, brightness=10, contrast=1.1, saturation=1.2),
        FrameAsset(id=1, name="test", image_bgra=overlay),
    )

    assert result.shape == base.shape
    assert result.dtype == np.uint8


def test_effect_settings_validation_bounds() -> None:
    with pytest.raises(ValidationError):
        EffectSettings(skin_smoothing=1.1)

    with pytest.raises(ValidationError):
        EffectSettings(brightness=-81)

    with pytest.raises(ValidationError):
        EffectSettings(eye_enlarge=0.5)


def test_face_skin_mask_is_limited_to_detected_face() -> None:
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[:, :] = [90, 120, 170]
    mask = build_face_skin_mask(frame, (DetectionBox(5, 5, 10, 10),))

    assert mask.shape == frame.shape[:2]
    assert mask[10, 10] > mask[0, 0]


def test_eye_zoom_and_debug_boxes_change_frame() -> None:
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[15:25, 15:25] = 180
    eye = DetectionBox(14, 14, 12, 12)

    zoomed = local_zoom(frame, eye, strength=0.25)
    debugged = draw_detection_debug(
        frame,
        FaceDetections(faces=(DetectionBox(8, 8, 24, 24),), eyes=(eye,)),
    )

    assert zoomed.shape == frame.shape
    assert debugged.shape == frame.shape
    assert not np.array_equal(debugged, frame)
