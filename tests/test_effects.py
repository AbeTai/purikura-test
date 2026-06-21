from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from purikura_test.api_models import EffectSettings
from purikura_test.effects import (
    DetectionBox,
    EffectPipeline,
    FACE_OVAL,
    LEFT_CHEEK,
    LEFT_EYE,
    LIPS,
    FaceDetections,
    FrameAsset,
    RIGHT_CHEEK,
    RIGHT_EYE,
    alpha_composite_bgra,
    build_face_skin_mask,
    draw_detection_debug,
    face_geometry_from_normalized_landmarks,
    face_geometry_from_points,
    local_translate,
    local_zoom,
)


class StaticTracker:
    def __init__(self, detections: FaceDetections) -> None:
        self.detections = detections

    def detect(self, frame_bgr: np.ndarray) -> FaceDetections:
        return self.detections


def synthetic_face(image_shape: tuple[int, int, int] = (160, 120, 3)) -> FaceDetections:
    points = np.zeros((478, 2), dtype=np.float32)
    points[:, :] = [60, 75]
    # Populate enough of the canonical face mesh indices used by the pipeline.
    for index, angle in zip(FACE_OVAL, np.linspace(-np.pi / 2, np.pi * 1.5, len(FACE_OVAL), endpoint=False)):
        points[index] = [60 + np.cos(angle) * 50, 78 + np.sin(angle) * 66]
    for index, angle in zip(LEFT_EYE, np.linspace(0, np.pi * 2, len(LEFT_EYE), endpoint=False)):
        points[index] = [84 + np.cos(angle) * 13, 58 + np.sin(angle) * 8]
    for index, angle in zip(RIGHT_EYE, np.linspace(0, np.pi * 2, len(RIGHT_EYE), endpoint=False)):
        points[index] = [36 + np.cos(angle) * 13, 58 + np.sin(angle) * 8]
    for index, angle in zip(LIPS, np.linspace(0, np.pi * 2, len(LIPS), endpoint=False)):
        points[index] = [60 + np.cos(angle) * 19, 103 + np.sin(angle) * 9]
    for index in LEFT_CHEEK:
        points[index] = [38, 86]
    for index in RIGHT_CHEEK:
        points[index] = [82, 86]
    return FaceDetections(faces=(face_geometry_from_points(points, image_shape),))


def test_alpha_composite_resizes_and_applies_alpha() -> None:
    base = np.zeros((4, 4, 3), dtype=np.uint8)
    overlay = np.zeros((2, 2, 4), dtype=np.uint8)
    overlay[:, :] = [0, 0, 255, 255]

    result = alpha_composite_bgra(base, overlay)

    assert result.shape == base.shape
    assert np.all(result[:, :, 2] == 255)
    assert np.all(result[:, :, :2] == 0)


def test_normalized_landmarks_build_face_and_eye_boxes() -> None:
    landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
    landmarks[10] = SimpleNamespace(x=0.5, y=0.1)
    landmarks[152] = SimpleNamespace(x=0.5, y=0.9)
    landmarks[234] = SimpleNamespace(x=0.1, y=0.5)
    landmarks[454] = SimpleNamespace(x=0.9, y=0.5)
    landmarks[33] = SimpleNamespace(x=0.28, y=0.38)
    landmarks[133] = SimpleNamespace(x=0.42, y=0.38)
    landmarks[362] = SimpleNamespace(x=0.58, y=0.38)
    landmarks[263] = SimpleNamespace(x=0.72, y=0.38)

    face = face_geometry_from_normalized_landmarks(landmarks, (200, 100, 3))

    assert face.bbox.width > 70
    assert face.bbox.height > 140
    assert face.right_eye.x < face.left_eye.x


def test_effect_pipeline_keeps_shape_with_frame_asset() -> None:
    base = np.full((160, 120, 3), 80, dtype=np.uint8)
    overlay = np.zeros((8, 8, 4), dtype=np.uint8)
    overlay[:, :, 1] = 255
    overlay[:, :, 3] = 128

    result = EffectPipeline(tracker=StaticTracker(synthetic_face())).apply(
        base,
        EffectSettings(skin_smoothing=0.8, brightness=10, contrast=1.0, saturation=1.2),
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
        EffectSettings(eye_enlarge=0.7)


def test_face_skin_mask_is_limited_to_detected_face_and_excludes_parts() -> None:
    frame = np.zeros((160, 120, 3), dtype=np.uint8)
    frame[:, :] = [90, 120, 170]
    detections = synthetic_face(frame.shape)
    face = detections.faces[0]
    mask = build_face_skin_mask(frame, detections)

    assert mask.shape == frame.shape[:2]
    assert mask[80, 60] > mask[0, 0]
    assert mask[face.left_eye.y + face.left_eye.height // 2, face.left_eye.x + face.left_eye.width // 2] < mask[80, 60]
    assert mask[face.lips.y + face.lips.height // 2, face.lips.x + face.lips.width // 2] < mask[80, 60]


def test_eye_zoom_face_slim_and_debug_boxes_change_frame() -> None:
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    frame[25:35, 25:35] = 180
    eye = DetectionBox(24, 24, 12, 12)

    zoomed = local_zoom(frame, eye, strength=0.35)
    slimmed = local_translate(frame, (28, 50), (6, 0), radius=20)
    debugged = draw_detection_debug(frame, synthetic_face((80, 80, 3)))

    assert zoomed.shape == frame.shape
    assert slimmed.shape == frame.shape
    assert debugged.shape == frame.shape
    assert not np.array_equal(debugged, frame)


def test_debug_overlay_reports_zero_faces() -> None:
    frame = np.zeros((80, 120, 3), dtype=np.uint8)

    debugged = draw_detection_debug(frame, FaceDetections(faces=()))

    assert not np.array_equal(debugged, frame)
