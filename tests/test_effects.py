from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from purikura_test.api_models import EffectSettings
from purikura_test.effects import (
    DetectionBox,
    EffectPipeline,
    FACE_OVAL,
    LEFT_EYE,
    LIPS,
    FaceDetections,
    FrameAsset,
    RIGHT_EYE,
    SegmentationMasks,
    alpha_composite_bgra,
    build_face_skin_mask,
    build_part_masks,
    CachedFaceTracker,
    draw_debug_overlay,
    draw_detection_debug,
    empty_masks,
    face_geometry_from_box,
    face_geometry_from_normalized_landmarks,
    head_roi,
    local_translate,
    local_zoom,
    masks_from_segmenter_result,
)


class StaticTracker:
    def __init__(self, detections: FaceDetections) -> None:
        self.detections = detections

    def detect(self, frame_bgr: np.ndarray) -> FaceDetections:
        return self.detections


class StaticSegmenter:
    def __init__(self, masks: SegmentationMasks) -> None:
        self.masks = masks

    def segment(self, frame_bgr: np.ndarray, detections: FaceDetections) -> SegmentationMasks:
        return self.masks


class DynamicSegmenter:
    def segment(self, frame_bgr: np.ndarray, detections: FaceDetections) -> SegmentationMasks:
        return synthetic_masks(frame_bgr.shape, detections)


class CountingTracker:
    def __init__(self, detections: FaceDetections) -> None:
        self.detections = detections
        self.calls = 0

    def detect(self, frame_bgr: np.ndarray) -> FaceDetections:
        self.calls += 1
        return self.detections


class FakeMask:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data

    def numpy_view(self) -> np.ndarray:
        return self.data


class FakeSegmenterResult:
    def __init__(self, masks: list[np.ndarray]) -> None:
        self.confidence_masks = [FakeMask(mask) for mask in masks]
        self.category_mask = None


def synthetic_face(image_shape: tuple[int, int, int] = (160, 120, 3)) -> FaceDetections:
    return FaceDetections(faces=(face_geometry_from_box(DetectionBox(18, 14, 84, 128), image_shape),))


def synthetic_masks(image_shape: tuple[int, int, int], detections: FaceDetections | None = None) -> SegmentationMasks:
    height, width = image_shape[:2]
    face_skin = np.zeros((height, width), dtype=np.float32)
    body_skin = np.zeros((height, width), dtype=np.float32)
    hair = np.zeros((height, width), dtype=np.float32)
    cv2_like_ellipse(face_skin, (width // 2, int(height * 0.48)), (width // 3, height // 3), 1.0)
    body_skin[int(height * 0.38) : int(height * 0.82), width // 8 : width - width // 8] = 0.85
    hair[int(height * 0.10) : int(height * 0.32), width // 5 : width - width // 5] = 0.9
    protected = build_part_masks(image_shape, detections or FaceDetections(faces=())).protected
    skin = np.maximum(face_skin, body_skin)
    head = np.maximum(skin, hair)
    return SegmentationMasks(
        head=head,
        skin=skin,
        hair=hair,
        protected=protected,
        face_skin=face_skin,
        body_skin=body_skin,
    )


def cv2_like_ellipse(mask: np.ndarray, center: tuple[int, int], axes: tuple[int, int], value: float) -> None:
    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    normalized = ((xx - center[0]) / max(1, axes[0])) ** 2 + ((yy - center[1]) / max(1, axes[1])) ** 2
    mask[normalized <= 1.0] = value


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
    detections = synthetic_face(base.shape)
    overlay = np.zeros((8, 8, 4), dtype=np.uint8)
    overlay[:, :, 1] = 255
    overlay[:, :, 3] = 128

    result = EffectPipeline(
        tracker=StaticTracker(detections),
        segmenter=StaticSegmenter(synthetic_masks(base.shape, detections)),
    ).apply(
        base,
        EffectSettings(skin_smoothing=0.8, brightness=10, contrast=1.0, saturation=1.2),
        FrameAsset(id=1, name="test", image_bgra=overlay),
    )

    assert result.shape == base.shape
    assert result.dtype == np.uint8


def test_fast_effect_pipeline_keeps_source_shape_with_frame_asset() -> None:
    base = np.full((240, 320, 3), 82, dtype=np.uint8)
    detections = synthetic_face((120, 160, 3))
    overlay = np.zeros((8, 8, 4), dtype=np.uint8)
    overlay[:, :, 2] = 255
    overlay[:, :, 3] = 96

    result = EffectPipeline(
        tracker=StaticTracker(detections),
        segmenter=DynamicSegmenter(),
    ).apply(
        base,
        EffectSettings(processing_profile="fast", skin_smoothing=0.7),
        FrameAsset(id=1, name="test", image_bgra=overlay),
    )

    assert result.shape == base.shape
    assert result.dtype == np.uint8
    assert not np.array_equal(result, base)


def test_effect_settings_validation_bounds() -> None:
    with pytest.raises(ValidationError):
        EffectSettings(skin_smoothing=1.1)

    with pytest.raises(ValidationError):
        EffectSettings(brightness=-81)

    with pytest.raises(ValidationError):
        EffectSettings(eye_enlarge=0.7)

    with pytest.raises(ValidationError):
        EffectSettings(debug_overlay="boxes")

    with pytest.raises(ValidationError):
        EffectSettings(processing_profile="turbo")


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


def test_segmenter_masks_include_body_skin_and_hair_near_face() -> None:
    image_shape = (120, 120, 3)
    detections = synthetic_face(image_shape)
    masks = [np.zeros(image_shape[:2], dtype=np.float32) for _ in range(6)]
    masks[3][28:96, 44:76] = 0.95
    masks[2][48:102, 22:42] = 0.9
    masks[1][14:38, 36:84] = 0.88

    result = masks_from_segmenter_result(FakeSegmenterResult(masks), image_shape, detections)

    assert result.skin[70, 32] > 0.25
    assert result.face_skin[60, 60] > 0.8
    assert result.hair[25, 60] > 0.25
    assert result.head[25, 60] >= result.hair[25, 60]


def test_protected_mask_covers_eyes_lips_and_brows() -> None:
    detections = synthetic_face((160, 120, 3))
    face = detections.faces[0]
    parts = build_part_masks((160, 120, 3), detections)

    for box in (face.left_eye, face.right_eye, face.lips):
        assert parts.protected[box.y + box.height // 2, box.x + box.width // 2] > 0.5
    brow_center = tuple(map(int, np.mean(face.left_brow, axis=0)))
    assert parts.protected[brow_center[1], brow_center[0]] > 0.1


def test_part_masks_for_nose_cheeks_forehead_and_chin() -> None:
    detections = synthetic_face((160, 120, 3))
    parts = build_part_masks((160, 120, 3), detections)
    face = detections.faces[0]

    assert parts.nose.shape == (160, 120)
    assert parts.cheeks.dtype == np.float32
    assert parts.nose[int(face.bbox.y + face.bbox.height * 0.55), int(face.bbox.center[0])] > 0.1
    assert parts.cheeks[int(face.bbox.y + face.bbox.height * 0.64), int(face.bbox.x + face.bbox.width * 0.36)] > 0.1
    assert np.max(parts.forehead) > 0.1
    assert np.max(parts.chin) > 0.1


def test_cached_face_tracker_skips_intermediate_frames() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    tracker = CountingTracker(synthetic_face(frame.shape))
    cached = CachedFaceTracker(tracker, detect_every_n_frames=2)

    for _ in range(5):
        assert cached.detect(frame).faces

    assert tracker.calls == 3


def test_head_roi_uses_face_and_mask_bounds() -> None:
    image_shape = (160, 120, 3)
    detections = synthetic_face(image_shape)
    masks = synthetic_masks(image_shape, detections)

    roi = head_roi(image_shape, detections, masks)

    assert roi is not None
    y1, y2, x1, x2 = roi
    assert 0 <= x1 < x2 <= image_shape[1]
    assert 0 <= y1 < y2 <= image_shape[0]


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


def test_debug_overlay_modes_change_frame() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = synthetic_face(frame.shape)
    masks = synthetic_masks(frame.shape, detections)

    for mode in ("landmarks", "masks", "parts", "all"):
        debugged = draw_debug_overlay(frame, detections, masks, mode)
        assert debugged.shape == frame.shape
        assert not np.array_equal(debugged, frame)


def test_debug_overlay_reports_zero_faces() -> None:
    frame = np.zeros((80, 120, 3), dtype=np.uint8)

    debugged = draw_debug_overlay(frame, FaceDetections(faces=()), empty_masks(frame.shape), "landmarks")

    assert not np.array_equal(debugged, frame)
