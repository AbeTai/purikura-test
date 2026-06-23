import time
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from purikura_test.api_models import EffectSettings
from purikura_test.effects import (
    DetectionBox,
    EffectPipeline,
    FACE_OVAL,
    HeadSegmenter,
    LEFT_EYE,
    LIPS,
    SEGMENT_MEDIUM_MOTION_RATIO,
    SEGMENT_MODERATE_EVERY_N_FRAMES,
    FaceDetections,
    FrameAsset,
    RIGHT_EYE,
    SegmentationMasks,
    alpha_composite_bgra,
    apply_background_high_key,
    apply_doll_eye_makeup,
    apply_doll_lip_gloss,
    apply_porcelain_skin,
    apply_soft_glow,
    build_face_skin_mask,
    build_part_masks,
    CachedFaceTracker,
    draw_debug_overlay,
    draw_detection_debug,
    empty_masks,
    face_geometry_from_box,
    face_geometry_from_normalized_landmarks,
    attenuate_settings_for_motion,
    detection_motion_ratio,
    detection_motion_delta,
    head_roi,
    local_eye_round,
    local_translate,
    local_zoom,
    masks_from_segmenter_result,
    motion_preview_scale,
    primary_face_motion_reference,
    segment_interval_for_motion,
    translate_masks,
)


class StaticTracker:
    def __init__(self, detections: FaceDetections) -> None:
        self.detections = detections
        self.last_detection_age_ms = 12.5

    def detect(self, frame_bgr: np.ndarray) -> FaceDetections:
        return self.detections


class StaticSegmenter:
    def __init__(self, masks: SegmentationMasks) -> None:
        self.masks = masks
        self.last_mask_age_ms = 45.5

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


class SequenceTracker:
    def __init__(self, detections: list[FaceDetections]) -> None:
        self.detections = detections
        self.calls = 0

    def detect(self, frame_bgr: np.ndarray) -> FaceDetections:
        index = min(self.calls, len(self.detections) - 1)
        self.calls += 1
        return self.detections[index]


class FakeMask:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data

    def numpy_view(self) -> np.ndarray:
        return self.data


class FakeSegmenterResult:
    def __init__(self, masks: list[np.ndarray]) -> None:
        self.confidence_masks = [FakeMask(mask) for mask in masks]
        self.category_mask = None


class FakeMP:
    class ImageFormat:
        SRGB = "srgb"

    class Image:
        def __init__(self, image_format: object, data: np.ndarray) -> None:
            self.image_format = image_format
            self.data = data


class FakeVideoSegmenter:
    def __init__(self, image_shape: tuple[int, int, int]) -> None:
        self.calls = 0
        self.image_shape = image_shape

    def segment_for_video(self, image: object, timestamp_ms: int) -> FakeSegmenterResult:
        self.calls += 1
        height, width = self.image_shape[:2]
        masks = [np.zeros((height, width), dtype=np.float32) for _ in range(6)]
        offset = min(width // 4, self.calls * 3)
        masks[3][height // 4 : height * 3 // 4, offset : min(width, offset + width // 3)] = 0.9
        masks[2][height // 2 : height * 7 // 8, offset : min(width, offset + width // 2)] = 0.8
        masks[1][height // 8 : height // 3, offset : min(width, offset + width // 3)] = 0.7
        masks[0][:] = 1.0 - np.maximum.reduce(masks[1:5])
        return FakeSegmenterResult(masks)


def fake_head_segmenter(
    image_shape: tuple[int, int, int],
    *,
    segment_every_n_frames: int = 8,
    max_reuse_age_ms: float = 10_000.0,
) -> HeadSegmenter:
    segmenter = object.__new__(HeadSegmenter)
    segmenter._mp = FakeMP
    segmenter._segmenter = FakeVideoSegmenter(image_shape)
    segmenter._segment_every_n_frames = segment_every_n_frames
    segmenter._ema_alpha = 0.0
    segmenter._max_reuse_age_ms = max_reuse_age_ms
    segmenter._medium_motion_ratio = SEGMENT_MEDIUM_MOTION_RATIO
    segmenter._moderate_motion_interval = SEGMENT_MODERATE_EVERY_N_FRAMES
    segmenter._frame_index = 0
    segmenter._last_timestamp_ms = 0
    segmenter._last_masks = None
    segmenter._last_detection_center = None
    segmenter._last_detection_width = None
    segmenter._last_segmented_perf = 0.0
    segmenter._last_segment_frame_index = 0
    segmenter._last_mask_age_ms = 0.0
    return segmenter


def synthetic_face(image_shape: tuple[int, int, int] = (160, 120, 3)) -> FaceDetections:
    return FaceDetections(faces=(face_geometry_from_box(DetectionBox(18, 14, 84, 128), image_shape),))


def synthetic_face_at(box: DetectionBox, image_shape: tuple[int, int, int] = (160, 120, 3)) -> FaceDetections:
    return FaceDetections(faces=(face_geometry_from_box(box, image_shape),))


def synthetic_masks(image_shape: tuple[int, int, int], detections: FaceDetections | None = None) -> SegmentationMasks:
    height, width = image_shape[:2]
    face_skin = np.zeros((height, width), dtype=np.float32)
    body_skin = np.zeros((height, width), dtype=np.float32)
    hair = np.zeros((height, width), dtype=np.float32)
    clothes = np.zeros((height, width), dtype=np.float32)
    background = np.ones((height, width), dtype=np.float32)
    cv2_like_ellipse(face_skin, (width // 2, int(height * 0.48)), (width // 3, height // 3), 1.0)
    body_skin[int(height * 0.38) : int(height * 0.82), width // 8 : width - width // 8] = 0.85
    hair[int(height * 0.10) : int(height * 0.32), width // 5 : width - width // 5] = 0.9
    clothes[int(height * 0.74) :, width // 9 : width - width // 9] = 0.8
    background = np.clip(background - np.maximum.reduce((face_skin, body_skin, hair, clothes)), 0.0, 1.0)
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
        background=background,
        clothes=clothes,
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
    assert face.left_iris.shape[1] == 2
    assert face.left_upper_eyelid.shape[1] == 2
    assert face.lip_inner.shape[1] == 2


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
        EffectSettings(skin_smoothing=0.8, purikura_intensity=0.8),
        FrameAsset(id=1, name="test", image_bgra=overlay),
    )

    assert result.shape == base.shape
    assert result.dtype == np.uint8


def test_effect_pipeline_exposes_analysis_age_metrics() -> None:
    base = np.full((160, 120, 3), 80, dtype=np.uint8)
    detections = synthetic_face(base.shape)
    pipeline = EffectPipeline(
        tracker=StaticTracker(detections),
        segmenter=StaticSegmenter(synthetic_masks(base.shape, detections)),
    )

    pipeline.apply(base, EffectSettings())

    assert pipeline.last_landmark_age_ms == 12.5
    assert pipeline.last_mask_age_ms == 45.5


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
        EffectSettings(eye_enlarge=0.7)

    with pytest.raises(ValidationError):
        EffectSettings(debug_overlay="boxes")

    with pytest.raises(ValidationError):
        EffectSettings(processing_profile="turbo")

    with pytest.raises(ValidationError):
        EffectSettings(doll_intensity=1.1)

    with pytest.raises(ValidationError):
        EffectSettings(lip_gloss=0.5)


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
    assert result.background.shape == image_shape[:2]
    assert result.clothes.shape == image_shape[:2]


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


def test_detection_motion_ratio_tracks_face_center_movement() -> None:
    previous = synthetic_face_at(DetectionBox(20, 20, 80, 100))
    moved = synthetic_face_at(DetectionBox(44, 20, 80, 100))
    previous_center, previous_width = primary_face_motion_reference(previous)

    ratio = detection_motion_ratio(moved, previous_center, previous_width)
    delta = detection_motion_delta(moved, previous_center)
    missing_ratio = detection_motion_ratio(FaceDetections(faces=()), previous_center, previous_width)

    assert ratio > 0.20
    assert delta[0] > 20
    assert abs(delta[1]) < 1
    assert missing_ratio == float("inf")


def test_motion_preview_scale_and_settings_attenuation() -> None:
    settings = EffectSettings(
        processing_profile="fast",
        skin_smoothing=0.8,
        purikura_intensity=0.9,
        eye_enlarge=0.4,
        face_slim=0.4,
        doll_intensity=0.9,
        background_high_key=0.8,
    )

    still_scale = motion_preview_scale(0.01)
    moving_scale = motion_preview_scale(0.18)
    attenuated = attenuate_settings_for_motion(settings, moving_scale)

    assert still_scale == 1.0
    assert moving_scale < 0.5
    assert attenuated.doll_intensity < settings.doll_intensity
    assert attenuated.background_high_key < settings.background_high_key
    assert attenuated.eye_enlarge < settings.eye_enlarge


def test_segment_interval_for_motion_adapts_to_face_movement() -> None:
    assert segment_interval_for_motion(0.01, 8) == 8
    assert segment_interval_for_motion(0.05, 8) == 2
    assert segment_interval_for_motion(0.11, 8) == 1
    assert segment_interval_for_motion(float("inf"), 8) == 1


def test_head_segmenter_refreshes_early_for_moderate_motion() -> None:
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    segmenter = fake_head_segmenter(frame.shape, segment_every_n_frames=8)
    first = synthetic_face_at(DetectionBox(10, 10, 40, 60), frame.shape)
    moved = synthetic_face_at(DetectionBox(13, 10, 40, 60), frame.shape)

    segmenter.segment(frame, first)
    assert segmenter._segmenter.calls == 1
    segmenter.segment(frame, first)
    assert segmenter._segmenter.calls == 1
    segmenter.segment(frame, moved)

    assert segmenter._segmenter.calls == 2


def test_head_segmenter_refreshes_when_cached_mask_is_too_old() -> None:
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    segmenter = fake_head_segmenter(frame.shape, segment_every_n_frames=8, max_reuse_age_ms=10.0)
    detections = synthetic_face_at(DetectionBox(10, 10, 40, 60), frame.shape)

    segmenter.segment(frame, detections)
    segmenter._last_segmented_perf = time.perf_counter() - 1.0
    segmenter.segment(frame, detections)

    assert segmenter._segmenter.calls == 2


def test_translate_masks_moves_foreground_and_recomputes_background() -> None:
    image_shape = (80, 90, 3)
    base = np.zeros(image_shape[:2], dtype=np.float32)
    base[30:42, 28:40] = 1.0
    masks = SegmentationMasks(
        head=base.copy(),
        skin=base.copy(),
        hair=np.zeros_like(base),
        protected=np.zeros_like(base),
        face_skin=base.copy(),
        body_skin=np.zeros_like(base),
        background=1.0 - base,
        clothes=np.zeros_like(base),
    )
    shifted = translate_masks(masks, image_shape, (12.0, 0.0))

    assert shifted.skin.shape == masks.skin.shape
    assert shifted.skin.dtype == np.float32
    assert shifted.skin[36, 45] > masks.skin[36, 45]
    assert shifted.skin[36, 33] < masks.skin[36, 33]
    assert shifted.background[36, 33] > shifted.background[36, 45]


def test_fast_pipeline_records_motion_ratio_for_preview() -> None:
    frame = np.full((160, 120, 3), 82, dtype=np.uint8)
    first = synthetic_face_at(DetectionBox(18, 14, 84, 128), frame.shape)
    second = synthetic_face_at(DetectionBox(42, 14, 84, 128), frame.shape)
    pipeline = EffectPipeline(
        tracker=SequenceTracker([first, second]),
        segmenter=DynamicSegmenter(),
    )
    settings = EffectSettings(processing_profile="fast")

    pipeline.apply(frame, settings)
    pipeline.apply(frame, settings)

    assert pipeline.last_motion_ratio > 0.20


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
    rounded = local_eye_round(frame, eye, strength=0.35)
    slimmed = local_translate(frame, (28, 50), (6, 0), radius=20)
    debugged = draw_detection_debug(frame, synthetic_face((80, 80, 3)))

    assert zoomed.shape == frame.shape
    assert rounded.shape == frame.shape
    assert slimmed.shape == frame.shape
    assert debugged.shape == frame.shape
    assert not np.array_equal(debugged, frame)


def test_doll_eye_lip_background_and_glow_effects_preserve_shape() -> None:
    frame = np.full((160, 120, 3), 96, dtype=np.uint8)
    detections = synthetic_face(frame.shape)
    face = detections.faces[0]
    masks = synthetic_masks(frame.shape, detections)
    settings = EffectSettings(
        doll_intensity=1.0,
        background_high_key=1.0,
    )

    eye = apply_doll_eye_makeup(frame, face, settings)
    lip = apply_doll_lip_gloss(frame, face, settings)
    background = apply_background_high_key(frame, settings, masks)
    glow = apply_soft_glow(frame, settings)

    for result in (eye, lip, background, glow):
        assert result.shape == frame.shape
        assert result.dtype == np.uint8
        assert not np.array_equal(result, frame)
    assert background[4, 4].mean() > frame[4, 4].mean()


def test_porcelain_skin_respects_protected_parts_more_than_cheeks() -> None:
    frame = np.full((160, 120, 3), 80, dtype=np.uint8)
    detections = synthetic_face(frame.shape)
    masks = synthetic_masks(frame.shape, detections)
    face = detections.faces[0]
    settings = EffectSettings(doll_intensity=1.0, purikura_intensity=1.0)

    result = apply_porcelain_skin(frame, settings, masks)

    cheek_point = (int(face.bbox.y + face.bbox.height * 0.64), int(face.bbox.x + face.bbox.width * 0.36))
    eye_point = (face.left_eye.y + face.left_eye.height // 2, face.left_eye.x + face.left_eye.width // 2)
    cheek_delta = int(np.abs(result[cheek_point] - frame[cheek_point]).sum())
    eye_delta = int(np.abs(result[eye_point] - frame[eye_point]).sum())
    assert result.shape == frame.shape
    assert cheek_delta > eye_delta


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
