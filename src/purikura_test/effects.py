from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from purikura_test.api_models import EffectSettings


FACE_OVAL = (
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
)
LEFT_EYE = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)
RIGHT_EYE = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
LIPS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
LEFT_CHEEK = (50, 101, 118, 117, 123, 147, 187, 205)
RIGHT_CHEEK = (280, 330, 347, 346, 352, 376, 411, 425)
LEFT_BROW = (276, 283, 282, 295, 285, 336, 296, 334, 293, 300)
RIGHT_BROW = (46, 53, 52, 65, 55, 107, 66, 105, 63, 70)
NOSE = (1, 2, 4, 5, 6, 45, 48, 64, 98, 97, 94, 326, 327, 294, 278, 275, 168, 197)
NOSE_BRIDGE = (6, 168, 197, 195, 5, 4, 1, 2)
FOREHEAD = (10, 67, 109, 338, 297, 151, 9)
CHIN = (152, 148, 176, 149, 150, 136, 172, 58, 288, 397, 365, 379, 378, 400, 377)
DEBUG_LANDMARK_STEP = 12

SEG_BACKGROUND = 0
SEG_HAIR = 1
SEG_BODY_SKIN = 2
SEG_FACE_SKIN = 3
SEG_CLOTHES = 4
SEG_OTHERS = 5
SEGMENT_EVERY_N_FRAMES = 4
MASK_EMA_ALPHA = 0.65
FAST_PROCESS_WIDTH = 640


@dataclass(frozen=True)
class FrameAsset:
    id: int
    name: str
    image_bgra: np.ndarray


@dataclass(frozen=True)
class DetectionBox:
    x: int
    y: int
    width: int
    height: int

    def clipped(self, image_shape: tuple[int, ...]) -> "DetectionBox":
        image_height, image_width = image_shape[:2]
        x1 = max(0, min(self.x, image_width - 1))
        y1 = max(0, min(self.y, image_height - 1))
        x2 = max(x1 + 1, min(self.x + self.width, image_width))
        y2 = max(y1 + 1, min(self.y + self.height, image_height))
        return DetectionBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2


@dataclass(frozen=True)
class FaceGeometry:
    bbox: DetectionBox
    left_eye: DetectionBox
    right_eye: DetectionBox
    lips: DetectionBox
    face_oval: np.ndarray
    left_cheek: np.ndarray
    right_cheek: np.ndarray
    left_brow: np.ndarray
    right_brow: np.ndarray
    nose: np.ndarray
    nose_bridge: np.ndarray
    forehead: np.ndarray
    chin: np.ndarray
    landmarks: np.ndarray

    @property
    def eyes(self) -> tuple[DetectionBox, DetectionBox]:
        return self.left_eye, self.right_eye


@dataclass(frozen=True)
class FaceDetections:
    faces: tuple[FaceGeometry, ...]


@dataclass(frozen=True)
class SegmentationMasks:
    head: np.ndarray
    skin: np.ndarray
    hair: np.ndarray
    protected: np.ndarray
    face_skin: np.ndarray
    body_skin: np.ndarray


@dataclass(frozen=True)
class PartMasks:
    cheeks: np.ndarray
    nose: np.ndarray
    nose_bridge: np.ndarray
    forehead: np.ndarray
    chin: np.ndarray
    brows: np.ndarray
    protected: np.ndarray


@dataclass(frozen=True)
class FrameAnalysis:
    detections: FaceDetections
    masks: SegmentationMasks


class FaceTracker(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> FaceDetections: ...


class HeadSegmenterProtocol(Protocol):
    def segment(self, frame_bgr: np.ndarray, detections: FaceDetections) -> SegmentationMasks: ...


class MediaPipeFaceTracker:
    def __init__(self, model_path: str | Path | None = None, max_cached_seconds: float = 0.25) -> None:
        resolved_model_path = Path(
            model_path
            or os.getenv("PURIKURA_FACE_LANDMARKER_MODEL", "")
            or default_model_path()
        )
        if not resolved_model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe face landmarker model not found: {resolved_model_path}. "
                "Run `uv run python scripts/download_models.py`."
            )

        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(resolved_model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=4,
            min_face_detection_confidence=0.32,
            min_face_presence_confidence=0.32,
            min_tracking_confidence=0.32,
        )
        self._mp = mp
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_timestamp_ms = 0
        self._last_detection_at = 0.0
        self._last_detections = FaceDetections(faces=())
        self._max_cached_seconds = max_cached_seconds

    def detect(self, frame_bgr: np.ndarray) -> FaceDetections:
        timestamp_ms = max(int(time.monotonic() * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        faces = tuple(
            face_geometry_from_normalized_landmarks(face_landmarks, frame_bgr.shape)
            for face_landmarks in result.face_landmarks
        )
        if faces:
            self._last_detections = FaceDetections(faces=faces)
            self._last_detection_at = time.monotonic()
            return self._last_detections
        if time.monotonic() - self._last_detection_at <= self._max_cached_seconds:
            return self._last_detections
        return FaceDetections(faces=())


class HeadSegmenter:
    """MediaPipe SelfieMulticlass segmenter. Missing models are fatal by design."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        segment_every_n_frames: int = SEGMENT_EVERY_N_FRAMES,
        ema_alpha: float = MASK_EMA_ALPHA,
    ) -> None:
        resolved_model_path = Path(
            model_path
            or os.getenv("PURIKURA_SELFIE_MULTICLASS_MODEL", "")
            or default_segmenter_model_path()
        )
        if not resolved_model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe selfie multiclass model not found: {resolved_model_path}. "
                "Run `uv run python scripts/download_models.py`."
            )

        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = vision.ImageSegmenterOptions(
            base_options=python.BaseOptions(model_asset_path=str(resolved_model_path)),
            running_mode=vision.RunningMode.VIDEO,
            output_category_mask=True,
            output_confidence_masks=True,
        )
        self._mp = mp
        self._segmenter = vision.ImageSegmenter.create_from_options(options)
        self._segment_every_n_frames = max(1, segment_every_n_frames)
        self._ema_alpha = float(np.clip(ema_alpha, 0.0, 1.0))
        self._frame_index = 0
        self._last_timestamp_ms = 0
        self._last_masks: SegmentationMasks | None = None

    def segment(self, frame_bgr: np.ndarray, detections: FaceDetections) -> SegmentationMasks:
        self._frame_index += 1
        if self._last_masks is not None and (self._frame_index - 1) % self._segment_every_n_frames != 0:
            return self._with_current_protection(self._last_masks, frame_bgr.shape, detections)

        timestamp_ms = max(int(time.monotonic() * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._segmenter.segment_for_video(mp_image, timestamp_ms)
        new_masks = masks_from_segmenter_result(result, frame_bgr.shape, detections)
        if self._last_masks is not None:
            new_masks = ema_masks(self._last_masks, new_masks, self._ema_alpha)
            new_masks = self._with_current_protection(new_masks, frame_bgr.shape, detections)
        self._last_masks = new_masks
        return new_masks

    @staticmethod
    def _with_current_protection(
        masks: SegmentationMasks,
        image_shape: tuple[int, ...],
        detections: FaceDetections,
    ) -> SegmentationMasks:
        protected = build_part_masks(image_shape, detections).protected
        return SegmentationMasks(
            head=masks.head,
            skin=masks.skin,
            hair=masks.hair,
            protected=protected,
            face_skin=masks.face_skin,
            body_skin=masks.body_skin,
        )


def default_model_path() -> Path:
    return Path(__file__).resolve().parents[2] / "models" / "face_landmarker.task"


def default_segmenter_model_path() -> Path:
    return Path(__file__).resolve().parents[2] / "models" / "selfie_multiclass_256x256.tflite"


class QualityEffectPipeline:
    """Applies purikura effects using MediaPipe face landmarks and multiclass masks."""

    def __init__(
        self,
        tracker: FaceTracker | None = None,
        detector: FaceTracker | None = None,
        segmenter: HeadSegmenterProtocol | None = None,
    ) -> None:
        self.tracker = tracker or detector or MediaPipeFaceTracker()
        self.segmenter = segmenter or HeadSegmenter()

    def apply(
        self,
        frame_bgr: np.ndarray,
        settings: EffectSettings,
        frame_asset: FrameAsset | None = None,
    ) -> np.ndarray:
        detections = self.tracker.detect(frame_bgr)
        masks = self.segmenter.segment(frame_bgr, detections)
        adjusted = frame_bgr.copy()
        adjusted = self._apply_face_slim(adjusted, detections, settings.face_slim * settings.purikura_intensity)
        adjusted = self._enlarge_eyes(adjusted, detections, settings.eye_enlarge * settings.purikura_intensity)
        adjusted = self._apply_purikura_skin(adjusted, settings, detections, masks)
        adjusted = self._apply_part_refinements(adjusted, settings, detections, masks)
        adjusted = self._apply_makeup(adjusted, settings, detections)
        adjusted = self._apply_purikura_tone(adjusted, settings)
        framed = adjusted if frame_asset is None else alpha_composite_bgra(adjusted, frame_asset.image_bgra)
        return draw_debug_overlay(framed, detections, masks, settings.debug_overlay)

    @staticmethod
    def _apply_purikura_skin(
        frame_bgr: np.ndarray,
        settings: EffectSettings,
        detections: FaceDetections,
        masks: SegmentationMasks,
    ) -> np.ndarray:
        intensity = settings.purikura_intensity
        smoothing = settings.skin_smoothing
        whitening = settings.skin_whitening

        skin_mask = np.clip(masks.skin * (1.0 - masks.protected), 0.0, 1.0)
        hair_mask = np.clip(masks.hair * (1.0 - masks.skin) * face_proximity_mask(frame_bgr.shape, detections), 0.0, 1.0)
        if float(np.max(skin_mask)) <= 0.01 and float(np.max(hair_mask)) <= 0.01:
            return frame_bgr

        diameter = 15 + int(14 * max(smoothing, intensity))
        if diameter % 2 == 0:
            diameter += 1
        smooth = cv2.bilateralFilter(frame_bgr, diameter, 120 + 56 * intensity, 120 + 56 * intensity)
        blurred = cv2.GaussianBlur(smooth, (0, 0), sigmaX=2.0 + 2.4 * smoothing, sigmaY=2.0 + 2.4 * smoothing)

        ivory = np.full_like(frame_bgr, (224, 230, 255))
        pink_white = np.full_like(frame_bgr, (218, 222, 255))
        whitening_target = cv2.addWeighted(blurred, 1.0 - 0.30 * whitening, ivory, 0.30 * whitening, 16 * whitening)
        whitening_target = cv2.addWeighted(whitening_target, 1.0 - 0.08 * intensity, pink_white, 0.08 * intensity, 0)

        lab = cv2.cvtColor(whitening_target, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] = np.clip(lab[:, :, 0] + 18 * whitening * intensity, 0, 255)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + 3.0 * intensity, 0, 255)
        whitening_target = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        skin_strength = np.clip(skin_mask * (0.70 * smoothing + 0.36 * intensity + 0.30 * whitening), 0, 0.96)
        result = blend_by_mask(frame_bgr, whitening_target, skin_strength)

        if float(np.max(hair_mask)) > 0.01:
            hair_target = cv2.addWeighted(result, 0.88, pink_white, 0.12, 4 * intensity)
            hair_strength = np.clip(hair_mask * (0.08 + 0.18 * intensity), 0, 0.28)
            result = blend_by_mask(result, hair_target, hair_strength)
        return result

    @staticmethod
    def _apply_part_refinements(
        frame_bgr: np.ndarray,
        settings: EffectSettings,
        detections: FaceDetections,
        masks: SegmentationMasks,
    ) -> np.ndarray:
        if not detections.faces:
            return frame_bgr

        intensity = settings.purikura_intensity
        part_masks = build_part_masks(frame_bgr.shape, detections)
        skin_available = np.clip(masks.skin * (1.0 - part_masks.protected), 0.0, 1.0)
        result = frame_bgr.copy()

        cheek_mask = np.clip(part_masks.cheeks * skin_available, 0.0, 1.0)
        if float(np.max(cheek_mask)) > 0.01:
            cheek_smooth = cv2.bilateralFilter(result, 17, 120, 120)
            cheek_smooth = cv2.GaussianBlur(cheek_smooth, (0, 0), sigmaX=2.0)
            result = blend_by_mask(result, cheek_smooth, cheek_mask * (0.25 + 0.35 * settings.skin_smoothing))
            warm = np.full_like(result, (172, 146, 255))
            result = blend_by_mask(result, cv2.addWeighted(result, 0.72, warm, 0.28, 0), cheek_mask * settings.blush * 0.30)

        nose_mask = np.clip(part_masks.nose * skin_available, 0.0, 1.0)
        if float(np.max(nose_mask)) > 0.01:
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
            local_l = cv2.GaussianBlur(lab[:, :, 0], (0, 0), sigmaX=6.0)
            lab[:, :, 0] = lab[:, :, 0] * (1.0 - nose_mask * 0.20 * intensity) + local_l * (nose_mask * 0.20 * intensity)
            nose_target = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
            result = blend_by_mask(result, nose_target, nose_mask * 0.65)

        bridge_mask = np.clip(part_masks.nose_bridge * skin_available, 0.0, 1.0)
        if float(np.max(bridge_mask)) > 0.01:
            highlight = cv2.addWeighted(result, 0.80, np.full_like(result, (232, 236, 255)), 0.20, 7 * intensity)
            result = blend_by_mask(result, highlight, bridge_mask * (0.28 + 0.25 * intensity))

        forehead_chin = np.clip((part_masks.forehead + part_masks.chin) * skin_available, 0.0, 1.0)
        if float(np.max(forehead_chin)) > 0.01:
            smooth = cv2.bilateralFilter(result, 15, 100, 100)
            soft = cv2.addWeighted(smooth, 0.84, np.full_like(result, (224, 228, 255)), 0.16, 5)
            result = blend_by_mask(result, soft, forehead_chin * (0.22 + 0.34 * settings.skin_smoothing))

        return result

    @staticmethod
    def _apply_purikura_tone(frame_bgr: np.ndarray, settings: EffectSettings) -> np.ndarray:
        intensity = settings.purikura_intensity
        contrast = max(0.60, settings.contrast - 0.17 * intensity)
        brightness = settings.brightness + int(42 * intensity + 18 * settings.skin_whitening)
        adjusted = cv2.convertScaleAbs(frame_bgr, alpha=contrast, beta=brightness)

        hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (settings.saturation + 0.18 * intensity), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + 9 * intensity, 0, 255)
        adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        pink_white = np.full_like(adjusted, (224, 226, 255))
        return cv2.addWeighted(adjusted, 1.0 - 0.14 * intensity, pink_white, 0.14 * intensity, 0)

    @staticmethod
    def _apply_makeup(frame_bgr: np.ndarray, settings: EffectSettings, detections: FaceDetections) -> np.ndarray:
        if not detections.faces:
            return frame_bgr
        result = frame_bgr.copy()
        for face in detections.faces:
            result = apply_lip_tint(result, face, settings.lip_tint * settings.purikura_intensity)
            result = apply_blush(result, face, settings.blush * settings.purikura_intensity)
            result = apply_eye_sparkle(result, face, settings.eye_sparkle * settings.purikura_intensity)
        return result

    @staticmethod
    def _apply_face_slim(frame_bgr: np.ndarray, detections: FaceDetections, strength: float) -> np.ndarray:
        if strength <= 0 or not detections.faces:
            return frame_bgr
        result = frame_bgr.copy()
        for face in detections.faces:
            cx, _ = face.bbox.center
            left_center = point_center(face.left_cheek)
            right_center = point_center(face.right_cheek)
            jaw_y = face.bbox.y + face.bbox.height * 0.7
            radius = max(face.bbox.width, face.bbox.height) * 0.34
            result = local_translate(result, (left_center[0], max(left_center[1], jaw_y)), (face.bbox.width * 0.075 * strength, 0), radius)
            result = local_translate(result, (right_center[0], max(right_center[1], jaw_y)), (-face.bbox.width * 0.075 * strength, 0), radius)
            result = local_translate(result, (cx, face.bbox.y + face.bbox.height * 0.9), (0, -face.bbox.height * 0.035 * strength), radius * 0.8)
        return result

    @staticmethod
    def _enlarge_eyes(frame_bgr: np.ndarray, detections: FaceDetections, strength: float) -> np.ndarray:
        if strength <= 0 or not detections.faces:
            return frame_bgr
        result = frame_bgr.copy()
        for face in detections.faces:
            for eye in face.eyes:
                result = local_zoom(result, eye, min(0.58, strength * 1.35), scale_x=2.25, scale_y=1.95)
        return result


class CachedFaceTracker:
    def __init__(self, tracker: FaceTracker, detect_every_n_frames: int = 2) -> None:
        self._tracker = tracker
        self._detect_every_n_frames = max(1, detect_every_n_frames)
        self._frame_index = 0
        self._last_detections = FaceDetections(faces=())

    def detect(self, frame_bgr: np.ndarray) -> FaceDetections:
        self._frame_index += 1
        if self._last_detections.faces and (self._frame_index - 1) % self._detect_every_n_frames != 0:
            return self._last_detections
        self._last_detections = self._tracker.detect(frame_bgr)
        return self._last_detections


class FastEffectPipeline:
    """Lower-latency purikura pipeline for live preview."""

    def __init__(
        self,
        tracker: FaceTracker | None = None,
        detector: FaceTracker | None = None,
        segmenter: HeadSegmenterProtocol | None = None,
        *,
        process_width: int = FAST_PROCESS_WIDTH,
    ) -> None:
        base_tracker = tracker or detector or MediaPipeFaceTracker()
        self.tracker = CachedFaceTracker(base_tracker, detect_every_n_frames=2)
        self.segmenter = segmenter or HeadSegmenter(segment_every_n_frames=8, ema_alpha=0.72)
        self.process_width = process_width

    def apply(
        self,
        frame_bgr: np.ndarray,
        settings: EffectSettings,
        frame_asset: FrameAsset | None = None,
    ) -> np.ndarray:
        source_height, source_width = frame_bgr.shape[:2]
        working = resize_for_processing(frame_bgr, self.process_width)
        detections = self.tracker.detect(working)
        masks = self.segmenter.segment(working, detections)

        adjusted = working.copy()
        adjusted = QualityEffectPipeline._apply_face_slim(adjusted, detections, settings.face_slim * settings.purikura_intensity * 0.82)
        adjusted = QualityEffectPipeline._enlarge_eyes(adjusted, detections, settings.eye_enlarge * settings.purikura_intensity * 0.9)
        adjusted = self._apply_fast_beauty(adjusted, settings, detections, masks)
        adjusted = self._apply_fast_tone(adjusted, settings)
        if settings.debug_overlay != "off":
            adjusted = draw_debug_overlay(adjusted, detections, masks, settings.debug_overlay)

        if adjusted.shape[:2] != (source_height, source_width):
            adjusted = cv2.resize(adjusted, (source_width, source_height), interpolation=cv2.INTER_LINEAR)
        return adjusted if frame_asset is None else alpha_composite_bgra(adjusted, frame_asset.image_bgra)

    @staticmethod
    def _apply_fast_beauty(
        frame_bgr: np.ndarray,
        settings: EffectSettings,
        detections: FaceDetections,
        masks: SegmentationMasks,
    ) -> np.ndarray:
        roi = head_roi(frame_bgr.shape, detections, masks)
        if roi is None:
            return frame_bgr

        result = frame_bgr.copy()
        y1, y2, x1, x2 = roi
        base = result[y1:y2, x1:x2]
        if base.size == 0:
            return result

        intensity = settings.purikura_intensity
        skin = np.clip(masks.skin[y1:y2, x1:x2] * (1.0 - masks.protected[y1:y2, x1:x2]), 0.0, 1.0)
        hair = np.clip(masks.hair[y1:y2, x1:x2] * (1.0 - masks.skin[y1:y2, x1:x2]), 0.0, 1.0)
        if float(np.max(skin)) <= 0.01 and float(np.max(hair)) <= 0.01:
            return result

        diameter = 7 + int(6 * max(settings.skin_smoothing, intensity))
        if diameter % 2 == 0:
            diameter += 1
        smooth = cv2.bilateralFilter(base, diameter, 86 + 34 * intensity, 86 + 34 * intensity)
        smooth = cv2.GaussianBlur(smooth, (0, 0), sigmaX=1.2 + settings.skin_smoothing)

        ivory = np.full_like(base, (226, 232, 255))
        pink = np.full_like(base, (208, 212, 255))
        skin_target = cv2.addWeighted(smooth, 1.0 - 0.26 * settings.skin_whitening, ivory, 0.26 * settings.skin_whitening, 12 * settings.skin_whitening)
        skin_target = cv2.addWeighted(skin_target, 1.0 - 0.08 * intensity, pink, 0.08 * intensity, 0)
        skin_strength = np.clip(skin * (0.62 * settings.skin_smoothing + 0.34 * settings.skin_whitening + 0.24 * intensity), 0.0, 0.88)
        blended = blend_by_mask(base, skin_target, skin_strength)

        part_masks = build_part_masks(frame_bgr.shape, detections)
        cheek = np.clip(part_masks.cheeks[y1:y2, x1:x2] * skin, 0.0, 1.0)
        if float(np.max(cheek)) > 0.01:
            blush_target = cv2.addWeighted(blended, 0.76, np.full_like(blended, (166, 140, 255)), 0.24, 0)
            blended = blend_by_mask(blended, blush_target, cheek * settings.blush * 0.30)

        bridge = np.clip(part_masks.nose_bridge[y1:y2, x1:x2] * skin, 0.0, 1.0)
        if float(np.max(bridge)) > 0.01:
            highlight = cv2.addWeighted(blended, 0.84, np.full_like(blended, (235, 238, 255)), 0.16, 4 * intensity)
            blended = blend_by_mask(blended, highlight, bridge * (0.18 + 0.22 * intensity))

        if float(np.max(hair)) > 0.01:
            hair_target = cv2.addWeighted(blended, 0.92, pink, 0.08, 2 * intensity)
            blended = blend_by_mask(blended, hair_target, hair * (0.06 + 0.12 * intensity))

        for face in detections.faces:
            blended_full = result.copy()
            blended_full[y1:y2, x1:x2] = blended
            blended_full = apply_lip_tint(blended_full, face, settings.lip_tint * intensity * 0.9)
            blended_full = apply_eye_sparkle(blended_full, face, settings.eye_sparkle * intensity)
            blended = blended_full[y1:y2, x1:x2]

        result[y1:y2, x1:x2] = blended
        return result

    @staticmethod
    def _apply_fast_tone(frame_bgr: np.ndarray, settings: EffectSettings) -> np.ndarray:
        intensity = settings.purikura_intensity
        contrast = max(0.66, settings.contrast - 0.13 * intensity)
        brightness = settings.brightness + int(34 * intensity + 14 * settings.skin_whitening)
        adjusted = cv2.convertScaleAbs(frame_bgr, alpha=contrast, beta=brightness)
        pink_white = np.full_like(adjusted, (224, 226, 255))
        return cv2.addWeighted(adjusted, 1.0 - 0.11 * intensity, pink_white, 0.11 * intensity, 0)


class EffectPipeline:
    """Dispatches to quality or fast purikura processing profiles."""

    def __init__(
        self,
        tracker: FaceTracker | None = None,
        detector: FaceTracker | None = None,
        segmenter: HeadSegmenterProtocol | None = None,
    ) -> None:
        self._tracker = tracker or detector
        self._segmenter = segmenter
        self._quality_segmenter: HeadSegmenterProtocol | None = segmenter
        self._fast_segmenter: HeadSegmenterProtocol | None = segmenter
        self._quality: QualityEffectPipeline | None = None
        self._fast: FastEffectPipeline | None = None

    def apply(
        self,
        frame_bgr: np.ndarray,
        settings: EffectSettings,
        frame_asset: FrameAsset | None = None,
    ) -> np.ndarray:
        if settings.processing_profile == "fast":
            return self._ensure_fast().apply(frame_bgr, settings, frame_asset)
        return self._ensure_quality().apply(frame_bgr, settings, frame_asset)

    def _ensure_quality(self) -> QualityEffectPipeline:
        if self._quality is None:
            if self._quality_segmenter is None:
                self._quality_segmenter = HeadSegmenter()
            self._quality = QualityEffectPipeline(tracker=self._ensure_tracker(), segmenter=self._quality_segmenter)
        return self._quality

    def _ensure_fast(self) -> FastEffectPipeline:
        if self._fast is None:
            if self._fast_segmenter is None:
                self._fast_segmenter = HeadSegmenter(segment_every_n_frames=8, ema_alpha=0.72)
            self._fast = FastEffectPipeline(tracker=self._ensure_tracker(), segmenter=self._fast_segmenter)
        return self._fast

    def _ensure_tracker(self) -> FaceTracker:
        if self._tracker is None:
            self._tracker = MediaPipeFaceTracker()
        return self._tracker


def resize_for_processing(frame_bgr: np.ndarray, process_width: int) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    if width <= process_width:
        return frame_bgr
    process_height = max(1, int(height * process_width / width))
    return cv2.resize(frame_bgr, (process_width, process_height), interpolation=cv2.INTER_AREA)


def head_roi(
    image_shape: tuple[int, ...],
    detections: FaceDetections,
    masks: SegmentationMasks,
    *,
    padding: int = 12,
) -> tuple[int, int, int, int] | None:
    height, width = image_shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    for face in detections.faces:
        expanded = expand_box(face.bbox, image_shape, scale_x=1.45, scale_y=1.35)
        boxes.append((expanded.x, expanded.y, expanded.x + expanded.width, expanded.y + expanded.height))

    head_uint8 = (masks.head > 0.08).astype(np.uint8)
    if int(cv2.countNonZero(head_uint8)) > 0:
        x, y, w, h = cv2.boundingRect(head_uint8)
        boxes.append((x, y, x + w, y + h))

    if not boxes:
        return None

    x1 = max(0, min(box[0] for box in boxes) - padding)
    y1 = max(0, min(box[1] for box in boxes) - padding)
    x2 = min(width, max(box[2] for box in boxes) + padding)
    y2 = min(height, max(box[3] for box in boxes) + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    return y1, y2, x1, x2


def face_geometry_from_normalized_landmarks(landmarks: object, image_shape: tuple[int, ...]) -> FaceGeometry:
    image_height, image_width = image_shape[:2]
    points = np.array(
        [
            (
                np.clip(float(landmark.x), 0.0, 1.0) * image_width,
                np.clip(float(landmark.y), 0.0, 1.0) * image_height,
            )
            for landmark in landmarks
        ],
        dtype=np.float32,
    )
    return face_geometry_from_points(points, image_shape)


def face_geometry_from_points(points: np.ndarray, image_shape: tuple[int, ...]) -> FaceGeometry:
    face_oval = points_for_indices(points, FACE_OVAL)
    left_eye = box_from_points(points_for_indices(points, LEFT_EYE), image_shape, padding=0.16)
    right_eye = box_from_points(points_for_indices(points, RIGHT_EYE), image_shape, padding=0.16)
    lips = box_from_points(points_for_indices(points, LIPS), image_shape, padding=0.16)
    bbox = box_from_points(face_oval, image_shape, padding=0.03)
    return FaceGeometry(
        bbox=bbox,
        left_eye=left_eye,
        right_eye=right_eye,
        lips=lips,
        face_oval=face_oval.astype(np.int32),
        left_cheek=points_for_indices(points, LEFT_CHEEK).astype(np.int32),
        right_cheek=points_for_indices(points, RIGHT_CHEEK).astype(np.int32),
        left_brow=points_for_indices(points, LEFT_BROW).astype(np.int32),
        right_brow=points_for_indices(points, RIGHT_BROW).astype(np.int32),
        nose=points_for_indices(points, NOSE).astype(np.int32),
        nose_bridge=points_for_indices(points, NOSE_BRIDGE).astype(np.int32),
        forehead=points_for_indices(points, FOREHEAD).astype(np.int32),
        chin=points_for_indices(points, CHIN).astype(np.int32),
        landmarks=points.astype(np.int32),
    )


def face_geometry_from_box(box: DetectionBox, image_shape: tuple[int, ...]) -> FaceGeometry:
    x, y, w, h = box.x, box.y, box.width, box.height
    points = np.zeros((478, 2), dtype=np.float32)
    points[:, :] = [x + w / 2, y + h / 2]
    for index, angle in zip(FACE_OVAL, np.linspace(-np.pi / 2, np.pi * 1.5, len(FACE_OVAL), endpoint=False)):
        points[index] = [x + w / 2 + np.cos(angle) * w * 0.52, y + h / 2 + np.sin(angle) * h * 0.62]
    for index, angle in zip(LEFT_EYE, np.linspace(0, np.pi * 2, len(LEFT_EYE), endpoint=False)):
        points[index] = [x + w * 0.66 + np.cos(angle) * w * 0.09, y + h * 0.42 + np.sin(angle) * h * 0.05]
    for index, angle in zip(RIGHT_EYE, np.linspace(0, np.pi * 2, len(RIGHT_EYE), endpoint=False)):
        points[index] = [x + w * 0.34 + np.cos(angle) * w * 0.09, y + h * 0.42 + np.sin(angle) * h * 0.05]
    for index, angle in zip(LIPS, np.linspace(0, np.pi * 2, len(LIPS), endpoint=False)):
        points[index] = [x + w * 0.5 + np.cos(angle) * w * 0.14, y + h * 0.72 + np.sin(angle) * h * 0.06]
    for index in LEFT_CHEEK:
        points[index] = [x + w * 0.36, y + h * 0.64]
    for index in RIGHT_CHEEK:
        points[index] = [x + w * 0.64, y + h * 0.64]
    for index, offset in zip(LEFT_BROW, np.linspace(-0.12, 0.12, len(LEFT_BROW))):
        points[index] = [x + w * (0.67 + offset), y + h * 0.31]
    for index, offset in zip(RIGHT_BROW, np.linspace(-0.12, 0.12, len(RIGHT_BROW))):
        points[index] = [x + w * (0.33 + offset), y + h * 0.31]
    for index, angle in zip(NOSE, np.linspace(0, np.pi * 2, len(NOSE), endpoint=False)):
        points[index] = [x + w * 0.5 + np.cos(angle) * w * 0.12, y + h * 0.55 + np.sin(angle) * h * 0.14]
    for index, offset in zip(NOSE_BRIDGE, np.linspace(-0.18, 0.18, len(NOSE_BRIDGE))):
        points[index] = [x + w * 0.5, y + h * (0.39 + offset)]
    for index, offset in zip(FOREHEAD, np.linspace(-0.20, 0.20, len(FOREHEAD))):
        points[index] = [x + w * (0.5 + offset), y + h * 0.18]
    for index, offset in zip(CHIN, np.linspace(-0.22, 0.22, len(CHIN))):
        points[index] = [x + w * (0.5 + offset), y + h * 0.86]
    return face_geometry_from_points(points, image_shape)


def points_for_indices(points: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    valid = [index for index in indices if index < len(points)]
    return points[valid]


def box_from_points(points: np.ndarray, image_shape: tuple[int, ...], padding: float = 0.0) -> DetectionBox:
    if points.size == 0:
        return DetectionBox(0, 0, 1, 1)
    min_xy = np.min(points, axis=0)
    max_xy = np.max(points, axis=0)
    width = max(1.0, max_xy[0] - min_xy[0])
    height = max(1.0, max_xy[1] - min_xy[1])
    pad_x = width * padding
    pad_y = height * padding
    return DetectionBox(
        x=int(min_xy[0] - pad_x),
        y=int(min_xy[1] - pad_y),
        width=int(width + pad_x * 2),
        height=int(height + pad_y * 2),
    ).clipped(image_shape)


def empty_masks(image_shape: tuple[int, ...]) -> SegmentationMasks:
    height, width = image_shape[:2]
    zero = np.zeros((height, width), dtype=np.float32)
    return SegmentationMasks(
        head=zero.copy(),
        skin=zero.copy(),
        hair=zero.copy(),
        protected=zero.copy(),
        face_skin=zero.copy(),
        body_skin=zero.copy(),
    )


def masks_from_segmenter_result(
    result: object,
    image_shape: tuple[int, ...],
    detections: FaceDetections,
) -> SegmentationMasks:
    face_skin = confidence_or_category_mask(result, SEG_FACE_SKIN, image_shape)
    body_skin = confidence_or_category_mask(result, SEG_BODY_SKIN, image_shape)
    hair = confidence_or_category_mask(result, SEG_HAIR, image_shape)
    proximity = face_proximity_mask(image_shape, detections)
    if detections.faces:
        body_skin_near = body_skin * proximity
        hair_near = hair * proximity
    else:
        body_skin_near = body_skin
        hair_near = hair

    face_skin = smooth_mask(face_skin, sigma=2.0)
    body_skin_near = smooth_mask(body_skin_near, sigma=2.5)
    hair_near = smooth_mask(hair_near, sigma=2.5)
    protected = build_part_masks(image_shape, detections).protected
    skin = np.clip(np.maximum(face_skin, body_skin_near), 0.0, 1.0)
    head = np.clip(np.maximum(skin, hair_near), 0.0, 1.0)
    head = np.maximum(smooth_mask(head, sigma=1.5), hair_near)
    return SegmentationMasks(
        head=np.clip(head, 0.0, 1.0).astype(np.float32),
        skin=smooth_mask(skin, sigma=1.5),
        hair=hair_near,
        protected=protected,
        face_skin=face_skin,
        body_skin=body_skin_near,
    )


def confidence_or_category_mask(result: object, category: int, image_shape: tuple[int, ...]) -> np.ndarray:
    confidence_masks = getattr(result, "confidence_masks", None)
    if confidence_masks and len(confidence_masks) > category:
        mask = confidence_masks[category].numpy_view()
        return resize_mask(mask, image_shape, interpolation=cv2.INTER_LINEAR)

    category_mask = getattr(result, "category_mask", None)
    if category_mask is None:
        raise RuntimeError("ImageSegmenter did not return category or confidence masks")
    mask = np.squeeze(category_mask.numpy_view())
    return resize_mask((mask == category).astype(np.float32), image_shape, interpolation=cv2.INTER_NEAREST)


def resize_mask(mask: np.ndarray, image_shape: tuple[int, ...], *, interpolation: int) -> np.ndarray:
    height, width = image_shape[:2]
    mask = np.squeeze(mask).astype(np.float32)
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=interpolation)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def ema_masks(previous: SegmentationMasks, current: SegmentationMasks, alpha: float) -> SegmentationMasks:
    def mix(old: np.ndarray, new: np.ndarray) -> np.ndarray:
        return np.clip(old * (1.0 - alpha) + new * alpha, 0.0, 1.0).astype(np.float32)

    return SegmentationMasks(
        head=mix(previous.head, current.head),
        skin=mix(previous.skin, current.skin),
        hair=mix(previous.hair, current.hair),
        protected=current.protected,
        face_skin=mix(previous.face_skin, current.face_skin),
        body_skin=mix(previous.body_skin, current.body_skin),
    )


def build_part_masks(image_shape: tuple[int, ...], detections: FaceDetections) -> PartMasks:
    height, width = image_shape[:2]
    cheeks = np.zeros((height, width), dtype=np.float32)
    nose = np.zeros((height, width), dtype=np.float32)
    nose_bridge = np.zeros((height, width), dtype=np.float32)
    forehead = np.zeros((height, width), dtype=np.float32)
    chin = np.zeros((height, width), dtype=np.float32)
    brows = np.zeros((height, width), dtype=np.float32)
    protected = np.zeros((height, width), dtype=np.float32)

    for face in detections.faces:
        cheeks = np.maximum(cheeks, cheek_masks(image_shape, face))
        nose = np.maximum(nose, polygon_mask(image_shape, face.nose, blur_sigma=3.0))
        nose_bridge = np.maximum(nose_bridge, line_mask(image_shape, face.nose_bridge, max(2, face.bbox.width // 36), blur_sigma=3.0))
        forehead = np.maximum(forehead, forehead_mask(image_shape, face))
        chin = np.maximum(chin, chin_mask(image_shape, face))
        brows = np.maximum(brows, polygon_mask(image_shape, face.left_brow, blur_sigma=2.0))
        brows = np.maximum(brows, polygon_mask(image_shape, face.right_brow, blur_sigma=2.0))

        for box in (face.left_eye, face.right_eye):
            expanded = expand_box(box, image_shape, scale_x=1.75, scale_y=1.95)
            eye = np.zeros((height, width), dtype=np.uint8)
            cv2.ellipse(eye, ellipse_from_box(expanded), 255, -1)
            protected = np.maximum(protected, smooth_mask(eye.astype(np.float32) / 255.0, sigma=3.0))
        lip_box = expand_box(face.lips, image_shape, scale_x=1.36, scale_y=1.52)
        lip = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(lip, ellipse_from_box(lip_box), 255, -1)
        protected = np.maximum(protected, smooth_mask(lip.astype(np.float32) / 255.0, sigma=3.0))

    protected = np.clip(np.maximum(protected, brows), 0.0, 1.0).astype(np.float32)
    return PartMasks(
        cheeks=cheeks,
        nose=nose,
        nose_bridge=nose_bridge,
        forehead=forehead,
        chin=chin,
        brows=brows,
        protected=protected,
    )


def build_face_skin_mask(frame_bgr: np.ndarray, detections: FaceDetections | tuple[DetectionBox, ...]) -> np.ndarray:
    if isinstance(detections, tuple):
        face_detections = FaceDetections(tuple(face_geometry_from_box(face, frame_bgr.shape) for face in detections))
    else:
        face_detections = detections
    face_mask = np.zeros(frame_bgr.shape[:2], dtype=np.float32)
    for face in face_detections.faces:
        face_mask = np.maximum(face_mask, polygon_mask(frame_bgr.shape, face.face_oval, blur_sigma=6.0))
    protected = build_part_masks(frame_bgr.shape, face_detections).protected
    return np.clip(face_mask * (1.0 - protected), 0.0, 1.0)


def face_proximity_mask(image_shape: tuple[int, ...], detections: FaceDetections) -> np.ndarray:
    height, width = image_shape[:2]
    if not detections.faces:
        return np.ones((height, width), dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.uint8)
    for face in detections.faces:
        x1 = int(face.bbox.x - face.bbox.width * 0.45)
        y1 = int(face.bbox.y - face.bbox.height * 0.45)
        x2 = int(face.bbox.x + face.bbox.width * 1.45)
        y2 = int(face.bbox.y + face.bbox.height * 1.32)
        roi = DetectionBox(x1, y1, x2 - x1, y2 - y1).clipped(image_shape)
        cv2.rectangle(mask, (roi.x, roi.y), (roi.x + roi.width, roi.y + roi.height), 255, -1)
        cv2.ellipse(
            mask,
            (int(face.bbox.center[0]), int(face.bbox.y + face.bbox.height * 0.45)),
            (max(1, int(face.bbox.width * 0.86)), max(1, int(face.bbox.height * 0.92))),
            0,
            0,
            360,
            255,
            -1,
        )
    return smooth_mask(mask.astype(np.float32) / 255.0, sigma=8.0)


def polygon_mask(image_shape: tuple[int, ...], points: np.ndarray, *, blur_sigma: float) -> np.ndarray:
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points) >= 3:
        hull = cv2.convexHull(points.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
    elif len(points) == 2:
        cv2.line(mask, tuple(points[0].astype(int)), tuple(points[1].astype(int)), 255, 2)
    elif len(points) == 1:
        cv2.circle(mask, tuple(points[0].astype(int)), 2, 255, -1)
    return smooth_mask(mask.astype(np.float32) / 255.0, sigma=blur_sigma)


def line_mask(image_shape: tuple[int, ...], points: np.ndarray, thickness: int, *, blur_sigma: float) -> np.ndarray:
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points) >= 2:
        cv2.polylines(mask, [points.astype(np.int32)], False, 255, thickness=thickness, lineType=cv2.LINE_AA)
    return smooth_mask(mask.astype(np.float32) / 255.0, sigma=blur_sigma)


def cheek_masks(image_shape: tuple[int, ...], face: FaceGeometry) -> np.ndarray:
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    axes = (max(5, face.bbox.width // 7), max(4, face.bbox.height // 12))
    for cheek in (face.left_cheek, face.right_cheek):
        center = tuple(map(int, point_center(cheek)))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return smooth_mask(mask.astype(np.float32) / 255.0, sigma=6.0)


def forehead_mask(image_shape: tuple[int, ...], face: FaceGeometry) -> np.ndarray:
    height, width = image_shape[:2]
    raw = np.zeros((height, width), dtype=np.uint8)
    center = (int(face.bbox.center[0]), int(face.bbox.y + face.bbox.height * 0.24))
    axes = (max(5, int(face.bbox.width * 0.28)), max(4, int(face.bbox.height * 0.10)))
    cv2.ellipse(raw, center, axes, 0, 0, 360, 255, -1)
    ellipse = smooth_mask(raw.astype(np.float32) / 255.0, sigma=4.0)
    if len(face.forehead) >= 3:
        base = np.maximum(ellipse, polygon_mask(image_shape, face.forehead, blur_sigma=5.0))
    else:
        base = ellipse
    return base


def chin_mask(image_shape: tuple[int, ...], face: FaceGeometry) -> np.ndarray:
    height, width = image_shape[:2]
    raw = np.zeros((height, width), dtype=np.uint8)
    center = (int(face.bbox.center[0]), int(face.bbox.y + face.bbox.height * 0.84))
    axes = (max(5, int(face.bbox.width * 0.26)), max(4, int(face.bbox.height * 0.11)))
    cv2.ellipse(raw, center, axes, 0, 0, 360, 255, -1)
    ellipse = smooth_mask(raw.astype(np.float32) / 255.0, sigma=4.0)
    if len(face.chin) >= 3:
        base = np.maximum(ellipse, polygon_mask(image_shape, face.chin, blur_sigma=5.0))
    else:
        base = ellipse
    return base


def smooth_mask(mask: np.ndarray, *, sigma: float) -> np.ndarray:
    if sigma > 0:
        mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def apply_lip_tint(frame_bgr: np.ndarray, face: FaceGeometry, strength: float) -> np.ndarray:
    if strength <= 0:
        return frame_bgr
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, ellipse_from_box(expand_box(face.lips, frame_bgr.shape, scale_x=1.22, scale_y=1.35)), 255, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=3, sigmaY=3).astype(np.float32) / 255.0
    tint = np.full_like(frame_bgr, (126, 84, 220))
    return blend_by_mask(frame_bgr, cv2.addWeighted(frame_bgr, 0.48, tint, 0.52, 0), mask * min(0.42, strength))


def apply_blush(frame_bgr: np.ndarray, face: FaceGeometry, strength: float) -> np.ndarray:
    if strength <= 0:
        return frame_bgr
    overlay = frame_bgr.copy()
    color = (164, 142, 255)
    for cheek in (face.left_cheek, face.right_cheek):
        center = tuple(map(int, point_center(cheek)))
        axes = (max(6, face.bbox.width // 10), max(5, face.bbox.height // 14))
        cv2.ellipse(overlay, center, axes, 0, 0, 360, color, -1)
    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=5, sigmaY=5)
    return cv2.addWeighted(frame_bgr, 1.0 - min(0.28, strength * 0.42), overlay, min(0.28, strength * 0.42), 0)


def apply_eye_sparkle(frame_bgr: np.ndarray, face: FaceGeometry, strength: float) -> np.ndarray:
    if strength <= 0:
        return frame_bgr
    result = frame_bgr.copy()
    for eye in face.eyes:
        cx, cy = eye.center
        radius = max(2, int(min(eye.width, eye.height) * 0.13))
        highlight = (int(cx - eye.width * 0.12), int(cy - eye.height * 0.17))
        cv2.circle(result, highlight, radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(result, (int(cx + eye.width * 0.16), int(cy + eye.height * 0.1)), max(1, radius // 2), (235, 240, 255), -1, lineType=cv2.LINE_AA)
    return cv2.addWeighted(frame_bgr, 1.0 - min(0.7, strength), result, min(0.7, strength), 0)


def expand_box(box: DetectionBox, image_shape: tuple[int, ...], *, scale_x: float, scale_y: float) -> DetectionBox:
    center_x = box.x + box.width / 2
    center_y = box.y + box.height / 2
    width = max(1.0, box.width * scale_x)
    height = max(1.0, box.height * scale_y)
    return DetectionBox(
        x=int(center_x - width / 2),
        y=int(center_y - height / 2),
        width=int(width),
        height=int(height),
    ).clipped(image_shape)


def ellipse_from_box(box: DetectionBox) -> tuple[tuple[int, int], tuple[int, int], float]:
    return (int(box.x + box.width / 2), int(box.y + box.height / 2)), (max(1, box.width // 2), max(1, box.height // 2)), 0.0


def point_center(points: np.ndarray) -> tuple[float, float]:
    if points.size == 0:
        return 0.0, 0.0
    center = np.mean(points, axis=0)
    return float(center[0]), float(center[1])


def blend_by_mask(base_bgr: np.ndarray, overlay_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        mask = mask[:, :, None]
    blended = base_bgr.astype(np.float32) * (1.0 - mask) + overlay_bgr.astype(np.float32) * mask
    return np.clip(blended, 0, 255).astype(np.uint8)


def local_zoom(
    frame_bgr: np.ndarray,
    box: DetectionBox,
    strength: float,
    *,
    scale_x: float = 1.9,
    scale_y: float = 1.65,
) -> np.ndarray:
    expanded = expand_box(box, frame_bgr.shape, scale_x=scale_x, scale_y=scale_y)
    roi = frame_bgr[expanded.y : expanded.y + expanded.height, expanded.x : expanded.x + expanded.width]
    if roi.size == 0:
        return frame_bgr

    height, width = roi.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    center_x = width / 2
    center_y = height / 2
    radius = max(1.0, min(width, height) / 2)
    dx = grid_x - center_x
    dy = grid_y - center_y
    distance = np.sqrt(dx * dx + dy * dy)
    influence = np.clip(1.0 - distance / radius, 0.0, 1.0) ** 2
    zoom = 1.0 + strength * influence
    map_x = center_x + dx / zoom
    map_y = center_y + dy / zoom
    warped = cv2.remap(roi, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = np.clip(influence[:, :, None] * 0.95, 0, 1)
    result = frame_bgr.copy()
    result[expanded.y : expanded.y + expanded.height, expanded.x : expanded.x + expanded.width] = blend_by_mask(roi, warped, mask)
    return result


def local_translate(
    frame_bgr: np.ndarray,
    center: tuple[float, float],
    shift: tuple[float, float],
    radius: float,
) -> np.ndarray:
    radius = max(1.0, radius)
    x1 = max(0, int(center[0] - radius))
    y1 = max(0, int(center[1] - radius))
    x2 = min(frame_bgr.shape[1], int(center[0] + radius))
    y2 = min(frame_bgr.shape[0], int(center[1] + radius))
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return frame_bgr

    height, width = roi.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    dx = grid_x + x1 - center[0]
    dy = grid_y + y1 - center[1]
    distance = np.sqrt(dx * dx + dy * dy)
    influence = np.clip(1.0 - distance / radius, 0.0, 1.0) ** 2
    map_x = grid_x - shift[0] * influence
    map_y = grid_y - shift[1] * influence
    warped = cv2.remap(roi, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    result = frame_bgr.copy()
    result[y1:y2, x1:x2] = blend_by_mask(roi, warped, influence[:, :, None])
    return result


def draw_debug_overlay(
    frame_bgr: np.ndarray,
    detections: FaceDetections,
    masks: SegmentationMasks,
    mode: str,
) -> np.ndarray:
    if mode == "off":
        return frame_bgr
    result = frame_bgr.copy()
    if mode in {"masks", "all"}:
        result = draw_mask_debug(result, masks)
    if mode in {"parts", "all"}:
        result = draw_part_debug(result, detections)
    if mode in {"landmarks", "all"}:
        result = draw_landmark_debug(result, detections)
    draw_debug_label(result, f"faces={len(detections.faces)} overlay={mode}")
    return result


def draw_detection_debug(frame_bgr: np.ndarray, detections: FaceDetections) -> np.ndarray:
    return draw_debug_overlay(frame_bgr, detections, empty_masks(frame_bgr.shape), "landmarks")


def draw_mask_debug(frame_bgr: np.ndarray, masks: SegmentationMasks) -> np.ndarray:
    result = frame_bgr.copy()
    overlays = (
        (masks.head, (0, 220, 255), 0.25),
        (masks.skin, (0, 255, 80), 0.32),
        (masks.hair, (255, 120, 0), 0.24),
        (masks.protected, (255, 0, 255), 0.45),
    )
    for mask, color, strength in overlays:
        color_frame = np.full_like(result, color)
        result = blend_by_mask(result, color_frame, np.clip(mask * strength, 0.0, 1.0))
    return result


def draw_part_debug(frame_bgr: np.ndarray, detections: FaceDetections) -> np.ndarray:
    result = frame_bgr.copy()
    for face in detections.faces:
        for label, points, color in (
            ("nose", face.nose, (0, 255, 255)),
            ("forehead", face.forehead, (255, 180, 0)),
            ("chin", face.chin, (80, 220, 255)),
            ("L brow", face.left_brow, (255, 0, 255)),
            ("R brow", face.right_brow, (255, 0, 255)),
        ):
            if len(points) >= 2:
                cv2.polylines(result, [cv2.convexHull(points.astype(np.int32))], True, color, 1)
            center = point_center(points)
            cv2.putText(result, label, (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        for label, cheek in (("L cheek", face.left_cheek), ("R cheek", face.right_cheek)):
            center = tuple(map(int, point_center(cheek)))
            cv2.circle(result, center, max(4, face.bbox.width // 11), (120, 120, 255), 1)
            cv2.putText(result, label, center, cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 255), 1)
    return result


def draw_landmark_debug(frame_bgr: np.ndarray, detections: FaceDetections) -> np.ndarray:
    result = frame_bgr.copy()
    for face in detections.faces:
        cv2.rectangle(result, (face.bbox.x, face.bbox.y), (face.bbox.x + face.bbox.width, face.bbox.y + face.bbox.height), (0, 255, 255), 2)
        cv2.polylines(result, [face.face_oval.astype(np.int32)], isClosed=True, color=(0, 220, 255), thickness=2)
        for label, box, color in (
            ("L eye", face.left_eye, (255, 160, 0)),
            ("R eye", face.right_eye, (255, 160, 0)),
            ("lips", face.lips, (180, 80, 255)),
        ):
            cv2.rectangle(result, (box.x, box.y), (box.x + box.width, box.y + box.height), color, 2)
            cv2.putText(result, label, (box.x, max(0, box.y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        for point in face.landmarks[::DEBUG_LANDMARK_STEP]:
            cv2.circle(result, tuple(map(int, point)), 1, (80, 255, 80), -1)
    return result


def draw_debug_label(frame_bgr: np.ndarray, text: str) -> None:
    width = min(frame_bgr.shape[1] - 1, max(172, 20 + len(text) * 14))
    cv2.rectangle(frame_bgr, (8, 8), (width, 44), (0, 0, 0), -1)
    cv2.putText(frame_bgr, text, (16, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 255), 2)


def alpha_composite_bgra(base_bgr: np.ndarray, overlay_bgra: np.ndarray) -> np.ndarray:
    if base_bgr.ndim != 3 or base_bgr.shape[2] != 3:
        raise ValueError("base_bgr must be an HxWx3 BGR image")
    if overlay_bgra.ndim != 3 or overlay_bgra.shape[2] != 4:
        raise ValueError("overlay_bgra must be an HxWx4 BGRA image")

    height, width = base_bgr.shape[:2]
    overlay = cv2.resize(overlay_bgra, (width, height), interpolation=cv2.INTER_AREA)
    overlay_rgb = overlay[:, :, :3].astype(np.float32)
    alpha = (overlay[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
    base = base_bgr.astype(np.float32)
    composed = overlay_rgb * alpha + base * (1.0 - alpha)
    return np.clip(composed, 0, 255).astype(np.uint8)
