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
DEBUG_LANDMARK_STEP = 12


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
    landmarks: np.ndarray

    @property
    def eyes(self) -> tuple[DetectionBox, DetectionBox]:
        return self.left_eye, self.right_eye


@dataclass(frozen=True)
class FaceDetections:
    faces: tuple[FaceGeometry, ...]


class FaceTracker(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> FaceDetections: ...


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
        faces = tuple(face_geometry_from_normalized_landmarks(face_landmarks, frame_bgr.shape) for face_landmarks in result.face_landmarks)
        if faces:
            self._last_detections = FaceDetections(faces=faces)
            self._last_detection_at = time.monotonic()
            return self._last_detections
        if time.monotonic() - self._last_detection_at <= self._max_cached_seconds:
            return self._last_detections
        return FaceDetections(faces=())


def default_model_path() -> Path:
    return Path(__file__).resolve().parents[2] / "models" / "face_landmarker.task"


class EffectPipeline:
    """Applies purikura effects using face landmarks when available."""

    def __init__(self, tracker: FaceTracker | None = None, detector: FaceTracker | None = None) -> None:
        self.tracker = tracker or detector or MediaPipeFaceTracker()

    def apply(
        self,
        frame_bgr: np.ndarray,
        settings: EffectSettings,
        frame_asset: FrameAsset | None = None,
    ) -> np.ndarray:
        detections = self.tracker.detect(frame_bgr)
        adjusted = frame_bgr.copy()
        adjusted = self._apply_face_slim(adjusted, detections, settings.face_slim * settings.purikura_intensity)
        adjusted = self._enlarge_eyes(adjusted, detections, settings.eye_enlarge * settings.purikura_intensity)
        adjusted = self._apply_purikura_skin(adjusted, settings, detections)
        adjusted = self._apply_makeup(adjusted, settings, detections)
        adjusted = self._apply_purikura_tone(adjusted, settings)
        framed = adjusted if frame_asset is None else alpha_composite_bgra(adjusted, frame_asset.image_bgra)
        if settings.face_debug_boxes:
            return draw_detection_debug(framed, detections)
        return framed

    @staticmethod
    def _apply_purikura_skin(
        frame_bgr: np.ndarray,
        settings: EffectSettings,
        detections: FaceDetections,
    ) -> np.ndarray:
        if not detections.faces:
            return frame_bgr

        intensity = settings.purikura_intensity
        smoothing = settings.skin_smoothing
        whitening = settings.skin_whitening
        diameter = 11 + int(10 * max(smoothing, intensity))
        if diameter % 2 == 0:
            diameter += 1

        smooth = cv2.bilateralFilter(frame_bgr, diameter, 92 + 48 * intensity, 92 + 48 * intensity)
        blurred = cv2.GaussianBlur(smooth, (0, 0), sigmaX=1.6 + 1.8 * smoothing)
        ivory = np.full_like(frame_bgr, (218, 226, 255))
        whitening_target = cv2.addWeighted(blurred, 1.0 - 0.22 * whitening, ivory, 0.22 * whitening, 16 * whitening)

        lab = cv2.cvtColor(whitening_target, cv2.COLOR_BGR2LAB)
        lab_float = lab.astype(np.float32)
        lab_float[:, :, 0] = np.clip(lab_float[:, :, 0] + 17 * whitening * intensity, 0, 255)
        whitening_target = cv2.cvtColor(lab_float.astype(np.uint8), cv2.COLOR_LAB2BGR)

        mask = build_face_skin_mask(frame_bgr, detections)
        strength = np.clip(mask * (0.58 * smoothing + 0.38 * intensity + 0.25 * whitening), 0, 0.95)
        return blend_by_mask(frame_bgr, whitening_target, strength)

    @staticmethod
    def _apply_purikura_tone(frame_bgr: np.ndarray, settings: EffectSettings) -> np.ndarray:
        intensity = settings.purikura_intensity
        contrast = max(0.62, settings.contrast - 0.14 * intensity)
        brightness = settings.brightness + int(38 * intensity + 16 * settings.skin_whitening)
        adjusted = cv2.convertScaleAbs(frame_bgr, alpha=contrast, beta=brightness)

        hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (settings.saturation + 0.16 * intensity), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + 8 * intensity, 0, 255)
        adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        pink_white = np.full_like(adjusted, (224, 226, 255))
        return cv2.addWeighted(adjusted, 1.0 - 0.12 * intensity, pink_white, 0.12 * intensity, 0)

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
        landmarks=points.astype(np.int32),
    )


def face_geometry_from_box(box: DetectionBox, image_shape: tuple[int, ...]) -> FaceGeometry:
    x, y, w, h = box.x, box.y, box.width, box.height
    points = np.zeros((478, 2), dtype=np.float32)
    points[:, :] = [x + w / 2, y + h / 2]
    for index, angle in zip(FACE_OVAL, np.linspace(-np.pi / 2, np.pi * 1.5, len(FACE_OVAL), endpoint=False)):
        points[index] = [x + w / 2 + np.cos(angle) * w * 0.52, y + h / 2 + np.sin(angle) * h * 0.62]
    for index in LEFT_EYE:
        points[index] = [x + w * 0.66, y + h * 0.42]
    for index in RIGHT_EYE:
        points[index] = [x + w * 0.34, y + h * 0.42]
    for index in LIPS:
        points[index] = [x + w * 0.5, y + h * 0.72]
    for index in LEFT_CHEEK:
        points[index] = [x + w * 0.36, y + h * 0.64]
    for index in RIGHT_CHEEK:
        points[index] = [x + w * 0.64, y + h * 0.64]
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


def build_face_skin_mask(frame_bgr: np.ndarray, detections: FaceDetections | tuple[DetectionBox, ...]) -> np.ndarray:
    face_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    exclusion_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    if isinstance(detections, tuple):
        faces = tuple(face_geometry_from_box(face, frame_bgr.shape) for face in detections)
    else:
        faces = detections.faces

    for face in faces:
        cv2.fillConvexPoly(face_mask, face.face_oval.astype(np.int32), 255)
        for part in (face.left_eye, face.right_eye, face.lips):
            expanded = expand_box(part, frame_bgr.shape, scale_x=1.55, scale_y=1.7)
            cv2.ellipse(exclusion_mask, ellipse_from_box(expanded), 255, -1)

    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    broad_skin = cv2.inRange(ycrcb, np.array([0, 125, 65], dtype=np.uint8), np.array([255, 185, 145], dtype=np.uint8))
    base = cv2.bitwise_and(face_mask, cv2.bitwise_or(broad_skin, cv2.convertScaleAbs(face_mask, alpha=0.68)))
    base = cv2.bitwise_and(base, cv2.bitwise_not(exclusion_mask))
    base = cv2.GaussianBlur(base, (0, 0), sigmaX=10, sigmaY=10)
    base = cv2.bitwise_and(base, face_mask)
    return base.astype(np.float32) / 255.0


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


def draw_detection_debug(frame_bgr: np.ndarray, detections: FaceDetections) -> np.ndarray:
    result = frame_bgr.copy()
    draw_debug_label(result, f"faces={len(detections.faces)}")
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
    cv2.rectangle(frame_bgr, (8, 8), (172, 44), (0, 0, 0), -1)
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
