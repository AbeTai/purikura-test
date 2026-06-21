from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from purikura_test.api_models import EffectSettings


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


@dataclass(frozen=True)
class FaceDetections:
    faces: tuple[DetectionBox, ...]
    eyes: tuple[DetectionBox, ...]


class FaceDetector:
    def __init__(
        self,
        face_cascade_path: str | Path | None = None,
        eye_cascade_path: str | Path | None = None,
    ) -> None:
        cascade_dir = Path(cv2.data.haarcascades)
        self._face_cascade = cv2.CascadeClassifier(str(face_cascade_path or cascade_dir / "haarcascade_frontalface_default.xml"))
        self._eye_cascade = cv2.CascadeClassifier(str(eye_cascade_path or cascade_dir / "haarcascade_eye.xml"))
        if self._face_cascade.empty():
            raise RuntimeError("OpenCV face cascade could not be loaded")
        if self._eye_cascade.empty():
            raise RuntimeError("OpenCV eye cascade could not be loaded")

    def detect(self, frame_bgr: np.ndarray) -> FaceDetections:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        min_face = max(40, min(frame_bgr.shape[:2]) // 8)
        raw_faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(min_face, min_face),
        )
        faces = tuple(DetectionBox(int(x), int(y), int(w), int(h)).clipped(frame_bgr.shape) for x, y, w, h in raw_faces)
        eyes: list[DetectionBox] = []
        for face in faces:
            roi = gray[face.y : face.y + face.height, face.x : face.x + face.width]
            if roi.size == 0:
                continue
            raw_eyes = self._eye_cascade.detectMultiScale(
                roi,
                scaleFactor=1.1,
                minNeighbors=7,
                minSize=(max(12, face.width // 9), max(12, face.height // 9)),
            )
            for ex, ey, ew, eh in raw_eyes:
                if ey > face.height * 0.62:
                    continue
                eyes.append(
                    DetectionBox(
                        x=face.x + int(ex),
                        y=face.y + int(ey),
                        width=int(ew),
                        height=int(eh),
                    ).clipped(frame_bgr.shape)
                )
        return FaceDetections(faces=faces, eyes=tuple(eyes))


class EffectPipeline:
    """Applies camera effects in a deterministic, testable order."""

    def __init__(self, detector: FaceDetector | None = None) -> None:
        self.detector = detector or FaceDetector()

    def apply(
        self,
        frame_bgr: np.ndarray,
        settings: EffectSettings,
        frame_asset: FrameAsset | None = None,
    ) -> np.ndarray:
        detections = self.detector.detect(frame_bgr)
        adjusted = self._apply_purikura_skin(frame_bgr, settings, detections.faces)
        adjusted = self._enlarge_eyes(adjusted, detections.eyes, settings.eye_enlarge)
        adjusted = self._apply_color_controls(
            adjusted,
            brightness=settings.brightness + int(18 * settings.purikura_intensity),
            contrast=settings.contrast + 0.08 * settings.purikura_intensity,
            saturation=settings.saturation + 0.18 * settings.purikura_intensity,
        )
        adjusted = self._apply_soft_pink_tone(adjusted, settings.purikura_intensity)
        if frame_asset is None:
            framed = adjusted
        else:
            framed = alpha_composite_bgra(adjusted, frame_asset.image_bgra)
        if settings.face_debug_boxes:
            return draw_detection_debug(framed, detections)
        return framed

    @staticmethod
    def _apply_purikura_skin(
        frame_bgr: np.ndarray,
        settings: EffectSettings,
        faces: tuple[DetectionBox, ...],
    ) -> np.ndarray:
        if not faces or (settings.skin_smoothing <= 0 and settings.purikura_intensity <= 0):
            return frame_bgr.copy()

        intensity = settings.purikura_intensity
        diameter = 7 + int((settings.skin_smoothing + intensity) * 8)
        if diameter % 2 == 0:
            diameter += 1
        smoothed = cv2.bilateralFilter(frame_bgr, diameter, 70 + intensity * 55, 70 + intensity * 55)
        brightened = cv2.convertScaleAbs(smoothed, alpha=1.0 + 0.04 * intensity, beta=18 * intensity)
        mask = build_face_skin_mask(frame_bgr, faces)
        blend_strength = np.clip(mask * (0.45 * settings.skin_smoothing + 0.35 * intensity), 0, 1)
        return blend_by_mask(frame_bgr, brightened, blend_strength)

    @staticmethod
    def _apply_skin_smoothing(frame_bgr: np.ndarray, strength: float) -> np.ndarray:
        if strength <= 0:
            return frame_bgr.copy()

        diameter = 5 + int(strength * 10)
        if diameter % 2 == 0:
            diameter += 1

        smoothed = cv2.bilateralFilter(frame_bgr, diameter, 50 + strength * 70, 50 + strength * 70)
        return cv2.addWeighted(smoothed, strength, frame_bgr, 1.0 - strength, 0)

    @staticmethod
    def _apply_color_controls(
        frame_bgr: np.ndarray,
        *,
        brightness: int,
        contrast: float,
        saturation: float,
    ) -> np.ndarray:
        adjusted = cv2.convertScaleAbs(frame_bgr, alpha=contrast, beta=brightness)
        if saturation == 1.0:
            return adjusted

        hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    @staticmethod
    def _apply_soft_pink_tone(frame_bgr: np.ndarray, intensity: float) -> np.ndarray:
        if intensity <= 0:
            return frame_bgr
        tint = np.full_like(frame_bgr, (205, 215, 255))
        return cv2.addWeighted(frame_bgr, 1.0 - 0.08 * intensity, tint, 0.08 * intensity, 0)

    @staticmethod
    def _enlarge_eyes(frame_bgr: np.ndarray, eyes: tuple[DetectionBox, ...], strength: float) -> np.ndarray:
        if strength <= 0 or not eyes:
            return frame_bgr
        result = frame_bgr.copy()
        for eye in eyes[:4]:
            result = local_zoom(result, eye, strength)
        return result


def build_face_skin_mask(frame_bgr: np.ndarray, faces: tuple[DetectionBox, ...]) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 173, 127], dtype=np.uint8))
    face_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    for face in faces:
        expanded = expand_box(face, frame_bgr.shape, scale_x=1.08, scale_y=1.18)
        center = (expanded.x + expanded.width // 2, expanded.y + expanded.height // 2)
        axes = (max(1, expanded.width // 2), max(1, expanded.height // 2))
        cv2.ellipse(face_mask, center, axes, 0, 0, 360, 255, -1)
    combined = cv2.bitwise_and(skin, face_mask)
    combined = cv2.GaussianBlur(combined, (0, 0), sigmaX=9, sigmaY=9)
    combined = cv2.bitwise_and(combined, face_mask)
    return combined.astype(np.float32) / 255.0


def expand_box(box: DetectionBox, image_shape: tuple[int, ...], *, scale_x: float, scale_y: float) -> DetectionBox:
    center_x = box.x + box.width / 2
    center_y = box.y + box.height / 2
    width = box.width * scale_x
    height = box.height * scale_y
    return DetectionBox(
        x=int(center_x - width / 2),
        y=int(center_y - height / 2),
        width=int(width),
        height=int(height),
    ).clipped(image_shape)


def blend_by_mask(base_bgr: np.ndarray, overlay_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        mask = mask[:, :, None]
    blended = base_bgr.astype(np.float32) * (1.0 - mask) + overlay_bgr.astype(np.float32) * mask
    return np.clip(blended, 0, 255).astype(np.uint8)


def local_zoom(frame_bgr: np.ndarray, box: DetectionBox, strength: float) -> np.ndarray:
    expanded = expand_box(box, frame_bgr.shape, scale_x=1.9, scale_y=1.65)
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
    mask = np.clip(influence[:, :, None] * 0.9, 0, 1)
    result = frame_bgr.copy()
    result[expanded.y : expanded.y + expanded.height, expanded.x : expanded.x + expanded.width] = blend_by_mask(roi, warped, mask)
    return result


def draw_detection_debug(frame_bgr: np.ndarray, detections: FaceDetections) -> np.ndarray:
    result = frame_bgr.copy()
    for face in detections.faces:
        cv2.rectangle(result, (face.x, face.y), (face.x + face.width, face.y + face.height), (0, 255, 255), 2)
        cv2.putText(result, "face", (face.x, max(0, face.y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    for eye in detections.eyes:
        cv2.rectangle(result, (eye.x, eye.y), (eye.x + eye.width, eye.y + eye.height), (255, 160, 0), 2)
        cv2.putText(result, "eye", (eye.x, max(0, eye.y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 160, 0), 1)
    return result


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
