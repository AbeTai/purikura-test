from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from purikura_test.api_models import EffectSettings


@dataclass(frozen=True)
class FrameAsset:
    id: int
    name: str
    image_bgra: np.ndarray


class EffectPipeline:
    """Applies camera effects in a deterministic, testable order."""

    def apply(
        self,
        frame_bgr: np.ndarray,
        settings: EffectSettings,
        frame_asset: FrameAsset | None = None,
    ) -> np.ndarray:
        adjusted = self._apply_skin_smoothing(frame_bgr, settings.skin_smoothing)
        adjusted = self._apply_color_controls(
            adjusted,
            brightness=settings.brightness,
            contrast=settings.contrast,
            saturation=settings.saturation,
        )
        if frame_asset is None:
            return adjusted
        return alpha_composite_bgra(adjusted, frame_asset.image_bgra)

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
