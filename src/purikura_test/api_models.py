from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CameraInfo(BaseModel):
    id: int
    name: str
    available: bool


class CameraSelection(BaseModel):
    camera_id: int = Field(ge=0)


class EffectSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    processing_profile: Literal["quality", "fast"] = "fast"
    skin_smoothing: float = Field(default=0.78, ge=0.0, le=1.0)
    purikura_intensity: float = Field(default=0.86, ge=0.0, le=1.0)
    eye_enlarge: float = Field(default=0.30, ge=0.0, le=0.6)
    face_slim: float = Field(default=0.30, ge=0.0, le=0.6)
    doll_intensity: float = Field(default=0.65, ge=0.0, le=1.0)
    background_high_key: float = Field(default=0.35, ge=0.0, le=1.0)
    debug_overlay: Literal["off", "landmarks", "masks", "parts", "all"] = "off"

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @property
    def skin_whitening(self) -> float:
        return self._clamp01(0.50 + 0.20 * self.purikura_intensity + 0.14 * self.doll_intensity)

    @property
    def eye_sparkle(self) -> float:
        return self._clamp01(0.30 + 0.62 * self.doll_intensity)

    @property
    def lip_tint(self) -> float:
        return self._clamp01(0.22 + 0.48 * self.doll_intensity)

    @property
    def blush(self) -> float:
        return self._clamp01(0.24 + 0.50 * self.doll_intensity)

    @property
    def brightness(self) -> int:
        return int(round(2 + 7 * self.purikura_intensity))

    @property
    def contrast(self) -> float:
        return max(0.78, 1.0 - 0.07 * self.purikura_intensity)

    @property
    def saturation(self) -> float:
        return 1.0 + 0.09 * self.purikura_intensity

    @property
    def porcelain_skin(self) -> float:
        return self._clamp01(0.28 + 0.70 * self.doll_intensity)

    @property
    def eye_roundness(self) -> float:
        return self._clamp01(0.15 + 0.62 * self.doll_intensity)

    @property
    def eye_liner(self) -> float:
        return self._clamp01(0.16 + 0.60 * self.doll_intensity)

    @property
    def lash_emphasis(self) -> float:
        return self._clamp01(0.12 + 0.51 * self.doll_intensity)

    @property
    def lower_eyelid(self) -> float:
        return self._clamp01(0.14 + 0.55 * self.doll_intensity)

    @property
    def iris_gloss(self) -> float:
        return self._clamp01(0.18 + 0.65 * self.doll_intensity)

    @property
    def cheek_gradient(self) -> float:
        return self._clamp01(0.15 + 0.54 * self.doll_intensity)

    @property
    def lip_gloss(self) -> float:
        return self._clamp01(0.17 + 0.58 * self.doll_intensity)

    @property
    def hair_silk(self) -> float:
        return self._clamp01(0.08 + 0.42 * self.doll_intensity)

    @property
    def soft_glow(self) -> float:
        return self._clamp01(0.12 + 0.51 * self.doll_intensity)


class FrameSummary(BaseModel):
    id: int
    name: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CurrentFrameSelection(BaseModel):
    frame_id: int | None = Field(default=None, ge=1)


class CaptureSummary(BaseModel):
    id: int
    created_at: datetime
    camera_id: int
    frame_id: int | None
    width: int
    height: int

    model_config = ConfigDict(from_attributes=True)


class CaptureCreated(BaseModel):
    id: int


class PerformanceSummary(BaseModel):
    processing_ms: float = 0.0
    encode_ms: float = 0.0
    effective_fps: float = 0.0
    dropped_frames: int = 0
    discarded_processed_frames: int = 0
    frame_age_ms: float = 0.0
    motion_factor: float = 0.0
    published_frame_id: int = 0
    latest_raw_frame_id: int = 0
    profile: Literal["quality", "fast"] = "quality"
