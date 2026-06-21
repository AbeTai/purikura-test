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
    processing_profile: Literal["quality", "fast"] = "quality"
    skin_smoothing: float = Field(default=0.78, ge=0.0, le=1.0)
    purikura_intensity: float = Field(default=0.86, ge=0.0, le=1.0)
    skin_whitening: float = Field(default=0.76, ge=0.0, le=1.0)
    eye_enlarge: float = Field(default=0.30, ge=0.0, le=0.6)
    face_slim: float = Field(default=0.30, ge=0.0, le=0.6)
    eye_sparkle: float = Field(default=0.68, ge=0.0, le=1.0)
    lip_tint: float = Field(default=0.34, ge=0.0, le=1.0)
    blush: float = Field(default=0.36, ge=0.0, le=1.0)
    brightness: int = Field(default=8, ge=-80, le=80)
    contrast: float = Field(default=0.94, ge=0.5, le=2.0)
    saturation: float = Field(default=1.08, ge=0.0, le=2.0)
    doll_intensity: float = Field(default=0.65, ge=0.0, le=1.0)
    porcelain_skin: float = Field(default=0.72, ge=0.0, le=1.0)
    eye_roundness: float = Field(default=0.55, ge=0.0, le=1.0)
    eye_liner: float = Field(default=0.55, ge=0.0, le=1.0)
    lash_emphasis: float = Field(default=0.45, ge=0.0, le=1.0)
    lower_eyelid: float = Field(default=0.50, ge=0.0, le=1.0)
    iris_gloss: float = Field(default=0.60, ge=0.0, le=1.0)
    cheek_gradient: float = Field(default=0.50, ge=0.0, le=1.0)
    lip_gloss: float = Field(default=0.55, ge=0.0, le=1.0)
    hair_silk: float = Field(default=0.35, ge=0.0, le=1.0)
    background_high_key: float = Field(default=0.35, ge=0.0, le=1.0)
    soft_glow: float = Field(default=0.45, ge=0.0, le=1.0)
    debug_overlay: Literal["off", "landmarks", "masks", "parts", "all"] = "off"


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
    profile: Literal["quality", "fast"] = "quality"
