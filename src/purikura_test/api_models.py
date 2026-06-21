from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class CameraInfo(BaseModel):
    id: int
    name: str
    available: bool


class CameraSelection(BaseModel):
    camera_id: int = Field(ge=0)


class EffectSettings(BaseModel):
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
    face_debug_boxes: bool = False


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
