import time

import cv2
import numpy as np

from purikura_test.api_models import EffectSettings
from purikura_test.repository import CaptureRepository
from purikura_test.runtime import FramePacket, PurikuraRuntime, max_publish_age_seconds, max_publish_lag_frames


class RecordingPipeline:
    last_motion_ratio = 0.0

    def __init__(self) -> None:
        self.settings: list[EffectSettings] = []

    def apply(self, frame_bgr: np.ndarray, settings: EffectSettings, frame_asset=None) -> np.ndarray:
        self.settings.append(settings)
        return frame_bgr.copy()


def test_max_publish_lag_frames_prefers_fresh_fast_preview() -> None:
    assert max_publish_lag_frames(EffectSettings(processing_profile="fast")) == 6
    assert max_publish_lag_frames(EffectSettings(processing_profile="quality")) == 10
    assert max_publish_age_seconds(EffectSettings(processing_profile="fast")) == 0.45
    assert max_publish_age_seconds(EffectSettings(processing_profile="quality")) == 0.9


def test_runtime_publish_freshness_uses_lag_and_age() -> None:
    runtime = PurikuraRuntime(CaptureRepository("sqlite:///:memory:"))
    runtime._latest_processed = np.zeros((2, 2, 3), dtype=np.uint8)
    runtime._last_publish_at = 99.8
    runtime._latest_raw_frame_id = 20

    fresh_by_age = FramePacket(id=1, captured_at=99.7, frame=np.zeros((2, 2, 3), dtype=np.uint8))
    assert runtime._is_packet_fresh_for_publish(
        fresh_by_age,
        EffectSettings(processing_profile="fast"),
        now=100.0,
    )

    stale = FramePacket(id=1, captured_at=99.0, frame=np.zeros((2, 2, 3), dtype=np.uint8))
    assert not runtime._is_packet_fresh_for_publish(stale, EffectSettings(processing_profile="fast"), now=100.0)
    assert not runtime._is_packet_fresh_for_publish(stale, EffectSettings(processing_profile="quality"), now=100.0)


def test_runtime_allows_publish_after_preview_stall() -> None:
    runtime = PurikuraRuntime(CaptureRepository("sqlite:///:memory:"))
    runtime._latest_processed = np.zeros((2, 2, 3), dtype=np.uint8)
    runtime._last_publish_at = 99.0
    runtime._latest_raw_frame_id = 20
    packet = FramePacket(id=1, captured_at=99.0, frame=np.zeros((2, 2, 3), dtype=np.uint8))

    assert runtime._is_packet_fresh_for_publish(packet, EffectSettings(processing_profile="fast"), now=100.0)


def test_capture_current_reprocesses_latest_raw_with_quality_profile() -> None:
    pipeline = RecordingPipeline()
    runtime = PurikuraRuntime(CaptureRepository("sqlite:///:memory:"), pipeline=pipeline)
    runtime.settings = EffectSettings(processing_profile="fast", purikura_intensity=0.9)
    runtime._latest_raw_packet = FramePacket(
        id=7,
        captured_at=time.perf_counter(),
        frame=np.full((12, 16, 3), 120, dtype=np.uint8),
    )

    encoded = runtime.capture_current()

    assert encoded is not None
    assert encoded.width == 16
    assert encoded.height == 12
    assert encoded.mime == "image/jpeg"
    assert encoded.settings.processing_profile == "quality"
    assert runtime.settings.processing_profile == "fast"
    assert pipeline.settings[-1].processing_profile == "quality"
    decoded = cv2.imdecode(np.frombuffer(encoded.blob, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (12, 16, 3)
