import time

import numpy as np

from purikura_test.api_models import EffectSettings
from purikura_test.repository import CaptureRepository
from purikura_test.runtime import FramePacket, PurikuraRuntime, max_publish_lag_frames


def test_max_publish_lag_frames_prefers_fresh_fast_preview() -> None:
    assert max_publish_lag_frames(EffectSettings(processing_profile="fast")) == 1
    assert max_publish_lag_frames(EffectSettings(processing_profile="quality")) == 2


def test_runtime_rejects_stale_processed_packet() -> None:
    runtime = PurikuraRuntime(CaptureRepository("sqlite:///:memory:"))
    packet = FramePacket(id=1, captured_at=time.perf_counter(), frame=np.zeros((2, 2, 3), dtype=np.uint8))

    runtime._latest_raw_frame_id = 2
    assert runtime._is_packet_fresh_for_publish(packet, EffectSettings(processing_profile="fast"))

    runtime._latest_raw_frame_id = 3
    assert not runtime._is_packet_fresh_for_publish(packet, EffectSettings(processing_profile="fast"))
    assert runtime._is_packet_fresh_for_publish(packet, EffectSettings(processing_profile="quality"))
