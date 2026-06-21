import numpy as np
import pytest
from pydantic import ValidationError

from purikura_test.api_models import EffectSettings
from purikura_test.effects import EffectPipeline, FrameAsset, alpha_composite_bgra


def test_alpha_composite_resizes_and_applies_alpha() -> None:
    base = np.zeros((4, 4, 3), dtype=np.uint8)
    overlay = np.zeros((2, 2, 4), dtype=np.uint8)
    overlay[:, :] = [0, 0, 255, 255]

    result = alpha_composite_bgra(base, overlay)

    assert result.shape == base.shape
    assert np.all(result[:, :, 2] == 255)
    assert np.all(result[:, :, :2] == 0)


def test_effect_pipeline_keeps_shape_with_frame_asset() -> None:
    base = np.full((8, 8, 3), 80, dtype=np.uint8)
    overlay = np.zeros((8, 8, 4), dtype=np.uint8)
    overlay[:, :, 1] = 255
    overlay[:, :, 3] = 128

    result = EffectPipeline().apply(
        base,
        EffectSettings(skin_smoothing=0, brightness=10, contrast=1.1, saturation=1.2),
        FrameAsset(id=1, name="test", image_bgra=overlay),
    )

    assert result.shape == base.shape
    assert result.dtype == np.uint8


def test_effect_settings_validation_bounds() -> None:
    with pytest.raises(ValidationError):
        EffectSettings(skin_smoothing=1.1)

    with pytest.raises(ValidationError):
        EffectSettings(brightness=-81)
