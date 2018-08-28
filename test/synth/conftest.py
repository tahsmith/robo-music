import pytest

from synth.model import ModelParams


@pytest.fixture
def params():
    params = ModelParams(
        channels=1,
        dilation_stack_depth=2,
        dilation_stack_count=2,
        residual_filters=8,
        conv_filters=8,
        skip_filters=16,
        quantisation=256,
        regularisation=False,
        dropout=False,
        conditioning=False,
        sample_rate=11025,
        feature_window=2048,
        n_mels=128
    )

    return params
