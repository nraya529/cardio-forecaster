import numpy as np
import pytest

from app.data.preprocessing import ChannelStats
from app.data.synthetic import generate_episode, streaming_window


def test_generate_episode_shape():
    signals, labels, patient = generate_episode(n_minutes=60, sampling_rate_hz=1.0, seed=0)
    assert signals.shape == (60 * 60, 6)
    assert labels.shape == (60 * 60,)
    assert patient.age >= 18


def test_generate_episode_deterioration_forced():
    signals, labels, _ = generate_episode(n_minutes=60, deterioration=True, seed=1)
    assert labels.sum() > 0
    assert signals.dtype == np.float32


def test_streaming_window_dimensions():
    signals, _, _ = generate_episode(n_minutes=20, seed=2)
    x, y = streaming_window(signals, history=360, horizon=120, stride=60)
    assert x.shape[1:] == (360, 6)
    assert y.shape[1:] == (120, 6)
    assert x.shape[0] == y.shape[0]


def test_streaming_window_rejects_short_episode():
    signals = np.zeros((100, 6), dtype=np.float32)
    with pytest.raises(ValueError):
        streaming_window(signals, history=360, horizon=120)


def test_channel_stats_roundtrip():
    signals, _, _ = generate_episode(n_minutes=30, seed=3)
    stats = ChannelStats.from_episodes([signals])
    normalized = stats.normalize(signals)
    reconstructed = stats.denormalize(normalized)
    np.testing.assert_allclose(reconstructed, signals, atol=1e-4)
