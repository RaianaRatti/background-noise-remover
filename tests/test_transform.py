import numpy as np

from src.transforms import stft, istft


def test_stft_istft_roundtrip():
    rng = np.random.default_rng(42)

    signal = rng.standard_normal(16000).astype(np.float32)

    S = stft(signal)

    reconstructed = istft(
        S,
        original_length=len(signal)
    )

    mse = np.mean((signal - reconstructed) ** 2)

    assert mse < 1e-6