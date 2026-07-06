import numpy as np

from .transforms import stft, istft
from .utils import hann_window

def spectral_subtraction(
    signal: np.ndarray,
    noise_profile: np.ndarray,
    alpha: float = 2.0,
    beta: float = 0.01,
    n_fft: int = 2048,
    hop_length: int = 512,
    window_fn = hann_window
) -> np.ndarray:
    
    # 1. STFT
    X = stft(signal, n_fft = n_fft, hop_length = hop_length, window_fn = window_fn)

    # 2. Magnitude and phase
    magnitude = np.abs(X)
    phase = np.angle(X)

    # 3. Noisy power
    power_noisy = magnitude ** 2

    # 4. Broadcast noise profile
    power_noise = noise_profile[:, np.newaxis]

    # 5. Spectral subtraction
    power_clean = power_noisy - alpha * power_noise

    # 6. Spectral floor
    power_clean = np.maximum(power_clean, beta * power_noisy)

    # 7. Recover magnitude
    magnitude_clean = np.sqrt(power_clean)

    # 8. Recombine magnitude + phase
    X_clean = magnitude_clean * np.exp(1j * phase)

    # 9. Inverse STFT
    return istft(
        X_clean,
        hop_length = hop_length,
        window_fn = window_fn,
        original_length = len(signal)
    )

def exponential_smooth(
    power_noise: np.ndarray,
    alpha: float = 0.9
) -> np.ndarray:
    smoothed = np.empty_like(power_noise)

    smoothed[:, 0] = power_noise[:, 0]
    
    for t in range(1, power_noise.shape[1]):
        smoothed[:, t] = alpha * smoothed[:, t - 1] + (1-alpha) * power_noise[:, t]
    
    return smoothed

def wiener_filter(
    signal: np.ndarray,
    noise_profile: np.ndarray,
    smoothing: int = 0.98,
    n_fft: int = 2048,
    hop_length: int = 512,
    window_fn = hann_window
) -> np.ndarray:

    # 1. STFT
    X = stft(signal, n_fft = n_fft, hop_length = hop_length, window_fn = window_fn)

    # 2. Noisy power spectrum
    power_noisy = np.abs(X) ** 2
    
    # 3. Broadcast noise profile
    power_noise = noise_profile[:, np.newaxis]

    # 4. Estimate SNR
    snr = (power_noisy - power_noise) / (power_noise + 1e-10)
    snr = np.maximum(snr, 0.0)

    # 5. Wiener gain
    wiener_gain = snr / (snr + 1)

    # 6. Temporal smoothing applied
    gain_smoothed = exponential_smooth(wiener_gain, smoothing)
    X_clean = gain_smoothed * X

    # 7. Reconstruct
    return istft(X_clean, hop_length = hop_length, window_fn = window_fn, original_length = len(signal))