import numpy as np

from .transforms import stft, istft
from .utils import hann_window
from scipy.signal import iirnotch, lfilter

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

def notch_comb_filter(
        signal: np.ndarray,
        sample_rate: int,
        fundamental: int = 60,
        n_harmonics: int = 10,
        notch_width: float = 2.0
) -> np.ndarray:
    
    filtered = signal.astype(np.float32).copy()
    nyquist = sample_rate / 2
    
    for k in range(1, n_harmonics + 1):
        frequency = k * fundamental

        # Ignore harmonics above Nyquist
        if frequency >= nyquist:
            break

        # Quality factor
        Q = frequency / notch_width
        w0 = frequency / nyquist

        b, a = iirnotch(w0, Q)
        filtered = lfilter(b, a, filtered)

    return filtered

def detect_hum(
        signal: np.ndarray,
        sample_rate: int,
        n_fft: int = 4096,
        hop_length: int = 1024,
        n_harmonics: int = 6
) -> bool:
    
    spec = stft(signal, n_fft=n_fft, hop_length=hop_length)
    power = np.abs(spec) ** 2
    average_power = power.mean(axis=1)
    freqs = np.fft.rfftfreq(
        n_fft,
        d=1/sample_rate
    )

    best_freq = None
    best_score = -np.inf

    for fundamental in (50, 60):
        score = 0

        for h in range(1, n_harmonics + 1):
            f = h * fundamental

            if f >= sample_rate / 2:
                break

            idx = np.argmin(np.abs(freqs - f))

            peak = average_power[idx]

            left = max(0, idx - 2)
            right = min(len(average_power), idx + 3)

            neighborhood = average_power[left:right]

            background = (neighborhood.sum() - peak) / max(len(neighborhood) - 1, 1)
            score += peak / (background + 1e-10)

        if score > best_score:
            best_score = score
            best_freq = fundamental

    confidence = min(best_score / (5 * n_harmonics), 1.0)

    if confidence < 0.4:
        return None, confidence
    
    return best_freq, confidence

def detect_clicks(
        signal: np.ndarray,
        sample_rate: int,
        threshold_factor: float = 6.0,
        window_ms: float = 5.0
) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    window = max(1, int(sample_rate * window_ms / 1000))

    squared = signal ** 2
    kernel = np.ones(window) / window

    local_power = np.convolve(
        squared,
        kernel,
        mode="same"
    )

    local_rms = np.sqrt(local_power)

    # Robust baseline
    baseline = np.median(local_rms)
    threshold = threshold_factor * baseline
    click_mask = local_rms > threshold

    # Dilate by 2 ms
    dilation = max(1, int(sample_rate * 0.002))
    kernel = np.ones(dilation)

    click_mask = (
        np.convolve(
            click_mask.astype(np.float32),
            kernel,
            mode="same"
        ) 
        > 0
    )
    return click_mask

def repair_clicks(
    signal: np.ndarray,
    click_mask: np.ndarray
) -> np.ndarray:
    
    repaired = signal.copy()
    indices = np.where(click_mask)[0]

    if len(indices) == 0:
        return repaired
    
    groups = np.split(
        indices,
        np.where(np.diff(indices) != 1)[0] + 1
    )

    for group in groups:
        start = group[0]
        end = group[-1]
        
        if start == 0 or end == len(signal) - 1:
            continue

        left = repaired[start-1]
        right = repaired[end+1]

        repaired[start:end+1] = np.linspace(
            left,
            right,
            end-start+1
        )
    
    return repaired