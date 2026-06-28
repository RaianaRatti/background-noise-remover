import numpy as np
from .utils import hann_window

# Short-time fourier transform
def stft(
        signal: np.ndarray, # input audio signal
        n_fft: int = 2048, # frame length / FFT size
        hop_length: int = 512, # number of samples between frames
        window_fn = hann_window # functioning returning window of length n_fft
) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    window = window_fn(n_fft)

    n_frames = 1 + int(np.ceil((len(signal) - n_fft) / hop_length))

    # zero-pad if necessary
    padded_length = (n_frames - 1) * hop_length + n_fft

    if padded_length > len(signal):
        signal = np.pad(signal, (0, padded_length - len(signal)))

    spectrogram = np.empty((n_fft // 2 + 1, n_frames), dtype=np.complex64)

    for i in range(n_frames):
        start = i * hop_length
        frame = signal[start:start + n_fft]

        frame = frame * window

        spectrogram[:, i] = np.fft.rfft(frame)
    
    return spectrogram

# Inverse short-time fourier transform using overlap_add
def istft(
        spectrogram: np.ndarray,
        hop_length: int = 512,
        window_fn = hann_window,
        original_length: int | None = None
) -> np.ndarray:
    n_fft = (spectrogram.shape[0] - 1) * 2
    n_frames = spectrogram.shape[1]

    window = window_fn(n_fft)

    output_length = hop_length * (n_frames - 1) + n_fft

    output = np.zeros(output_length, dtype=np.float32)
    window_sum = np.zeros(output_length, dtype=np.float32)

    for i in range(n_frames):

        start = i * hop_length

        frame = np.fft.irfft(spectrogram[:, i], n=n_fft)

        frame *= window

        output[start:start + n_fft] += frame

        window_sum[start:start + n_fft] += window ** 2

    nonzero = window_sum > 1e-8
    output[nonzero] /= window_sum[nonzero]

    if original_length is not None:
        output = output[:original_length]

    return output