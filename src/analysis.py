import numpy as np
import matplotlib.pyplot as plt

def plot_spectogram(
    complex_spec: np.ndarray,
    sample_rate: int,
    hop_length: int,
    title: str = ""
):
    magnitude = np.abs(complex_spec)
    magnitude_db = 20 * np.log10(magnitude + 1e-8)
    n_bins, n_frames = magnitude.shape
    duration = n_frames * hop_length / sample_rate
    max_freq = sample_rate / 2
    plt.figure(figsize=(12,6))

    plt.imshow(
        magnitude_db,
        origin="lower",
        aspect="auto",
        extent=[0, duration, 0, max_freq],
        cmap="magma"
    )

    plt.yscale("symlog")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

    plt.title(title)

    plt.colorbar(label = "Magnitude (dB)")

    plt.tight_layout()
    plt.show()