# Load / save audio and resample

import soundfile as sf
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
import math

def load_audio(path: Path) -> tuple[np.ndarray, int]: # samples, sample_rate
    samples, sample_rate = sf.read(path, dtype="float32")

    # Convert stereo / multi-channel audio to mono if needed
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    return samples, sample_rate

def save_audio(path: Path, samples: np.ndarray, sample_rate: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, sample_rate)

# sinc resampling
def resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return samples

    g = math.gcd(orig_sr, target_sr)

    return resample_poly(samples, up = target_sr // g, down = orig_sr // g)

# peak normalize to -1/+1
def normalize(samples: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(samples))

    if peak == 0:
        return samples.copy()
    
    return samples / peak