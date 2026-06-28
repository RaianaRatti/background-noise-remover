# clearwave

**Classical audio denoising — built from first principles.**

`clearwave` converts real-world recordings into clean, studio-quality audio using signal processing algorithms implemented from scratch. No pretrained models. No black boxes. Every filter is derived from math you can follow.

---

## What it does

Real recordings carry noise that falls into four distinct categories, each requiring a different approach:

| Noise Type | Characteristics | Examples |
|---|---|---|
| Stationary | Consistent spectrum over time | HVAC hum, fan noise, tape hiss |
| Non-stationary | Spectrum changes over time | Traffic, crowd noise, wind |
| Impulsive | Short, sharp transients | Clicks, pops, mic bumps |
| Narrowband | Energy at discrete frequencies | 50/60 Hz electrical hum and harmonics |

`clearwave` handles all four through a composable pipeline of classical DSP techniques: spectral subtraction, Wiener filtering, comb notch filters, autoregressive click interpolation, and a hand-built voice activity detector for adaptive noise tracking.

---

## Why from scratch?

Most noise reduction libraries are one function call. That call hides the short-time Fourier transform, the noise power estimation, the gain function, the overlap-add reconstruction — everything that matters. This project builds each of those pieces explicitly, so the tradeoffs are visible and the algorithms are understandable.

The only external dependencies are `numpy`, `scipy` (filter math only), `soundfile` (I/O), and `matplotlib`.

---

## Installation

```bash
git clone https://github.com/yourname/clearwave.git
cd clearwave
pip install -r requirements.txt
```

**Requirements:**
```
numpy
scipy
soundfile
matplotlib
```

---

## Quick Start

**Inspect a recording's spectrogram:**
```bash
python tools/inspect.py recording.wav
```

**Run the full denoising pipeline:**
```bash
python tools/clean.py recording.wav output.wav
```

**Compare before and after:**
```bash
python tools/compare.py recording.wav output.wav
```

**With a custom config:**
```bash
python tools/clean.py recording.wav output.wav --config config.yaml
```

---

## Pipeline

Processing stages run in a deliberate order. Each step is independently configurable or disableable.

```
Input audio
    │
    ▼
1. Normalize input
    │
    ▼
2. Click / pop removal          ← time domain; must precede STFT processing
    │
    ▼
3. Hum removal (comb filter)   ← deterministic narrowband; before broadband
    │
    ▼
4. Adaptive noise estimation   ← VAD-gated; tracks changing backgrounds
    │
    ▼
5. Wiener filtering            ← broadband suppression with smoothed gain
    │
    ▼
6. Normalize output
    │
    ▼
Output audio
```

**Why this order matters:** Clicks corrupt entire STFT frames, so they are removed before any frequency-domain processing. Electrical hum is deterministic and would be partially preserved by a Wiener filter that mistakes it for signal. Broadband noise removal runs last, using the most complete noise estimate.

---

## Configuration

The pipeline is driven by a YAML config file. All parameters have documented defaults.

```yaml
pipeline:
  normalize_input: true

  click_removal:
    enabled: true
    threshold_factor: 6.0       # std devs above local RMS to classify as click

  hum_removal:
    enabled: true
    auto_detect: true           # detect 50 or 60 Hz automatically
    notch_width_hz: 2.0         # bandwidth of each notch

  noise_reduction:
    method: wiener              # "wiener" or "spectral_subtraction"
    noise_estimation: adaptive  # "adaptive", "first_n_frames", or "percentile"
    n_fft: 2048
    hop_length: 512
    smoothing: 0.98             # temporal gain smoothing factor

  normalize_output: true
```

---

## Project Structure

```
clearwave/
├── src/
│   ├── io.py               # Load, save, resample, normalize
│   ├── transforms.py       # STFT, iSTFT (numpy.fft only)
│   ├── analysis.py         # Noise profiling, spectrogram visualization
│   ├── filters.py          # Spectral subtraction, Wiener, comb, click removal
│   ├── pipeline.py         # Composable processing chain
│   └── utils.py            # Windowing functions, overlap-add
├── tools/
│   ├── inspect.py          # CLI: render spectrogram of a file
│   ├── clean.py            # CLI: run pipeline on a file
│   └── compare.py          # CLI: side-by-side spectrogram + SNR metrics
├── tests/
│   ├── test_transforms.py  # STFT perfect reconstruction (MSE < 1e-6)
│   └── test_filters.py     # Filter gain and phase correctness
└── notebooks/
    ├── 01_stft_exploration.ipynb
    ├── 02_noise_profile.ipynb
    ├── 03_spectral_subtraction.ipynb
    ├── 04_wiener_filter.ipynb
    └── 05_full_pipeline.ipynb
```

---

## Algorithms

### Short-Time Fourier Transform

The STFT is implemented from scratch using `numpy.fft`. Reconstruction uses the overlap-add method with synthesis window normalization — the iSTFT divides by the sum of squared windows to guarantee perfect reconstruction when no processing is applied.

**Correctness check:** `istft(stft(x)) ≈ x` with mean squared error below `1e-6`. This is enforced in the test suite.

### Spectral Subtraction

Based on Boll (1979). The estimated noise power spectrum is subtracted from the noisy signal's power spectrum on a per-frequency-bin, per-frame basis:

```
P_clean(f, t) = max(P_noisy(f, t) − α · P_noise(f), β · P_noisy(f, t))
```

`α` controls oversubtraction aggressiveness; `β` sets the spectral floor to prevent silence artifacts. Temporal smoothing of the noise estimate reduces musical noise.

### Wiener Filter

Derived from minimum mean-squared-error estimation. A frequency-dependent gain function H(f, t) is computed per frame:

```
SNR(f, t) = (P_noisy(f, t) − P_noise(f)) / P_noise(f)
H(f, t)   = max(SNR, 0) / (max(SNR, 0) + 1)
```

H approaches 1 in high-SNR bins (preserve signal) and 0 in low-SNR bins (suppress noise). Temporal smoothing of H across frames eliminates the musical noise that spectral subtraction introduces.

### Comb Notch Filter

Removes electrical hum at a fundamental frequency (50 or 60 Hz, auto-detected) and its harmonics. Each harmonic is notched with a narrow IIR filter. The fundamental is detected by looking for harmonic series peaks in the long-term average power spectrum.

### Click Removal

Clicks are detected by comparing local RMS energy to a slowly varying baseline. Detected regions are replaced using autoregressive interpolation: an AR model is fit on surrounding clean samples and used to predict forward and backward into the gap, with the two predictions blended at the center.

### Voice Activity Detection

A frame-level classifier using three features — short-term energy, zero-crossing rate, and spectral flatness — combined with hysteresis to prevent rapid toggling. Used to gate noise estimate updates: the noise profile is updated only on frames classified as silence.

---

## Evaluation

The `tools/compare.py` script produces a side-by-side spectrogram and computes:

- **SNR (dB):** `10 · log₁₀( Σclean² / Σ(clean − noisy)² )`
- **Segmental SNR (dB):** SNR averaged over short segments; more correlated with perceived quality

To build a test set: take clean speech recordings, mix in recorded background noise at known SNR levels (e.g. 5, 10, 20 dB), and measure the pipeline's output SNR improvement across the range.

---

## Background Reading

The algorithms here have a short, well-documented literature. These are the primary sources:

- **Spectral Audio Signal Processing** — Julius O. Smith III. Free online: [ccrma.stanford.edu/~jos/sasp](https://ccrma.stanford.edu/~jos/sasp/). Covers STFT, windowing, and overlap-add rigorously.
- **Boll (1979)** — "Suppression of Acoustic Noise in Speech Using Spectral Subtraction." The original paper. Short and readable.
- **Ephraim & Malah (1984)** — MMSE-based spectral estimation. The statistical foundation behind the Wiener gain derivation.
- **Proakis & Manolakis** — *Digital Signal Processing*. Reference for IIR filter design and notch filter theory.

---

## Extending with a Learned Mask

The Wiener gain H(f, t) is a mask: a value in [0, 1] applied to each frequency bin. A small neural network can learn to predict this mask directly from noisy spectral features, bypassing the Gaussianity assumptions in the classical derivation.

See `notebooks/05_full_pipeline.ipynb` for a minimal two-layer network (implemented in NumPy or PyTorch for autodiff only) trained on synthetically mixed data. The learned mask consistently outperforms the Wiener filter on non-stationary noise once trained on sufficient variety.

---

## License

MIT