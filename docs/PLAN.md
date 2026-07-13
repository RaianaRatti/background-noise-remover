# Audio Noise Reduction: Real Recording to Studio Quality
### A From-Scratch Learning Project

---

## Project Philosophy

This project is deliberately low-dependency. The goal is to understand *why* each step works, not just run a pretrained model. You will build noise reduction using classical signal processing and, optionally, a small neural network you design yourself. Every algorithm here is something you can derive on paper.

**Allowed dependencies:** `numpy`, `scipy`, `soundfile` (I/O only), `matplotlib` (visualization). No pretrained models, no `librosa` for core processing (you'll reimplement what you need).

---

## What Is "Background Noise"?

Before writing code, understand what you're fighting:

| Noise Type | Characteristics | Example |
|---|---|---|
| Stationary | Consistent spectrum over time | HVAC hum, fan, tape hiss |
| Non-stationary | Changes over time | Traffic, crowd, wind gusts |
| Impulsive | Short sharp spikes | Clicks, pops, mic bumps |
| Narrowband | Concentrated at one frequency | 50/60 Hz electrical hum |

Different types need different tools. This project will handle all four.

---

## Concepts to Master (in order)

Before implementing each module, read these concepts. Links to free resources are noted.

1. **Sampling and the Fourier Transform** - How digital audio works, what FFT gives you
2. **The Short-Time Fourier Transform (STFT)** - Analyzing how a signal's frequency content changes over time
3. **Spectrograms** - Visualizing STFT magnitude; your main debugging tool
4. **Power Spectral Density** - Measuring the "energy" at each frequency
5. **Wiener Filtering** - The statistical foundation of noise reduction
6. **Spectral Subtraction** - The simplest practical noise suppression
7. **Psychoacoustics basics** - Why some artifacts are more annoying than others (musical noise)

---

## Project Structure

```
audio-cleaner/
    README.md
    PLAN.md
    requirements.txt
    
    src/
        io.py               # Load/save audio, resample
        transforms.py       # STFT, iSTFT, mel filterbank
        analysis.py         # Noise profiling, spectrograms
        filters.py          # The core algorithms
        pipeline.py         # Chain filters together
        utils.py            # Windowing, overlap-add
    
    tools/
        inspect.py          # CLI: view spectrogram of a file
        clean.py            # CLI: run full pipeline on a file
        compare.py          # CLI: A/B plot before/after
    
    tests/
        test_transforms.py  # Verify STFT roundtrip
        test_filters.py     # Verify filter math
    
    notebooks/
        01_stft_exploration.ipynb
        02_noise_profile.ipynb
        03_spectral_subtraction.ipynb
        04_wiener_filter.ipynb
        05_full_pipeline.ipynb
```

---

## Phase 1: Foundation - Audio I/O and the STFT

**Goal:** Load audio, compute STFT, reconstruct perfectly, visualize.

### 1.1 - Audio I/O (`src/io.py`)

Use `soundfile` only for reading/writing raw PCM data. Implement everything else yourself.

```
load_audio(path) -> (samples: np.ndarray, sample_rate: int)
save_audio(path, samples, sample_rate)
resample(samples, orig_sr, target_sr) -> np.ndarray   # implement sinc resampling
normalize(samples) -> np.ndarray                      # peak normalize to -1/+1
```

**Learn:** Why do we normalize? What is sample rate? What does resampling do in the frequency domain?

### 1.2 - Windowing (`src/utils.py`)

Windows are multiplied onto each audio frame before FFT to reduce spectral leakage.

```
hann_window(n) -> np.ndarray       # implement from formula: 0.5*(1 - cos(2pi*k/N))
hamming_window(n) -> np.ndarray
blackman_window(n) -> np.ndarray
```

**Learn:** What is spectral leakage? Plot the frequency response of each window. The Hann window will be your default for everything.

### 1.3 - STFT and Inverse STFT (`src/transforms.py`)

The Short-Time Fourier Transform is the heart of this project. Implement it from scratch using only `numpy.fft`.

```
stft(
    signal,
    n_fft=2048,
    hop_length=512,
    window_fn=hann_window
) -> complex_spectrogram   # shape: (n_fft//2 + 1, n_frames)

istft(
    complex_spectrogram,
    hop_length=512,
    window_fn=hann_window,
    original_length=None
) -> signal
```

**Key implementation detail - Overlap-Add (OLA):**
The iSTFT uses overlap-add reconstruction. Each frame is windowed, then overlapping frames are summed. You must divide by the sum of squared windows to normalize correctly. This is called the "synthesis window normalization."

```python
# Pseudocode for iSTFT OLA:
output = np.zeros(output_length)
window_sum = np.zeros(output_length)
for i, frame in enumerate(frames):
    start = i * hop_length
    output[start:start+n_fft] += np.real(ifft(frame)) * window
    window_sum[start:start+n_fft] += window**2
output /= np.where(window_sum > 1e-8, window_sum, 1.0)
```

**Milestone test:** `istft(stft(signal))` should reconstruct the original signal with less than 1e-6 mean squared error. Write this as a unit test.

### 1.4 - Spectrogram Visualization (`src/analysis.py`, `tools/inspect.py`)

```
plot_spectrogram(complex_spec, sample_rate, hop_length, title="")
    # Convert to magnitude, then to dB: 20*log10(|X| + 1e-8)
    # Plot with time on x-axis, frequency on y-axis
    # Use log frequency scale for readability
```

Build the `tools/inspect.py` CLI first. You'll use it constantly:
```
python tools/inspect.py recording.wav
```

---

## Phase 2: Stationary Noise Removal

**Goal:** Remove consistent background noise like fans, hiss, hum.

### 2.1 - Noise Profiling

Stationary noise removal works by estimating the noise spectrum during a "noise-only" segment, then subtracting it everywhere.

```
estimate_noise_profile(
    signal,
    sample_rate,
    method="first_n_frames",  # or "percentile" or "manual_segment"
    n_frames=20,
    percentile=10
) -> noise_power_spectrum   # shape: (n_fft//2 + 1,)
```

**Three noise estimation strategies:**

1. **First N frames:** Average power spectrum of the first N frames. Works if the recording starts with silence.
2. **Percentile method:** For each frequency bin, take the Pth percentile power across all frames. Since speech is intermittent, the low percentile captures noise-floor moments. This is more robust.
3. **Manual segment:** User specifies start/end time in seconds of a known-noise-only region.

**Learn:** Why percentile? A frequency bin that consistently reads low across time is likely just noise. Bins with speech activity will spike intermittently.

### 2.2 - Spectral Subtraction (`src/filters.py`)

The oldest and most intuitive noise reduction algorithm (Boll, 1979).

**Core idea:** If we know the average noise power at each frequency, subtract it from the signal's power spectrum.

```
spectral_subtraction(
    signal,
    noise_profile,
    alpha=2.0,      # oversubtraction factor (>1 = more aggressive)
    beta=0.01       # spectral floor (prevents going to zero)
) -> cleaned_signal
```

**Algorithm:**
```
1. Compute STFT of noisy signal: X = stft(signal)
2. Get magnitude: |X| and phase: angle(X)
3. Compute noisy power: P_noisy = |X|^2
4. Compute noise power: P_noise = noise_profile (broadcast across frames)
5. Subtract: P_clean = P_noisy - alpha * P_noise
6. Apply floor: P_clean = max(P_clean, beta * P_noisy)
7. Recover magnitude: |X_clean| = sqrt(P_clean)
8. Recombine with original phase: X_clean = |X_clean| * exp(j * angle(X))
9. Reconstruct: signal_clean = istft(X_clean)
```

**The "musical noise" problem:** Spectral subtraction introduces a characteristic artifact - random isolated spikes in the spectrogram that create a musical, warbling quality. This is because noise estimation is imperfect and some bins get over-subtracted while others are under-subtracted frame-to-frame.

**Fix - Temporal smoothing:** Before applying subtraction, smooth the noise estimate across adjacent frames using a moving average or exponential smoothing:
```
P_noise_smooth[t] = alpha_smooth * P_noise_smooth[t-1] + (1-alpha_smooth) * P_noise[t]
```

### 2.3 - Wiener Filter (`src/filters.py`)

More principled than spectral subtraction. Derived from minimizing mean squared error between clean and estimated signal.

```
wiener_filter(
    signal,
    noise_profile,
    smoothing=0.98
) -> cleaned_signal
```

**Algorithm:**
The Wiener gain H(f, t) is applied to each frequency bin:
```
SNR(f,t) = (|X(f,t)|^2 - P_noise(f)) / P_noise(f)
SNR_max = max(SNR, 0)   # don't go negative
H(f,t) = SNR_max / (SNR_max + 1)    # ranges from 0 to 1
X_clean(f,t) = H(f,t) * X(f,t)
```

**Why this works:** When SNR is high (loud speech), H approaches 1 (pass through). When SNR is low (mostly noise), H approaches 0 (suppress). The filter is adaptive per frame and per frequency bin.

**Gain smoothing:** Apply temporal smoothing to H(f,t) across frames to avoid rapid fluctuations. This greatly reduces musical noise.

**Compare:** Run both spectral subtraction and Wiener filter on the same file. Plot the spectrograms side by side. The Wiener filter should have fewer artifacts.

---

## Phase 3: Narrowband Noise (Hum Removal)

**Goal:** Remove 50/60 Hz electrical hum and its harmonics.

### 3.1 - Comb Filter (`src/filters.py`)

Electrical hum sits at exactly 50 or 60 Hz, plus harmonics (100, 150, 200... or 120, 180, 240...). A comb filter notches out exactly those frequencies.

```
notch_comb_filter(
    signal,
    sample_rate,
    fundamental=60,     # Hz
    n_harmonics=10,
    notch_width=2.0     # Hz bandwidth of each notch
) -> cleaned_signal
```

**Implementation using scipy.signal (allowed as a math utility):**
Chain individual IIR notch filters. For each harmonic frequency f_k:
```python
from scipy.signal import iirnotch, lfilter

b, a = iirnotch(f_k / (sample_rate/2), Q=f_k/notch_width)
signal = lfilter(b, a, signal)
```

**Or implement from scratch:** A notch filter in the frequency domain is simply zeroing (or attenuating) the STFT bins nearest to each harmonic frequency. Less precise but instructive.

**Learn:** What is Q factor? Why do harmonics exist? (Nonlinear distortion in power supplies.)

### 3.2 - Hum Detection

Auto-detect whether hum is present and at what frequency:

```
detect_hum(signal, sample_rate) -> (fundamental_hz or None, confidence)
```

Compute the long-term average power spectrum. Look for peaks at 50 or 60 Hz with harmonic structure. A true hum will have peaks at f, 2f, 3f... that stand out from the noise floor.

---

## Phase 4: Impulsive Noise (Click/Pop Removal)

**Goal:** Remove short transient artifacts like vinyl clicks, mic pops.

### 4.1 - Click Detection (`src/filters.py`)

Clicks are characterized by a very sudden, brief energy spike across all frequencies. Detect them in the time domain:

```
detect_clicks(
    signal,
    threshold_factor=6.0    # how many std devs above local RMS = click
) -> click_mask   # boolean array, True where clicks are
```

**Algorithm:**
```
1. Compute local RMS energy with a short window (e.g., 5ms)
2. Compute global or slowly-varying RMS for baseline
3. Where local_rms > threshold_factor * baseline_rms: mark as click
4. Dilate the mask slightly (clicks have a ringing tail)
```

### 4.2 - Click Interpolation

Replace detected click regions with interpolated signal:

```
repair_clicks(signal, click_mask) -> repaired_signal
```

**Two approaches:**
1. **Linear interpolation:** Simple, just draw a straight line across the gap. Fine for short clicks.
2. **AR (autoregressive) interpolation:** Model the signal as a linear combination of past samples, predict forward across the gap. Much better quality.

For AR interpolation:
```
1. Fit an AR model on the clean samples surrounding the click region
2. Forward-predict into the gap from the left
3. Backward-predict into the gap from the right  
4. Blend the two predictions
```

The AR order should be approximately sample_rate / 1000 (one per millisecond of pitch period).

---

## Phase 5: Non-Stationary Noise

**Goal:** Handle noise that changes over time (traffic, crowd, wind).

### 5.1 - Voice Activity Detection (VAD)

To handle non-stationary noise, you need to know when speech is happening so you can update the noise estimate from silence frames.

```
simple_vad(
    signal,
    sample_rate,
    frame_length_ms=25,
    hop_length_ms=10,
    energy_threshold=None   # auto if None
) -> vad_labels   # array of 0/1 per frame
```

**Features for VAD decision (compute per frame):**
1. **Short-term energy:** `sum(frame^2)`
2. **Zero crossing rate:** `sum(|sign(x[n]) - sign(x[n-1])|) / N` - speech has more ZCR than silence
3. **Spectral flatness:** ratio of geometric to arithmetic mean of spectrum - noise tends to be flatter than speech

Combine with a simple threshold or hysteresis state machine (don't flip between speech/silence too fast).

This is simpler than VADNet but sufficient for noise estimation purposes.

### 5.2 - Adaptive Noise Estimation

Instead of estimating noise once at the start, continuously update the noise estimate using VAD labels:

```
adaptive_noise_estimation(
    complex_spectrogram,
    vad_labels,
    smoothing=0.95
) -> noise_profile_per_frame   # updated estimate at each frame
```

**Algorithm:**
```
For each frame t:
    if vad_labels[t] == 0 (silence):
        noise_estimate[t] = smoothing * noise_estimate[t-1] + (1-smoothing) * |X[t]|^2
    else (speech):
        noise_estimate[t] = noise_estimate[t-1]  # hold last value
```

This allows noise reduction to track slowly-changing backgrounds like outdoor recordings.

---

## Phase 6: The Pipeline

**Goal:** Chain everything together intelligently.

### 6.1 - Pipeline Design (`src/pipeline.py`)

```python
class AudioCleaningPipeline:
    def __init__(self, config: dict):
        self.steps = []
    
    def add_step(self, name, fn, enabled=True, params={}):
        ...
    
    def run(self, signal, sample_rate) -> (cleaned_signal, diagnostics):
        ...
```

**Recommended processing order:**
```
1. Load and normalize
2. Click/pop removal          (time domain, before STFT)
3. Hum removal                (comb notch filter)
4. Noise profile estimation   (adaptive, using VAD)
5. Wiener filtering           (main broadband noise removal)
6. Output normalization
```

**Why this order matters:**
- Remove impulsive noise before STFT-based processing; clicks corrupt entire STFT frames.
- Remove deterministic hum before broadband filtering; the Wiener filter treats hum as "signal."
- Do broadband last, with the most accurate noise estimate.

### 6.2 - Configuration File

Drive the pipeline from a YAML/JSON config:

```yaml
pipeline:
  normalize_input: true
  click_removal:
    enabled: true
    threshold_factor: 6.0
  hum_removal:
    enabled: true
    auto_detect: true
    notch_width_hz: 2.0
  noise_reduction:
    method: wiener           # or "spectral_subtraction"
    noise_estimation: adaptive   # or "first_n_frames" or "percentile"
    n_fft: 2048
    hop_length: 512
    smoothing: 0.98
  normalize_output: true
```

---

## Phase 7: Evaluation

**Goal:** Objectively measure improvement.

### 7.1 - Metrics (implement from scratch)

These require a clean reference signal (you'll need to create or download test pairs):

```
snr(clean, noisy) -> dB           # signal to noise ratio
segmental_snr(clean, noisy) -> dB # SNR averaged per short segment (more perceptually relevant)
```

**Signal-to-Noise Ratio:**
```
SNR = 10 * log10( sum(clean^2) / sum((clean - noisy)^2) )
```

### 7.2 - Test Dataset

Create your own test set:
1. Find clean speech recordings (e.g., audiobook recordings or studio podcasts)
2. Record your own background noise (fan, outside ambience)
3. Mix them at known SNRs (e.g., 5 dB, 10 dB, 20 dB)
4. Run your pipeline and measure output SNR improvement

This lets you plot "input SNR vs output SNR" curves - your pipeline should always improve it.

### 7.3 - Perceptual Evaluation

Implement a simple A/B test script:
```
python tools/compare.py input.wav cleaned.wav
```
This should:
- Plot spectrograms of both side by side
- Play them back alternately (using `sounddevice` or write a wav and open it)
- Optionally compute and display SNR metrics

---

## Optional Phase 8: Learn by Building a Simple Spectral Mask Neural Network

If you want to go further without using pretrained models, build a tiny neural network yourself that learns to predict a spectral mask. This extends everything you've already built.

### Architecture

Input: noisy STFT magnitude frame (n_fft//2 + 1 features) + a few context frames
Output: soft mask M(f) in [0,1] for each frequency bin
Apply: `X_clean = M * X_noisy`

**Network (implement in pure numpy, or use PyTorch only for autodiff):**
```
Linear(n_freq, 256) -> ReLU
Linear(256, 256) -> ReLU  
Linear(256, n_freq) -> Sigmoid
```

### Training Data

Generate it synthetically:
- Clean: audiobook speech clips
- Noise: recordings of fans, outdoor ambience, etc.
- Mix at random SNRs between -5 and 20 dB
- Label: the ideal binary/soft mask = `|clean| / (|clean| + |noise|)`

### Loss Function

```
loss = mean((predicted_mask - ideal_mask)^2)
```
or optionally signal-approximation loss:
```
loss = mean((predicted_mask * |X_noisy| - |X_clean|)^2)
```

### Why This Is Educational

You are directly observing the connection between the classical Wiener filter (which you built in Phase 2) and learned approaches. The Wiener gain H(f,t) IS a mask. The neural network learns to predict it without assuming Gaussianity.

---

## Milestones and Suggested Order

| Week | Deliverable |
|---|---|
| 1 | Phase 1 complete: `stft(istft(x)) == x` passes, spectrograms render |
| 2 | Phase 2 complete: stationary noise removed, Wiener filter working |
| 3 | Phases 3 and 4 complete: hum and clicks handled |
| 4 | Phase 5 complete: adaptive noise estimation, VAD working |
| 5 | Phase 6 complete: full pipeline, config file, CLI tools |
| 6 | Phase 7 complete: evaluation framework, SNR plots |
| 7+ | Optional Phase 8: learned mask network |

---

## Key Reading (Free)

- **"Spectral Audio Signal Processing"** by Julius O. Smith (online, free): https://ccrma.stanford.edu/~jos/sasp/ — covers STFT, windows, overlap-add rigorously
- **Boll (1979)** "Suppression of acoustic noise in speech using spectral subtraction" - the original spectral subtraction paper
- **Ephraim & Malah (1984)** on MMSE-based Wiener filtering - slightly harder math but very illuminating
- **"Digital Signal Processing"** by Proakis & Manolakis - for IIR filter design (notch filters)

---

## Notes on What Not to Use

These are the tempting libraries to avoid using as black boxes (for learning purposes):

| Library | What it does | What to build instead |
|---|---|---|
| `noisereduce` | Entire spectral subtraction pipeline | Phases 2-3 of this plan |
| `librosa` core (stft, mel) | STFT, mel filterbank | `src/transforms.py` |
| `pyannote.audio` | VAD, diarization | Phase 5 VAD |
| Pretrained denoising models | End-to-end learned denoising | Optional Phase 8 (your own) |

`scipy.signal` is acceptable for filter design math (IIR coefficients) — that math is not the interesting part. The interesting parts are the STFT, noise estimation, and gain computation, which you build yourself.