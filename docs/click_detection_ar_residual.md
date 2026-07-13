# Alternative Click Detection: Linear Prediction (AR) Residual

## Why look for another method

The current `detect_clicks` (`src/filters.py`) flags a sample as a click when its
local RMS energy exceeds `threshold_factor * baseline`, where `baseline` is a
statistic (median, or the 25th percentile) of local RMS over the *whole* clip.

This is fundamentally an **energy/loudness** test: "is this moment louder than
usual?" That test is only reliable when clicks are louder than everything else
in the recording. In real speech recordings, loud voiced segments (vowels,
plosives) can rival or exceed a click's energy, so any global energy threshold
is a compromise between missing quiet clicks and false-triggering on loud
speech — tuning the percentile just moves that compromise around, it doesn't
remove it.

## The alternative: predict each sample, look at what you couldn't predict

A click isn't just loud — it's *unpredictable*. Speech (and most natural
sound) is locally smooth: each sample is well approximated by a linear
combination of the samples just before it, because the vocal tract behaves
like a slowly-varying resonant filter over short windows. A click is a sudden
discontinuity that breaks that local structure, regardless of how loud the
surrounding speech is.

This is the classic approach used in vinyl/tape restoration (Vaseghi & Rayner,
*"Detection and suppression of impulsive noise in speech communication
systems"*), and it fits naturally with `PLAN.md`'s existing mention of
AR-based interpolation in Phase 4.2 — the same AR model can be reused for
detection.

### Algorithm

```
1. Choose an AR order p (rule of thumb: ~1 coefficient per 0.5-1 ms,
   e.g. p = sample_rate // 1000 for a 1 ms model).

2. Slide a window across the signal (e.g. 20-50 ms, hop a few ms).
   Within each window, fit an AR(p) model via least squares / Yule-Walker:

       x[n] ≈ a1*x[n-1] + a2*x[n-2] + ... + ap*x[n-p]

   Fitting can use scipy/numpy lstsq on the windowed samples
   (Levinson-Durbin is the classic O(p^2) way, but a plain lstsq solve
   is fine for a from-scratch implementation).

3. Compute the one-step-ahead prediction for every sample in the window
   using the fitted coefficients, and take the residual:

       residual[n] = x[n] - predicted[n]

4. A click makes the AR model's prediction fail badly at that instant,
   so |residual[n]| spikes there even if the surrounding speech is loud
   (the model already "expects" loud speech - it was fit on it).

5. Flag click samples where |residual[n]| exceeds a threshold based on a
   robust *local* statistic of the residual itself, e.g.:

       robust_sigma = 1.4826 * MAD(residual in this window)   # MAD = median absolute deviation
       click_mask[n] = |residual[n]| > threshold_factor * robust_sigma

   Using MAD instead of std/RMS matters here too: a few click samples in
   the window shouldn't be allowed to inflate the "normal residual" estimate
   used to detect them.

6. Dilate the mask slightly, same as today, to catch the short decaying
   tail that follows the impulse.
```

### Why this fixes the speech-loudness problem

The AR model is fit *per window*, so it adapts to whatever the local speech
energy and spectral shape are. The threshold is applied to the **prediction
residual**, not the raw signal energy — a loud but smooth vowel is still
predictable (small residual), while a click is unpredictable regardless of
whether it lands during loud speech or silence. This removes the core
trade-off in the RMS/percentile approach, rather than just shifting it.

### Trade-offs vs. the current RMS approach

| | RMS + baseline percentile (current) | AR residual |
|---|---|---|
| Cost | O(n), one convolution | O(n * p^2) or O(n * p) with Levinson-Durbin, still cheap for small p |
| Sensitive to loud speech | Yes — this is the bug we hit | No — adapts locally |
| Extra tuning parameter | `threshold_factor`, `window_ms` | AR order `p`, window size, `threshold_factor` |
| Complexity to implement from scratch | Very low | Moderate (needs a linear solve per window) |
| False positives on sharp phonemes (plosives, sibilants) | Possible if loud | Lower — these are usually still linearly predictable over 1 ms, unlike a true click |

### Where it would plug in

Same interface as today, so it's a drop-in alternative rather than a rewrite
of the pipeline:

```
detect_clicks_ar(
    signal: np.ndarray,
    sample_rate: int,
    ar_order: int = None,       # default sample_rate // 1000
    window_ms: float = 30.0,
    hop_ms: float = 5.0,
    threshold_factor: float = 6.0
) -> click_mask
```

`repair_clicks` (linear or AR interpolation) stays unchanged — only the
*detection* step changes.
