import argparse
from pathlib import Path

import yaml

from src.io import load_audio, save_audio, normalize
from src.filters import (
    detect_clicks,
    repair_clicks,
    detect_hum,
    notch_comb_filter,
    wiener_filter,
    spectral_subtraction,
    simple_vad,
    adaptive_noise_estimation
)
from src.transforms import stft
from src.pipeline import AudioCleaningPipeline


# --------------------------------------------------
# Step wrappers
#
# Each function must accept (signal, sample_rate, **params)
# and return the processed signal, since that is the call
# signature AudioCleaningPipeline.run() uses.
# --------------------------------------------------

def normalize_step(signal, sample_rate):
    return normalize(signal)


def remove_clicks(signal, sample_rate, threshold_factor=6.0, window_ms=5.0):
    click_mask = detect_clicks(
        signal,
        sample_rate,
        threshold_factor=threshold_factor,
        window_ms=window_ms
    )
    return repair_clicks(signal, click_mask)


def remove_hum(signal, sample_rate, auto_detect=True, fundamental=60, notch_width_hz=2.0):
    if auto_detect:
        detected_fundamental, confidence = detect_hum(signal, sample_rate)
        if detected_fundamental is None:
            # No hum detected with enough confidence, leave signal untouched
            return signal
        fundamental = detected_fundamental

    return notch_comb_filter(
        signal,
        sample_rate,
        fundamental=fundamental,
        notch_width=notch_width_hz
    )


def reduce_noise(
    signal,
    sample_rate,
    method="wiener",
    noise_estimation="adaptive",
    n_fft=2048,
    hop_length=512,
    smoothing=0.98
):
    complex_spec = stft(signal, n_fft=n_fft, hop_length=hop_length)

    if noise_estimation == "adaptive":
        vad_labels = simple_vad(signal, sample_rate)
        noise_profile = adaptive_noise_estimation(
            complex_spec,
            vad_labels,
            smoothing=smoothing
        )
    else:
        raise ValueError(f"Unsupported noise estimation method: {noise_estimation}")

    if method == "wiener":
        return wiener_filter(
            signal,
            noise_profile,
            smoothing=smoothing,
            n_fft=n_fft,
            hop_length=hop_length
        )
    elif method == "spectral_subtraction":
        return spectral_subtraction(
            signal,
            noise_profile,
            n_fft=n_fft,
            hop_length=hop_length
        )
    else:
        raise ValueError(f"Unsupported noise reduction method: {method}")


# --------------------------------------------------
# Pipeline construction
# --------------------------------------------------

def build_pipeline(config):
    pipeline_config = config["pipeline"]
    pipeline = AudioCleaningPipeline(config)

    pipeline.add_step(
        "normalize_input",
        normalize_step,
        enabled=pipeline_config.get("normalize_input", True)
    )

    click_cfg = pipeline_config.get("click_removal", {})
    pipeline.add_step(
        "click_removal",
        remove_clicks,
        enabled=click_cfg.get("enabled", False),
        params={
            "threshold_factor": click_cfg.get("threshold_factor", 6.0),
            "window_ms": click_cfg.get("window_ms", 5.0)
        }
    )

    hum_cfg = pipeline_config.get("hum_removal", {})
    pipeline.add_step(
        "hum_removal",
        remove_hum,
        enabled=hum_cfg.get("enabled", False),
        params={
            "auto_detect": hum_cfg.get("auto_detect", True),
            "fundamental": hum_cfg.get("fundamental", 60),
            "notch_width_hz": hum_cfg.get("notch_width_hz", 2.0)
        }
    )

    noise_cfg = pipeline_config.get("noise_reduction", {})
    pipeline.add_step(
        "noise_reduction",
        reduce_noise,
        enabled=noise_cfg.get("enabled", False),
        params={
            "method": noise_cfg.get("method", "wiener"),
            "noise_estimation": noise_cfg.get("noise_estimation", "adaptive"),
            "n_fft": noise_cfg.get("n_fft", 2048),
            "hop_length": noise_cfg.get("hop_length", 512),
            "smoothing": noise_cfg.get("smoothing", 0.98)
        }
    )

    pipeline.add_step(
        "normalize_output",
        normalize_step,
        enabled=pipeline_config.get("normalize_output", True)
    )

    return pipeline


def clean_audio(signal, sample_rate, config):
    pipeline = build_pipeline(config)
    return pipeline.run(signal, sample_rate)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_file",
        help="Path to noisy audio file"
    )

    parser.add_argument(
        "output_file",
        help="Path to cleaned audio file"
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    signal, sample_rate = load_audio(input_path)

    cleaned_signal, diagnostics = clean_audio(signal, sample_rate, config)

    save_audio(output_path, cleaned_signal, sample_rate)

    print(f"Saved cleaned audio to {output_path}")
    print(f"Steps run: {list(diagnostics.keys())}")


if __name__ == "__main__":
    main()