import argparse
import yaml
from pathlib import Path

from src.io import load_audio, save_audio, normalize
from src.filters import (
    detect_clicks,
    repair_clicks,
    detect_hum,
    notch_comb_filter,
    wiener_filter,
    simple_vad,
    adaptive_noise_estimation
)
from src.transforms import stft


def clean_audio(signal, sample_rate, config):
    pipeline_config = config["pipeline"]

    # --------------------------------------------------
    # Input normalization
    # --------------------------------------------------
    if pipeline_config.get("normalize_input", True):
        signal = normalize(signal)

    # --------------------------------------------------
    # Click removal
    # --------------------------------------------------
    click_cfg = pipeline_config["click_removal"]

    if click_cfg["enabled"]:
        click_mask = detect_clicks(
            signal,
            sample_rate,
            threshold_factor=click_cfg["threshold_factor"]
        )

        signal = repair_clicks(
            signal,
            click_mask
        )

    # --------------------------------------------------
    # Hum removal
    # --------------------------------------------------
    hum_cfg = pipeline_config["hum_removal"]

    if hum_cfg["enabled"]:

        if hum_cfg["auto_detect"]:
            fundamental, confidence = detect_hum(
                signal,
                sample_rate
            )
        else:
            fundamental = 60

        if fundamental is not None:
            signal = notch_comb_filter(
                signal,
                sample_rate,
                fundamental=fundamental,
                notch_width=hum_cfg["notch_width_hz"]
            )

    # --------------------------------------------------
    # Noise reduction
    # --------------------------------------------------
    noise_cfg = pipeline_config["noise_reduction"]

    complex_spec = stft(
        signal,
        n_fft=noise_cfg["n_fft"],
        hop_length=noise_cfg["hop_length"]
    )

    if noise_cfg["noise_estimation"] == "adaptive":

        vad_labels = simple_vad(
            signal,
            sample_rate
        )

        noise_profile = adaptive_noise_estimation(
            complex_spec,
            vad_labels,
            smoothing=noise_cfg["smoothing"]
        )

    else:
        raise ValueError(
            f"Unsupported noise estimation method: "
            f"{noise_cfg['noise_estimation']}"
        )

    if noise_cfg["method"] == "wiener":
        signal = wiener_filter(
            signal,
            noise_profile,
            smoothing=noise_cfg["smoothing"],
            n_fft=noise_cfg["n_fft"],
            hop_length=noise_cfg["hop_length"]
        )

    else:
        raise ValueError(
            f"Unsupported noise reduction method: "
            f"{noise_cfg['method']}"
        )

    # --------------------------------------------------
    # Output normalization
    # --------------------------------------------------
    if pipeline_config.get("normalize_output", True):
        signal = normalize(signal)

    return signal


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

    signal, sample_rate = load_audio(
        input_path
    )

    cleaned_signal = clean_audio(
        signal,
        sample_rate,
        config
    )

    save_audio(
        output_path,
        cleaned_signal,
        sample_rate
    )

    print(f"Saved cleaned audio to {output_path}")


if __name__ == "__main__":
    main()