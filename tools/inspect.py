import argparse

from src.io import load_audio
from src.transforms import stft
from src.analysis import plot_spectrogram

def main():

    parser = argparse.ArgumentParser(
        description = "Inspect an audio file"
    )

    parser.add_argument(
        "audio_file",
        help = "Path to WAV file"
    )

    parser.add_argument(
        "--n-fft",
        type = int,
        default = 2048
    )

    parser.add_argument(
        "--hop",
        type=int,
        default=512
    )

    args = parser.parse_args()

    signal, sample_rate = load_audio(args.audio_file)

    spec = stft(
        signal,
        n_fft = args.n_fft,
        hop_length = args.hop
    )

    plot_spectrogram(
        spec,
        sample_rate,
        args.hop,
        title = args.audio_file
    )

if __name__ == "__main__":
    main()