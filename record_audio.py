import sounddevice as sd
import soundfile as sf
import numpy as np

SAMPLE_RATE = 16000

filename = "speaker1.wav"

print("Recording... Press Ctrl+C to stop.")

recording = []

try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=lambda indata, frames, time, status: recording.append(indata.copy())
    ):
        while True:
            sd.sleep(100)

except KeyboardInterrupt:
    pass

audio = np.concatenate(recording, axis=0)
sf.write(filename, audio, SAMPLE_RATE)

print(f"Saved {filename}")