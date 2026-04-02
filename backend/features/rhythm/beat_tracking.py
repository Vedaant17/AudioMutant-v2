import librosa

import librosa
import numpy as np

def extract_beat_grid(y, sr):

    # ✅ ALWAYS convert to mono
    y_mono = librosa.to_mono(y) if y.ndim > 1 else y

    tempo, beats = librosa.beat.beat_track(y=y_mono, sr=sr)

    beat_times = librosa.frames_to_time(beats, sr=sr)

    return {
        "tempo": float(tempo),  # ensure scalar
        "beat_times": beat_times[::4].tolist()
    }