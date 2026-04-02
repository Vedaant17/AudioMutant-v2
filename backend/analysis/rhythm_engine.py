import librosa
import numpy as np

def analyze_rhythm(y, sr):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

    beat_times = librosa.frames_to_time(beats, sr=sr)

    intervals = np.diff(beat_times)
    groove = np.std(intervals) if len(intervals) else 0

    return {
        "tempo": float(tempo),
        "beat_times": beat_times[::4].tolist(),
        "groove_stability": float(groove)
    }