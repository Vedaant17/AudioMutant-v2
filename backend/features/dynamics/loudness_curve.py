import librosa
import numpy as np

def extract_loudness_curve(y):

    rms = librosa.feature.rms(y=y)[0]

    return {
        "mean": float(np.mean(rms)),
        "std": float(np.std(rms)),
        "max": float(np.max(rms)),
        "min": float(np.min(rms)),
        "curve_small": rms[::100].tolist()
    }