import numpy as np
import librosa


def extract_stereo_features(y, sr):

    # -----------------------------
    # MONO CASE
    # -----------------------------
    if y.ndim < 2:
        return {
            "stereo_width": 0.0,
            "phase_correlation": 1.0,
            "stereo_field": None
        }

    left = y[0]
    right = y[1]

    # -----------------------------
    # BASIC STEREO METRICS
    # -----------------------------
    correlation = np.corrcoef(left, right)[0, 1]
    correlation = np.clip(correlation, -1, 1)

    stereo_width = float(1 - correlation)
    phase_corr = float(correlation)

    # -----------------------------
    # MID / SIDE
    # -----------------------------
    mid = (left + right) / 2
    side = (left - right) / 2

    # -----------------------------
    # STFT
    # -----------------------------
    mid_stft = np.abs(librosa.stft(mid))
    side_stft = np.abs(librosa.stft(side))

    freqs = librosa.fft_frequencies(sr=sr)

    # -----------------------------
    # FREQUENCY BANDS
    # -----------------------------
    bands = {
        "low": (20, 250),
        "mid": (250, 4000),
        "high": (4000, 20000)
    }

    stereo_distribution = {}

    for band, (low, high) in bands.items():
        idx = np.where((freqs >= low) & (freqs < high))

        mid_energy = np.mean(mid_stft[idx])
        side_energy = np.mean(side_stft[idx])

        width = side_energy / (mid_energy + 1e-8)

        stereo_distribution[band] = float(width)

    return {
        "stereo_width": stereo_width,
        "phase_correlation": phase_corr,
        "stereo_field": stereo_distribution
    }