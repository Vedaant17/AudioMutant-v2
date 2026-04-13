import numpy as np
import librosa


def extract_transient_features(y, sr):

    # -----------------------------
    # MONO
    # -----------------------------
    y_mono = librosa.to_mono(y) if y.ndim > 1 else y

    # -----------------------------
    # ONSET STRENGTH (core transient signal)
    # -----------------------------
    onset_env = librosa.onset.onset_strength(y=y_mono, sr=sr)

    # -----------------------------
    # BASIC TRANSIENT METRICS
    # -----------------------------
    transient_strength = float(np.mean(onset_env))
    transient_variation = float(np.std(onset_env))

    # -----------------------------
    # ATTACK SHARPNESS
    # -----------------------------
    # Measure how steep peaks are
    try:
        peaks = librosa.util.peak_pick(
            onset_env,
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=5,
            delta=0.5,
            wait=5
        )
    except TypeError:
        peaks = librosa.util.peak_pick(onset_env, 3, 3, 3, 5, 0.5, 5)

    if len(peaks) > 1:
        peak_diffs = np.diff(onset_env[peaks])
        attack_sharpness = float(np.mean(np.abs(peak_diffs)))
    else:
        attack_sharpness = 0.0

    # -----------------------------
    # KICK PUNCH (LOW FREQ TRANSIENTS)
    # -----------------------------
    S = np.abs(librosa.stft(y_mono))
    freqs = librosa.fft_frequencies(sr=sr)

    low_band = S[(freqs >= 20) & (freqs < 150)]
    low_energy = np.mean(low_band, axis=0)

    kick_punch = float(np.mean(np.abs(np.diff(low_energy))))

    return {
        "transient_strength": transient_strength,
        "transient_variation": transient_variation,
        "attack_sharpness": attack_sharpness,
        "kick_punch": kick_punch
    }