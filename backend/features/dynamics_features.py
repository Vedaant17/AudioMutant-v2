import numpy as np
import librosa
import pyloudnorm as pyln


def extract_dynamics(y, sr):

    # -------------------------
    # MONO
    # -------------------------
    y_mono = librosa.to_mono(y) if y.ndim > 1 else y

    # -------------------------
    # RMS
    # -------------------------
    frame_length = 2048
    hop_length = 512

    rms = librosa.feature.rms(
        y=y_mono,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    # -------------------------
    # PEAK
    # -------------------------
    peak = float(np.max(np.abs(y_mono)))

    # -------------------------
    # LOUDNESS (LUFS)
    # -------------------------
    meter = pyln.Meter(sr)
    lufs = float(meter.integrated_loudness(y_mono.astype(np.float32)))

    # -------------------------
    # BASIC STATS
    # -------------------------
    mean_rms = float(np.mean(rms))
    dynamic_range = float(np.percentile(rms, 95) - np.percentile(rms, 5))

    # -------------------------
    # CREST FACTOR (GLOBAL)
    # -------------------------
    crest_factor = peak / (mean_rms + 1e-8)

    # -------------------------
    # CREST OVER TIME (FIXED)
    # -------------------------
    frames = librosa.util.frame(
        y_mono,
        frame_length=frame_length,
        hop_length=hop_length
    )

    frame_peak = np.max(np.abs(frames), axis=0)

    min_len = min(len(rms), len(frame_peak))

    frame_rms = rms[:min_len]
    frame_peak = frame_peak[:min_len]

    eps = 1e-6
    frame_rms = np.maximum(frame_rms, eps)
    frame_peak = np.maximum(frame_peak, eps)

    crest_over_time = frame_peak / frame_rms

    # 🔥 clip outliers (VERY IMPORTANT)
    crest_over_time = np.clip(crest_over_time, 0, 20)

    crest_features = {
        "mean": float(np.mean(crest_over_time)),
        "std": float(np.std(crest_over_time)),
        "max": float(np.max(crest_over_time)),
        "min": float(np.min(crest_over_time))
    }

    # -------------------------
    # LOUDNESS CURVE (UI SAFE)
    # -------------------------
    loudness_curve_small = rms[::100].tolist()

    # -------------------------
    # RETURN
    # -------------------------
    return {
        "lufs": lufs,
        "loudness_rms": mean_rms,
        "peak": peak,
        "dynamic_range": dynamic_range,
        "crest_factor": crest_factor,
        "crest_over_time": crest_features,
        "loudness_curve_small": loudness_curve_small
    }