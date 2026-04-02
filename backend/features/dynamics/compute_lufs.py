import numpy as np
import pyloudnorm as pyln
import librosa

def compute_lufs(y, sr):
    
    # -------------------------
    # MONO (important)
    # -------------------------
    if y.ndim > 1:
        y = librosa.to_mono(y)

    # -------------------------
    # RESAMPLE (recommended: 48k for accuracy)
    # -------------------------
    if sr != 48000:
        y = librosa.resample(y, orig_sr=sr, target_sr=48000)
        sr = 48000

    # -------------------------
    # CREATE METER
    # -------------------------
    meter = pyln.Meter(sr)  # EBU R128 meter

    # -------------------------
    # CALCULATE LUFS
    # -------------------------
    loudness = meter.integrated_loudness(y)

    return float(loudness)