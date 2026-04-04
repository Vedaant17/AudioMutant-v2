import math

import librosa
import numpy as np
import pyloudnorm as pyln
import json
import os


# ==================================================
# 🔹 CORE FEATURE EXTRACTION
# ==================================================
def extract_base(y, sr):

    # -----------------------------
    # MONO / STEREO
    # -----------------------------
    if y.ndim > 1:
        y_mono = librosa.to_mono(y)
        left, right = y[0], y[1]
    else:
        y_mono = y
        left = right = y

    # -----------------------------
    # HPSS (Harmonic / Percussive)
    # -----------------------------
    y_harmonic, y_percussive = librosa.effects.hpss(y_mono)

    harmonic_energy = float(np.mean(np.abs(y_harmonic)))
    percussive_energy = float(np.mean(np.abs(y_percussive)))
    hpr = harmonic_energy / (percussive_energy + 1e-8)

    # -----------------------------
    # TEMPO
    # -----------------------------
    tempo, beat_frames = librosa.beat.beat_track(y=y_mono, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    tempo_stability = (
        float(np.std(np.diff(beat_times)))
        if len(beat_times) > 1 else 0.0
    )

    # -----------------------------
    # KEY DETECTION
    # -----------------------------
    chroma = librosa.feature.chroma_cqt(y=y_mono, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    key_idx = chroma_mean.argmax()
    key_signature = f"{notes[key_idx]} major"

    # -----------------------------
    # STFT (ONLY FOR DERIVED FEATURES)
    # -----------------------------
    stft = np.abs(librosa.stft(y_mono))
    freqs = librosa.fft_frequencies(sr=sr)

    # Spectral tilt
    log_freqs = np.log(freqs + 1e-8)
    log_energy = np.log(np.mean(stft, axis=1) + 1e-8)
    spectral_tilt = float(np.polyfit(log_freqs, log_energy, 1)[0])

    # -----------------------------
    # FREQUENCY BALANCE
    # -----------------------------
    low = np.mean(stft[(freqs >= 20) & (freqs < 250)])
    mid = np.mean(stft[(freqs >= 250) & (freqs < 4000)])
    high = np.mean(stft[(freqs >= 4000)])

    # -----------------------------
    # TRANSIENTS
    # -----------------------------
    onset_env = librosa.onset.onset_strength(y=y_mono, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env)

    duration = librosa.get_duration(y=y_mono, sr=sr)
    transient_density = float(len(onsets) / duration)

    # Groove swing
    swing = (
        float(np.std(np.diff(beat_times)))
        if len(beat_times) > 1 else 0.0
    )

    # -----------------------------
    # STRUCTURE SEGMENTS (LIGHTWEIGHT)
    # -----------------------------
    mfcc = librosa.feature.mfcc(y=y_mono, sr=sr)
    segments = librosa.segment.agglomerative(mfcc.T, k=6)

    # -----------------------------
    # STEREO
    # -----------------------------
    if y.ndim > 1:
        corr = np.corrcoef(left, right)[0, 1]
        stereo_width = float(1 - corr)
        phase_corr = float(corr)
    else:
        stereo_width = 0.0
        phase_corr = 1.0

    # -----------------------------
    # RETURN (CLEAN)
    # -----------------------------
    return {
        "tempo_bpm": float(tempo),
        "tempo_stability": tempo_stability,
        "key_signature": key_signature,

        "harmonic_energy": harmonic_energy,
        "percussive_energy": percussive_energy,
        "harmonic_percussive_ratio": hpr,

        "spectral_tilt": spectral_tilt,

        "frequency_balance": {
            "low": float(low),
            "mid": float(mid),
            "high": float(high)
        },

        "transient_density": transient_density,
        "groove_swing": swing,

        "structure_segments": segments.tolist() if segments is not None else [],

        "stereo_width": stereo_width,
        "phase_correlation": phase_corr
    }
# ==================================================
# 🔹 CLEAN NUMPY TYPES (CRITICAL)
# ==================================================
def clean_features(obj):
    if isinstance(obj, np.generic):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj.item()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, dict):
        return {k: clean_features(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [clean_features(v) for v in obj]

    return obj


# ==================================================
# 🔹 PRINT JSON (DEBUG TOOL)
# ==================================================
def print_features_json(features):
    cleaned = clean_features(features)

    print("\n🎧 Extracted Features:\n")
    print(json.dumps(cleaned, indent=4))

    return cleaned


# ==================================================
# 🔹 SAVE REFERENCE TRACK (V2 VERSION)
# ==================================================
def save_reference(track_name, artist, genre, data):

    # ✅ Clean ONLY the features part (if needed)
    if "features" in data:
        data["features"] = clean_features(data["features"])

    # 📁 Save path
    folder = os.path.join("reference_data", genre)
    os.makedirs(folder, exist_ok=True)

    filename = f"{track_name.lower().replace(' ', '_')}.json"
    path = os.path.join(folder, filename)

    # 💾 Save FULL object (no restructuring)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"✅ Saved reference: {path}")

    return path