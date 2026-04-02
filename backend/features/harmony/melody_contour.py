import librosa
import numpy as np


def extract_melody_contour(y, sr):

    # -----------------------------
    # PITCH TRACKING (PYIN)
    # -----------------------------
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7")
    )

    # -----------------------------
    # CLEAN DATA
    # -----------------------------
    f0_clean = np.nan_to_num(f0)

    # Remove unvoiced frames
    voiced_f0 = f0_clean[voiced_flag] if voiced_flag is not None else f0_clean

    if len(voiced_f0) == 0:
        return {
            "melody_present": False
        }

    # -----------------------------
    # CONVERT TO MUSICAL SCALE (MIDI)
    # -----------------------------
    midi = librosa.hz_to_midi(voiced_f0)

    # -----------------------------
    # SUMMARY FEATURES
    # -----------------------------
    melody_mean = float(np.mean(midi))
    melody_std = float(np.std(midi))
    melody_range = float(np.max(midi) - np.min(midi))

    # -----------------------------
    # DOWNSAMPLED CONTOUR (UI SAFE)
    # -----------------------------
    contour_small = midi[::50]  # reduce size

    return {
        "melody_present": True,
        "melody_mean_pitch": melody_mean,
        "melody_pitch_std": melody_std,
        "melody_range": melody_range,
        "melody_contour_small": contour_small.tolist()
    }