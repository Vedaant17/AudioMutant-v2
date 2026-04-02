import numpy as np
import librosa

def analyze_arrangement(y, sr):
    harmonic, percussive = librosa.effects.hpss(y)

    harmonic_energy = np.mean(np.abs(harmonic))
    percussive_energy = np.mean(np.abs(percussive))

    return {
        "harmonic_energy": float(harmonic_energy),
        "percussive_energy": float(percussive_energy),
        "ratio": float(harmonic_energy / (percussive_energy + 1e-6))
    }