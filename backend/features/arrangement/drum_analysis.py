import numpy as np
import librosa

def analyze_drums(y, sr):

    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    def band_energy(low, high):
        idx = np.where((freqs >= low) & (freqs <= high))
        return float(np.mean(stft[idx]))

    return {
        "kick_strength": band_energy(20, 120),
        "snare_presence": band_energy(150, 250),
        "hihat_energy": band_energy(5000, 10000)
    }