import numpy as np
import librosa

def analyze_drums(y, sr):
    # Convert to mono for analysis
    y_mono = librosa.to_mono(y) if y.ndim > 1 else y

    # Compute STFT properly
    stft = np.abs(librosa.stft(y_mono, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    def band_energy(low, high):
        idx = np.where((freqs >= low) & (freqs <= high))[0]
        if len(idx) == 0:
            return 0.0
        return float(np.mean(stft[idx]))

    return {
        "kick_strength": band_energy(20, 120),
        "snare_strength": band_energy(120, 2500),
        "hihat_strength": band_energy(5000, 10000),
    }