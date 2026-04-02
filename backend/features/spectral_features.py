import numpy as np
import librosa


def extract_spectral(y, sr):

    # -----------------------------
    # MONO FOR ANALYSIS
    # -----------------------------
    y_mono = librosa.to_mono(y) if y.ndim > 1 else y

    # -----------------------------
    # STFT
    # -----------------------------
    stft = np.abs(librosa.stft(y_mono, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr)

    # -----------------------------
    # BASIC SPECTRAL FEATURES
    # -----------------------------
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y_mono, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y_mono, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y_mono, sr=sr)))

    # -----------------------------
    # SPECTRAL FLATNESS
    # -----------------------------
    flatness = librosa.feature.spectral_flatness(y=y_mono)
    flatness_mean = float(np.mean(flatness))
    flatness_std = float(np.std(flatness))

    # -----------------------------
    # SPECTRAL FLUX
    # -----------------------------
    spectral_flux = np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0))
    flux_mean = float(np.mean(spectral_flux))
    flux_std = float(np.std(spectral_flux))

    # -----------------------------
    # SPECTRAL CONTRAST
    # -----------------------------
    contrast = librosa.feature.spectral_contrast(y=y_mono, sr=sr)
    spectral_contrast = {
        "mean": float(np.mean(contrast)),
        "std": float(np.std(contrast))
    }
    contrast_mean = np.mean(contrast, axis=1).tolist()
    contrast_std = np.std(contrast, axis=1).tolist()

    # -----------------------------
    # FREQUENCY BALANCE
    # -----------------------------
    low_idx = np.where((freqs >= 20) & (freqs < 250))
    mid_idx = np.where((freqs >= 250) & (freqs < 4000))
    high_idx = np.where(freqs >= 4000)

    low_energy = float(np.mean(stft[low_idx]))
    mid_energy = float(np.mean(stft[mid_idx]))
    high_energy = float(np.mean(stft[high_idx]))

    frequency_balance = {
        "low": low_energy,
        "mid": mid_energy,
        "high": high_energy
    }

    # -----------------------------
    # SUB BASS
    # -----------------------------
    sub_idx = np.where(freqs < 60)
    sub_bass_energy = float(np.mean(stft[sub_idx]))

    # -----------------------------
    # SPECTRAL TILT (IMPORTANT)
    # -----------------------------
    log_freqs = np.log(freqs + 1e-8)
    log_energy = np.log(np.mean(stft, axis=1) + 1e-8)

    spectral_tilt = float(np.polyfit(log_freqs, log_energy, 1)[0])

    return {
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff": rolloff,
        "spectral_flatness_mean": flatness_mean,
        "spectral_flatness_std": flatness_std,

        "spectral_flux_mean": flux_mean,
        "spectral_flux_std": flux_std,

        "spectral_contrast_mean": contrast_mean,
        "spectral_contrast_std": contrast_std,

        "frequency_balance": frequency_balance,
        "sub_bass_energy": sub_bass_energy,

        "spectral_tilt": spectral_tilt,

        # 🔥 IMPORTANT for other modules
        #"stft": stft,
        #"freqs": freqs
    }