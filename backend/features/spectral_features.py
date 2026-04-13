import numpy as np
import librosa
import pyloudnorm as pyln


def extract_spectral(y, sr):

    # -----------------------------
    # MONO FOR ANALYSIS
    # -----------------------------
    y_mono = librosa.to_mono(y) if y.ndim > 1 else y

    # -----------------------------
    # STFT
    # -----------------------------
    stft = np.abs(librosa.stft(y_mono, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

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
    # SPECTRAL TILT
    # -----------------------------
    log_freqs = np.log(freqs + 1e-8)
    log_energy = np.log(np.mean(stft, axis=1) + 1e-8)
    spectral_tilt = float(np.polyfit(log_freqs, log_energy, 1)[0])

    # -----------------------------
    # ZERO CROSSING RATE 🔥
    # -----------------------------
    zcr = librosa.feature.zero_crossing_rate(y_mono)
    zcr_mean = float(np.mean(zcr))

    # -----------------------------
    # RMS ENERGY 🔥
    # -----------------------------
    rms = librosa.feature.rms(y=y_mono)
    rms_mean = float(np.mean(rms))

    # -----------------------------
    # HARMONIC vs PERCUSSIVE 🔥🔥
    # -----------------------------
    y_harmonic, y_percussive = librosa.effects.hpss(y_mono)

    harmonic_ratio = float(np.mean(np.abs(y_harmonic)))
    percussive_ratio = float(np.mean(np.abs(y_percussive)))

    meter = pyln.Meter(sr)

    # Integrated LUFS
    integrated_lufs = meter.integrated_loudness(y_mono)

    # True peak
    true_peak = np.max(np.abs(y_mono))

    # -----------------------------
    # RETURN
    # -----------------------------
    return {
        # Core spectral
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff": rolloff,

        # Texture
        "spectral_flatness_mean": flatness_mean,
        "spectral_flatness_std": flatness_std,

        # Dynamics in spectrum
        "spectral_flux_mean": flux_mean,
        "spectral_flux_std": flux_std,

        # Tonal detail
        "spectral_contrast_mean": contrast_mean,
        "spectral_contrast_std": contrast_std,

        # Frequency distribution
        "frequency_balance": frequency_balance,
        "sub_bass_energy": sub_bass_energy,

        # Tonal slope
        "spectral_tilt": spectral_tilt,

        # 🔥 ML-critical features
        "zero_crossing_rate": zcr_mean,
        "rms": rms_mean,
        "harmonic_ratio": harmonic_ratio,
        "percussive_ratio": percussive_ratio,
        "integrated_lufs": float(integrated_lufs),
        "true_peak": float(true_peak)
    }