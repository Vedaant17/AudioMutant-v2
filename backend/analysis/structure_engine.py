import librosa

def analyze_structure(y, sr):
    rms = librosa.feature.rms(y=y)[0]

    energy_curve = rms.tolist()

    return {
        "energy_curve": energy_curve
    }