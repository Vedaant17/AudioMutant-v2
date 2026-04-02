import librosa
import numpy as np
from features.harmony import detect_key, detect_chords

def analyze_harmony(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    key = detect_key(chroma)
    chords = detect_chords(chroma, sr)

    return {
        "key": key["key"],
        "confidence": key["confidence"],
        "chords": chords
    }