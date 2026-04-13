import numpy as np
import librosa
from features.dynamics.section_loudness import extract_section_loudness


def extract_section_embeddings(y, sr, sections, extractor):

    section_data = []

    # 🔥 NEW: compute LUFS first
    section_lufs = extract_section_loudness(y, sr, sections)

    for i, sec in enumerate(sections):
        try:
            start = int(sec["start"] * sr)
            end = int(sec["end"] * sr)

            y_sec = y[:, start:end] if y.ndim > 1 else y[start:end]

            if y_sec is None or len(y_sec) == 0:
                continue

            # Convert to mono
            if y_sec.ndim > 1:
                y_sec = librosa.to_mono(y_sec)

            emb = extractor.extract_embedding(y_sec, sr)

            if emb is None or np.isnan(emb).any():
                continue

            section_data.append({
                "type": sec.get("type"),
                "start": sec.get("start"),
                "end": sec.get("end"),
                "lufs": section_lufs[i],   # 🔥 NEW
                "embedding": emb.tolist()
            })

        except Exception as e:
            print(f"⚠️ Section embedding failed: {e}")
            continue

    return section_data