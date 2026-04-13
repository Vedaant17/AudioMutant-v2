import numpy as np
import librosa
import pyloudnorm as pyln


def extract_section_loudness(y, sr, sections):
    """
    Returns list of LUFS values per section
    """

    meter = pyln.Meter(sr)
    section_loudness = []

    for i, sec in enumerate(sections):
        try:
            start = int(sec["start"] * sr)
            end = int(sec["end"] * sr)

            y_sec = y[:, start:end] if y.ndim > 1 else y[start:end]

            if y_sec is None or len(y_sec) == 0:
                section_loudness.append(None)
                continue

            # Convert to mono
            if y_sec.ndim > 1:
                y_sec = librosa.to_mono(y_sec)

            # LUFS calculation
            lufs = meter.integrated_loudness(y_sec)

            if np.isnan(lufs) or np.isinf(lufs):
                section_loudness.append(None)
            else:
                section_loudness.append(float(lufs))

        except Exception as e:
            print(f"⚠️ Section LUFS failed at index {i}: {e}")
            section_loudness.append(None)

    return section_loudness