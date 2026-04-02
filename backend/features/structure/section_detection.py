import librosa
import numpy as np
from librosa.segment import agglomerative, recurrence_matrix


# -------------------------
# MAIN FUNCTION
# -------------------------
def detect_sections(y, sr, genre="general", embed_fn=None):

    # -------------------------
    # FEATURES
    # -------------------------
    y_mono = librosa.to_mono(y) if y.ndim > 1 else y

    chroma = librosa.feature.chroma_cqt(y=y_mono, sr=sr)
    rms = librosa.feature.rms(y=y_mono)[0]
    onset_env = librosa.onset.onset_strength(y=y_mono, sr=sr)

    # -------------------------
    # BEAT TRACKING
    # -------------------------
    tempo, beats = librosa.beat.beat_track(y=y_mono, sr=sr)

    if len(beats) < 2:
        # fallback: entire track as one section
        duration = len(y_mono) / sr
        return [{
            "start": 0.0,
            "end": duration,
            "type": "full",
            "energy": float(np.mean(rms))
        }]

    # -------------------------
    # CREATE COARSE SEGMENTS
    # -------------------------
    # Split beats into ~6 groups
    num_segments = min(6, len(beats) - 1)
    split_points = np.linspace(0, len(beats) - 1, num_segments + 1, dtype=int)

    sections = []

    for i in range(len(split_points) - 1):
        start_beat = beats[split_points[i]]
        end_beat = beats[split_points[i + 1]]

        start_time = float(librosa.frames_to_time(start_beat, sr=sr))
        end_time = float(librosa.frames_to_time(end_beat, sr=sr))

        start_frame = start_beat
        end_frame = end_beat

        if end_frame <= start_frame:
            continue

        segment_rms = rms[start_frame:end_frame]
        segment_flux = onset_env[start_frame:end_frame]
        if len(segment_rms) == 0:
            continue

        energy = float(np.mean(segment_rms))
        flux = float(np.mean(segment_flux))

        if np.isnan(energy):
            energy = 0.0

        sections.append({
            "start": start_time,
            "end": end_time,
            "energy": energy,
            "flux": flux
        })

    # -------------------------
    # 🔥 MIN DURATION FILTER (SAFE)
    # -------------------------
    MIN_DURATION = 3.0  # start smaller than 5

    filtered = []
    for sec in sections:
        if (sec["end"] - sec["start"]) >= MIN_DURATION:
            filtered.append(sec)

    # fallback if everything removed
    if len(filtered) == 0:
        duration = len(y_mono) / sr
        filtered = [{
            "start": 0.0,
            "end": duration,
            "energy": float(np.mean(rms)),
            "flux": float(np.mean(onset_env))
        }]

    sections = filtered

    sections[0]["start"] = 0.0

    # -------------------------
    # LABELING
    # -------------------------
    sections = label_sections_genre(sections, genre)

# -------------------------
# 🔥 ADD IDS + EMBEDDINGS
# -------------------------
    final_sections = []

    for idx, sec in enumerate(sections):

        start_sample = int(sec["start"] * sr)
        end_sample = int(sec["end"] * sr)

        if end_sample <= start_sample:
            continue

        segment = y[:, start_sample:end_sample] if y.ndim > 1 else y[start_sample:end_sample]

        if len(segment) == 0:
            continue

        embedding = None
        if embed_fn is not None:
            try:
                embedding = embed_fn(segment, sr)
            except:
                embedding = None

        final_sections.append({
        "id": idx,
        "type": sec["type"],
        "start": sec["start"],
        "end": sec["end"],
        "energy": sec["energy"],
        "embedding": embedding
        })

    return final_sections
# -------------------------
# GENRE-AWARE LABELING
# -------------------------
def label_sections_genre(sections, genre):

    if len(sections) == 0:
        return []

    energies = np.array([s["energy"] for s in sections])
    fluxes = np.array([s["flux"] for s in sections])

    max_energy = np.max(energies) if len(energies) else 1.0
    mean_energy = np.mean(energies) if len(energies) else 0.0

    energy_diff = np.diff(energies, prepend=energies[0])

    labeled = []

    for i, sec in enumerate(sections):

        e = sec["energy"]
        f = sec["flux"]
        diff = energy_diff[i]

        if genre == "edm":

            if i == 0:
                label = "intro"

            elif i == len(sections) - 1:
                label = "outro"

            elif diff > 0.25 * max_energy:
                label = "drop"

            elif e < 0.4 * max_energy:
                label = "breakdown"

            elif e < 0.01:
                label = "silence"

            elif f > np.mean(fluxes) and e > mean_energy:
                label = "build"

            elif e > 0.75 * max_energy:
                label = "drop"

            else:
                label = "build"

        elif genre == "hiphop":

            if i == 0:
                label = "intro"
            elif i == len(sections) - 1:
                label = "outro"
            elif e > 0.7 * max_energy:
                label = "hook"
            else:
                label = "verse"

        elif genre == "rock":

            if i == 0:
                label = "intro"
            elif i == len(sections) - 1:
                label = "outro"
            elif e > 0.75 * max_energy:
                label = "chorus"
            elif e < 0.5 * max_energy:
                label = "verse"
            elif diff > 0.2 * max_energy:
                label = "chorus"
            else:
                label = "bridge"

        else:

            if i == 0:
                label = "intro"
            elif i == len(sections) - 1:
                label = "outro"
            elif e > 0.75 * max_energy:
                label = "high_energy"
            elif e < 0.4 * max_energy:
                label = "low_energy"
            else:
                label = "mid_energy"

        labeled.append({
            "start": sec["start"],
            "end": sec["end"],
            "type": label,
            "energy": e
        })

    return labeled