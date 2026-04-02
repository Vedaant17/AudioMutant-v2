import numpy as np
import librosa


def get_sections(y, sr, k=6):

    y_mono = librosa.to_mono(y) if y.ndim > 1 else y

    # MFCC for structure
    mfcc = librosa.feature.mfcc(y=y_mono, sr=sr)

    segments = librosa.segment.agglomerative(mfcc.T, k=k)

    # Convert to time
    boundaries = []
    prev = segments[0]

    for i, seg in enumerate(segments):
        if seg != prev:
            boundaries.append(i)
            prev = seg

    boundaries = [0] + boundaries + [len(segments)]

    times = librosa.frames_to_time(boundaries, sr=sr)

    sections = []
    for i in range(len(times) - 1):
        sections.append({
            "start": float(times[i]),
            "end": float(times[i + 1])
        })

    return sections

def analyze_section(y, sr, start, end):

    start_sample = int(start * sr)
    end_sample = int(end * sr)

    segment = y[start_sample:end_sample]

    if len(segment) == 0:
        return None

    y_mono = librosa.to_mono(segment) if segment.ndim > 1 else segment

    # ENERGY
    rms = librosa.feature.rms(y=y_mono)[0]
    energy = float(np.mean(rms))

    # LOUDNESS APPROX
    peak = float(np.max(np.abs(y_mono)))

    # SPECTRAL BALANCE
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y_mono, sr=sr)))

    # TRANSIENTS
    onset_env = librosa.onset.onset_strength(y=y_mono, sr=sr)
    transients = float(np.mean(onset_env))

    return {
        "energy": energy,
        "peak": peak,
        "centroid": centroid,
        "transients": transients
    }

def detect_section_issues(section_features, global_mix):

    issues = []

    energy = section_features["energy"]
    transients = section_features["transients"]
    centroid = section_features["centroid"]

    avg_energy = global_mix.get("avg_energy", energy)
    avg_transients = global_mix.get("avg_transients", transients)

    # -------------------------
    # LOW ENERGY (WEAK SECTION)
    # -------------------------
    if energy < avg_energy * 0.6:
        issues.append({
            "type": "low_energy",
            "message": "Section lacks energy",
            "fix": "Add layers, drums, or automation"
        })

    # -------------------------
    # LOW PUNCH
    # -------------------------
    if transients < avg_transients * 0.6:
        issues.append({
            "type": "low_punch",
            "message": "Weak transients",
            "fix": "Add transient shaping or reduce compression"
        })

    # -------------------------
    # DULL SECTION
    # -------------------------
    if centroid < 1500:
        issues.append({
            "type": "dull_mix",
            "message": "Lacks high frequency content",
            "fix": "Boost 5–10kHz or add brightness"
        })

    return issues

def label_section(current, prev):

    if prev is None:
        return "intro"

    if current["energy"] > prev["energy"] * 1.3:
        return "drop"

    if current["energy"] < prev["energy"] * 0.7:
        return "breakdown"

    return "build"

def detect_timeline_issues(y, sr, stft, freqs, mix):

    sections = get_sections(y, sr)

    results = []

    section_features_list = []

    # -------------------------
    # ANALYZE ALL SECTIONS
    # -------------------------
    for sec in sections:
        features = analyze_section(y, sr, sec["start"], sec["end"])
        section_features_list.append(features)

    # -------------------------
    # GLOBAL AVERAGES
    # -------------------------
    energies = [f["energy"] for f in section_features_list if f]
    transients = [f["transients"] for f in section_features_list if f]

    global_mix = {
        "avg_energy": np.mean(energies),
        "avg_transients": np.mean(transients)
    }

    # -------------------------
    # DETECT ISSUES PER SECTION
    # -------------------------
    prev = None

    for i, sec in enumerate(sections):

        features = section_features_list[i]
        if not features:
            continue

        issues = detect_section_issues(features, global_mix)

        label = label_section(features, prev)

        results.append({
            "start": sec["start"],
            "end": sec["end"],
            "type": label,
            "issues": issues
        })

        prev = features

    return results