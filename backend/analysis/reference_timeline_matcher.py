import os
import json
import librosa
import numpy as np


def load_references(genre):

    folder = f"reference_data/{genre}"
    refs = []

    for file in os.listdir(folder):
        if file.endswith(".json"):
            with open(os.path.join(folder, file), "r") as f:
                refs.append(json.load(f))

    return refs


def extract_section_features(y, sr, sections):

    features = []

    for sec in sections:
        start = int(sec["start"] * sr)
        end = int(sec["end"] * sr)

        segment = y[start:end]
        if len(segment) == 0:
            continue

        y_mono = librosa.to_mono(segment) if segment.ndim > 1 else segment

        rms = librosa.feature.rms(y=y_mono)[0]
        energy = float(np.mean(rms))

        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y_mono, sr=sr)))

        onset_env = librosa.onset.onset_strength(y=y_mono, sr=sr)
        transients = float(np.mean(onset_env))

        features.append({
            "energy": energy,
            "centroid": centroid,
            "transients": transients
        })

    return features

def build_reference_profile(refs):

    energies = []
    centroids = []
    transients = []

    for ref in refs:
        f = ref["features"]

        energies.append(f["energy"]["mean"])
        centroids.append(f["spectral_centroid"])
        transients.append(f["transient_density"])

    return {
        "energy": np.mean(energies),
        "centroid": np.mean(centroids),
        "transients": np.mean(transients)
    }

def compare_section(section_feat, ref_profile):

    results = []

    # -------------------------
    # ENERGY
    # -------------------------
    energy_diff = (section_feat["energy"] - ref_profile["energy"]) / ref_profile["energy"]

    if energy_diff < -0.2:
        results.append(f"Energy is {abs(int(energy_diff*100))}% lower than reference")

    # -------------------------
    # TRANSIENTS
    # -------------------------
    trans_diff = (section_feat["transients"] - ref_profile["transients"]) / ref_profile["transients"]

    if trans_diff < -0.2:
        results.append("Transients weaker than reference")

    # -------------------------
    # BRIGHTNESS
    # -------------------------
    bright_diff = (section_feat["centroid"] - ref_profile["centroid"]) / ref_profile["centroid"]

    if bright_diff < -0.2:
        results.append("Mix is darker than reference")

    elif bright_diff > 0.2:
        results.append("Mix is brighter than reference")

    return results


def reference_timeline_matcher(y, sr, sections, reference_sections):

    results = []

    for sec in sections:

        # Find matching type in reference
        ref_matches = [r for r in reference_sections if r["type"] == sec["type"]]

        if not ref_matches:
            continue

        ref_energy = np.mean([r["energy"] for r in ref_matches])

        diff = sec["energy"] - ref_energy
        percent = (diff / (ref_energy + 1e-6)) * 100

        results.append({
            "section": sec["type"],
            "your_energy": sec["energy"],
            "reference_energy": ref_energy,
            "difference_percent": percent,
            "message": generate_section_feedback(sec["type"], percent)
        })

    return results

def generate_section_feedback(section, percent):

    if percent < -15:
        return f"Your {section} is weaker than reference. Consider adding layers or increasing energy."

    elif percent > 20:
        return f"Your {section} is stronger than reference. Watch for over-compression or harshness."

    else:
        return f"Your {section} energy is well balanced."