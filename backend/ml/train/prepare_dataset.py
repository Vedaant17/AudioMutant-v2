import os
import json
import numpy as np

DATASET_PATH = "reference_data"

track_X = []
track_y = []

section_X = []
section_y = []


# -------------------------
# TRACK FEATURES (9 dims)
# -------------------------
def build_feature_vector(f):
    return np.array([
        f.get("tempo_bpm", 0),
        f.get("mid_energy", 0),
        f.get("side_ratio", 0),
        f.get("integrated_lufs", 0),
        f.get("kick_punch", 0),
        f.get("transient_strength", 0),
        f.get("energy_mean", 0),
        f.get("stereo_width", 0),
        f.get("spectral_centroid", 0),
    ], dtype=np.float32)


# -------------------------
# SECTION FEATURES (4 dims)
# -------------------------
def build_section_vector(s):
    return np.array([
        s.get("kick_punch", 0),
        s.get("mid_energy", 0),
        s.get("side_ratio", 0),
        s.get("lufs", 0),
    ], dtype=np.float32)


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# -------------------------
# LOAD DATASET
# -------------------------
for root, _, files in os.walk(DATASET_PATH):
    for file in files:
        if not file.endswith(".json"):
            continue

        path = os.path.join(root, file)

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except:
            continue

        features = data.get("features", {})
        sections = data.get("sections", [])

        if not features:
            continue

        # -------------------------
        # TRACK
        # -------------------------
        vec = build_feature_vector(features)
        noise = np.random.normal(0, 0.05, size=vec.shape)
        input_vec = vec + noise
        similarity = cosine_similarity(input_vec, vec)

        track_X.append(input_vec)
        track_y.append(similarity)  # placeholder label

        # -------------------------
        # SECTIONS
        # -------------------------
        for sec in sections:
            sec_vec = build_section_vector(sec)
            noise = np.random.normal(0, 0.05, size=sec_vec.shape)
            input_sec = sec_vec + noise

            sim = cosine_similarity(input_sec, sec_vec)

            if len(sec_vec) == 0:
                continue

            section_X.append(input_sec)
            section_y.append(sim)


# -------------------------
# FINALIZE
# -------------------------
track_X = np.array(track_X)
track_y = np.array(track_y)

section_X = np.array(section_X)
section_y = np.array(section_y)

print("Track dataset:", track_X.shape)
print("Section dataset:", section_X.shape)

np.save("ml/train/X.npy", track_X)
np.save("ml/train/y.npy", track_y)

np.save("ml/train/section_X.npy", section_X)
np.save("ml/train/section_y.npy", section_y)

print("✅ Dataset saved!")

# -------------------------
# NORMALIZATION STATS
# -------------------------
track_mean = np.mean(track_X, axis=0)
track_std = np.std(track_X, axis=0) + 1e-6

section_mean = np.mean(section_X, axis=0)
section_std = np.std(section_X, axis=0) + 1e-6

np.save("ml/train/track_mean.npy", track_mean)
np.save("ml/train/track_std.npy", track_std)

np.save("ml/train/section_mean.npy", section_mean)
np.save("ml/train/section_std.npy", section_std)

print("✅ Normalization stats saved!")