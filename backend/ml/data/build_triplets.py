import numpy as np
import os
import json
import random

DATASET_PATH = "reference_data"

data = []

# -------------------------
# LOAD DATA
# -------------------------
for root, _, files in os.walk(DATASET_PATH):
    for file in files:
        if not file.endswith(".json"):
            continue

        path = os.path.join(root, file)

        with open(path, "r") as f:
            d = json.load(f)

        features = d.get("features", {})
        genre = os.path.basename(root)

        vec = np.array([
            features.get("tempo_bpm", 0),
            features.get("mid_energy", 0),
            features.get("side_ratio", 0),
            features.get("integrated_lufs", 0),
            features.get("kick_punch", 0),
            features.get("transient_strength", 0),
            features.get("energy_mean", 0),
            features.get("stereo_width", 0),
            features.get("spectral_centroid", 0),
        ], dtype=np.float32)

        data.append((vec, genre))


# -------------------------
# BUILD TRIPLETS
# -------------------------
anchor_X = []
pos_X = []
neg_X = []

for i in range(len(data)):
    anchor_vec, anchor_genre = data[i]

    # positive = same genre
    pos_candidates = [d for d in data if d[1] == anchor_genre and not np.array_equal(d[0], anchor_vec)]
    if not pos_candidates:
        continue
    pos_vec = random.choice(pos_candidates)[0]

    # negative = different genre
    neg_candidates = [d for d in data if d[1] != anchor_genre]
    neg_vec = random.choice(neg_candidates)[0]

    anchor_X.append(anchor_vec)
    pos_X.append(pos_vec)
    neg_X.append(neg_vec)


# -------------------------
# SAVE
# -------------------------
np.save("ml/train/anchor_X.npy", np.array(anchor_X))
np.save("ml/train/pos_X.npy", np.array(pos_X))
np.save("ml/train/neg_X.npy", np.array(neg_X))

print("✅ Triplets saved!")