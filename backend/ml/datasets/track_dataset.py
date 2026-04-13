import os
import json
import numpy as np

def build_track_dataset(reference_folder):

    X, y = [], []

    for genre in os.listdir(reference_folder):
        gpath = os.path.join(reference_folder, genre)

        for file in os.listdir(gpath):
            if not file.endswith(".json"):
                continue

            with open(os.path.join(gpath, file)) as f:
                data = json.load(f)

            fvec = np.array(list(data["features"].values()), dtype=np.float32)
            emb = np.array(data["embedding"], dtype=np.float32)

            x = np.concatenate([fvec, emb])

            score = np.clip(
                data["features"].get("kick_punch", 0) / 10 +
                data["features"].get("energy_mean", 0),
                0, 1
            )

            X.append(x)
            y.append(score)

    return np.array(X), np.array(y)