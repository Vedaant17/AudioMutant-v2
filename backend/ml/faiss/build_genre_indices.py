import os
import json
import numpy as np
import faiss
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "reference_data")
FAISS_DIR = os.path.join(BASE_DIR, "ml", "faiss", "genres")

os.makedirs(FAISS_DIR, exist_ok=True)


def l2_normalize(vec):
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def build_feature_vector(features):
    base_vec = np.array([
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

    embedding = np.array(
        features.get("learned_embedding", []),
        dtype=np.float32,
    )

    return l2_normalize(np.concatenate([base_vec, embedding]))


def build_genre_indices():
    genre_vectors = defaultdict(list)
    genre_metadata = defaultdict(list)

    for root, _, files in os.walk(DATASET_DIR):
        for file in files:
            if not file.endswith(".json"):
                continue

            with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                track = json.load(f)

            genre = track.get("genre", "unknown")
            vec = build_feature_vector(track["features"])

            genre_vectors[genre].append(vec)
            genre_metadata[genre].append(track)

    for genre, vectors in genre_vectors.items():
        vectors = np.vstack(vectors).astype(np.float32)
        dimension = vectors.shape[1]

        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)

        faiss.write_index(
            index,
            os.path.join(FAISS_DIR, f"{genre}_index.faiss")
        )

        with open(
            os.path.join(FAISS_DIR, f"{genre}_metadata.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(genre_metadata[genre], f, indent=2)

        print(f"✅ Built genre index: {genre}")