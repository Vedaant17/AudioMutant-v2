import os
import json
import numpy as np
from sklearn.neighbors import NearestNeighbors

from ml.embedding_extractor import EmbeddingExtractor


class HybridMatcher:

    def __init__(self, reference_folder="reference_data"):

        self.reference_folder = reference_folder
        self.feature_vectors = []
        self.embedding_vectors = []
        self.metadata = []

        self.embedding_extractor = EmbeddingExtractor()

        print("📚 Loading reference dataset...")
        self._load_references()

        print("🧠 Building similarity models...")
        self._build_models()

    # -------------------------
    # LOAD REFERENCES
    # -------------------------
    def _load_references(self):

        for genre in os.listdir(self.reference_folder):
            genre_path = os.path.join(self.reference_folder, genre)

            if not os.path.isdir(genre_path):
                continue

            for file in os.listdir(genre_path):
                if not file.endswith(".json"):
                    continue

                path = os.path.join(genre_path, file)

                with open(path, "r") as f:
                    data = json.load(f)

                features = data["features"]

                # 👉 Convert to vector
                feature_vector = self._feature_to_vector(features)

                # 👉 Load embedding (if exists)
                embedding = features.get("embedding", None)

                if embedding is not None:
                    embedding = np.array(embedding)

                self.feature_vectors.append(feature_vector)
                self.embedding_vectors.append(embedding)
                self.metadata.append(data)

        self.feature_vectors = np.array(self.feature_vectors)
        self.embedding_vectors = np.array(self.embedding_vectors)

    # -------------------------
    # VECTOR BUILDER
    # -------------------------
    def _feature_to_vector(self, f):

        return np.array([
            f["tempo_bpm"],
            f["LUFS"],
            f["spectral_tilt"],
            f["low_mid_ratio"],
            f["mid_high_ratio"],
            f["transient_density"],
            f["stereo_width"],
            f["energy"]["mean"],
            f["energy"]["dynamic_range"],
        ])

    # -------------------------
    # BUILD MODELS
    # -------------------------
    def _build_models(self):

        self.feature_model = NearestNeighbors(
            n_neighbors=3,
            metric="cosine"
        ).fit(self.feature_vectors)

        self.embedding_model = NearestNeighbors(
            n_neighbors=3,
            metric="cosine"
        ).fit(self.embedding_vectors)

    # -------------------------
    # MATCH FUNCTION
    # -------------------------
    def find_best_match(self, features, y, sr):

        # Feature vector
        input_vec = self._feature_to_vector(features).reshape(1, -1)

        # Embedding vector
        embedding_vec = self.embedding_extractor.extract_embedding(y, sr).reshape(1, -1)

        # -------------------------
        # FEATURE MATCH
        # -------------------------
        f_dist, f_idx = self.feature_model.kneighbors(input_vec)

        # -------------------------
        # EMBEDDING MATCH
        # -------------------------
        e_dist, e_idx = self.embedding_model.kneighbors(embedding_vec)

        # -------------------------
        # COMBINE SCORES
        # -------------------------
        scores = {}

        for i, idx in enumerate(f_idx[0]):
            scores[idx] = scores.get(idx, 0) + (1 - f_dist[0][i]) * 0.6

        for i, idx in enumerate(e_idx[0]):
            scores[idx] = scores.get(idx, 0) + (1 - e_dist[0][i]) * 0.4

        # Sort
        best_idx = max(scores, key=scores.get)

        return {
            "best_match": self.metadata[best_idx],
            "score": scores[best_idx]
        }