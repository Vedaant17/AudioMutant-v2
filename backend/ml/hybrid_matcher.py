import os
import json
import numpy as np
from sklearn.neighbors import NearestNeighbors

from ml.embedding_extractor import EmbeddingExtractor


class HybridMatcher:

    def __init__(self, reference_folder=None):

        # ✅ Absolute path fix
        if reference_folder is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            reference_folder = os.path.join(base_dir, "..", "reference_data")

        self.reference_folder = os.path.abspath(reference_folder)

        self.feature_vectors = []
        self.embedding_vectors = []
        self.metadata = []

        # ✅ FIX 1: Always initialize
        self.embedding_extractor = EmbeddingExtractor()

        print(f"📚 Loading reference dataset from: {self.reference_folder}")
        self._load_references()

        print("🧠 Building similarity models...")
        self._build_models()

    # -------------------------
    # LOAD REFERENCES
    # -------------------------
    def _load_references(self):

        if not os.path.exists(self.reference_folder):
            raise FileNotFoundError(f"Reference folder not found: {self.reference_folder}")

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

                # Feature vector
                feature_vector = self._feature_to_vector(features)

                # Embedding
                embedding = features.get("embedding", None)

                if embedding is not None:
                    embedding = np.array(embedding, dtype=np.float32)

                self.feature_vectors.append(feature_vector)
                self.embedding_vectors.append(embedding)
                self.metadata.append(data)

        # ✅ Convert safely
        self.feature_vectors = np.array(self.feature_vectors, dtype=np.float32)

        # Handle empty embeddings safely
        if any(e is not None for e in self.embedding_vectors):
            self.embedding_vectors = np.array(
                [e if e is not None else np.zeros_like(self.embedding_vectors[0])
                 for e in self.embedding_vectors],
                dtype=np.float32
            )
        else:
            self.embedding_vectors = None

        print(f"✅ Loaded {len(self.feature_vectors)} reference tracks")

    # -------------------------
    # VECTOR BUILDER
    # -------------------------
    def _feature_to_vector(self, f):

        return np.array([
            f.get("tempo_bpm", 0),
            f.get("LUFS", 0),
            f.get("spectral_tilt", 0),
            f.get("low_mid_ratio", 0),
            f.get("mid_high_ratio", 0),
            f.get("transient_density", 0),
            f.get("stereo_width", 0),
            f.get("energy", {}).get("mean", 0),
            f.get("energy", {}).get("dynamic_range", 0),
        ], dtype=np.float32)

    # -------------------------
    # BUILD MODELS
    # -------------------------
    def _build_models(self):

        self.feature_model = NearestNeighbors(
            n_neighbors=min(3, len(self.feature_vectors)),
            metric="cosine"
        ).fit(self.feature_vectors)

        if self.embedding_vectors is not None:
            self.embedding_model = NearestNeighbors(
                n_neighbors=min(3, len(self.embedding_vectors)),
                metric="cosine"
            ).fit(self.embedding_vectors)
        else:
            self.embedding_model = None

    # -------------------------
    # MATCH FUNCTION
    # -------------------------
    def find_best_match(self, features, y, sr):

        # Feature vector
        input_vec = self._feature_to_vector(features).reshape(1, -1)

        # Feature similarity
        f_dist, f_idx = self.feature_model.kneighbors(input_vec)

        scores = {}

        for i, idx in enumerate(f_idx[0]):
            scores[idx] = scores.get(idx, 0) + (1 - f_dist[0][i]) * 0.7

        # -------------------------
        # EMBEDDING MATCH (SAFE)
        # -------------------------
        if self.embedding_model is not None:
            try:
                embedding_vec = self.embedding_extractor.extract_embedding(y, sr).reshape(1, -1)

                e_dist, e_idx = self.embedding_model.kneighbors(embedding_vec)

                for i, idx in enumerate(e_idx[0]):
                    scores[idx] = scores.get(idx, 0) + (1 - e_dist[0][i]) * 0.3

            except Exception as e:
                print("⚠️ Embedding failed, skipping:", e)

        # -------------------------
        # FINAL RESULT
        # -------------------------
        best_idx = max(scores, key=scores.get)

        return {
            "best": self.metadata[best_idx],
            "top_k": [self.metadata[i] for i in scores.keys()]
        }