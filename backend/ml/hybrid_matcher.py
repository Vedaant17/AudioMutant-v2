import os
import json
import numpy as np
from sklearn.neighbors import NearestNeighbors

from ml.embedding_extractor import EmbeddingExtractor


class HybridMatcher:

    def __init__(self, reference_folder=None):

        if reference_folder is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            reference_folder = os.path.join(base_dir, "..", "reference_data")

        self.reference_folder = os.path.abspath(reference_folder)

        self.feature_vectors = []
        self.embedding_vectors = []
        self.section_embeddings = []   # 🔥 NEW
        self.metadata = []

        self.embedding_extractor = EmbeddingExtractor()

        print(f"📚 Loading reference dataset from: {self.reference_folder}")
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

                # -------------------------
                # FEATURE VECTOR
                # -------------------------
                feature_vector = self._feature_to_vector(features)

                # -------------------------
                # FIXED EMBEDDING LOAD
                # -------------------------
                embedding = data.get("embedding", None)
                if embedding is not None:
                    embedding = np.array(embedding, dtype=np.float32)

                # -------------------------
                # SECTION EMBEDDINGS 🔥
                # -------------------------
                sections = data.get("sections", [])
                sec_embs = []

                for sec in sections:
                    emb = sec.get("embedding", None)
                    if emb is not None:
                        sec_embs.append(np.array(emb, dtype=np.float32))

                self.feature_vectors.append(feature_vector)
                self.embedding_vectors.append(embedding)
                self.section_embeddings.append(sec_embs)
                self.metadata.append(data)

        self.feature_vectors = np.array(self.feature_vectors, dtype=np.float32)

        # Handle embeddings safely
        if any(e is not None for e in self.embedding_vectors):
            first_valid = next(e for e in self.embedding_vectors if e is not None)

            self.embedding_vectors = np.array([
                e if e is not None else np.zeros_like(first_valid)
                for e in self.embedding_vectors
            ], dtype=np.float32)
        else:
            self.embedding_vectors = None

        print(f"✅ Loaded {len(self.metadata)} reference tracks")

    # -------------------------
    # FEATURE VECTOR BUILDER
    # -------------------------
    def _feature_to_vector(self, f):

        return np.array([
            f.get("tempo_bpm", 0),
            f.get("spectral_centroid", 0),
            f.get("stereo_width", 0),
            f.get("energy_mean", 0),
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
    # SECTION SIMILARITY 🔥
    # -------------------------
    def _section_similarity(self, input_sections):

        scores = {}

        for i, ref_sections in enumerate(self.section_embeddings):

            if not ref_sections:
                continue

            total_sim = 0
            count = 0

            for inp in input_sections:
                for ref in ref_sections:
                    sim = self._cosine_similarity(inp, ref)
                    total_sim += sim
                    count += 1

            if count > 0:
                scores[i] = total_sim / count

        return scores

    # -------------------------
    # COSINE SIM
    # -------------------------
    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)

    # -------------------------
    # MAIN MATCH FUNCTION
    # -------------------------
    def find_best_match(self, features, y, sr, sections=None):

        input_vec = self._feature_to_vector(features).reshape(1, -1)

        f_dist, f_idx = self.feature_model.kneighbors(input_vec)

        scores = {}

        # -------------------------
        # FEATURE SCORE (0.4)
        # -------------------------
        for i, idx in enumerate(f_idx[0]):
            scores[idx] = (1 - f_dist[0][i]) * 0.4

        # -------------------------
        # EMBEDDING SCORE (0.3)
        # -------------------------
        if self.embedding_model is not None:
            try:
                emb = self.embedding_extractor.extract_embedding(y, sr)

                if emb is not None:
                    emb = emb.reshape(1, -1)
                    e_dist, e_idx = self.embedding_model.kneighbors(emb)

                    for i, idx in enumerate(e_idx[0]):
                        scores[idx] = scores.get(idx, 0) + (1 - e_dist[0][i]) * 0.3

            except Exception as e:
                print("⚠️ Embedding failed:", e)

        # -------------------------
        # SECTION SCORE (0.2) 🔥
        # -------------------------
        if sections is not None and len(sections) > 0:
            sec_scores = self._section_similarity(sections)

            for idx, val in sec_scores.items():
                scores[idx] = scores.get(idx, 0) + val * 0.2

        # -------------------------
        # GENRE BOOST (0.1) 🔥
        # -------------------------
        input_genre = features.get("genre", None)

        if input_genre:
            for i, meta in enumerate(self.metadata):
                if meta.get("genre") == input_genre:
                    scores[i] = scores.get(i, 0) + 0.1

        # -------------------------
        # FINAL
        # -------------------------
        best_idx = max(scores, key=scores.get)

        return {
            "best": self.metadata[best_idx],
            "top_k": [self.metadata[i] for i in scores.keys()]
        }