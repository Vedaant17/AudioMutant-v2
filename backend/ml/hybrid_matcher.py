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

                feature_vector = self._feature_to_vector(features)

                embedding = data.get("embedding", None)
                if embedding is not None:
                    embedding = np.array(embedding, dtype=np.float32)

                self.feature_vectors.append(feature_vector)
                self.embedding_vectors.append(embedding)
                self.metadata.append(data)

        self.feature_vectors = np.array(self.feature_vectors, dtype=np.float32)

        # Fix missing embeddings
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
    # FEATURE VECTOR
    # -------------------------
    def _feature_to_vector(self, f):

        return np.array([
            f.get("tempo_bpm", 0) / 200,
            f.get("spectral_centroid", 0) / 8000,
            f.get("stereo_width", 0),
            f.get("energy_mean", 0),
            f.get("dynamics_range", 0) / 60,

            f.get("spectral_bandwidth", 0) / 8000,
            f.get("spectral_rolloff", 0) / 8000,
            f.get("zero_crossing_rate", 0),
            f.get("rms", 0),
            f.get("harmonic_ratio", 0),
            f.get("percussive_ratio", 0),
            f.get("integrated_lufs", 0),
            f.get("true_peak", 0),
            f.get("transient_strength", 0) / 5,
            f.get("attack_sharpness", 0) / 5,
            f.get("kick_punch", 0) / 10,
            f.get("side_ratio", 0),
            f.get("mid_side_balance", 0),
        ], dtype=np.float32)

    # -------------------------
    # BUILD MODELS
    # -------------------------
    def _build_models(self):

        self.feature_model = NearestNeighbors(
            n_neighbors=min(5, len(self.feature_vectors)),
            metric="cosine"
        ).fit(self.feature_vectors)

        if self.embedding_vectors is not None:
            self.embedding_model = NearestNeighbors(
                n_neighbors=min(5, len(self.embedding_vectors)),
                metric="cosine"
            ).fit(self.embedding_vectors)
        else:
            self.embedding_model = None

    # -------------------------
    # COSINE SIM
    # -------------------------
    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)

    # -------------------------
    # FEATURE SIM (for sections)
    # -------------------------
    def _feature_similarity(self, a, b, key, scale=1.0):
        return 1 - min(abs(a.get(key, 0) - b.get(key, 0)) / scale, 1)

    # -------------------------
    # SECTION SIMILARITY 🔥🔥
    # -------------------------
    def _section_similarity(self, input_sections):

        scores = {}

        for i, ref_track in enumerate(self.metadata):

            ref_secs = ref_track.get("sections", [])
            total_sim = 0
            weight_sum = 0

            for inp in input_sections:

                inp_type = inp.get("type")
                inp_emb = inp.get("embedding")

                if inp_emb is None:
                    continue

                inp_emb = np.array(inp_emb)

                best_sim = 0

                for ref in ref_secs:

                    if ref.get("type") != inp_type:
                        continue

                    ref_emb = ref.get("embedding")
                    if ref_emb is None:
                        continue

                    ref_emb = np.array(ref_emb)

                    # --- embedding sim ---
                    emb_sim = self._cosine_similarity(inp_emb, ref_emb)

                    # --- feature sims ---
                    lufs_sim = self._feature_similarity(inp, ref, "lufs", 10)
                    punch_sim = self._feature_similarity(inp, ref, "kick_punch", 10)

                    sim = (0.7 * emb_sim) + (0.15 * lufs_sim) + (0.15 * punch_sim)

                    best_sim = max(best_sim, sim)

                if best_sim > 0:
                    weight = 2.0 if inp_type == "chorus" else 1.0
                    total_sim += best_sim * weight
                    weight_sum += weight

            if weight_sum > 0:
                scores[i] = total_sim / weight_sum

        return scores

    # -------------------------
    # MAIN MATCH FUNCTION
    # -------------------------
    def find_best_match(self, features, y, sr, sections=None):

        FEATURE_W = 0.3
        EMB_W = 0.25
        SEC_W = 0.35
        GENRE_W = 1.05  # multiplicative

        input_vec = self._feature_to_vector(features).reshape(1, -1)
        f_dist, f_idx = self.feature_model.kneighbors(input_vec)

        scores = {}

        # FEATURE SCORE
        for i, idx in enumerate(f_idx[0]):
            scores[idx] = (1 - f_dist[0][i]) * FEATURE_W

        # EMBEDDING SCORE
        if self.embedding_model is not None:
            try:
                emb = self.embedding_extractor.extract_embedding(y, sr)

                if emb is not None:
                    emb = emb.reshape(1, -1)
                    e_dist, e_idx = self.embedding_model.kneighbors(emb)

                    for i, idx in enumerate(e_idx[0]):
                        scores[idx] = scores.get(idx, 0) + (1 - e_dist[0][i]) * EMB_W

            except Exception as e:
                print("⚠️ Embedding failed:", e)

        # SECTION SCORE 🔥
        if sections:
            sec_scores = self._section_similarity(sections)

            for idx, val in sec_scores.items():
                scores[idx] = scores.get(idx, 0) + val * SEC_W

        # GENRE BOOST
        input_genre = features.get("genre", None)

        if input_genre:
            for i, meta in enumerate(self.metadata):
                if meta.get("genre") == input_genre and i in scores:
                    scores[i] *= GENRE_W

        # SORT
        sorted_indices = sorted(scores, key=scores.get, reverse=True)

        best_idx = sorted_indices[0]

        return {
            "best": self.metadata[best_idx],
            "top_k": [self.metadata[i] for i in sorted_indices[:5]],
            "scores": scores
        }
    
    def predict_genre(self, scores):

      # 🔥 Safety check
        if not scores or not isinstance(scores, dict):
             return "unknown", 0.0

    # remove invalid values
        scores = {k: float(v) for k, v in scores.items() if v is not None}

        if not scores:
            return "unknown", 0.0

        best_genre = max(scores, key=scores.get)
        confidence = float(scores[best_genre])

        return best_genre, confidence