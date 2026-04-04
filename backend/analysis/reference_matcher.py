"""import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity


REFERENCE_FOLDER = "reference_data"


# --------------------------------------------------
# LOAD ALL REFERENCE JSON FILES
# --------------------------------------------------

def load_references():

    data = []
    metadata = []

    for genre in os.listdir(REFERENCE_FOLDER):

        genre_path = os.path.join(REFERENCE_FOLDER, genre)

        if not os.path.isdir(genre_path):
            continue

        for file in os.listdir(genre_path):

            if not file.endswith(".json"):
                continue

            path = os.path.join(genre_path, file)

            with open(path, "r") as f:
                ref = json.load(f)

                features = ref["features"]

                vector = build_feature_vector(features)

                data.append(vector)

                metadata.append({
                    "track": ref.get("track"),
                    "artist": ref.get("artist"),
                    "genre": ref.get("genre"),
                    "features": features
                })

    return np.array(data), metadata


# --------------------------------------------------
# FEATURE VECTOR BUILDER
# (VERY IMPORTANT)
# --------------------------------------------------

def build_feature_vector(features):

    return np.array([

        # 🎧 CORE MIX FEATURES
        features.get("tempo_bpm", 0),
        features.get("tempo_stability", 0),
        features.get("LUFS", 0),
        features.get("crest_factor", 0),
        features.get("dynamic_range", 0),
        features.get("compression_ratio", 0),

        # 🎚 TONAL BALANCE
        features.get("low_mid_ratio", 0),
        features.get("mid_high_ratio", 0),

        # 🎛 SPECTRAL
        features.get("spectral_centroid", 0),
        features.get("spectral_bandwidth", 0),
        features.get("spectral_rolloff", 0),
        features.get("spectral_tilt", 0),

        # 🥁 TRANSIENTS
        features.get("transient_density", 0),
        features.get("beat_strength", 0),

        # 🎧 STEREO
        features.get("stereo_width", 0),
        features.get("phase_correlation", 0),

    ])


# --------------------------------------------------
# MAIN MATCHER CLASS
# --------------------------------------------------

class ReferenceMatcher:

    def _init_(self):

        print("🔄 Loading reference dataset...")

        self.X, self.metadata = load_references()

        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

        # ML MODEL (KNN)
        self.model = NearestNeighbors(
            n_neighbors=5,
            metric="euclidean"
        )

        self.model.fit(self.X_scaled)

        print(f"✅ Loaded {len(self.metadata)} reference tracks")


    # --------------------------------------------------
    # FIND MATCHES
    # --------------------------------------------------

    def find_matches(self, input_features, top_k=3):

        input_vector = build_feature_vector(input_features)

        input_scaled = self.scaler.transform([input_vector])

        # -------------------------
        # ML MATCH (KNN)
        # -------------------------
        distances, indices = self.model.kneighbors(input_scaled)

        knn_matches = []

        for dist, idx in zip(distances[0], indices[0]):
            knn_matches.append({
                "score": float(1 / (1 + dist)),  # convert distance → similarity
                "meta": self.metadata[idx]
            })

        # -------------------------
        # COSINE MATCH
        # -------------------------
        cosine_scores = cosine_similarity(input_scaled, self.X_scaled)[0]

        cosine_matches = []

        for i, score in enumerate(cosine_scores):
            cosine_matches.append({
                "score": float(score),
                "meta": self.metadata[i]
            })

        # -------------------------
        # HYBRID MERGE
        # -------------------------
        combined = []

        for i in range(len(self.metadata)):

            cos_score = cosine_scores[i]

            # find if in knn
            knn_score = 0
            for match in knn_matches:
                if match["meta"] == self.metadata[i]:
                    knn_score = match["score"]

            final_score = 0.6 * cos_score + 0.4 * knn_score

            combined.append({
                "score": final_score,
                "meta": self.metadata[i]
            })

        # Sort best matches
        combined = sorted(combined, key=lambda x: x["score"], reverse=True)

        return combined[:top_k]"""

import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


class ReferenceMatcher:

    def __init__(self, reference_dir="reference_data"):
        self.reference_dir = reference_dir
        self.references = []
        self.feature_matrix = []
        self.scaler = StandardScaler()

        self._load_references()
        self._build_feature_matrix()

    # ----------------------------------
    # LOAD JSON FILES
    # ----------------------------------
    def _load_references(self):
        for genre in os.listdir(self.reference_dir):
            genre_path = os.path.join(self.reference_dir, genre)

            if not os.path.isdir(genre_path):
                continue

            for file in os.listdir(genre_path):
                if file.endswith(".json"):
                    path = os.path.join(genre_path, file)

                    with open(path, "r") as f:
                        data = json.load(f)

                    self.references.append(data)

    # ----------------------------------
    # FEATURE VECTOR BUILDER
    # ----------------------------------
    def _feature_vector(self, features):

        return np.array([
            features.get("tempo_bpm", 0),
            features.get("LUFS", 0),
            features.get("spectral_tilt", 0),
            features.get("low_mid_ratio", 0),
            features.get("mid_high_ratio", 0),
            features.get("transient_density", 0),
            features.get("stereo_width", 0),
            features.get("harmonic_percussive_ratio", 0),
            features.get("dynamic_range", 0),
        ])

    # ----------------------------------
    # BUILD MATRIX + NORMALIZE
    # ----------------------------------
    def _build_feature_matrix(self):
        vectors = []

        for ref in self.references:
            vec = self._feature_vector(ref["features"])
            vectors.append(vec)

        self.feature_matrix = np.array(vectors)

        # Normalize (CRITICAL)
        self.feature_matrix = self.scaler.fit_transform(self.feature_matrix)

    # ----------------------------------
    # FIND BEST MATCHES
    # ----------------------------------
    def find_matches(self, input_features, top_k=3):

        input_vec = self._feature_vector(input_features).reshape(1, -1)
        input_vec = self.scaler.transform(input_vec)

        similarities = cosine_similarity(input_vec, self.feature_matrix)[0]

        indices = np.argsort(similarities)[::-1][:top_k]

        results = []

        for idx in indices:
            ref = self.references[idx]

            results.append({
                "track": ref["track"],
                "artist": ref["artist"],
                "genre": ref["genre"],
                "similarity": float(similarities[idx])
            })

        return results