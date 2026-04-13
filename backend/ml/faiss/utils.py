import numpy as np
import faiss

def normalize_vector(vec):
    """Normalize a vector for cosine similarity."""
    vec = np.asarray(vec, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec[0]

def compute_feature_similarity(query_features, track_features):
    """
    Compute similarity between structured audio features.
    Returns a value between 0 and 1.
    """
    def safe_diff(a, b, scale):
        return abs(a - b) / scale

    tempo_sim = 1 - safe_diff(
        query_features.get("tempo_bpm", 0),
        track_features.get("tempo_bpm", 0),
        200,
    )

    energy_sim = 1 - safe_diff(
        query_features.get("energy_mean", 0),
        track_features.get("energy_mean", 0),
        1,
    )

    stereo_sim = 1 - safe_diff(
        query_features.get("stereo_width", 0),
        track_features.get("stereo_width", 0),
        1,
    )

    return max(0.0, (tempo_sim + energy_sim + stereo_sim) / 3)