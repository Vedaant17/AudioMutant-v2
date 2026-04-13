import os
import json
import numpy as np
import faiss
from tqdm import tqdm

# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "reference_data")
FAISS_DIR = os.path.join(BASE_DIR, "ml", "faiss")

INDEX_PATH = os.path.join(FAISS_DIR, "faiss_index.ivf")
METADATA_PATH = os.path.join(FAISS_DIR, "faiss_metadata.json")

NLIST = 16  # Number of Voronoi cells for IVF index
VERSION = "1.0"


# --------------------------------------------------
# NORMALIZATION
# --------------------------------------------------
def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Normalize a vector for cosine similarity."""
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


# --------------------------------------------------
# FEATURE VECTOR BUILDER
# --------------------------------------------------
def build_feature_vector(features: dict) -> np.ndarray:
    """
    Builds the hybrid feature vector (structured features + learned embedding).
    """
    base_vec = np.array(
        [
            features.get("tempo_bpm", 0),
            features.get("mid_energy", 0),
            features.get("side_ratio", 0),
            features.get("integrated_lufs", 0),
            features.get("kick_punch", 0),
            features.get("transient_strength", 0),
            features.get("energy_mean", 0),
            features.get("stereo_width", 0),
            features.get("spectral_centroid", 0),
        ],
        dtype=np.float32,
    )

    embedding = np.array(
        features.get("learned_embedding", []),
        dtype=np.float32,
    )

    if embedding.size == 0:
        raise ValueError(
            "Missing 'learned_embedding' in features."
        )

    return np.concatenate([base_vec, embedding]).astype(np.float32)


# --------------------------------------------------
# LOAD REFERENCE TRACKS
# --------------------------------------------------
def load_reference_tracks():
    """
    Loads all JSON reference tracks and constructs vectors.
    Returns:
        tracks (list): List of track dictionaries.
        vectors (np.ndarray): Feature vectors.
        dimension (int): Vector dimension.
    """
    tracks = []
    vectors = []
    json_files = []

    for root, _, files in os.walk(DATASET_DIR):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))

    if not json_files:
        raise ValueError(f"No JSON files found in {DATASET_DIR}")

    print(f"📂 Found {len(json_files)} reference tracks.")

    for file_path in tqdm(json_files, desc="Loading tracks"):
        with open(file_path, "r", encoding="utf-8") as f:
            track = json.load(f)

        if "features" not in track:
            print(f"⚠️ Skipping {file_path} (missing features).")
            continue

        try:
            vector = build_feature_vector(track["features"])
            vector = l2_normalize(vector)

            vectors.append(vector)
            tracks.append(track)
        except Exception as e:
            print(f"⚠️ Skipping {file_path}: {e}")

    if not vectors:
        raise ValueError("No valid vectors found.")

    vectors = np.vstack(vectors).astype(np.float32)
    dimension = vectors.shape[1]

    return tracks, vectors, dimension


# --------------------------------------------------
# BUILD FAISS INDEX
# --------------------------------------------------
def build_index():
    print("🚀 Building FAISS Index...")

    tracks, vectors, dimension = load_reference_tracks()

    os.makedirs(FAISS_DIR, exist_ok=True)

    # Create IVF index with cosine similarity
    quantizer = faiss.IndexFlatIP(dimension)
    ivf_index = faiss.IndexIVFFlat(
        quantizer,
        dimension,
        NLIST,
        faiss.METRIC_INNER_PRODUCT,
    )

    index = faiss.IndexIDMap(ivf_index)

    # Train index
    if not ivf_index.is_trained:
        print("🔧 Training FAISS index...")
        ivf_index.train(vectors)

    # Add vectors
    ids = np.arange(len(vectors)).astype(np.int64)
    index.add_with_ids(vectors, ids)

    # Save index
    faiss.write_index(index, INDEX_PATH)
    print(f"💾 FAISS index saved to: {INDEX_PATH}")

    # Save metadata with versioning
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "version": VERSION,
                "dimension": dimension,
                "metadata": tracks,
            },
            f,
            indent=2,
        )

    print(f"💾 Metadata saved to: {METADATA_PATH}")
    print(f"📊 Total vectors indexed: {index.ntotal}")
    print(f"📏 Vector dimension: {dimension}")
    print("✅ FAISS index successfully built!")


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    build_index()