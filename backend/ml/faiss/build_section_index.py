import os
import json
import numpy as np
import faiss

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

REFERENCE_DATA_DIR = os.path.join(BASE_DIR, "reference_data")
INDEX_PATH = os.path.join(BASE_DIR, "ml", "faiss", "section_index.faiss")
METADATA_PATH = os.path.join(BASE_DIR, "ml", "faiss", "section_metadata.json")

EMBEDDING_DIM = 32  # Ensure this matches your SectionEmbeddingModel


def load_reference_sections():
    """Loads all section embeddings from genre-based reference folders."""
    vectors = []
    metadata = []
    idx = 0

    print("📂 Scanning reference data...")

    for genre in os.listdir(REFERENCE_DATA_DIR):
        genre_path = os.path.join(REFERENCE_DATA_DIR, genre)
        if not os.path.isdir(genre_path):
            continue

        for file in os.listdir(genre_path):
            if not file.endswith(".json"):
                continue

            file_path = os.path.join(genre_path, file)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            track_name = data.get("track", file.replace(".json", ""))
            sections = data.get("sections", [])

            for sec in sections:
                embedding = sec.get("embedding")
                if embedding and len(embedding) == EMBEDDING_DIM:
                    vectors.append(np.array(embedding, dtype=np.float32))
                    metadata.append({
                        "id": idx,
                        "track": track_name,
                        "genre": genre,
                        "section_type": sec.get("type"),
                        "start": sec.get("start"),
                        "end": sec.get("end"),
                        "features": sec,
                    })
                    idx += 1

    if not vectors:
        raise ValueError("No valid section embeddings found.")

    print(f"✅ Loaded {len(vectors)} section embeddings.")
    return np.vstack(vectors), metadata


def build_section_index():
    print("🚀 Building Section FAISS Index...")

    vectors, metadata = load_reference_sections()
    dim = vectors.shape[1]

    # L2 normalize for cosine similarity
    faiss.normalize_L2(vectors)

    # Choose appropriate number of clusters
    nlist = max(1, int(np.sqrt(len(vectors))))
    print(f"🔧 Using nlist={nlist}")

    # Create IVF index
    quantizer = faiss.IndexFlatIP(dim)
    ivf_index = faiss.IndexIVFFlat(
        quantizer,
        dim,
        nlist,
        faiss.METRIC_INNER_PRODUCT
    )

    # Wrap with ID mapping BEFORE training/adding vectors
    index = faiss.IndexIDMap(ivf_index)

    print("🔧 Training FAISS index...")
    index.train(vectors)

    print("📥 Adding vectors to index...")
    ids = np.arange(len(vectors)).astype(np.int64)
    index.add_with_ids(vectors, ids)

    # Save index
    faiss.write_index(index, INDEX_PATH)
    print(f"💾 Section index saved to: {INDEX_PATH}")

    # Save metadata
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"💾 Section metadata saved to: {METADATA_PATH}")
    print("✅ Section FAISS index built successfully!")


if __name__ == "__main__":
    build_section_index()