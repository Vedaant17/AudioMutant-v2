import os
import json
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional


class FAISSMatcher:
    """
    Production-level FAISS matcher using IndexIVFFlat with cosine similarity.
    Supports:
    - Hybrid retrieval (structured features + learned embeddings)
    - Metadata mapping via IndexIDMap
    - Genre and tempo filtering
    - Batch search
    - Persistence with versioning
    - Backward compatibility with legacy indices
    """

    VERSION = "1.0"

    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------
    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        dimension: int,
        nlist: int = 100,
        nprobe: int = 10,
        use_cosine: bool = True,
    ):
        """
        Initialize the FAISS matcher.

        Args:
            index_path: Path to the FAISS index file.
            metadata_path: Path to the metadata JSON file.
            dimension: Dimensionality of the vectors.
            nlist: Number of IVF clusters.
            nprobe: Number of clusters to probe during search.
            use_cosine: Whether to use cosine similarity.
            """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dimension = dimension
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_cosine = use_cosine

    # -------------------------
    # LOAD OR CREATE FAISS INDEX
    # -------------------------
        if os.path.exists(index_path):
            print("✅ Loading FAISS index...")
            self.index = faiss.read_index(index_path)

        # Ensure ID mapping
            if not isinstance(self.index, faiss.IndexIDMap):
                self.index = faiss.IndexIDMap(self.index)
        else:
            print("⚠️ No existing index found. Creating new one...")
            metric = (
                faiss.METRIC_INNER_PRODUCT
                if self.use_cosine
                else faiss.METRIC_L2
                )
            quantizer = faiss.IndexFlatIP(self.dimension)
            ivf_index = faiss.IndexIVFFlat(
                quantizer,
                self.dimension,
                self.nlist,
                metric,
            )
            self.index = faiss.IndexIDMap(ivf_index)
            print("🆕 Created new IndexIVFFlat with ID mapping.")

        print(f"🔎 FAISS index type: {type(self.index).__name__}")

        # -------------------------
        # LOAD METADATA
        # -------------------------
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle versioned metadata
            if isinstance(data, dict) and "metadata" in data:
                self.metadata = data["metadata"]
            else:
                self.metadata = data

            # Ensure metadata is a list
            if not isinstance(self.metadata, list):
                print("🔄 Converting metadata to list...")
                self.metadata = list(self.metadata.values())

            print("✅ Metadata loaded.")
            print(f"Total Vectors in FAISS Index: {self.index.ntotal}")
            print(f"Total Metadat Entries: {len(self.metadata)}")
        else:
            print("⚠️ No metadata found. Starting fresh.")
            self.metadata = []

        # Build ID map
        self.id_map = {i: item for i, item in enumerate(self.metadata)}
    # --------------------------------------------------
    # Index Initialization
    # --------------------------------------------------
    def _create_new_index(self) -> faiss.Index:
        """Create a new IVF index wrapped with ID mapping."""
        metric = (
            faiss.METRIC_INNER_PRODUCT
            if self.use_cosine
            else faiss.METRIC_L2
        )

        quantizer = faiss.IndexFlatIP(self.dimension)
        ivf_index = faiss.IndexIVFFlat(
            quantizer,
            self.dimension,
            self.nlist,
            metric
        )

        index = faiss.IndexIDMap(ivf_index)
        print("🆕 Created new IndexIVFFlat with ID mapping.")
        return index

    def _load_or_create_index(self):
        """Load an existing index or create a new one."""
        if os.path.exists(self.index_path):
            print("✅ Loading FAISS index...")
            self.index = faiss.read_index(self.index_path)

            # Validate dimension
            if self.index.d != self.dimension:
                print("⚠️ Dimension mismatch. Rebuilding index...")
                self.index = self._create_new_index()

            # Ensure index supports ID mapping
            if not isinstance(self.index, faiss.IndexIDMap):
                print("⚠️ Wrapping existing index with IndexIDMap...")
                self.index = faiss.IndexIDMap(self.index)

        else:
            print("⚠️ No existing index found. Creating new one...")
            self.index = self._create_new_index()

        print(f"🔎 FAISS index type: {type(self.index).__name__}")

    # --------------------------------------------------
    # Metadata Handling
    # --------------------------------------------------
    def _load_metadata(self):
        """Load metadata and rebuild ID mapping."""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict) and "metadata" in data:
                self.metadata = data["metadata"]
                self.dimension = data.get("dimension", self.dimension)
            else:
                self.metadata = data

            self.id_map = {i: item for i, item in enumerate(self.metadata)}
            print("✅ Metadata loaded.")
        else:
            print("⚠️ No metadata found. Starting fresh.")
            self.metadata = []
            self.id_map = {}

    # --------------------------------------------------
    # Utility Functions
    # --------------------------------------------------
    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity."""
        if self.use_cosine:
            faiss.normalize_L2(vectors)
        return vectors

    def _build_feature_vector(self, features: Dict) -> np.ndarray:
        """Build hybrid feature + embedding vector."""
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
            dtype=np.float32
        )

        if embedding.size == 0:
            raise ValueError("Missing 'learned_embedding' in features.")

        vector = np.concatenate([base_vec, embedding]).astype(np.float32)

        if vector.shape[0] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, "
                f"got {vector.shape[0]}"
            )

        return vector

    # --------------------------------------------------
    # Add Tracks
    # --------------------------------------------------
    def add_tracks(self, tracks: List[Dict]):
        """Add tracks to the FAISS index."""
        if not tracks:
            print("⚠️ No tracks provided for indexing.")
            return

        vectors = []
        ids = []

        start_id = len(self.metadata)

        for i, track in enumerate(tracks):
            vec = self._build_feature_vector(track["features"])
            vectors.append(vec)
            ids.append(start_id + i)

        vectors = np.vstack(vectors).astype(np.float32)
        vectors = self._normalize(vectors)
        ids = np.array(ids, dtype=np.int64)

        # Train IVF index if required
        base_index = (
            self.index.index
            if isinstance(self.index, faiss.IndexIDMap)
            else self.index
        )

        if isinstance(base_index, faiss.IndexIVF) and not base_index.is_trained:
            print("🔧 Training FAISS index...")
            base_index.train(vectors)

        self.index.add_with_ids(vectors, ids)

        for i, track in zip(ids, tracks):
            self.metadata.append(track)
            self.id_map[int(i)] = track

        print(f"✅ Added {len(tracks)} tracks to FAISS index.")

    # --------------------------------------------------
    # Similarity Search
    # --------------------------------------------------
    def find_similar(
        self,
        features: Dict,
        top_k: int = 5,
        genre_filter: Optional[str] = None,
        tempo_range: Optional[Tuple[float, float]] = None,
    ) -> List[Dict]:
        """Perform similarity search with optional filtering."""
        if self.index.ntotal == 0:
            print("⚠️ FAISS index is empty.")
            return []
        
        print(f"Starting FAISS search")
        print(f"Genre filter: {genre_filter}")
        print(f"Tempo range: {tempo_range}")

        query = self._build_feature_vector(features).reshape(1, -1)
        query = self._normalize(query)

        # Set nprobe only for IVF indices
        base_index = (
            self.index.index
            if isinstance(self.index, faiss.IndexIDMap)
            else self.index
        )

        if isinstance(base_index, faiss.IndexIVF):
            base_index.nprobe = min(self.nprobe, self.nlist)

        distances, indices = self.index.search(query, top_k * 5)
        print(f"Raw FAISS indices: {indices}")
        print(f"Raw FAISS distances: {distances}")


        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            track = self.id_map.get(int(idx))
            if track is None:
                continue

            # Genre filtering
            if genre_filter and track.get("genre") != genre_filter:
                continue

            # Tempo filtering
            if tempo_range:
                tempo = track["features"].get("tempo_bpm", 0)
                if not (tempo_range[0] <= tempo <= tempo_range[1]):
                    continue

            similarity = (
                float(dist) if self.use_cosine else float(1 / (1 + dist))
            )
            print(f"Accepted track ID {idx} with similarity {similarity}")


            results.append({
                "track": track,
                "similarity": similarity
            })

            if len(results) >= top_k:
                break
        print(f"Total similar tracks found: {len(results)}")
        return results

    # Backward compatibility
    def search(self, *args, **kwargs):
        """Alias for backward compatibility."""
        return self.find_similar(*args, **kwargs)

    # --------------------------------------------------
    # Batch Search
    # --------------------------------------------------
    def batch_search(
        self,
        features_list: List[Dict],
        top_k: int = 5
    ) -> List[List[Dict]]:
        """Perform batch similarity search."""
        if not features_list:
            return []

        queries = [
            self._build_feature_vector(f) for f in features_list
        ]

        queries = np.vstack(queries).astype(np.float32)
        queries = self._normalize(queries)

        base_index = (
            self.index.index
            if isinstance(self.index, faiss.IndexIDMap)
            else self.index
        )

        if isinstance(base_index, faiss.IndexIVF):
            base_index.nprobe = min(self.nprobe, self.nlist)

        distances, indices = self.index.search(queries, top_k)

        batch_results = []
        for q_idx in range(len(features_list)):
            results = []
            for dist, idx in zip(distances[q_idx], indices[q_idx]):
                if idx == -1:
                    continue

                track = self.id_map.get(int(idx))
                if track:
                    similarity = (
                        float(dist)
                        if self.use_cosine
                        else float(1 / (1 + dist))
                    )
                    results.append({
                        "track": track,
                        "distance": similarity
                    })
            batch_results.append(results)

        return batch_results

    # --------------------------------------------------
    # Persistence
    # --------------------------------------------------
    def save(self):
        """Save index and metadata with versioning."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        faiss.write_index(self.index, self.index_path)

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": self.VERSION,
                    "dimension": self.dimension,
                    "metadata": self.metadata,
                },
                f,
                indent=2,
            )

        print("💾 FAISS index and metadata saved.")


    # --------------------------------------------------
    # VECTOR-BASED SEARCH (FOR SECTION SIMILARITY)
    # --------------------------------------------------
    def find_similar_by_vector(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> List[Dict]:
        """
        Perform FAISS similarity search using a precomputed embedding vector.
        Primarily used for section-level similarity.
        """
        if self.index is None or self.index.ntotal == 0:
            print("⚠️ FAISS index is empty.")
            return []

        # Ensure correct shape and dtype
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        # Validate embedding dimension
        if vector.shape[1] != self.index.d:
            raise ValueError(
            f"Embedding dimension mismatch: expected {self.index.d}, got {vector.shape[1]}"
            )

        # Normalize for cosine similarity
        if self.use_cosine:
            faiss.normalize_L2(vector)

        # Set nprobe for IVF indices
        base_index = (
            self.index.index
            if isinstance(self.index, faiss.IndexIDMap)
            else self.index
            )
        if isinstance(base_index, faiss.IndexIVF):
            base_index.nprobe = min(self.nprobe, self.nlist)

        # Perform search
        distances, indices = self.index.search(vector, top_k)

        print(f"🔍 FAISS returned indices: {indices}")
        print(f"📏 Distances/Similarities: {distances}")

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            metadata = self.id_map.get(int(idx))
            if metadata is None:
                print(f"⚠️ No metadata found for ID: {idx}")
                continue

            similarity = (
                float(dist) if self.use_cosine
                else float(1 / (1 + dist))
            )

            if similarity < min_similarity:
                continue

            results.append({
                "id": int(idx),
                "similarity": similarity,
                "metadata": metadata
            })

            print(
                f"✅ Match found: {metadata.get('title', 'Unknown')} "
                f"(Similarity: {similarity:.4f})"
            )

        return results