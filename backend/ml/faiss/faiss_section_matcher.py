import json
import numpy as np
import faiss


class SectionMatcher:
    def __init__(self, index_path, metadata_path):
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def _normalize(self, vec):
        norm = np.linalg.norm(vec)
        return vec if norm == 0 else vec / norm

    def search(self, embedding, top_k=5, genre=None, section_type=None):
        query = self._normalize(
            np.array(embedding, dtype=np.float32).reshape(1, -1)
        )

        distances, indices = self.index.search(query, top_k * 5)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            meta = self.metadata[idx]

            if genre and meta["genre"] != genre:
                continue
            if section_type and meta["section_type"] != section_type:
                continue

            results.append({
                "track": meta["track"],
                "genre": meta["genre"],
                "section_type": meta["section_type"],
                "start": meta["start"],
                "end": meta["end"],
                "similarity": float(dist),
            })

            if len(results) >= top_k:
                break

        return results