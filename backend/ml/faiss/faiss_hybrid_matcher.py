import numpy as np


class HybridMatcher:
    def __init__(self, track_matcher, section_matcher, alpha=0.7):
        self.track_matcher = track_matcher
        self.section_matcher = section_matcher
        self.alpha = alpha

    def find_similar(
        self,
        track_features,
        section_embeddings,
        top_k=5,
        genre_filter=None
    ):
        track_results = self.track_matcher.find_similar(
            track_features,
            top_k=top_k * 3,
            genre_filter=genre_filter
        )

        hybrid_results = []

        for result in track_results:
            candidate = result["track"]
            track_score = result["distance"]

            # Compute best section similarity
            best_section_score = 0.0
            for section in candidate.get("sections", []):
                candidate_emb = np.array(
                    section["embedding"], dtype=np.float32
                )

                for query_emb in section_embeddings:
                    sim = float(
                        np.dot(query_emb, candidate_emb)
                        / (
                            np.linalg.norm(query_emb)
                            * np.linalg.norm(candidate_emb)
                            + 1e-10
                        )
                    )
                    best_section_score = max(
                        best_section_score, sim
                    )

            final_score = (
                self.alpha * track_score
                + (1 - self.alpha) * best_section_score
            )

            hybrid_results.append({
                "track": candidate,
                "track_similarity": track_score,
                "section_similarity": best_section_score,
                "hybrid_score": final_score,
            })

        hybrid_results.sort(
            key=lambda x: x["hybrid_score"], reverse=True
        )
        return hybrid_results[:top_k]