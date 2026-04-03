import numpy as np


class SectionMatcher:

    def __init__(self, reference_data):
        self.reference_data = reference_data

    # -------------------------
    # Cosine similarity
    # -------------------------
    def similarity(self, a, b):
        a = np.array(a)
        b = np.array(b)

        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0

        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # -------------------------
    # Main matching
    # -------------------------
    def find_best_section_match(self, input_sections):

        results = []

        for ref in self.reference_data:

            ref_sections = ref.get("features", {}).get("section_embeddings", [])

            total_score = 0
            matches = 0

            for inp_sec in input_sections:

                best_score = 0

                for ref_sec in ref_sections:

                    # Match only same type
                    if inp_sec["type"] != ref_sec["type"]:
                        continue

                    score = self.similarity(
                        inp_sec["embedding"],
                        ref_sec["embedding"]
                    )

                    best_score = max(best_score, score)

                if best_score > 0:
                    total_score += best_score
                    matches += 1

            if matches > 0:
                avg_score = total_score / matches

                results.append({
                    "track": ref.get("track", "unknown"),
                    "score": avg_score
                })

        # Sort best matches
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:3]