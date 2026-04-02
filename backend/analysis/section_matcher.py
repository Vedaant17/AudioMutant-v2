import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SectionMatcher:

    def __init__(self, reference_data):
        self.reference_data = reference_data

    def find_best_section_match(self, input_sections):

        results = []

        for sec in input_sections:

            best_score = -1
            best_match = None

            for ref in self.reference_data:

                for ref_sec in ref["features"].get("section_embeddings", []):

                    if ref_sec["type"] != sec["type"]:
                        continue

                    sim = cosine_similarity(
                        [sec["embedding"]],
                        [ref_sec["embedding"]]
                    )[0][0]

                    if sim > best_score:
                        best_score = sim
                        best_match = {
                            "track": ref["track"],
                            "section": ref_sec["type"],
                            "similarity": float(sim),
                            "reference_energy": ref_sec.get("energy", 0.0)
                        }

            if best_match:
                results.append({
                    "section": sec["type"],
                    "match": best_match
                })

        return results