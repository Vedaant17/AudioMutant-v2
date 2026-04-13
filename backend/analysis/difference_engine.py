import numpy as np

class DifferenceEngine:
    
    def _cosine_similarity(self, a, b):
        a = np.array(a)
        b = np.array(b)

        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0

        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _safe_get(self, d, key, default=0):
        val = d.get(key, default)
        return 0 if val is None else val

    def _compute_diff(self, inp, ref):
        return inp - ref

    def _severity(self, diff, scale=1.0):
        val = abs(diff) / (scale + 1e-6)

        if val > 0.5:
            return "high"
        elif val > 0.25:
            return "medium"
        else:
            return "low"

    # -------------------------
    # TRACK DIFFERENCE
    # -------------------------
    def compare_track(self, input_feat, ref_feat, input_score=None, ref_score=None):

        diffs = {}

        keys = [
            "kick_punch",
            "transient_strength",
            "energy_mean",
            "stereo_width",
            "spectral_centroid"
        ]

        for k in keys:
            inp = self._safe_get(input_feat, k)
            ref = self._safe_get(ref_feat, k)

            if not isinstance(inp, (int, float)) or not isinstance(ref, (int, float)):
                continue

            diff = self._compute_diff(inp, ref)

            diffs[k] = {
                "input": inp,
                "reference": ref,
                "diff": diff,
                "severity": self._severity(diff, scale=ref if ref != 0 else 1)
            }

        # 🔥 ML SCORE DIFFERENCE
        if input_score is not None and ref_score is not None:
            score_diff = input_score - ref_score

            diffs["track_score"] = {
                "input": input_score,
                "reference": ref_score,
                "diff": score_diff,
                "severity": self._severity(score_diff, scale=1)
            }

        return diffs

    # -------------------------
    # SECTION DIFFERENCE
    # -------------------------
    def compare_sections(self, input_sections, ref_sections, section_scores=None):

        results = []

        for i, inp in enumerate(input_sections):

            best = None
            best_score = -1

            for ref in ref_sections:
                if ref.get("type") != inp.get("type"):
                    continue

                score = self._simple_similarity(inp, ref)

                if score > best_score:
                    best_score = score
                    best = ref

            if best is None:
                continue

            kick_diff = inp.get("kick_punch", 0) - best.get("kick_punch", 0)

            diff = {
                "type": inp.get("type"),
                "kick_diff": kick_diff,
                "energy_diff": inp.get("mid_energy", 0) - best.get("mid_energy", 0),
                "width_diff": inp.get("side_ratio", 0) - best.get("side_ratio", 0),
                "severity": self._severity(
                    kick_diff,
                    scale=best.get("kick_punch", 1)
                )
            }

            # ✅ SAFE score attach (NO POP)
            if section_scores is not None and i < len(section_scores):
                diff["score"] = section_scores[i]

            results.append(diff)

        return results

    def _simple_similarity(self, a, b):
        return 1 - abs(a.get("kick_punch", 0) - b.get("kick_punch", 0))