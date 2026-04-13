class AdviceEngine:

    # -------------------------
    # TRACK ADVICE
    # -------------------------
    def generate_track_advice(self, diffs, score=None):

        advice = []

        # -------------------------
        # SCORE FEEDBACK
        # -------------------------
        if score is not None:
            score_100 = int(score * 100)

            if score_100 > 80:
                advice.append(f"Your track is strong overall ({score_100}/100)")
            elif score_100 > 60:
                advice.append(f"Your track is decent but has room for improvement ({score_100}/100)")
            else:
                advice.append(f"Your track needs improvement ({score_100}/100)")

        # -------------------------
        # FEATURE DIFFERENCES (STRUCTURED)
        # -------------------------
        for key, value in diffs.items():

            # ✅ Extract diff safely
            if not isinstance(value, dict):
                continue

            diff = value.get("diff", 0)
            severity = value.get("severity", "low")

            if key == "kick_punch" and diff < -1:
                advice.append(f"Kick lacks punch ({severity})")

            elif key == "stereo_width" and diff < -0.1:
                advice.append(f"Mix is too narrow ({severity})")

            elif key == "spectral_centroid" and diff < -500:
                advice.append(f"Mix sounds too dark ({severity})")

            elif key == "energy_mean" and diff < -0.05:
                advice.append(f"Overall energy is lower than professional tracks ({severity})")

        return advice

    # -------------------------
    # SECTION ADVICE
    # -------------------------
    def generate_section_advice(self, section_diffs):

        advice = []

        for diff in section_diffs:

            sec_type = diff.get("type", "section")
            score = diff.get("score")

            score_text = ""
            if score is not None:
                score_text = f" ({int(score * 100)}/100)"

            if diff.get("kick_diff", 0) < -1:
                advice.append(f"{sec_type} lacks punch{score_text}")

            if diff.get("width_diff", 0) < -0.1:
                advice.append(f"{sec_type} is not wide enough{score_text}")

            if diff.get("energy_diff", 0) < -0.05:
                advice.append(f"{sec_type} lacks energy{score_text}")

        return advice