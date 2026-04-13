class MixInsightEngine:

    def analyze(self, features):

        insights = []

        energy = features.get("energy_mean", 0)
        transient = features.get("transient_strength", 0)
        side_ratio = features.get("side_ratio", 0)
        rms = features.get("rms", 0)
        centroid = features.get("spectral_centroid", 0)
        attack = features.get("attack_sharpness", 0)

        # 🔥 Kick vs Bass masking
        if features.get("kick_punch", 0) < 4 and rms > 0.25:
            insights.append("Kick may be masked by bass → try sidechain or EQ separation")

        # 🔥 Weak transients but high energy
        if transient < 1.2 and energy > 0.3:
           insights.append("Mix is energetic but lacks punch → transient shaping needed")

        # 🔥 Narrow but loud
        if side_ratio < 0.15 and rms > 0.25:
            insights.append("Mix is loud but narrow → widen stereo elements")

        # 🔥 Too bright + harsh risk
        if centroid > 3000 and attack > 3:
           insights.append("Mix may sound harsh → control high frequencies or transients")

        # 🔥 Too dark
        if centroid < 2000:
            insights.append("Mix is dark → add brightness (EQ or saturation)")

        return insights

    # -------------------------
    # DROP STRENGTH
    # -------------------------
    def analyze_drop_strength(self, input_sections, ref_sections):

        def get_avg(sections, key, sec_type):
            vals = [s[key] for s in sections if s["type"] == sec_type]
            return sum(vals) / len(vals) if vals else 0

        inp_drop = get_avg(input_sections, "kick_punch", "chorus")
        ref_drop = get_avg(ref_sections, "kick_punch", "chorus")

        if inp_drop < ref_drop * 0.8:
            return "Your drop hits weaker than reference → increase kick punch and transient shaping"

        return None

    # -------------------------
    # ENERGY CURVE
    # -------------------------
    def energy_curve(self, sections):
        return [s["mid_energy"] for s in sections]

    # -------------------------
    # ENERGY COMPARISON
    # -------------------------
    def compare_energy_curve(self, inp, ref):

        inp_curve = self.energy_curve(inp)   # ✅ FIX
        ref_curve = self.energy_curve(ref)   # ✅ FIX

        if max(inp_curve) - min(inp_curve) < max(ref_curve) - min(ref_curve):
            return "Your track has less energy variation → arrangement feels flatter than reference"

        return None