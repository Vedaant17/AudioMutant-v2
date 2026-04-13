from analysis.mix_insights_engine import MixInsightEngine
from analysis.structure_analyzer import StructureAnalyzer


class AdvancedAdvisor:

    def __init__(self):
        self.mix_engine = MixInsightEngine()
        self.structure = StructureAnalyzer()

    def analyze(self, input_data, ref_data):

        output = {}

        # -------------------------
        # SAFE ACCESS
        # -------------------------
        features = input_data.get("features", {})
        sections = input_data.get("sections", [])
        ref_sections = ref_data.get("sections", [])

        # -------------------------
        # TRACK-LEVEL INSIGHTS
        # -------------------------
        output["mix_insights"] = self.mix_engine.analyze(features)

        # -------------------------
        # STRUCTURE INSIGHTS
        # -------------------------
        transitions = self.structure.compute_transitions(sections)

        structure_advice = set()

        for t in transitions:
            if t["to"] == "chorus":

                if t["punch_change"] < 0:
                    structure_advice.add("Drop loses punch instead of increasing")

                if t["energy_change"] < 0:
                    structure_advice.add("Energy decreases into chorus → weak transition")

                if t["width_change"] < 0:
                    structure_advice.add("Chorus is not wider than previous section")

        output["structure_advice"] = list(structure_advice)

        # -------------------------
        # REFERENCE COMPARISON
        # -------------------------
        ref_advice = []

        drop_msg = self.mix_engine.analyze_drop_strength(sections, ref_sections)
        if drop_msg:
            ref_advice.append(drop_msg)

        energy_msg = self.mix_engine.compare_energy_curve(sections, ref_sections)
        if energy_msg:
            ref_advice.append(energy_msg)

        output["reference_advice"] = ref_advice

        # -------------------------
        # FALLBACKS
        # -------------------------
        if not output["mix_insights"]:
            output["mix_insights"] = ["Mix is well-balanced compared to reference"]

        if not output["structure_advice"]:
            output["structure_advice"] = ["Song structure transitions are effective"]

        if not output["reference_advice"]:
            output["reference_advice"] = ["Track is comparable to reference"]

        return output