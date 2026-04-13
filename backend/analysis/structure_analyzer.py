class StructureAnalyzer:

    def compute_transitions(self, sections):

        transitions = []

        for i in range(1, len(sections)):

            prev = sections[i - 1]
            curr = sections[i]

            delta_energy = curr["mid_energy"] - prev["mid_energy"]
            delta_punch = curr["kick_punch"] - prev["kick_punch"]
            delta_width = curr["side_ratio"] - prev["side_ratio"]
            delta_lufs = curr["lufs"] - prev["lufs"]

            transitions.append({
                "from": prev["type"],
                "to": curr["type"],
                "energy_change": delta_energy,
                "punch_change": delta_punch,
                "width_change": delta_width,
                "loudness_change": delta_lufs
            })

        return transitions