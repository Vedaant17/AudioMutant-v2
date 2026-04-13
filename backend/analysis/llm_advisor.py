class LLMAdvisor:

    def generate(self, track_advice, section_advice):

        messages = []

        for a in track_advice:

            if a["issue"] == "low_kick_punch":
                messages.append(
                    "Your mix lacks punch because the kick is weaker than professional tracks."
                )

            if a["issue"] == "low_energy":
                messages.append(
                    "Your track feels less energetic compared to reference tracks."
                )

            if a["issue"] == "low_overall_quality":
                messages.append(
                    "Overall, your track is below professional quality standards."
                )

        for a in section_advice:

            if a["issue"] == "weak_drop":
                messages.append(
                    "Your drop lacks impact and does not hit as hard as reference tracks."
                )

            if a["issue"] == "low_energy_drop":
                messages.append(
                    "Your drop does not build enough energy compared to the reference."
                )

        return messages