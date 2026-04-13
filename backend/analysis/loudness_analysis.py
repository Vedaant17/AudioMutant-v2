def analyze_section_loudness(sections):

    insights = []

    lufs_values = [
        s.get("lufs") for s in sections if s.get("lufs") is not None
    ]

    if not lufs_values:
        return insights

    avg_lufs = sum(lufs_values) / len(lufs_values)

    # Find loudest & quietest
    loudest = min(lufs_values)   # more negative = quieter
    quietest = max(lufs_values)

    # Contrast
    dynamic_range = quietest - loudest

    if dynamic_range < 2:
        insights.append("⚠️ Low section contrast → mix may feel flat")

    # Drop vs others
    drops = [s for s in sections if s.get("type") in ["drop", "chorus"]]

    if drops:
        drop_lufs = [d["lufs"] for d in drops if d.get("lufs") is not None]

        if drop_lufs:
            avg_drop = sum(drop_lufs) / len(drop_lufs)

            if avg_drop > avg_lufs:
                insights.append("⚠️ Drop is not louder than average → weak impact")

    return insights