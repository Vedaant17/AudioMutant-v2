def apply_genre_rules(genre, sections):

    advice = []

    if genre == "edm":

        has_drop = any(s["type"] == "drop" for s in sections)

        if not has_drop:
            advice.append({
                "issue": "missing_drop",
                "suggestion": "EDM tracks should include a drop section"
            })

    elif genre == "hiphop":

        advice.append({
            "tip": "Focus on groove, drum swing, and vocal space"
        })

    elif genre == "rock":

        advice.append({
            "tip": "Use dynamic contrast between verse and chorus"
        })

    return advice