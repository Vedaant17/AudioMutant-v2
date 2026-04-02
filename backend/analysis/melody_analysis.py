import numpy as np

def analyze_melody_by_section(melody, sections):

    pitches = np.array(melody.get("pitch_contour", []))
    times = np.array(melody.get("times", []))

    results = []

    for sec in sections:
        start = sec["start"]
        end = sec["end"]

        idx = np.where((times >= start) & (times <= end))[0]

        if len(idx) == 0:
            continue

        segment = pitches[idx]

        results.append({
            "section": sec["type"],
            "pitch_range": float(np.max(segment) - np.min(segment)),
            "variation": float(np.std(segment))
        })

    return results