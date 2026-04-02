import numpy as np

def analyze_rhythm_patterns(beat_grid):

    beat_times = np.array(beat_grid.get("beat_times", []))

    if len(beat_times) < 2:
        return {"groove_variation": 0}

    intervals = np.diff(beat_times)

    return {
        "groove_variation": float(np.std(intervals)),
        "tightness": float(1 / (np.std(intervals) + 1e-6))
    }