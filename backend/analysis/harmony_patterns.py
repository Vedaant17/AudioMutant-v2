import numpy as np

NOTE_TO_DEGREE = {
    "C": 1, "C#": 2, "D": 3, "D#": 4, "E": 5,
    "F": 6, "F#": 7, "G": 8, "G#": 9, "A": 10,
    "A#": 11, "B": 12
}

def detect_progression(chords, key):

    if not chords or key == "Unknown":
        return {"pattern": "unknown", "complexity": 0}

    root = key.split()[0]

    progression = []

    for c in chords:
        note = c["chord"]
        if note in NOTE_TO_DEGREE:
            degree = (NOTE_TO_DEGREE[note] - NOTE_TO_DEGREE.get(root, 0)) % 12
            progression.append(degree)

    # Simplify pattern
    unique = list(dict.fromkeys(progression))

    return {
        "pattern": unique,
        "complexity": len(unique)
    }