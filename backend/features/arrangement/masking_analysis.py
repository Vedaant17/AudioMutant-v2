import numpy as np

def detect_masking(stft, freqs):

    issues = []

    bands = {
        "low": (20, 250),
        "low_mid": (250, 1000),
        "presence": (2000, 5000),
        "high": (5000, 12000)
    }

    for name, (low, high) in bands.items():

        idx = np.where((freqs >= low) & (freqs <= high))

        energy = np.mean(stft[idx])

        if energy > 0.6:  # threshold tweak later
            issues.append({
                "band": name,
                "severity": "high",
                "message": f"Possible masking in {name} range",
                "fix": "Reduce competing instruments or EQ separation"
            })

    return issues