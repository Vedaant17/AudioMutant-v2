import numpy as np


# --------------------------------------------------
# 🔹 HELPER: SAFE DIFF
# --------------------------------------------------
def diff(a, b):
    return a - b


# --------------------------------------------------
# 🔥 CORE MIX ADVISOR
# --------------------------------------------------
def analyze_loudness(features):
    lufs = features.get("LUFS", -14)

    if lufs > -6:
        return "Mix is extremely loud. Reduce limiter gain."
    elif lufs < -16:
        return "Mix is too quiet. Increase overall gain."
    return "Loudness is within acceptable range."


def analyze_tonal_balance(features):
    low = features["frequency_balance"]["low"]
    mid = features["frequency_balance"]["mid"]
    high = features["frequency_balance"]["high"]

    issues = []

    if low > mid * 2:
        issues.append("Too much low-end. Cut 60–200Hz.")

    if high < mid * 0.5:
        issues.append("High frequencies weak. Boost 6–10kHz.")

    return issues


def analyze_stereo(stereo):
    issues = []

    width = stereo.get("stereo_width", 0)
    field = stereo.get("stereo_field", {})

    if width < 0.2:
        issues.append("Mix is too narrow. Add stereo widening.")

    if field:
        if field.get("low", 0) > 0.5:
            issues.append("Low frequencies too wide. Keep bass mono.")

        if field.get("high", 0) < 0.3:
            issues.append("High frequencies too narrow. Add width to hats/reverbs.")

    return issues


def analyze_dynamics(features):
    crest = features.get("crest_factor", 0)
    dynamic_range = features.get("dynamic_range", 0)

    issues = []

    if crest < 3:
        issues.append("Over-compressed. Increase dynamics.")

    if dynamic_range < 0.2:
        issues.append("Low dynamic range. Add punch.")

    return issues


def analyze_melody_mix(melody):
    if not melody.get("melody_present"):
        return ["No clear lead element. Add vocals or lead instrument."]

    if melody.get("melody_pitch_std", 0) < 2:
        return ["Melody lacks presence. Boost 2–5kHz range."]

    return []

def mix_advisor(
    base,
    spectral,
    dynamics,
    stereo,
    masking=None,
    drums=None,
    melody=None
):
    advice = []

    # Merge everything into one feature dict (for legacy functions)
    features = {
        **base,
        **spectral,
        **dynamics,
        **stereo
    }

    # -------------------------
    # LOUDNESS
    # -------------------------
    advice.append(analyze_loudness(features))

    # -------------------------
    # TONAL BALANCE
    # -------------------------
    advice.extend(analyze_tonal_balance(features))

    # -------------------------
    # STEREO FIELD
    # -------------------------
    advice.extend(analyze_stereo(stereo))

    # -------------------------
    # DYNAMICS
    # -------------------------
    advice.extend(analyze_dynamics(features))

    # -------------------------
    # MELODY IN MIX
    # -------------------------
    if melody is not None:
        advice.extend(analyze_melody_mix(melody))

    # -------------------------
    # MASKING (NEW)
    # -------------------------
    if masking is not None:
        advice.append({
            "type": "masking",
            "message": f"Masking detected in {len(masking)} regions"
        })

    # -------------------------
    # DRUMS (NEW)
    # -------------------------
    if drums is not None:
        advice.append({
            "type": "drums",
            "message": f"Kick: {drums.get('kick_strength', 0):.2f}, "
                       f"Snare: {drums.get('snare_strength', 0):.2f}"
        })

    return {
        "issues": advice
    }