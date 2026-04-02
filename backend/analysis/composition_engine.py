import numpy as np
from analysis.harmony_patterns import detect_progression
from analysis.melody_analysis import analyze_melody_by_section
from analysis.rhythm_analysis_advanced import analyze_rhythm_patterns
from analysis.genre_rules import apply_genre_rules


NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']


def get_scale_notes(key_signature):
    root, mode = key_signature.split()

    root_idx = NOTE_NAMES.index(root)

    if mode == "major":
        intervals = [0,2,4,5,7,9,11]
    else:
        intervals = [0,2,3,5,7,8,10]

    return [(root_idx + i) % 12 for i in intervals]


def suggest_chords(scale_notes):
    # Simple diatonic triads
    chords = []

    for i in range(len(scale_notes)):
        root = scale_notes[i]
        third = scale_notes[(i + 2) % len(scale_notes)]
        fifth = scale_notes[(i + 4) % len(scale_notes)]

        chords.append(f"{NOTE_NAMES[root]}-{NOTE_NAMES[third]}-{NOTE_NAMES[fifth]}")

    return chords


def analyze_melody(melody_data):
    if not melody_data.get("melody_present"):
        return {
            "melody_type": "none",
            "suggestion": "No strong melody detected. Consider adding lead elements or vocals."
        }

    pitch_std = melody_data["melody_pitch_std"]
    pitch_range = melody_data["melody_range"]

    if pitch_std < 2:
        return {
            "melody_type": "monotone",
            "suggestion": "Melody is too flat. Add pitch variation or jumps."
        }

    if pitch_range < 5:
        return {
            "melody_type": "narrow",
            "suggestion": "Melody range is limited. Expand across more octaves."
        }

    return {
        "melody_type": "dynamic",
        "suggestion": "Melody has good movement. Consider layering harmonies."
    }


def suggest_bassline(key_signature, tempo):
    if tempo > 120:
        return "Try rhythmic bass with octave jumps (EDM style)"
    else:
        return "Use sustained bass notes following root notes"


def suggest_rhythm(groove_stability):
    if groove_stability < 0.02:
        return "Very tight rhythm. Add swing or humanization."
    else:
        return "Groove is natural. Maintain this feel."

def composition_engine(harmony, rhythm, melody, sections, section_matches, genre):

    advice = []

    chords = harmony.get("chords", [])
    key = harmony.get("key", "Unknown")

    tempo = rhythm.get("tempo", rhythm.get("beat_grid", {}).get("tempo", 0))
    swing = rhythm.get("groove_swing", 0)

    pitch_var = melody.get("pitch_variation", 0)
    pitch_range = melody.get("pitch_range", 0)

    # -------------------------
    # 🔥 HARMONY INTELLIGENCE
    # -------------------------
    progression = detect_progression(chords, key)

    unique_chords = len(set([c["chord"] for c in chords])) if chords else 0

    if progression["complexity"] <= 2 or unique_chords <= 2:
        advice.append({
            "level": "global",
            "issue": "simple_progression",
            "suggestion": "Chord progression is too simple — add variation or extensions"
        })

    # -------------------------
    # 🎵 GLOBAL MELODY
    # -------------------------
    if pitch_var < 20:
        advice.append({
            "level": "global",
            "issue": "flat_melody",
            "suggestion": "Melody lacks variation — introduce jumps or rhythm changes"
        })

    if pitch_range < 5:
        advice.append({
            "level": "global",
            "issue": "narrow_range",
            "suggestion": "Expand melodic range for more emotional impact"
        })

    # -------------------------
    # 🥁 RHYTHM
    # -------------------------
    rhythm_adv = analyze_rhythm_patterns(rhythm.get("beat_grid", {}))

    if swing < 0.01 or rhythm_adv["groove_variation"] < 0.01:
        advice.append({
            "level": "global",
            "issue": "rigid_rhythm",
            "suggestion": "Add swing or groove variation"
        })

    if tempo > 150:
        advice.append({
            "level": "global",
            "tip": "High tempo — ensure clarity in dense sections"
        })

    # -------------------------
    # 🎵 SECTION MELODY
    # -------------------------
    melody_sections = analyze_melody_by_section(melody, sections)

    # -------------------------
    # 🔥 SECTION ANALYSIS
    # -------------------------
    for sec in sections:

        sec_type = sec["type"]
        sec_energy = sec.get("energy", 0)

        match = next(
            (m for m in section_matches if m["section"] == sec_type),
            None
        )

        # find melody for this section
        mel = next((m for m in melody_sections if m["section"] == sec_type), None)

        # -------------------------
        # DROP
        # -------------------------
        if sec_type == "drop":

            if match:
                ref_energy = match["match"]["reference_energy"]

                if sec_energy < ref_energy:
                    advice.append({
                        "section": "drop",
                        "issue": "weak_drop",
                        "suggestion": "Increase drop energy (layers, sub, stereo width)"
                    })

            if mel and mel["variation"] < 25:
                advice.append({
                    "section": "drop",
                    "issue": "weak_melody",
                    "suggestion": "Increase pitch movement or rhythm density"
                })

            if swing < 0.01:
                advice.append({
                    "section": "drop",
                    "tip": "Add groove variation (hi-hats/percussion)"
                })

            advice.append({
                "section": "drop",
                "tip": "Add pre-drop silence or riser"
            })

        # -------------------------
        # BUILD
        # -------------------------
        elif sec_type == "build":

            advice.append({
                "section": "build",
                "tip": "Use automation (filter, reverb, pitch risers)"
            })

            if progression["complexity"] <= 2:
                advice.append({
                    "section": "build",
                    "issue": "harmonic_static",
                    "suggestion": "Add chord movement for tension"
                })

        # -------------------------
        # VERSE / HOOK
        # -------------------------
        elif sec_type in ["verse", "hook"]:

            advice.append({
                "section": sec_type,
                "tip": "Ensure contrast with other sections"
            })

            if mel and mel["pitch_range"] < 5:
                advice.append({
                    "section": sec_type,
                    "issue": "melody_flat",
                    "suggestion": "Use wider melodic range"
                })

    # -------------------------
    # 🎯 GENRE RULES
    # -------------------------
    advice.extend(apply_genre_rules(genre, sections))

    return advice