import librosa
import numpy as np
from collections import Counter

from utils.audio_loader import load_audio

# FEATURES
from features.base_features import extract_base
from features.spectral_features import extract_spectral
from features.dynamics_features import extract_dynamics
from features.stereo_features import extract_stereo_features
from features.harmony.melody_contour import extract_melody_contour
from features.harmony.chord_detection import detect_chords
from features.rhythm.beat_tracking import extract_beat_grid
from features.structure.section_detection import detect_sections
from features.arrangement.drum_analysis import analyze_drums
from features.arrangement.masking_analysis import detect_masking
from features.dynamics.loudness_curve import extract_loudness_curve

# ANALYSIS
from analysis.mix_engine import analyze_mix
from analysis.mix_advisor import mix_advisor
from analysis.composition_engine import composition_engine
from analysis.section_matcher import SectionMatcher

# ML
from ml.hybrid_matcher import HybridMatcher
from ml.embedding_extractor import EmbeddingExtractor
from ml.section_embedding import extract_section_embeddings

# OTHER
from issues.detect_timeline_issues import detect_timeline_issues


matcher = HybridMatcher()
extractor = EmbeddingExtractor()


async def run_full_analysis(file_path):

    # -------------------------
    # LOAD AUDIO
    # -------------------------
    y, sr = load_audio(file_path)
    y_mono = librosa.to_mono(y) if y.ndim > 1 else y

    # -------------------------
    # CORE FEATURES
    # -------------------------
    base = extract_base(y, sr)
    spectral = extract_spectral(y, sr)
    dynamics = extract_dynamics(y, sr)
    stereo = extract_stereo_features(y, sr)

    features = {
        **base,
        **spectral,
        **dynamics,
        **stereo
    }

    # -------------------------
    # 🔥 ADD TRACK EMBEDDING
    # -------------------------
    track_embedding = extractor.extract_embedding(y, sr)
    features["embedding"] = track_embedding.tolist()

    # -------------------------
    # TRACK MATCHING (HYBRID ML)
    # -------------------------
    match_data = matcher.find_best_match(features)

    best_match = match_data["best"]
    matches = match_data["top_k"]

    # -------------------------
    # GENRE PREDICTION
    # -------------------------
    if matches:
        genres = [m["genre"] for m in matches[:3]]
        counter = Counter(genres)
        predicted_genre, count = counter.most_common(1)[0]
        confidence = count / len(genres)
    else:
        predicted_genre = "Unknown"
        confidence = 0.0

    # -------------------------
    # 🎼 MUSICAL FEATURES
    # -------------------------
    melody = extract_melody_contour(y_mono, sr)
    chords = detect_chords(y, sr)
    beat_grid = extract_beat_grid(y, sr)

    # -------------------------
    # 🔥 STRUCTURE (GENRE-AWARE)
    # -------------------------
    sections = detect_sections(y, sr, genre=predicted_genre)

    # -------------------------
    # 🔥 SECTION EMBEDDINGS
    # -------------------------
    section_embeddings = extract_section_embeddings(
        y, sr, sections, extractor
    )

    # -------------------------
    # 🔥 SECTION MATCHING (NEW)
    # -------------------------
    section_matcher = SectionMatcher(matcher.reference_data)

    section_matches = section_matcher.find_best_section_match(
        section_embeddings
    )

    # -------------------------
    # MIX + ARRANGEMENT
    # -------------------------
    drums = analyze_drums(y, sr)
    masking = detect_masking(spectral["stft"], spectral["freqs"])
    loudness_curve = extract_loudness_curve(y)

    mix = analyze_mix(base, spectral, dynamics, stereo)

    # -------------------------
    # ISSUES
    # -------------------------
    timeline_issues = detect_timeline_issues(
        y, sr, spectral["stft"], spectral["freqs"], mix
    )

    # -------------------------
    # 🔥 COMPOSITION ENGINE (UPGRADED)
    # -------------------------
    composition_advice = composition_engine(
        harmony={"key": base["key_signature"], "chords": chords},
        rhythm=beat_grid,
        melody=melody,
        sections=sections,
        section_matches=section_matches
    )

    # -------------------------
    # 🔥 MIX ADVISOR (UPGRADED)
    # -------------------------
    mix_advice = mix_advisor(
        base=base,
        spectral=spectral,
        dynamics=dynamics,
        stereo=stereo,
        masking=masking,
        drums=drums
    )

    # -------------------------
    # FINAL OUTPUT
    # -------------------------
    return {
        "mix": mix,

        "predicted_genre": predicted_genre,
        "confidence": confidence,

        "harmony": {
            "key": base["key_signature"],
            "chords": chords
        },

        "melody": melody,

        "rhythm": {
            "beat_grid": beat_grid
        },

        "structure": {
            "sections": sections
        },

        "arrangement": {
            "drums": drums,
            "masking": masking
        },

        "stereo": stereo,

        "dynamics": {
            "loudness": loudness_curve
        },

        "timeline_issues": timeline_issues,

        "reference_matches": matches,
        "best_match": best_match,

        # 🔥 NEW INTELLIGENCE
        "section_matches": section_matches,
        "composition_advice": composition_advice,
        "mix_advice": mix_advice
    }