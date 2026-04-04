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


# -------------------------
# GENRE NORMALIZATION
# -------------------------
def normalize_genre(g):
    if not g:
        return "general"

    g = g.lower()

    if "edm" in g or "house" in g or "trance" in g:
        return "edm"
    elif "rock" in g:
        return "rock"
    elif "hiphop" in g or "rap" in g:
        return "hiphop"
    else:
        return "general"


# -------------------------
# MAIN PIPELINE (SYNC FOR TESTING)
# -------------------------
def run_full_analysis(file_path):

    print("\n🎧 Loading audio...")
    y, sr = load_audio(file_path)
    y_mono = librosa.to_mono(y) if y.ndim > 1 else y
    stft = np.abs(librosa.stft(y_mono, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    print(f"Sample rate: {sr}")
    print(f"Audio shape: {y.shape}")

    # -------------------------
    # CORE FEATURES
    # -------------------------
    print("\n⚙️ Extracting core features...")
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
    # TRACK EMBEDDING
    # -------------------------
    print("\n🧠 Extracting track embedding...")
    track_embedding = extractor.extract_embedding(y, sr)
    features["embedding"] = track_embedding.tolist()

    # -------------------------
    # TRACK MATCHING
    # -------------------------
    print("\n🔍 Matching reference tracks...")
    match_data = matcher.find_best_match(features, y, sr, sections=section_embeddings)

    best_match = match_data.get("best")
    matches = match_data.get("top_k", [])

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

    normalized_genre = normalize_genre(predicted_genre)

    print(f"🎼 Predicted genre: {predicted_genre} → {normalized_genre}")

    # -------------------------
    # MUSICAL FEATURES
    # -------------------------
    print("\n🎼 Extracting musical features...")
    melody = extract_melody_contour(y_mono, sr)
    chords = detect_chords(y, sr)
    beat_grid = extract_beat_grid(y, sr)

    # -------------------------
    # STRUCTURE
    # -------------------------
    print("\n🧱 Detecting sections...")
    sections = detect_sections(y, sr, genre=normalized_genre)

    print(f"Sections found: {len(sections)}")

    # -------------------------
    # SECTION EMBEDDINGS
    # -------------------------
    print("\n🧠 Extracting section embeddings...")
    section_embeddings = extract_section_embeddings(
        y, sr, sections, extractor
    )

    print(f"Section embeddings: {len(section_embeddings)}")

    # -------------------------
    # ATTACH EMBEDDINGS SAFELY
    # -------------------------
    for i, sec in enumerate(sections):
        if i < len(section_embeddings):
            sec["embedding"] = section_embeddings[i].get("embedding")
        else:
            sec["embedding"] = None

    # -------------------------
    # SECTION MATCHING
    # -------------------------
    print("\n🔗 Matching sections...")
    if hasattr(matcher, "reference_data") and matcher.reference_data:
        section_matcher = SectionMatcher(matcher.reference_data)
        section_matches = section_matcher.find_best_section_match(
            section_embeddings
        )
    else:
        section_matches = []

    # -------------------------
    # MIX + ARRANGEMENT
    # -------------------------
    print("\n🎚️ Analyzing mix and arrangement...")
    drums = analyze_drums(y, sr)
    masking = detect_masking(stft, freqs)
    loudness_curve = extract_loudness_curve(y)

    mix = analyze_mix(base, spectral, dynamics, stereo)

    # -------------------------
    # ISSUES
    # -------------------------
    print("\n⚠️ Detecting issues...")
    timeline_issues = detect_timeline_issues(
        y, sr, stft, freqs, mix
    )

    # -------------------------
    # COMPOSITION ENGINE
    # -------------------------
    print("\n🎼 Generating composition advice...")
    composition_advice = composition_engine(
        harmony={"key": base["key_signature"], "chords": chords},
        rhythm=beat_grid,
        melody=melody,
        sections=sections,
        section_matches=section_matches,
        genre=predicted_genre
    )

    # -------------------------
    # MIX ADVISOR
    # -------------------------
    print("\n🎧 Generating mix advice...")
    mix_advice = mix_advisor(
        base=base,
        spectral=spectral,
        dynamics=dynamics,
        stereo=stereo,
        masking=masking,
        drums=drums,
        melody=melody
    )

    print("\n✅ Analysis complete!")

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

        "section_matches": section_matches,
        "composition_advice": composition_advice,
        "mix_advice": mix_advice
    }