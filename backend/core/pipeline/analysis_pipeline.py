import os
import librosa
import numpy as np
import torch
from typing import List, Dict

from utils.audio_loader import load_audio

# FEATURES
from scripts.generate_references import normalize_section_type
from features.base_features import extract_base
from features.spectral_features import extract_spectral
from features.dynamics_features import extract_dynamics
from features.stereo_features import extract_stereo_features
from features.structure.section_detection import detect_sections
from features.dynamics.transient_features import extract_transient_features
from features.stereo.mid_side_features import extract_mid_side
from features.dynamics.section_loudness import extract_section_loudness

# ANALYSIS
from analysis.difference_engine import DifferenceEngine
from analysis.advice_engine import AdviceEngine
from analysis.structure_analyzer import StructureAnalyzer
from analysis.advanced_advisor import AdvancedAdvisor

# ML
from ml.models.track_model import TrackScoringModel
from ml.models.section_model import SectionScoringModel
from ml.models.embedding_model import EmbeddingModel
from ml.models.section_embedding_model import SectionEmbeddingModel
from ml.faiss.faiss_matcher import FAISSMatcher

# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Track-level FAISS
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "ml", "faiss", "faiss_index.ivf")
FAISS_METADATA_PATH = os.path.join(BASE_DIR, "ml", "faiss", "faiss_metadata.json")

# Section-level FAISS
SECTION_FAISS_INDEX_PATH = os.path.join(
    BASE_DIR, "ml", "faiss", "section_index.faiss"
)
SECTION_FAISS_METADATA_PATH = os.path.join(
    BASE_DIR, "ml", "faiss", "section_metadata.json"
)

# Models
TRACK_MODEL_PATH = os.path.join(BASE_DIR, "ml/models/track_model.pth")
SECTION_MODEL_PATH = os.path.join(BASE_DIR, "ml/models/section_model.pth")
EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "ml/models/embedding_model.pth")
SECTION_EMBEDDING_MODEL_PATH = os.path.join(
    BASE_DIR, "ml/models/section_embedding_model.pth"
)

# --------------------------------------------------
# INIT SYSTEMS
# --------------------------------------------------
matcher = FAISSMatcher(
    index_path=FAISS_INDEX_PATH,
    metadata_path=FAISS_METADATA_PATH,
    dimension=73,  # 9 structured + 64 learned embedding
    use_cosine=True,
    nprobe=10,
)

section_matcher = FAISSMatcher(
    index_path=SECTION_FAISS_INDEX_PATH,
    metadata_path=SECTION_FAISS_METADATA_PATH,
    dimension=32,  # Section embedding dimension
    use_cosine=True,
    nprobe=10,
)

diff_engine = DifferenceEngine()
advisor = AdviceEngine()
structure_analyzer = StructureAnalyzer()
advanced_advisor = AdvancedAdvisor()

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
def load_model(model, path, name):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        print(f"✅ Loaded {name}")
    else:
        raise FileNotFoundError(f"{name} not found at {path}")

track_model = TrackScoringModel(input_dim=9)
section_model = SectionScoringModel(input_dim=4)
embedding_model = EmbeddingModel(input_dim=9)
section_embedding_model = SectionEmbeddingModel(input_dim=7, embedding_dim=32)

load_model(track_model, TRACK_MODEL_PATH, "TrackScoringModel")
load_model(section_model, SECTION_MODEL_PATH, "SectionScoringModel")
load_model(embedding_model, EMBEDDING_MODEL_PATH, "EmbeddingModel")
load_model(section_embedding_model, SECTION_EMBEDDING_MODEL_PATH, "SectionEmbeddingModel")

# --------------------------------------------------
# FEATURE BUILDERS
# --------------------------------------------------
def build_feature_vector(f: Dict) -> np.ndarray:
    return np.array([
        f.get("tempo_bpm", 0),
        f.get("mid_energy", 0),
        f.get("side_ratio", 0),
        f.get("integrated_lufs", 0),
        f.get("kick_punch", 0),
        f.get("transient_strength", 0),
        f.get("energy_mean", 0),
        f.get("stereo_width", 0),
        f.get("spectral_centroid", 0),
    ], dtype=np.float32)

def build_section_feature_vector(s: Dict) -> np.ndarray:
    return np.array([
        s.get("kick_punch", 0),
        s.get("mid_energy", 0),
        s.get("side_ratio", 0),
        s.get("lufs", 0),
        s.get("transient_strength", 0),
        s.get("transient_variation", 0),
        s.get("attack_sharpness", 0),
    ], dtype=np.float32)

def normalize_features(vec: np.ndarray) -> np.ndarray:
    return (vec - np.mean(vec)) / (np.std(vec) + 1e-6)

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-10)

# --------------------------------------------------
# SECTION EMBEDDING GENERATION
# --------------------------------------------------
def generate_section_embeddings(y, sr, sections):
    section_lufs = extract_section_loudness(y, sr, sections)
    embeddings = []

    for i, sec in enumerate(sections):
        start = int(sec["start"] * sr)
        end = int(sec["end"] * sr)
        y_sec = y[:, start:end] if y.ndim > 1 else y[start:end]
        y_mono = librosa.to_mono(y_sec) if y_sec.ndim > 1 else y_sec

        mid = extract_mid_side(y_sec)
        trans = extract_transient_features(y_mono, sr)

        sec.update({
            "mid_energy": mid.get("mid_energy", 0),
            "side_ratio": mid.get("side_ratio", 0),
            "kick_punch": trans.get("kick_punch", 0),
            "transient_strength": trans.get("transient_strength", 0),
            "transient_variation": trans.get("transient_variation", 0),
            "attack_sharpness": trans.get("attack_sharpness", 0),
            "lufs": section_lufs[i] if i < len(section_lufs) else 0,
        })

        vec = normalize_features(build_section_feature_vector(sec))
        with torch.no_grad():
            emb = section_embedding_model(
                torch.tensor(vec).unsqueeze(0)
            ).cpu().numpy()[0]

        emb = l2_normalize(emb)
        sec["embedding"] = emb.tolist()
        embeddings.append(emb)

    return sections, embeddings

# --------------------------------------------------
# SECTION SIMILARITY USING FAISS
# --------------------------------------------------
def compute_section_similarity_faiss(input_embeddings: List[np.ndarray], top_k: int = 3):
    similarities = []
    matched_sections = []

    for i, emb in enumerate(input_embeddings):
        matches = section_matcher.find_similar_by_vector(
            vector=emb.astype(np.float32),
            top_k=top_k
        )
        print(f"Matches found for section {i}: {len(matches)}")
        if matches:
            similarities.append(matches[0]["similarity"])
            matched_sections.extend(matches)

    if not similarities:
        return 0.0, []

    return float(np.mean(similarities)), matched_sections

# --------------------------------------------------
# SIMILARITY EXPLANATION
# --------------------------------------------------
def explain_similarity(inp, ref):
    reasons = []
    if abs(inp.get("tempo_bpm", 0) - ref.get("tempo_bpm", 0)) < 5:
        reasons.append("Similar tempo")
    if abs(inp.get("energy_mean", 0) - ref.get("energy_mean", 0)) < 0.05:
        reasons.append("Similar energy")
    if abs(inp.get("stereo_width", 0) - ref.get("stereo_width", 0)) < 0.1:
        reasons.append("Similar stereo width")
    return reasons

def suggest_improvements(inp, ref):
    tips = []
    if inp.get("kick_punch", 0) < ref.get("kick_punch", 0):
        tips.append("Increase kick punch")
    if inp.get("stereo_width", 0) < ref.get("stereo_width", 0):
        tips.append("Widen stereo image")
    if inp.get("energy_mean", 0) < ref.get("energy_mean", 0):
        tips.append("Increase energy")
    return tips

# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------
def run_full_analysis(file_path: str, genre: str = None, top_k: int = 5):
    print("\n🎧 Loading audio...")
    y, sr = load_audio(file_path)

    print("\n⚙️ Extracting features...")
    base = extract_base(y, sr)
    spectral = extract_spectral(y, sr)
    dynamics = extract_dynamics(y, sr)
    stereo = extract_stereo_features(y, sr)
    transients = extract_transient_features(y, sr)
    mid_side = extract_mid_side(y)

    features = {
        **base,
        **spectral,
        **dynamics,
        **stereo,
        "energy_mean": dynamics.get("loudness_rms", 0),
        "transient_strength": transients.get("transient_strength", 0),
        "kick_punch": transients.get("kick_punch", 0),
        "mid_energy": mid_side.get("mid_energy", 0),
        "side_ratio": mid_side.get("side_ratio", 0),
        "genre": genre,
    }

    # -------------------------
    # TRACK EMBEDDING
    # -------------------------
    vec = normalize_features(build_feature_vector(features))
    with torch.no_grad():
        learned_embedding = embedding_model(
            torch.tensor(vec).unsqueeze(0)
        ).cpu().numpy()[0]

    learned_embedding = l2_normalize(learned_embedding)
    features["learned_embedding"] = learned_embedding.tolist()

    # -------------------------
    # TRACK SCORE
    # -------------------------
    with torch.no_grad():
        track_score = track_model(
            torch.tensor(vec).unsqueeze(0)
        ).item()
    track_score = round(track_score * 100, 2)

    # -------------------------
    # SECTION DETECTION
    # -------------------------
    print("\n🧱 Detecting sections...")
    sections = detect_sections(y, sr, genre) or []
    for sec in sections:
        sec["type"] = normalize_section_type(sec.get("type"))

    sections, input_section_embeddings = generate_section_embeddings(
        y, sr, sections
    )

    # -------------------------
    # TRACK-LEVEL FAISS SEARCH
    # -------------------------
    print("\n🔍 Finding similar tracks...")
    print("\n🧪 Query Track Debug Info:")
    print(f"🎵 Tempo: {features.get('tempo_bpm')}")
    print(f"🎼 Genre: {genre}")
    print(f"📐 Feature Vector Dimension: {len(build_feature_vector(features))}")
    print(f"🧠 Embedding Dimension: {len(features.get('learned_embedding', []))}")

    matches = matcher.find_similar(
        features=features,
        top_k=top_k,
        genre_filter=genre,
        tempo_range=(
            features.get("tempo_bpm", 0) - 5,
            features.get("tempo_bpm", 0) + 5,
        ),
    )
    print(f"\n🔎 Number of raw matches from FAISS: {len(matches)}")
    if matches:
        print("🏆 Top match preview:")
        print(matches[0])

    # -------------------------
    # HYBRID SCORING
    # -------------------------
    section_similarity, section_matches = compute_section_similarity_faiss(
        input_section_embeddings
    )

    hybrid_results = []
    for m in matches:
        ref_track = m["track"]
        track_similarity = m["similarity"]

        final_score = 0.7 * track_similarity + 0.3 * section_similarity

        hybrid_results.append({
            "track": ref_track,
            "track_similarity": float(track_similarity),
            "section_similarity": float(section_similarity),
            "hybrid_similarity": float(final_score),
            "matched_sections": section_matches,
            "why_similar": explain_similarity(
                features,
                ref_track.get("features", {})
            ),
            "what_to_copy": suggest_improvements(
                features,
                ref_track.get("features", {})
            )
        })

    hybrid_results.sort(
        key=lambda x: x["hybrid_similarity"], reverse=True
    )

    best_match = hybrid_results[0]["track"] if hybrid_results else None

    # -------------------------
    # DIFFERENCE + ADVICE
    # -------------------------
    if best_match:
        track_diffs = diff_engine.compare_track(
            features,
            best_match.get("features", {})
        )
        section_diffs = diff_engine.compare_sections(
            sections,
            best_match.get("sections", [])
        )
    else:
        track_diffs, section_diffs = {}, []

    track_advice = advisor.generate_track_advice(
        track_diffs,
        score=track_score
    )
    section_advice = advisor.generate_section_advice(section_diffs)

    print("\n✅ Analysis complete!")

    return {
        "ml_scores": {
            "track_score": track_score,
        },
        "features": features,
        "sections": sections,
        "similar_tracks": hybrid_results[:top_k],
        "track_advice": track_advice,
        "section_advice": section_advice,
    }