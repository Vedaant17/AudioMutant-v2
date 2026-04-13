import os
import numpy as np
import librosa
import traceback
import torch

from features.dynamics.transient_features import extract_transient_features
from utils.audio_loader import load_audio
from features.base_features import extract_base, save_reference
from features.spectral_features import extract_spectral
from features.dynamics_features import extract_dynamics
from features.stereo_features import extract_stereo_features
from features.stereo.mid_side_features import extract_mid_side
from features.dynamics.section_loudness import extract_section_loudness
from features.structure.section_detection import detect_sections

# ML Models
from ml.models.embedding_model import EmbeddingModel
from ml.models.section_embedding_model import SectionEmbeddingModel

SUPPORTED_FORMATS = (".wav", ".mp3", ".flac")

# -------------------------
# 🔧 PATH CONFIGURATION
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACK_MODEL_PATH = os.path.join(BASE_DIR, "ml", "models", "embedding_model.pth")
SECTION_MODEL_PATH = os.path.join(
    BASE_DIR, "ml", "models", "section_embedding_model.pth"
)

# -------------------------
# 🤖 LOAD TRACK EMBEDDING MODEL
# -------------------------
embedding_model = EmbeddingModel(input_dim=9)
embedding_model.load_state_dict(
    torch.load(TRACK_MODEL_PATH, map_location=torch.device("cpu"))
)
embedding_model.eval()
print("✅ Loaded EmbeddingModel for reference generation")

# -------------------------
# 🤖 LOAD SECTION EMBEDDING MODEL
# -------------------------
section_embedding_model = SectionEmbeddingModel(
    input_dim=7, embedding_dim=32
)
section_embedding_model.load_state_dict(
    torch.load(SECTION_MODEL_PATH, map_location=torch.device("cpu"))
)
section_embedding_model.eval()
print("✅ Loaded SectionEmbeddingModel for reference generation")


# -------------------------
# 🔢 FEATURE VECTOR BUILDERS
# -------------------------
def build_feature_vector(features: dict) -> np.ndarray:
    """Builds the 9-D feature vector used by the track EmbeddingModel."""
    return np.array([
        features.get("tempo_bpm", 0),
        features.get("mid_energy", 0),
        features.get("side_ratio", 0),
        features.get("integrated_lufs", 0),
        features.get("kick_punch", 0),
        features.get("transient_strength", 0),
        features.get("energy_mean", 0),
        features.get("stereo_width", 0),
        features.get("spectral_centroid", 0),
    ], dtype=np.float32)


def build_section_feature_vector(section: dict) -> np.ndarray:
    """Builds the 7-D feature vector used by the SectionEmbeddingModel."""
    return np.array([
        section.get("kick_punch", 0.0),
        section.get("mid_energy", 0.0),
        section.get("side_ratio", 0.0),
        section.get("lufs", 0.0),
        section.get("transient_strength", 0.0),
        section.get("transient_variation", 0.0),
        section.get("attack_sharpness", 0.0),
    ], dtype=np.float32)


def normalize_features(vec: np.ndarray) -> np.ndarray:
    """Standard score normalization."""
    return (vec - np.mean(vec)) / (np.std(vec) + 1e-6)


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2 normalization for cosine similarity."""
    return vec / (np.linalg.norm(vec) + 1e-10)


# -------------------------
# 🧠 EMBEDDING GENERATORS
# -------------------------
def generate_learned_embedding(features: dict) -> np.ndarray:
    """Generates a learned track embedding."""
    vec = normalize_features(build_feature_vector(features))

    with torch.no_grad():
        embedding = embedding_model(
            torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
        ).cpu().numpy()[0]

    return l2_normalize(embedding)


def generate_section_embedding(section_features: dict) -> list:
    vec = normalize_features(build_section_feature_vector(section_features))
    with torch.no_grad():
        embedding = section_embedding_model(
            torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
        ).cpu().numpy()[0]
    embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
    return embedding.tolist()

# -------------------------
# 🎯 SECTION EMBEDDINGS
# -------------------------
def extract_section_embeddings(y, sr, sections):
    section_data = []
    section_lufs = extract_section_loudness(y, sr, sections)

    for i, sec in enumerate(sections):
        try:
            start = int(sec["start"] * sr)
            end = int(sec["end"] * sr)

            y_sec_raw = y[:, start:end] if y.ndim > 1 else y[start:end]
            if y_sec_raw is None or len(y_sec_raw) == 0:
                continue

            y_sec_mono = (
                librosa.to_mono(y_sec_raw)
                if y_sec_raw.ndim > 1
                else y_sec_raw
            )

            # Feature extraction
            transients = extract_transient_features(y_sec_mono, sr)
            mid_side = extract_mid_side(y_sec_raw)
            lufs_val = (
                section_lufs[i] if i < len(section_lufs) else 0.0
            )

            # Build feature dict for embedding
            section_feature_dict = {
                "kick_punch": transients.get("kick_punch", 0),
                "mid_energy": mid_side.get("mid_energy", 0),
                "side_ratio": mid_side.get("side_ratio", 0),
                "lufs": lufs_val,
                "transient_strength": transients.get(
                    "transient_strength", 0
                ),
                "transient_variation": transients.get(
                    "transient_variation", 0
                ),
                "attack_sharpness": transients.get(
                    "attack_sharpness", 0
                ),
            }

            # Generate learned embedding
            section_embedding = generate_section_embedding(
                section_feature_dict
            )

            section_data.append({
                "type": sec.get("type"),
                "start": sec.get("start"),
                "end": sec.get("end"),
                "lufs": lufs_val,
                "transient_strength": section_feature_dict["transient_strength"],
                "transient_variation": section_feature_dict["transient_variation"],
                "attack_sharpness": section_feature_dict["attack_sharpness"],
                "kick_punch": section_feature_dict["kick_punch"],
                "mid_energy": mid_side.get("mid_energy", 0),
                "side_energy": mid_side.get("side_energy", 0),
                "side_ratio": mid_side.get("side_ratio", 0),
                "mid_side_balance": mid_side.get(
                    "mid_side_balance", 0
                ),
                "embedding": section_embedding,
            })

        except Exception as e:
            print(f"⚠️ Section processing failed: {e}")
            continue

    return section_data


# -------------------------
# 🔄 NORMALIZE SECTION TYPE
# -------------------------
def normalize_section_type(t):
    if t in ["drop", "high_energy"]:
        return "chorus"
    if t in ["build", "mid_energy"]:
        return "verse"
    return t


# -------------------------
# 🎧 PROCESS SINGLE FILE
# -------------------------
def process_file(file_path, genre):
    try:
        print(f"\n🎧 Processing: {file_path}")
        y, sr = load_audio(file_path)

        # FEATURE EXTRACTION
        base = extract_base(y, sr)
        spectral = extract_spectral(y, sr)
        dynamics = extract_dynamics(y, sr)
        stereo = extract_stereo_features(y, sr)
        transients = extract_transient_features(y, sr)
        mid_side_features = extract_mid_side(y)

        # STRUCTURE
        sections_raw = detect_sections(y, sr, genre)
        sections_clean = [{
            "type": normalize_section_type(s.get("type")),
            "start": s.get("start"),
            "end": s.get("end")
        } for s in sections_raw]

        # FEATURE DICTIONARY
        features = {
            "tempo_bpm": base.get("tempo_bpm"),
            "key_signature": base.get("key_signature"),
            "energy_mean": dynamics.get("loudness_rms"),
            "dynamics_range": dynamics.get("dynamic_range"),
            "spectral_centroid": spectral.get("spectral_centroid"),
            "spectral_bandwidth": spectral.get("spectral_bandwidth"),
            "spectral_rolloff": spectral.get("spectral_rolloff"),
            "stereo_width": stereo.get("stereo_width"),
            "transient_strength": transients.get("transient_strength", 0),
            "transient_variation": transients.get("transient_variation", 0),
            "attack_sharpness": transients.get("attack_sharpness", 0),
            "kick_punch": transients.get("kick_punch", 0),
            "mid_energy": mid_side_features.get("mid_energy", 0),
            "side_energy": mid_side_features.get("side_energy", 0),
            "side_ratio": mid_side_features.get("side_ratio", 0),
            "mid_side_balance": mid_side_features.get("mid_side_balance", 0),
            "zero_crossing_rate": spectral.get("zero_crossing_rate"),
            "rms": spectral.get("rms"),
            "harmonic_ratio": spectral.get("harmonic_ratio"),
            "percussive_ratio": spectral.get("percussive_ratio"),
            "integrated_lufs": spectral.get("integrated_lufs"),
            "true_peak": spectral.get("true_peak"),
        }

        # 🧠 TRACK EMBEDDING
        learned_embedding = generate_learned_embedding(features)
        features["learned_embedding"] = learned_embedding.tolist()

        # 🧠 SECTION EMBEDDINGS
        section_embeddings = extract_section_embeddings(
            y, sr, sections_clean
        )

        # FINAL JSON STRUCTURE
        track_name = os.path.splitext(
            os.path.basename(file_path)
        )[0]

        reference_data = {
            "track": track_name,
            "artist": "Unknown",
            "genre": genre,
            "features": features,
            "sections": section_embeddings
        }

        save_reference(
            track_name=track_name,
            artist="Unknown",
            genre=genre,
            data=reference_data
        )

        print(f"✅ Done: {track_name}")
        print(f"   Sections saved: {len(section_embeddings)}")

    except Exception as e:
        print(f"❌ Failed: {file_path}")
        print("Error:", e)
        traceback.print_exc()


# -------------------------
# 🔁 PROCESS ALL GENRES
# -------------------------
def process_all_genres(base_folder):
    for genre in os.listdir(base_folder):
        genre_path = os.path.join(base_folder, genre)
        if not os.path.isdir(genre_path):
            continue

        print(f"\n🔥 Processing GENRE: {genre}")

        files = os.listdir(genre_path)
        for i, file in enumerate(files):
            if file.lower().endswith(SUPPORTED_FORMATS):
                print(f"[{i+1}/{len(files)}]")
                process_file(os.path.join(genre_path, file), genre)


# -------------------------
# 🚀 ENTRY POINT
# -------------------------
if __name__ == "__main__":
    DATASET_ROOT = os.path.join(BASE_DIR, "datasets")  # Update if needed
    process_all_genres(DATASET_ROOT)
    print("\n🚀 ALL TRACKS PROCESSED")