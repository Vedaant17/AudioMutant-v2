import os
import numpy as np
import traceback

from utils.audio_loader import load_audio
from features.base_features import extract_base, save_reference
from features.spectral_features import extract_spectral
from features.dynamics_features import extract_dynamics
from features.stereo_features import extract_stereo_features

from ml.embedding_extractor import EmbeddingExtractor
from ml.section_embedding import extract_section_embeddings

from features.structure.section_detection import detect_sections
from features.harmony.chord_detection import detect_chords
from features.rhythm.beat_tracking import extract_beat_grid


SUPPORTED_FORMATS = (".wav", ".mp3", ".flac")

# 🔥 Initialize ONCE
extractor = EmbeddingExtractor()


def process_file(file_path, genre):
    try:
        print(f"\n🎧 Processing: {file_path}")

        y, sr = load_audio(file_path)

        # -------------------------
        # BASE FEATURES
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
        # 🎼 MUSICAL FEATURES
        # -------------------------
        chords = detect_chords(y, sr)
        beat_grid = extract_beat_grid(y, sr)
        sections = detect_sections(y, sr, genre)

        # -------------------------
        # 🤖 EMBEDDINGS
        # -------------------------
        embedding = extractor.extract_embedding(y, sr)

        # 🚨 SAFETY CHECK
        if np.linalg.norm(embedding) == 0:
            print("⚠️ Skipping invalid embedding")
            return

        section_embeddings = extract_section_embeddings(y, sr, sections, extractor)

        # -------------------------
        # 📦 FORMAT FOR JSON
        # -------------------------
        features["embedding"] = embedding.tolist()

        # Convert section embeddings to list
        features["section_embeddings"] = [
            {
                "section": sec["type"],
                "embedding": sec["embedding"] if isinstance(sec["embedding"], list) else sec["embedding"].tolist()
            }
            for sec in section_embeddings
        ]

        # Store musical structure
        features["sections"] = sections
        features["chords"] = chords
        features["beat_grid"] = beat_grid

        # -------------------------
        # SAVE
        # -------------------------
        track_name = os.path.splitext(os.path.basename(file_path))[0]

        save_reference(
            track_name=track_name,
            artist="Unknown",
            genre=genre,
            features=features
        )

        print(f"✅ Done: {track_name}")

    except Exception as e:
        print(f"❌ Failed: {file_path}")
        print("Error:", e)
        traceback.print_exc()


def process_all_genres(base_folder):

    genres = os.listdir(base_folder)

    for genre in genres:

        genre_path = os.path.join(base_folder, genre)

        if not os.path.isdir(genre_path):
            continue

        print(f"\n🔥 Processing GENRE: {genre}")

        files = os.listdir(genre_path)

        for i, file in enumerate(files):

            if file.lower().endswith(SUPPORTED_FORMATS):
                print(f"[{i+1}/{len(files)}]")

                file_path = os.path.join(genre_path, file)
                process_file(file_path, genre)


# ------------------------------
# ENTRY
# ------------------------------
if __name__ == "__main__":

    DATASET_ROOT = "datasets"

    process_all_genres(DATASET_ROOT)