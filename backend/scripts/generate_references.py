import os
import numpy as np
import traceback

from utils.audio_loader import load_audio
from features.base_features import extract_base, save_reference
from features.spectral_features import extract_spectral
from features.dynamics_features import extract_dynamics
from features.stereo_features import extract_stereo_features

from ml.embedding_extractor import EmbeddingExtractor
from features.structure.section_detection import detect_sections


SUPPORTED_FORMATS = (".wav", ".mp3", ".flac")

# 🔥 Initialize ONCE
extractor = EmbeddingExtractor()


# -------------------------
# 🎯 SECTION EMBEDDING HELPER
# -------------------------
def extract_section_embeddings(y, sr, sections, extractor):
    section_data = []

    for sec in sections:
        try:
            start = int(sec["start"] * sr)
            end = int(sec["end"] * sr)

            y_sec = y[:, start:end] if y.ndim > 1 else y[start:end]

            # convert to mono
            if y_sec.ndim > 1:
                import librosa
                y_sec = librosa.to_mono(y_sec)

            emb = extractor.extract_embedding(y_sec, sr)

            if emb is None or np.isnan(emb).any():
                continue

            section_data.append({
                "type": sec.get("type"),
                "start": sec.get("start"),
                "end": sec.get("end"),
                "embedding": emb.tolist()
            })

        except Exception as e:
            print(f"⚠️ Section embedding failed: {e}")
            continue

    return section_data


# -------------------------
# 🎧 PROCESS SINGLE FILE
# -------------------------
def process_file(file_path, genre):
    try:
        print(f"\n🎧 Processing: {file_path}")

        y, sr = load_audio(file_path)

        # -------------------------
        # 🎧 BASIC FEATURES (LIGHT)
        # -------------------------
        base = extract_base(y, sr)
        spectral = extract_spectral(y, sr)
        dynamics = extract_dynamics(y, sr)
        stereo = extract_stereo_features(y, sr)

        features = {
            "tempo": base.get("tempo_bpm"),
            "key": base.get("key_signature"),
            "energy": dynamics.get("dynamic_range"),
            "spectral_centroid": spectral.get("spectral_centroid"),
            "stereo_width": stereo.get("stereo_width"),
        }

        # -------------------------
        # 🎼 STRUCTURE
        # -------------------------
        sections_raw = detect_sections(y, sr, genre)

        sections_clean = [
            {
                "type": s.get("type"),
                "start": s.get("start"),
                "end": s.get("end")
            }
            for s in sections_raw
        ]

        # -------------------------
        # 🤖 TRACK EMBEDDING
        # -------------------------
        embedding = extractor.extract_embedding(y, sr)

        if embedding is None:
            print("⚠️ Skipping (invalid embedding)")
            return

        if np.isnan(embedding).any():
            print("⚠️ Skipping (NaN embedding)")
            return

        # -------------------------
        # 🧠 SECTION EMBEDDINGS (🔥 NEW)
        # -------------------------
        section_embeddings = extract_section_embeddings(
            y, sr, sections_clean, extractor
        )

        embedding = extractor.extract_embedding(y, sr)

        # 🚨 SAFETY CHECKS (ADD HERE)
        if embedding is None:
            print("⚠️ Skipping (embedding failed)")
            return

        print("Embedding length:", len(embedding))  # Debug check

        if np.linalg.norm(embedding) == 0:
           print("⚠️ Skipping (zero embedding)")
           return

        if np.isnan(embedding).any():
           print("⚠️ Skipping (NaN embedding)")
           return

        # -------------------------
        # 📦 FINAL STRUCTURE (FLAT)
        # -------------------------
        track_name = os.path.splitext(os.path.basename(file_path))[0]

        reference_data = {
            "track": track_name,
             "artist": "Unknown",
             "genre": genre,

            # ✅ PURE FEATURES ONLY
            "features": {
                "tempo_bpm": base.get("tempo_bpm"),
                "key_signature": base.get("key_signature"),
                "energy_mean": dynamics.get("dynamic_range"),
                "spectral_centroid": spectral.get("spectral_centroid"),
                "stereo_width": stereo.get("stereo_width"),
            },

            # ✅ SEPARATE EMBEDDING
            "embedding": embedding.tolist(),

            # ✅ SEPARATE STRUCTURE
            "sections": section_embeddings
        }

        save_reference(
            track_name=track_name,
            artist="Unknown",
            genre=genre,
            data=reference_data   # now passing FULL object
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
# 🚀 ENTRY
# ------------------------------
if __name__ == "__main__":

    DATASET_ROOT = "datasets"  # ⚠️ ensure correct folder name

    process_all_genres(DATASET_ROOT)

    print("\n🚀 ALL TRACKS PROCESSED")