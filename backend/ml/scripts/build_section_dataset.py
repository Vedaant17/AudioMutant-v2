import os
import json

# --------------------------------------------------
# 📁 PATH CONFIGURATION
# --------------------------------------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
REFERENCE_DIR = os.path.join(BASE_DIR, "reference_data")
OUTPUT_FILE = os.path.join(BASE_DIR, "ml", "data", "section_dataset.json")

# --------------------------------------------------
# 🔢 FEATURE EXTRACTION FROM SECTIONS
# --------------------------------------------------
def extract_section_features(section: dict) -> dict:
    """Extracts the 7 features used for section embeddings."""
    return {
        "kick_punch": section.get("kick_punch", 0.0),
        "mid_energy": section.get("mid_energy", 0.0),
        "side_ratio": section.get("side_ratio", 0.0),
        "lufs": section.get("lufs", 0.0),
        "transient_strength": section.get("transient_strength", 0.0),
        "transient_variation": section.get("transient_variation", 0.0),
        "attack_sharpness": section.get("attack_sharpness", 0.0),
    }

# --------------------------------------------------
# 📦 DATASET BUILDER
# --------------------------------------------------
def build_section_dataset():
    dataset = []
    total_files = 0
    total_sections = 0

    print(f"📂 Scanning reference data in: {REFERENCE_DIR}")

    for root, _, files in os.walk(REFERENCE_DIR):
        genre_from_folder = os.path.basename(root)

        for file in files:
            if not file.lower().endswith(".json"):
                continue

            file_path = os.path.join(root, file)
            total_files += 1

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                track = data.get("track", os.path.splitext(file)[0])
                genre = data.get("genre", genre_from_folder)

                for section in data.get("sections", []):
                    # Ensure essential values exist
                    features = extract_section_features(section)

                    # Skip sections with missing loudness or invalid data
                    if features["lufs"] is None:
                        continue

                    dataset.append({
                        "track": track,
                        "genre": genre,
                        "type": section.get("type", "unknown"),
                        "features": features
                    })
                    total_sections += 1

            except Exception as e:
                print(f"⚠️ Error processing {file_path}: {e}")

    # --------------------------------------------------
    # 💾 SAVE DATASET
    # --------------------------------------------------
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print("\n✅ Section dataset successfully created!")
    print(f"📄 Output file: {OUTPUT_FILE}")
    print(f"🎵 Tracks processed: {total_files}")
    print(f"🎼 Sections collected: {total_sections}")


# --------------------------------------------------
# 🚀 ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    build_section_dataset()