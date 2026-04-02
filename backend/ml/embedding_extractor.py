import numpy as np
import librosa
import tensorflow_hub as hub

class EmbeddingExtractor:
    def __init__(self):
        print("🔄 Loading VGGish model...")
        self.model = hub.load("https://tfhub.dev/google/vggish/1")

    def extract_embedding(self, y, sr):

    # -------------------------
    # PREPROCESS
    # -------------------------
        if y.ndim > 1:
            y = librosa.to_mono(y)

        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000

        y = y.astype(np.float32)

    # -------------------------
    # CHUNK AUDIO (IMPORTANT)
    # -------------------------
        chunk_size = sr  # 1 second
        embeddings_list = []

        for i in range(0, len(y), chunk_size):
            chunk = y[i:i + chunk_size]

            if len(chunk) < chunk_size:
                continue

            emb = self.model(chunk)  # ✅ CORRECT INPUT
            embeddings_list.append(emb.numpy())

        if len(embeddings_list) == 0:
            return np.zeros(128)

        embeddings = np.vstack(embeddings_list)

    # -------------------------
    # FINAL EMBEDDING
    # -------------------------
        return np.mean(embeddings, axis=0)