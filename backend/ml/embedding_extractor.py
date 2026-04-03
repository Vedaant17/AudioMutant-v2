import numpy as np
import librosa


class EmbeddingExtractor:
    def extract_embedding(self, y, sr):

        # -------------------------
        # Safety check
        # -------------------------
        if y is None or len(y) == 0:
            return None

        # Convert to mono (IMPORTANT)
        if y.ndim > 1:
            y = librosa.to_mono(y)

        # Remove silence
        if np.max(np.abs(y)) < 1e-4:
            return None

        # -------------------------
        # Feature Extraction
        # -------------------------
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        except Exception as e:
            print("⚠️ Feature extraction failed:", e)
            return None

        # -------------------------
        # Stats function
        # -------------------------
        def stats(x):
            return np.array([
                np.mean(x),
                np.std(x)
            ])

        # -------------------------
        # Build embedding
        # -------------------------
        embedding = np.concatenate([
            stats(mfcc),         # 2
            stats(chroma),       # 2
            stats(centroid),     # 2
            stats(bandwidth),    # 2
            stats(rolloff)       # 2
        ])

        # -------------------------
        # Normalize
        # -------------------------
        norm = np.linalg.norm(embedding)

        if norm == 0 or np.isnan(norm):
            return None

        embedding = embedding / norm

        return embedding.astype(np.float32)