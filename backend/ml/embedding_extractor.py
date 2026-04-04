import numpy as np
import librosa


class EmbeddingExtractor:
    TARGET_DIM = 10

    def extract_embedding(self, y, sr):

        # -------------------------
        # 🚨 Silence check
        # -------------------------
        if np.max(np.abs(y)) < 1e-4:
            return None

        try:
            # -------------------------
            # 🎧 MONO (IMPORTANT)
            # -------------------------
            y_mono = librosa.to_mono(y) if y.ndim > 1 else y

            # -------------------------
            # 🎼 CORE FEATURES
            # -------------------------
            mel = librosa.feature.melspectrogram(y=y_mono, sr=sr, n_mels=32)
            mel_db = librosa.power_to_db(mel)

            mfcc = librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=13)

            centroid = librosa.feature.spectral_centroid(y=y_mono, sr=sr)
            bandwidth = librosa.feature.spectral_bandwidth(y=y_mono, sr=sr)

            zcr = librosa.feature.zero_crossing_rate(y_mono)

            rms = librosa.feature.rms(y=y_mono)

            # -------------------------
            # 🎼 Harmonic / Percussive split
            # -------------------------
            y_harm, y_perc = librosa.effects.hpss(y_mono)

            harm_energy = np.mean(np.abs(y_harm))
            perc_energy = np.mean(np.abs(y_perc)) + 1e-6

            harmonic_ratio = harm_energy / (harm_energy + perc_energy)

            # -------------------------
            # 📊 BUILD SMART EMBEDDING
            # -------------------------
            embedding = np.array([
                np.mean(mel_db),
                np.std(mel_db),

                np.mean(mfcc),
                np.std(mfcc),

                np.mean(centroid),
                np.std(centroid),

                np.mean(bandwidth),

                np.mean(zcr),

                np.mean(rms),

                harmonic_ratio
            ], dtype=np.float32)

            # -------------------------
            # 🔒 DIMENSION LOCK
            # -------------------------
            embedding = self._fix_dimension(embedding)

            # -------------------------
            # 📏 NORMALIZE
            # -------------------------
            norm = np.linalg.norm(embedding)
            if norm == 0 or np.isnan(norm):
                return None

            embedding = embedding / norm

            # -------------------------
            # 🚨 FINAL CHECK
            # -------------------------
            if len(embedding) != self.TARGET_DIM:
                raise ValueError(
                    f"Embedding size mismatch: {len(embedding)} != {self.TARGET_DIM}"
                )

            return embedding

        except Exception as e:
            print("⚠️ Embedding extraction failed:", e)
            return None

    # -------------------------
    # 🧱 DIMENSION FIXER
    # -------------------------
    def _fix_dimension(self, embedding):

        if len(embedding) > self.TARGET_DIM:
            return embedding[:self.TARGET_DIM]

        elif len(embedding) < self.TARGET_DIM:
            pad_size = self.TARGET_DIM - len(embedding)
            return np.pad(embedding, (0, pad_size))

        return embedding