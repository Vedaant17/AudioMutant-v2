import librosa
import numpy as np


def load_audio(file_path, sr=22050, mono=False, duration=300):
    """
    Load audio safely for analysis

    Args:
        file_path (str)
        sr (int): sample rate
        mono (bool): keep stereo if False
        duration (int): limit duration (seconds)

    Returns:
        y (np.ndarray)
        sr (int)
    """

    try:
        y, sr = librosa.load(
            file_path,
            sr=sr,
            mono=mono,
            duration=duration
        )
        y = y / (np.max(np.abs(y)) + 1e-8)

        # Ensure float32 (important for performance)
        y = y.astype(np.float32)

        return y, sr

    except Exception as e:
        print(f"❌ Error loading audio: {file_path}")
        print(e)
        return None, None