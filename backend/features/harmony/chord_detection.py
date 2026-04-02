import numpy as np
import librosa

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

import numpy as np
import librosa

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

import numpy as np
import librosa

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def detect_chords(y, sr):

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    hop_length = 512
    times = librosa.frames_to_time(
        np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length
    )

    chords = []
    window = 50  # frames (~1–2 sec)

    for i in range(0, chroma.shape[1], window):

        if i >= len(times):
            break

        segment = chroma[:, i:i+window]

        if segment.shape[1] == 0:
            continue

        segment_mean = segment.mean(axis=1)

        if np.isnan(segment_mean).any():
            continue

        # 🔥 NEW: top 3 notes instead of single note
        top_notes = np.argsort(segment_mean.flatten())[-3:]
        top_notes = sorted(top_notes.tolist())
        chord = "-".join([NOTE_NAMES[n % 12] for n in top_notes])

        chords.append({
            "time": float(times[i]),
            "chord": chord
        })

    return chords