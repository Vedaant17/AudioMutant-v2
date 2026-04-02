def detect_key(chroma):
    mean_chroma = chroma.mean(axis=1)
    key_index = mean_chroma.argmax()

    keys = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

    return {
        "key": keys[key_index],
        "confidence": float(mean_chroma[key_index])
    }