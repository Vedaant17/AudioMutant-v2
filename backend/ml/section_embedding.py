import numpy as np

def extract_section_embeddings(y, sr, sections, extractor):

    section_embeddings = []

    for sec in sections:
        start_sample = max(0, int(sec["start"] * sr))
        end_sample = min(len(y[0]) if y.ndim > 1 else len(y), int(sec["end"] * sr))

        if end_sample <= start_sample:
           continue

        # Convert to mono
        if y.ndim > 1:
            segment = np.mean(y[:, start_sample:end_sample], axis=0)
        else:
            segment = y[start_sample:end_sample]

        # Validate segment
        if segment.size == 0:
           continue

        if len(segment) < sr * 2:  # at least 2 seconds
            continue

        emb = extractor.extract_embedding(segment, sr)

        section_embeddings.append({
            "type": sec["type"],
            "embedding": emb.tolist(),
            "energy": sec.get("energy", 0.0)
        })

    return section_embeddings