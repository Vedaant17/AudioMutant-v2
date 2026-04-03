import numpy as np

def extract_section_embeddings(y, sr, sections, extractor):

    section_embeddings = []

    for section in sections:
        start = int(section["start"] * sr)
        end = int(section["end"] * sr)

        y_section = y[start:end]

        if len(y_section) == 0:
            continue
       
        embedding = extractor.extract_embedding(y_section, sr)

        if embedding is None:
            embedding = np.zeros(10, dtype=np.float32)  # same size as your embedding

        section_embeddings.append({
            "type": section["type"],
           "embedding": embedding.tolist()
        })
        
    return section_embeddings