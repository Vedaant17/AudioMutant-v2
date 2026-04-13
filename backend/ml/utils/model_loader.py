import torch
from ml.models.section_embedding_model import SectionEmbeddingModel

def load_section_embedding_model(model_path: str):
    """
    Loads a trained SectionEmbeddingModel from disk.
    """
    model = SectionEmbeddingModel(input_dim=7, embedding_dim=32)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print("✅ Loaded section embedding model")
    return model