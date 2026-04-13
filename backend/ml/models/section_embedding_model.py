import torch
import torch.nn as nn
import torch.nn.functional as F

class SectionEmbeddingModel(nn.Module):
    """
    Generates dense embeddings for individual song sections.
    Designed to mirror the logic of the TrackEmbeddingModel.
    """

    def __init__(self, input_dim: int = 7, embedding_dim: int = 32):
        super(SectionEmbeddingModel, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate normalized embeddings.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: L2-normalized embedding of shape (batch_size, embedding_dim)
        """
        embedding = self.network(x)
        embedding = F.normalize(embedding, p=2, dim=1)  # Normalize for cosine similarity
        return embedding