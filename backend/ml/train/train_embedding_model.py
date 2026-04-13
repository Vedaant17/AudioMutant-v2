import numpy as np
import torch
import torch.optim as optim

from ml.models.embedding_model import EmbeddingModel
from ml.losses.triplet_loss import triplet_loss

# -------------------------
# LOAD DATA
# -------------------------
anchor_X = np.load("ml/train/anchor_X.npy")
pos_X = np.load("ml/train/pos_X.npy")
neg_X = np.load("ml/train/neg_X.npy")

# -------------------------
# MODEL
# -------------------------
model = EmbeddingModel(input_dim=anchor_X.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------
# TRAIN
# -------------------------
for epoch in range(50):

    anchor = torch.tensor(anchor_X)
    positive = torch.tensor(pos_X)
    negative = torch.tensor(neg_X)

    anchor_emb = model(anchor)
    pos_emb = model(positive)
    neg_emb = model(negative)

    loss = triplet_loss(anchor_emb, pos_emb, neg_emb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# -------------------------
# SAVE
# -------------------------
torch.save(model.state_dict(), "ml/models/embedding_model.pth")

print("✅ Embedding model trained!")