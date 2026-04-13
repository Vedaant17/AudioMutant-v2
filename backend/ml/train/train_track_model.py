import torch
import torch.nn as nn
import numpy as np

from ml.models.track_model import TrackScoringModel


# -------------------------
# LOAD DATA
# -------------------------
X = np.load("ml/train/X.npy")
y = np.load("ml/train/y.npy")

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)


# -------------------------
# MODEL
# -------------------------
model = TrackScoringModel(input_dim=X.shape[1])

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    inputs = torch.tensor(X)
    targets = torch.tensor(y).unsqueeze(1)

    outputs = model(inputs)

    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# -------------------------
# SAVE MODEL
# -------------------------
torch.save(model.state_dict(), "ml/models/track_model.pth")

print("✅ Track model trained & saved")