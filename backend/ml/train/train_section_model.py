import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ml.models.section_model import SectionScoringModel

# ✅ LOAD CORRECT DATA
X = np.load("ml/train/section_X.npy")
y = np.load("ml/train/section_y.npy")

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

model = SectionScoringModel(input_dim=X.shape[1])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------
# TRAIN
# -------------------------
for epoch in range(50):
    optimizer.zero_grad()

    outputs = model(X)
    loss = criterion(outputs, y)

    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# -------------------------
# SAVE
# -------------------------
torch.save(model.state_dict(), "ml/models/section_model.pth")

print("✅ Section model trained & saved")