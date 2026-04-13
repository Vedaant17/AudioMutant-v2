import torch
from ml.models.track_model import TrackScoringModel
from ml.datasets.track_dataset import build_track_dataset

X, y = build_track_dataset("reference_data")

model = TrackScoringModel(input_dim=X.shape[1])

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

for epoch in range(20):
    preds = model(X)
    loss = loss_fn(preds, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}: {loss.item()}")

torch.save(model.state_dict(), "track_model.pth")