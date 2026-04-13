import os
import torch
from torch.utils.data import DataLoader
from torch.nn import TripletMarginLoss
import torch.optim as optim

from ml.models.section_embedding_model import SectionEmbeddingModel
from ml.data.section_dataset_loader import SectionDataset


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = os.path.join(BASE_DIR, "ml", "data", "section_dataset.json")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "ml", "models", "section_embedding_model.pth")

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
EMBEDDING_DIM = 32

def train():
    dataset = SectionDataset(DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SectionEmbeddingModel(input_dim=7, embedding_dim=EMBEDDING_DIM)
    criterion = TripletMarginLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("🚀 Starting Section Embedding Training")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for anchor, positive, negative in dataloader:
            anchor, positive, negative = (
                anchor.to(device),
                positive.to(device),
                negative.to(device),
            )

            optimizer.zero_grad()

            emb_anchor = model(anchor)
            emb_positive = model(positive)
            emb_negative = model(negative)

            loss = criterion(emb_anchor, emb_positive, emb_negative)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()