import json
import torch
from torch.utils.data import Dataset
import numpy as np

class SectionDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.label_map = {
            "intro": 0,
            "verse": 1,
            "chorus": 2,
            "bridge": 3,
            "outro": 4
        }

    def __len__(self):
        return len(self.data)

    def _build_feature_vector(self, features):
        vec = np.array([
            features.get("kick_punch", 0.0),
            features.get("mid_energy", 0.0),
            features.get("side_ratio", 0.0),
            features.get("lufs", 0.0),
            features.get("transient_strength", 0.0),
            features.get("transient_variation", 0.0),
            features.get("attack_sharpness", 0.0),
        ], dtype=np.float32)

        return (vec - vec.mean()) / (vec.std() + 1e-6)

    def __getitem__(self, idx):
        item = self.data[idx]
        anchor = torch.tensor(self._build_feature_vector(item["features"]))

        # Positive: same section type
        positives = [
            d for d in self.data
            if d["type"] == item["type"] and d != item
        ]
        negative = [
            d for d in self.data
            if d["type"] != item["type"]
        ]

        pos = torch.tensor(self._build_feature_vector(
            positives[np.random.randint(len(positives))]["features"]
        ))
        neg = torch.tensor(self._build_feature_vector(
            negative[np.random.randint(len(negative))]["features"]
        ))

        return anchor, pos, neg