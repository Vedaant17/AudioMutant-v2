import torch
import numpy as np

from ml.models.track_model import TrackScoringModel
from ml.models.section_model import SectionScoringModel
from ml.models.sequence_model import SequenceModel


class MLScorer:

    def __init__(self, track_dim, section_dim, seq_dim):

        self.track_model = TrackScoringModel(track_dim)
        self.track_model.load_state_dict(torch.load("track_model.pth"))
        self.track_model.eval()

        self.section_model = SectionScoringModel(section_dim)
        self.section_model.load_state_dict(torch.load("section_model.pth"))
        self.section_model.eval()

        self.seq_model = SequenceModel(seq_dim)
        self.seq_model.load_state_dict(torch.load("sequence_model.pth"))
        self.seq_model.eval()

    def score_track(self, vec):
        x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
        return self.track_model(x).item()

    def score_sections(self, section_vecs):
        scores = []
        for v in section_vecs:
            x = torch.tensor(v, dtype=torch.float32).unsqueeze(0)
            scores.append(self.section_model(x).item())
        return scores

    def score_sequence(self, sequence):
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
        return self.seq_model(x).item()