"""The trained SoVITS output loudness condition."""

import torch
from torch import nn


class LoudnessCondition(nn.Module):
    def __init__(self):
        super().__init__()
        self.curve = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.curve.weight.copy_(torch.tensor([[0.7, 0.0]]))

    def forward(self, logits, db):
        logits, db = torch.broadcast_tensors(logits, db)
        calibrated = self.curve(torch.stack((db / 6, db.abs() * db / 36), dim=-1)).squeeze(-1)
        return logits * torch.exp(calibrated)
