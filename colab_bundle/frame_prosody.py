"""Estimate local acoustic values from GPT-SoVITS semantic frames."""

import argparse
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

SEMANTIC_CODES = 1024


class FrameProsody(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(SEMANTIC_CODES, 48)
        self.acoustic = nn.Sequential(
            nn.Conv1d(48, 64, 5, padding=2), nn.GELU(),
            nn.Conv1d(64, 48, 3, padding=1), nn.GELU(),
            nn.Conv1d(48, 2, 1),
        )

    def forward(self, tokens, valid):
        embedded = self.embedding(tokens)
        return self.acoustic((embedded * valid.unsqueeze(-1)).transpose(1, 2))

    def from_logits(self, logits, valid):
        # GPT's final class is EOS. Supervise acoustic frames only.
        probs = logits[:, :SEMANTIC_CODES, :].float().transpose(1, 2).softmax(dim=-1)
        embedded = probs @ self.embedding.weight
        return self.acoustic((embedded * valid.unsqueeze(-1)).transpose(1, 2))


def frame_batch(examples, device):
    tokens = pad_sequence([item[0] for item in examples], batch_first=True).to(device)
    wanted = pad_sequence([item[2].T for item in examples], batch_first=True).transpose(1, 2).to(device)
    voiced = pad_sequence([item[1] for item in examples], batch_first=True).to(device)
    lengths = torch.tensor([item[0].numel() for item in examples], device=device)
    valid = torch.arange(tokens.shape[1], device=device)[None, :] < lengths[:, None]
    return tokens, wanted, voiced, valid


def evaluate(model, examples, device):
    error = torch.zeros(2, device=device)
    baseline = torch.zeros_like(error)
    with torch.inference_mode():
        for offset in range(0, len(examples), 32):
            tokens, target, voiced, valid = frame_batch(examples[offset:offset+32], device)
            pred = model(tokens, valid)
            error += ((pred-target).square() * voiced[:,None]).sum((0,2))
            baseline += (target.square() * voiced[:,None]).sum((0,2))
    return (1-error/baseline.clamp_min(1e-5)).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    torch.manual_seed(73)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(args.data, map_location="cpu", weights_only=True)
    groups = {"train": [], "holdout": []}
    for uid, split, _, tokens, _ in data["items"]:
        _, voiced, target = data["frame_targets"][uid]
        if voiced.any():
            groups[split].append((tokens.long(), voiced, target))
    model = FrameProsody().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=2e-3)
    best = -float("inf")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        train = list(groups["train"])
        random.Random(73 + epoch).shuffle(train)
        for i in range(0, len(train), 32):
            tokens, target, voiced, valid = frame_batch(train[i:i+32], device)
            loss = F.smooth_l1_loss(model(tokens, valid), target, reduction="none")
            loss = (loss * voiced[:,None]).sum() / (2 * voiced.sum()).clamp_min(1)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
        scores = evaluate(model, groups["holdout"], device)
        print(json.dumps({"epoch": epoch+1, "pitch_r2": scores[0], "loudness_r2": scores[1]}), flush=True)
        if sum(scores) > best:
            best = sum(scores)
            pending = args.out.with_suffix(".pending")
            torch.save({"state_dict": model.state_dict(), "scores": scores}, pending)
            pending.replace(args.out)


if __name__ == "__main__":
    main()
