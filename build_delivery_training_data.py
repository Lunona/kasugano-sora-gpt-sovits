"""Build vowel-aligned local delivery controls from the original performances."""

from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "colab_data"
REFERENCES = (
    "SR010709", "SR010316", "SR010688", "SR030252",
    "SR011523", "SR010304", "SR011403", "SR011014",
)
MIN_ALIGNMENT = 0.85


def read_rows(path):
    with path.open(encoding="utf-8-sig") as handle:
        return {row["id"]: row for row in map(json.loads, handle)}


def vowel_controls(conditions, windows, energy_db):
    result = torch.zeros(2, conditions.shape[1])
    ids = [index for index, _, _, _ in windows]
    times = np.array([(start + end) / 2 for _, _, start, end in windows])
    centers = np.clip(np.rint(times * 25).astype(int), 0, len(energy_db) - 1)
    energy = np.array([np.median(energy_db[max(0, center-1):center+2]) for center in centers])
    pitch = conditions[22, ids].float()
    result[0, ids] = ((pitch - pitch.median()) * (0.7143 * 12 / 6)).clamp(-1, 1)
    result[1, ids] = torch.from_numpy(np.clip((energy - np.median(energy)) / 8, -1, 1)).float()
    return result


def aligned_frame_targets(analysis, windows, length):
    owners = torch.full((length,), -1, dtype=torch.long)
    for index, _, start, end in windows:
        lo = max(0, round(start * 25))
        hi = min(length, max(lo + 1, round(end * 25)))
        owners[lo:hi] = index
    voiced = torch.as_tensor(analysis["voiced"], dtype=torch.bool) & (owners >= 0)
    pitch = torch.as_tensor(analysis["log_f0"]).float()
    energy = torch.as_tensor(analysis["energy_db"]).float()
    if not voiced.any():
        return owners, voiced, torch.zeros(2, length)
    target = torch.stack(((pitch - pitch[voiced].median()) * 2, (energy - energy[voiced].median()) / 8))
    return owners, voiced, target.clamp(-1, 1)


def main():
    source = DATA / "training-data-v2-phoneme-aligned.pt"
    dataset = torch.load(source, weights_only=True)
    report = read_rows(source.with_suffix(".jsonl"))
    acoustic = torch.load(ROOT / "aligned_prosody.pt", weights_only=False)
    items = []
    refs = {}
    vowel_frames = {}
    frame_targets = {}

    for uid, split, phones, semantic, cond in dataset:
        if uid in REFERENCES:
            if split != "train":
                raise ValueError(f"Reference is outside the original training partition: {uid}")
            refs[uid] = (phones[phones < 732], semantic)
            continue
        windows = report[uid]["pitch_phones"]
        if report[uid]["match"] < MIN_ALIGNMENT or len(windows) < 3 or len(semantic) > 300:
            continue
        delivery = vowel_controls(cond, windows, acoustic[uid]["energy_db"])
        if not bool(delivery.abs().any()):
            continue
        focus = torch.zeros(len(semantic), dtype=torch.bool)
        for index, _, start, end in windows:
            if delivery[:, index].abs().max() >= 0.35:
                focus[max(0, int(start * 25) - 1):min(len(semantic), int(np.ceil(end * 25)) + 1)] = True
        vowel_frames[uid] = focus
        frame_targets[uid] = aligned_frame_targets(acoustic[uid], windows, len(semantic))
        items.append((uid, split, phones, semantic, torch.cat((cond.float(), delivery))))
    assert len(refs) == len(REFERENCES)
    assert all(torch.isfinite(row[-1]).all() and row[-1].shape == (28, len(row[2])) for row in items)
    path = DATA / "training-data-delivery.pt"
    torch.save({"items": items, "references": refs, "vowel_frames": vowel_frames,
                "frame_targets": frame_targets}, path)
    counts = Counter(split for _, split, *_ in items)
    print(f"{path}: {dict(counts)}")


if __name__ == "__main__":
    main()
