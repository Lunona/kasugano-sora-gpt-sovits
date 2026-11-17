"""Rebuild colab_data/training-data.pt with the 7 non-verbal vocalization tokens inserted at annotated event offsets."""
import json
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "colab_bundle"))

from performance import CHANNELS, EMOTIONS, features
from vocal_tokens import VOCAL_TOKEN_IDS, insert_events_into_text_and_plan

data_dir = ROOT / "colab_data"
src_pt = data_dir / "training-data.pt"
backup_pt = data_dir / "training-data-base732.pt"
if not backup_pt.exists():
    shutil.copy2(src_pt, backup_pt)

rows = {}
for line in (data_dir / "逐条表演标注.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    d = json.loads(line)
    rows[d["id"]] = d

existing = torch.load(backup_pt, map_location="cpu", weights_only=True)
updated = []
tagged_utterances = 0
total_tags = 0

for uid, split, old_phones, semantic, old_controls in existing:
    row = rows[uid]
    text = row["text_ja"]
    spans = row.get("spans") or [
        {
            "start": 0,
            "end": len(text),
            "primary": None,
            "secondary": None,
            "intensity": 0.45,
            "pause_s": 0.0,
        }
    ]
    clean_spans = []
    for sp in spans:
        clean_spans.append(
            {
                "start": int(sp["start"]),
                "end": int(sp["end"]),
                "primary": sp.get("primary") if sp.get("primary") in EMOTIONS else None,
                "secondary": sp.get("secondary") if sp.get("secondary") in EMOTIONS else None,
                "additional_emotions": [e for e in sp.get("additional_emotions", []) if e in EMOTIONS],
                "intensity": min(1.0, max(0.0, float(sp.get("intensity", 0.45)))),
                "pause_s": min(1.0, max(0.0, float(sp.get("pause_s", 0.0)))),
            }
        )
    events = row.get("events", [])
    new_text, new_plan = insert_events_into_text_and_plan(text, clean_spans, events)
    if new_text != text:
        tagged_utterances += 1
        ph_list, cond = features(new_text, new_plan)
        total_tags += sum(1 for pid in ph_list if pid >= 732)
        new_phones = torch.tensor(ph_list, dtype=old_phones.dtype)
        new_controls = cond.to(old_controls.dtype)
    else:
        new_phones = old_phones
        new_controls = old_controls
    assert new_controls.shape == (CHANNELS, len(new_phones)), f"{uid}: shape mismatch"
    updated.append((uid, split, new_phones, semantic, new_controls))

torch.save(updated, src_pt)
print(
    f"Rebuilt {src_pt}: {len(updated)} utterances, "
    f"tagged_utterances={tagged_utterances}, inserted_vocal_tokens={total_tags}"
)
