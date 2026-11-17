"""Rebuild V2 pitch targets from Japanese CTC alignments and the original WAVs."""

import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
import sys

import librosa
import numpy as np
import torch
from torchaudio.functional import forced_align, merge_tokens
from transformers import HubertForCTC

from rebuild_v2_training_data import apply_aligned_pitch


ROOT = Path(__file__).resolve().parent
PITCH_RATE = 100
VOCALIC_PHONES = frozenset("aeiouN")
MIN_CLIP_MATCH = 0.72
MIN_PHONE_SCORE = 0.50


def estimate_pitch(audio):
    hop = 32000 // PITCH_RATE
    frame = 1280
    rms = librosa.feature.rms(y=audio, frame_length=frame, hop_length=hop, center=True)[0]
    f0 = librosa.yin(
        audio, fmin=80, fmax=950, sr=32000, frame_length=frame, hop_length=hop, center=True
    )
    voiced = (rms > max(0.008, 0.12 * float(rms.max()))) & (f0 >= 85) & (f0 <= 920)
    return {"voiced": voiced, "f0_hz": f0}


def align_phones(logits, symbols, vocab, *, vowels_only=True):
    indices = [i for i, symbol in enumerate(symbols) if symbol in vocab]
    windows = [None] * len(symbols)
    if not indices:
        return windows, 0.0
    expected = [vocab[symbols[i]] for i in indices]
    blank = vocab["PAD"]
    scores = logits.float().log_softmax(-1).cpu()
    # Ignore recorded silence and allow either voicing for /i/ and /u/.
    for target, variants in (("PAD", ["pau", "sil"]), ("i", ["I"]), ("u", ["U"])):
        target_id = vocab[target]
        variant_ids = [vocab[key] for key in variants]
        scores[:, target_id] = scores[:, [target_id, *variant_ids]].logsumexp(-1)
        scores[:, variant_ids] = -torch.inf
    predicted = scores.argmax(-1).unique_consecutive().tolist()
    observed = [symbol for symbol in predicted if symbol != blank]
    similarity = SequenceMatcher(None, expected, observed, autojunk=False).ratio()
    repeats = sum(a == b for a, b in zip(expected, expected[1:]))
    if similarity < MIN_CLIP_MATCH or len(scores) < len(expected) + repeats:
        return windows, similarity

    targets = torch.tensor([expected], dtype=torch.int64)
    path, path_scores = forced_align(scores.unsqueeze(0), targets, blank=blank)
    spans = merge_tokens(path[0], path_scores[0].exp(), blank=blank)
    assert [span.token for span in spans] == expected
    for index, span in zip(indices, spans):
        if (not vowels_only or symbols[index] in VOCALIC_PHONES) and span.score >= MIN_PHONE_SCORE:
            # HuBERT's 320-sample stride and 400-sample receptive field at 16 kHz.
            windows[index] = (0.020 * span.start + 0.0025, 0.020 * span.end + 0.0025)
    return windows, similarity


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    project = args.project.resolve()
    sys.path.insert(0, str(project))
    from extract_alignment_and_prosody import load_wav_32k
    from text.symbols2 import symbols

    data_dir = project / "colab_data"
    source = data_dir / "training-data-v2.pt"
    out = args.out or data_dir / "training-data-v2-phoneme-aligned.pt"
    if out.resolve() == source.resolve() or (args.limit and not args.out):
        parser.error("Use a separate output path, explicitly set for a limited run.")
    mask_path = out.with_name(out.stem + "-pitch-mask.pt")
    report_path = out.with_suffix(".jsonl")
    model_path = args.model or project / "tmp/japanese-hubert-base-phoneme-ctc-v4"
    vocab = json.loads((model_path / "vocab.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HubertForCTC.from_pretrained(model_path, local_files_only=True, use_safetensors=True).to(device).eval()
    items = torch.load(source, map_location="cpu", weights_only=True)
    if args.limit:
        focus = [row for row in items if row[0] == "SR011444"]
        items = (focus + [row for row in items if row[0] != "SR011444"])[:args.limit]
    print(f"Aligning {len(items)} clips on {device}", flush=True)
    result, masks, report = [], {}, []
    for index, (uid, split, phones, semantic, controls) in enumerate(items, 1):
        names = [symbols[phone] if phone < 732 else "" for phone in phones.tolist()]
        audio = load_wav_32k(project / "prep/5-wav32k" / f"{uid}.wav")
        # The model expects unnormalized 16 kHz mono audio.
        sample = librosa.resample(audio, orig_sr=32000, target_sr=16000)
        windows = [None] * len(names)
        similarity = 0.0
        if len(sample) >= 400 and any(name in VOCALIC_PHONES for name in names):
            logits = model(torch.from_numpy(sample).to(device).unsqueeze(0)).logits[0]
            windows, similarity = align_phones(logits, names, vocab)
        cond = controls.clone()
        if any(windows):
            masks[uid] = apply_aligned_pitch(cond[22:26], estimate_pitch(audio), windows, frame_rate=PITCH_RATE)
        else:
            cond[22:24].zero_()
            masks[uid] = torch.zeros((2, len(names)), dtype=torch.bool)
        result.append((uid, split, phones, semantic, cond))
        report.append({
            "id": uid,
            "match": round(similarity, 4),
            "pitch_phones": [
                [i, names[i], round(windows[i][0], 4), round(windows[i][1], 4)]
                for i in masks[uid][0].nonzero().flatten().tolist()
            ],
        })
        if index % 250 == 0 or index == len(items):
            print(f"{index}/{len(items)} | usable F0: {sum(bool(m[0].any()) for m in masks.values())}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("".join(json.dumps(row) + "\n" for row in report), encoding="utf-8")
    torch.save(masks, mask_path)
    torch.save(result, out)
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
