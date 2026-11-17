"""Build separate 26-channel V2 data and a mask for word-aligned pitch targets."""
import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent

LOG_F0_MEAN = 8.6059
LOG_F0_STD = 0.7143
ENERGY_MEAN = -38.81
ENERGY_STD = 29.85


def map_phones_to_times(new_text: str, char_spans: list, dur: float):
    """Keep the original windows for the energy and silence channels."""
    from text.cleaner import clean_text
    from text import cleaned_text_to_sequence
    from vocal_tokens import VOCAL_TAG_RE, VOCAL_TOKEN_IDS

    # Map every char index in new_text to a timestamp in char_spans (ignoring <tag> tokens)
    clean_idx = 0
    times = []
    for part in VOCAL_TAG_RE.split(new_text):
        if not part:
            continue
        if part in VOCAL_TOKEN_IDS:
            t_anchor = char_spans[min(clean_idx, len(char_spans) - 1)][0] if char_spans else 0.0
            times.append((max(0.0, t_anchor - 0.08), min(dur, t_anchor + 0.08)))
        elif part.strip():
            part_spans = char_spans[clean_idx : clean_idx + len(part)]
            clean_idx += len(part)
            symbols, _, _ = clean_text(part, "ja", "v2")
            pids = cleaned_text_to_sequence(symbols, "v2")
            t0 = part_spans[0][0] if part_spans else 0.0
            t1 = part_spans[-1][1] if part_spans else dur
            m = max(1, len(pids))
            for k in range(m):
                ps = t0 + (t1 - t0) * (k / m)
                pe = t0 + (t1 - t0) * ((k + 1) / m)
                times.append((ps, pe))
    return times


def matched_character_times(text, words, duration):
    positions = [i for i, char in enumerate(text) if char.isalnum()]
    transcript = []
    times = []
    for start, end, word in words:
        chars = [char for char in word if char.isalnum()]
        start = max(0.0, start)
        end = min(duration, end)
        if not chars or end <= start:
            continue
        step = (end - start) / len(chars)
        transcript.extend(chars)
        times.extend((start + i * step, start + (i + 1) * step) for i in range(len(chars)))
    reference = ''.join(text[i] for i in positions)
    matches = SequenceMatcher(None, reference, ''.join(transcript), autojunk=False).get_matching_blocks()
    exact = {positions[a + k]: (b + k, *times[b + k]) for a, b, size in matches for k in range(size)}
    return exact if len(exact) >= 0.5 * max(len(positions), 1) else {}


def word_pitch_windows(text, alignment):
    from text import cleaned_text_to_sequence
    from text import japanese as ja
    from text.cleaner import clean_text
    from vocal_tokens import VOCAL_TAG_RE, VOCAL_TOKEN_IDS

    original = alignment["text_ja"]
    exact = matched_character_times(original, alignment["words"], alignment["duration_s"])
    cursor = 0
    output = []
    for part in VOCAL_TAG_RE.split(text):
        if part in VOCAL_TOKEN_IDS:
            output.append(None)
            continue
        origin = cursor
        cursor += len(part)
        if not part.strip():
            continue
        symbols, _, _ = clean_text(part, "ja", "v2")
        target = cleaned_text_to_sequence(symbols, "v2")
        anchors, times = [], []
        chunks = re.finditer(r'[^\s、。！？…‥,.!?「」『』]+', part) if exact else ()
        for match in chunks:
            start = match.start()
            for node in ja.pyopenjtalk.run_frontend(match.group()):
                surface, pronunciation = node["string"], node["pron"]
                at = part.find(surface, start, match.end())
                if at < 0:
                    continue
                start = at + len(surface)
                if not pronunciation or pronunciation == '、':
                    continue
                phones = ja.pyopenjtalk.g2p(pronunciation.replace('’', '')).split()
                begin, end = origin + at, origin + start
                aligned = [exact.get(i) for i in range(begin, end) if original[i].isalnum()]
                window = None
                if aligned and all(aligned) and aligned[-1][0] - aligned[0][0] == len(aligned) - 1:
                    window = (aligned[0][1], aligned[-1][2])
                anchors.extend(cleaned_text_to_sequence(phones, "v2"))
                times.extend(window for _ in phones)
        mapped = [None] * len(target)
        for a, b, count in SequenceMatcher(None, anchors, target, autojunk=False).get_matching_blocks():
            mapped[b:b + count] = times[a:a + count]
        output.extend(mapped)
    assert cursor == len(original)
    return output


def apply_aligned_pitch(prosody, alignment, windows, *, frame_rate=25, min_voiced_frames=2):
    mask = torch.zeros((2, len(windows)), dtype=torch.bool)
    f0 = np.asarray(alignment["f0_hz"], dtype=np.float32)
    voiced = np.asarray(alignment["voiced"], dtype=bool) & np.isfinite(f0) & (f0 > 0)
    log_pitch = np.log2(np.where(voiced, f0, 1))
    prosody[:2].zero_()
    for i, window in enumerate(windows):
        if window is None:
            continue
        start, end = window
        lo = max(0, int(np.ceil(start * frame_rate)))
        hi = min(len(f0), int(np.ceil(end * frame_rate)))
        if hi <= lo or voiced[lo:hi].sum() < min_voiced_frames:
            continue
        phones_f0 = log_pitch[lo:hi]
        valid = voiced[lo:hi]
        prosody[0, i] = float((phones_f0[valid].mean() - LOG_F0_MEAN) / LOG_F0_STD)
        mask[0, i] = True
        adjacent = valid[1:] & valid[:-1]
        if adjacent.any():
            slope_per_frame = np.diff(phones_f0)[adjacent].mean() / LOG_F0_STD
            prosody[1, i] = float(slope_per_frame * frame_rate / 25)
            mask[1, i] = True
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    sys.path[:0] = [str(project / "colab_bundle"), str(project)]
    from performance import EMOTIONS, features
    from vocal_tokens import VOCAL_TAG_RE, insert_events_into_text_and_plan
    data_dir = project / "colab_data"
    out_pt = args.out or data_dir / "training-data-v2-word-aligned.pt"
    if out_pt.resolve() == (data_dir / "training-data-v2.pt").resolve():
        parser.error("Use a separate output path for word-aligned data.")
    aligned = torch.load(project / "aligned_prosody.pt", map_location="cpu", weights_only=False)
    base_pt = torch.load(data_dir / "training-data-base732.pt", map_location="cpu", weights_only=True)

    rows = {}
    for line in (data_dir / "逐条表演标注.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            rows[d["id"]] = d

    updated = []
    pitch_masks = {}
    for uid, split, _, semantic, _ in base_pt:
        row = rows[uid]
        al = aligned[uid]
        text = row["text_ja"]
        char_spans = al["char_spans"]
        dur = float(al["duration_s"])
        energy_db = np.asarray(al["energy_db"], dtype=np.float32)
        silence = np.asarray(al["silence"], dtype=np.float32)
        t_sem = len(energy_db)

        spans = row.get("spans") or [
            {"start": 0, "end": len(text), "primary": None, "secondary": None, "intensity": 0.45, "pause_s": 0.0}
        ]
        clean_spans = []
        for idx, sp in enumerate(spans):
            s_i, e_i = int(sp["start"]), int(sp["end"])
            # Measure real silence gap after this span using aligned char_spans
            real_pause = float(sp.get("pause_s", 0.0))
            if idx + 1 < len(spans) and e_i - 1 < len(char_spans) and int(spans[idx + 1]["start"]) < len(char_spans):
                gap = max(0.0, char_spans[int(spans[idx + 1]["start"])][0] - char_spans[max(0, e_i - 1)][1])
                real_pause = min(1.0, max(real_pause, gap))
            clean_spans.append(
                {
                    "start": s_i,
                    "end": e_i,
                    "primary": sp.get("primary") if sp.get("primary") in EMOTIONS else None,
                    "secondary": sp.get("secondary") if sp.get("secondary") in EMOTIONS else None,
                    "additional_emotions": [e for e in sp.get("additional_emotions", []) if e in EMOTIONS],
                    "intensity": min(1.0, max(0.0, float(sp.get("intensity", 0.45)))),
                    "pause_s": min(1.0, max(0.0, real_pause)),
                }
            )

        events = row.get("events", [])
        new_text, new_plan = insert_events_into_text_and_plan(text, clean_spans, events)
        ph_list, conditions = features(new_text, new_plan)
        ph_times = map_phones_to_times(new_text, char_spans, dur)
        assert len(ph_times) == len(ph_list), f"{uid}: {len(ph_times)} != {len(ph_list)}"

        prosody4 = torch.zeros(4, len(ph_list), dtype=torch.float32)
        for p_idx, (ts, te) in enumerate(ph_times):
            f_start = max(0, min(t_sem - 1, int(ts * 25.0)))
            f_end = max(f_start + 1, min(t_sem, int(np.ceil(te * 25.0))))
            prosody4[2, p_idx] = float(np.mean((energy_db[f_start:f_end] - ENERGY_MEAN) / ENERGY_STD))
            prosody4[3, p_idx] = float(np.mean(silence[f_start:f_end]))

        assert al["text_ja"] == VOCAL_TAG_RE.sub("", new_text), uid
        windows = word_pitch_windows(new_text, al)
        assert len(windows) == len(ph_list), uid
        pitch_masks[uid] = apply_aligned_pitch(prosody4, al, windows)
        cond26 = torch.cat([conditions[:22].float(), prosody4], dim=0).to(torch.float16)
        updated.append((uid, split, torch.tensor(ph_list, dtype=torch.long), semantic, cond26))

    out_pt.parent.mkdir(parents=True, exist_ok=True)
    mask_path = out_pt.with_name(out_pt.stem + "-pitch-mask.pt")
    torch.save(pitch_masks, mask_path)
    torch.save(updated, out_pt)
    print(f"Pitch supervision: {sum(bool(m[0].any()) for m in pitch_masks.values())}/{len(updated)} utterances; masks: {mask_path}")
    print(f"Built {out_pt}: {len(updated)} utterances, controls_shape={tuple(updated[0][4].shape)}")


if __name__ == "__main__":
    main()
