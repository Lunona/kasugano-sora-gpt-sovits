"""Select same-speaker references independently of the target utterance."""

from collections import Counter, defaultdict
from functools import lru_cache
from hashlib import sha256
import json
import random
import wave

import torch
from torch.nn.utils.rnn import pad_sequence

from performance import EMOTIONS


class CrossReference:
    def __init__(self, dataset, manifest_path, seed):
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
        self.fingerprint = sha256(raw).hexdigest()
        self.dataset = dataset
        self.seed = seed
        self.info = manifest["items"]
        self.presets = manifest["presets"]
        self.emotions = {}
        self.by_emotion = defaultdict(list)
        self.candidates = []
        self.counts = Counter()
        self.last_refs = {}
        preset_ids = set(self.presets.values())
        target_ids = {uid for uid, _, _ in dataset.items}
        if target_ids != self.info.keys():
            missing = sorted(self.info.keys() - target_ids)
            extra = sorted(target_ids - self.info.keys())
            raise ValueError(f"Training data and manifest differ: missing={missing[:5]}, extra={extra[:5]}")

        for uid, _, controls in dataset.items:
            row = self.info[uid]
            primary = controls[:len(EMOTIONS)].sum(dim=1)
            emotion = EMOTIONS[int(primary.argmax())]
            self.emotions[uid] = emotion
            if not row["text"] or not row["reference"]:
                continue
            ref_emotion = row.get("reference_emotion", emotion)
            if ref_emotion == "neutral" and not row["neutral_reference"]:
                continue
            with wave.open(str(dataset.wav_dir / f"{uid}.wav"), "rb") as audio:
                duration = audio.getnframes() / audio.getframerate()
            min_duration = 2.5 if uid in preset_ids else 3
            if min_duration <= duration <= 10:
                self.by_emotion[ref_emotion].append(uid)
                self.candidates.append(uid)
        missing_presets = preset_ids.difference(self.candidates)
        if missing_presets:
            raise ValueError(f"Presets failed reference screening: {sorted(missing_presets)}")
        self.pairs = {
            uid: self._pools(uid, emotion) for uid, emotion in self.emotions.items()
        }
        print(
            f"Cross-reference bank: {len(self.candidates)} candidates for {len(self.pairs)} targets; "
            f"neutral={len(self.by_emotion['neutral'])}; "
            f"usable presets={sum(uid in self.candidates for uid in self.presets.values())}",
            flush=True,
        )

    def _pools(self, target, emotion):
        target_text = self.info[target]["text"]
        eligible = [
            uid for uid in self.candidates
            if uid != target and self.info[uid]["text"] != target_text
        ]
        if not eligible:
            raise ValueError(f"No distinct reference available for {target}")

        def prefer_other_script(ids):
            other_script = [
                uid for uid in ids
                if self.info[uid]["script"] != self.info[target]["script"]
            ]
            return other_script or ids

        eligible_ids = set(eligible)
        same = prefer_other_script([uid for uid in self.by_emotion[emotion] if uid in eligible_ids])
        neutral = prefer_other_script([uid for uid in self.by_emotion["neutral"] if uid in eligible_ids])
        preset = self.presets.get(emotion)
        return same, [preset] if preset in eligible else [], neutral, prefer_other_script(eligible)

    def begin_epoch(self, epoch):
        self.rng = random.Random(self.seed + epoch)
        self.counts.clear()

    def select(self, target):
        same, preset, neutral, other = self.pairs[target]
        tier = self.rng.random()
        preferred = ("same", same) if tier < 0.6 else ("preset", preset) if tier < 0.8 else ("neutral", neutral)
        for source, pool in (preferred, ("same", same), ("neutral", neutral), ("other", other)):
            alternatives = [uid for uid in pool if uid != self.last_refs.get(target)]
            if alternatives:
                choice = self.rng.choice(alternatives)
                self.last_refs[target] = choice
                self.counts[source] += 1
                return choice
        raise ValueError(f"No alternate reference available for {target}")

    @lru_cache(maxsize=64)
    def _load(self, uid):
        spec, _ = self.dataset._read_wav_spec(self.dataset.wav_dir / f"{uid}.wav")
        sv = torch.load(self.dataset.sv_dir / f"{uid}.wav.pt", map_location="cpu", weights_only=True)
        return spec[:704].contiguous().T, sv.detach().reshape(20480).float()

    def batch(self, targets, device):
        ref_ids = [self.select(target) for target in targets]
        specs, embeddings = zip(*(self._load(uid) for uid in ref_ids))
        spec = pad_sequence(specs, batch_first=True).transpose(1, 2).contiguous().to(device)
        lengths = torch.tensor([x.size(0) for x in specs], device=device)
        mask = torch.arange(spec.size(2), device=device)[None, None] < lengths[:, None, None]
        return (spec, mask.to(spec.dtype)), torch.stack(embeddings).to(device), ref_ids
