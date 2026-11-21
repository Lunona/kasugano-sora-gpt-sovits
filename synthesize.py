from functools import lru_cache
from io import BytesIO
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from performance import (
    APP, ROOT, DATA, PRE, S1, S2, CHANNELS, DELIVERY_CHANNELS, EMOTIONS, DELIVERIES, DIRECTOR_EVENTS,
    features, install, reference_phones, validate,
    plan_with_deepseek, plan_with_ollama, torch,
)
sys.path.insert(0, str(ROOT / "colab_bundle"))
from prosody_director import ProsodyDirector, predict_controls_and_tags
from sovits_emotion import install_sovits_emotion, load_sovits_emotion_state, set_sovits_controls
from vocal_tokens import VOCAL_TAG_RE, expand_gpt_model, expand_sovits_model

EMOTION_REFS = {
    "neutral": "SR011455", "anger": "SR030252", "hurt": "SR011523", "fear": "SR011452",
    "joy": "SR011312", "shy": "SR010746", "tender": "SR011403", "surprise": "SR011733",
    "resolve": "SR011444", "impatient": "SR011065",
}
PERFORMANCE_REFS = {
    **EMOTION_REFS,
    "hurt": "SR010053", "fear": "SR010423", "joy": "SR011014", "shy": "SR010446",
    "surprise": "SR010304", "resolve": "SR011322",
}
DELIVERY_REFS = {
    "tearful": "SR011523", "trembling": "SR010423", "soft": "SR011403",
    "bright": "SR011014", "hesitant": "SR010446", "startled": "SR010304",
}
CLEAR_ACOUSTIC_REF = "SR011455"
ASMR_DATA_DIR = Path(os.environ.get("SORAVOICE_ASMR_DATA", ROOT.parent / "春日野穹_ASMR数据"))
ASMR_SOVITS_PATH = ROOT / "v3_asmr_clean_checkpoints/sora-sovits-v2proplus-epoch5.pth"
WHISPER_ACOUSTIC_REF = "ASMR_0290"
def _resolve_asmr_ref(stem: str) -> dict:
    local_wav = ROOT / f"asmr_refs/{stem}.wav"
    local_sv = ROOT / f"asmr_refs/{stem}.wav.pt"
    if local_wav.is_file() and local_sv.is_file():
        return {"wav": local_wav, "sv": local_sv}
    return {
        "wav": ASMR_DATA_DIR / f"prep_asmr_clean/5-wav32k/{stem}.wav",
        "sv": ASMR_DATA_DIR / f"prep_asmr_clean/7-sv_cn/{stem}.wav.pt",
    }


ASMR_WHISPER_REFS = {
    "ASMR_0290": _resolve_asmr_ref("ASMR_0290"),
    "ASMR_0009": _resolve_asmr_ref("ASMR_0009"),
}
# Expressive references whose own acoustics already carry heavy creak/fry (>20%).
CREAKY_EXPRESSIVE_REFS = {"SR011532", "SR011322", "SR011065", "SR010423", "SR011444", "SR011523"}

_BASE_API_REFS = tuple(dict.fromkeys((
    *PERFORMANCE_REFS.values(), *DELIVERY_REFS.values(),
    "SR010392", "SR011532", "SR000124", "SR000150",
)))


def _load_ref_profiles() -> tuple[list[dict], dict[str, dict]]:
    p = ROOT / "ref_profiles.json"
    if not p.is_file():
        return [], {}
    try:
        raw_items = json.loads(p.read_text(encoding="utf-8"))
        items = [
            it for it in raw_items
            if isinstance(it, dict) and "id" in it and 3.005 <= float(it.get("dur_s", 0.0)) <= 9.95
        ]
        by_id = {it["id"]: it for it in items}
        return items, by_id
    except Exception:
        return [], {}


_REF_PROFILES_LIST, _REF_PROFILES_BY_ID = _load_ref_profiles()
API_PERFORMANCE_REFS = tuple(dict.fromkeys(
    ([it["id"] for it in _REF_PROFILES_LIST] or list(_BASE_API_REFS)) + list(_BASE_API_REFS)
))


def _sovits_file_key(path: Path):
    path = Path(path).resolve()
    stat = path.stat()
    return path, (stat.st_mtime_ns, stat.st_size, stat.st_ino)


@lru_cache(maxsize=2)
def _read_sovits_checkpoint(path: Path, signature):
    """Keep deserialized inference weights on CPU for repeat mode switches."""
    with path.open("rb") as file:
        header = file.read(2)
        if header == b"06":
            checkpoint = torch.load(BytesIO(b"PK" + file.read()), map_location="cpu", weights_only=False)
        else:
            file.seek(0)
            checkpoint = torch.load(file, map_location="cpu", weights_only=False)
    return header, {key: checkpoint[key] for key in ("weight", "emotion_adapter", "vocal_tokens") if key in checkpoint}


def load_sovits_weights(tts, sovits_path: Path):
    """Switch SoVITS generator weights in-place when switching between normal and whisper modes."""
    key = _sovits_file_key(sovits_path)
    if getattr(tts, "_loaded_sovits_key", None) == key:
        return
    _, ckpt = _read_sovits_checkpoint(*key)
    tts._loaded_sovits_key = None
    tts._loaded_sovits_path = None
    tts.vits_model.load_state_dict(
        {k: v.to(device=tts.configs.device, dtype=tts.precision) for k, v in ckpt["weight"].items()},
        strict=False,
    )
    if ckpt.get("emotion_adapter"):
        load_sovits_emotion_state(tts.vits_model, ckpt["emotion_adapter"])
    tts.vits_model.eval()
    tts._loaded_sovits_path = str(key[0])
    tts._loaded_sovits_key = key


# Continuous acoustic softness [0..1] per reference so extra downward energy/pitch push
# attenuates smoothly instead of hard-zeroing emotion expression.
REF_SOFTNESS = {
    "ASMR_0290": 0.75,
    "ASMR_0009": 0.70,
    "SR011403": 0.45,
    "SR010446": 0.30,
    "SR011523": 0.25,
}


def adapt_controls_for_reference(controls: torch.Tensor, emo_scale: float, ref_id: str, acoustic_ref_id: str = None, whisper_mode: bool = False):
    """Per-dimension adaptive control attenuation and conflict clamping."""
    out = controls.clone()
    scale = float(emo_scale)
    ref_soft = max(REF_SOFTNESS.get(ref_id, 0.0), REF_SOFTNESS.get(acoustic_ref_id, 0.0))
    if whisper_mode and ref_soft < 0.65:
        ref_soft = 0.70
    is_creaky_ref = (ref_id in CREAKY_EXPRESSIVE_REFS) or (acoustic_ref_id in CREAKY_EXPRESSIVE_REFS)

    # 1. Emotion channels (0..19) & intensity (20): preserve multi-emotion nuance, cap only overload.
    emo_cap = 1.25 if whisper_mode else (1.35 if is_creaky_ref else 1.50)
    emo_sum = out[:20].abs().sum(dim=0, keepdim=True).clamp(min=1.0)
    out[:20] = out[:20] * (emo_cap / emo_sum).clamp(max=1.0)
    if out.shape[0] > 20 and whisper_mode:
        out[20] = out[20] * (1.0 - 0.35 * ref_soft)

    # 2. Pitch (22, 23): independent clamp so whisper/creaky refs don't get over-bent.
    if out.shape[0] >= 24 and (whisper_mode or is_creaky_ref):
        out[22] = out[22].clamp(-1.5, 1.5)
        out[23] = out[23].clamp(-0.5, 0.5)

    # 3. Energy (24): continuously attenuate negative downward push by (1 - ref_soft).
    if out.shape[0] >= 25 and ref_soft > 0.0:
        neg_atten = max(0.15, 1.0 - ref_soft)
        out[24] = torch.where(out[24] < 0, out[24] * neg_atten, out[24])

    # 4. Prosody silence_prob (25): prevent excessive pause/stutter pressure in whisper mode.
    if out.shape[0] >= 26 and whisper_mode:
        out[25] = out[25].clamp(max=0.65)

    # 5. Delivery channels (26+): attenuate negative pitch/energy shift continuously.
    if out.shape[0] >= DELIVERY_CHANNELS and ref_soft > 0.0:
        neg_atten = max(0.15, 1.0 - ref_soft)
        out[26] = torch.where(out[26] < 0, out[26] * neg_atten, out[26])
        out[27] = torch.where(out[27] < 0, out[27] * neg_atten, out[27])

    try:
        from self_optimize import load_adaptive_weights

        opt_w = load_adaptive_weights()
        soft_delta = float((opt_w.get("ref_softness_delta") or {}).get(ref_id, 0.0))
        if soft_delta > 0 and out.shape[0] >= 25:
            extra_atten = max(0.25, 1.0 - soft_delta)
            out[24] = torch.where(out[24] < 0, out[24] * extra_atten, out[24])
        # Keep normal director pause_s (ch 21) & silence_prob (ch 25) intact so expressive pacing is never rushed.
        cap_delta = float(opt_w.get("emo_scale_cap_delta", 0.0))
        if cap_delta != 0.0:
            scale = max(0.85, scale + cap_delta)
    except Exception:
        pass

    if whisper_mode:
        # Smooth scale attenuation instead of hard cutoff.
        scale = min(scale, max(0.45, 1.0 - 0.50 * ref_soft))
    elif is_creaky_ref:
        scale = min(scale, 1.45)
    return out, scale


def pick_ref_id(plan, ref_id="auto"):
    if ref_id and ref_id != "auto":
        return ref_id
    best = max(plan, key=lambda s: float(s.get("intensity", 0.0)), default={})
    emo = best.get("primary") or best.get("secondary") or "neutral"
    return EMOTION_REFS.get(emo, "SR011455")


def _desired_pace_and_burst(text: str, span: dict) -> tuple[str, bool]:
    emo = span.get("primary") or "neutral"
    deliv = span.get("delivery", "plain")
    inten = float(span.get("intensity", 0.5) or 0.5)
    frag = text[span["start"]:span["end"]] if "start" in span and "end" in span else text
    has_ellipsis = ("…" in frag or "‥" in frag)
    has_bang = ("！" in frag or "!" in frag)
    is_burst = (
        deliv in ("emphatic", "startled")
        or (emo in ("anger", "impatient", "surprise") and inten >= 0.72)
        or (has_bang and inten >= 0.70)
    )
    if deliv in ("tearful", "restrained", "soft", "hesitant") or (
        has_ellipsis and not is_burst and emo in ("hurt", "shy", "tender", "fear", "resolve", "neutral")
    ):
        pace = "slow"
    elif is_burst or (deliv == "bright" and inten >= 0.65):
        pace = "fast"
    else:
        pace = "mid"
    return pace, is_burst


def _score_ref_profile(item: dict, text: str, span: dict) -> float:
    emo = span.get("primary") or "neutral"
    sec = span.get("secondary")
    deliv = span.get("delivery", "plain")
    inten = float(span.get("intensity", 0.5) or 0.5)
    pace, is_burst = _desired_pace_and_burst(text, span)

    score = 0.0
    if item["emotion"] == emo:
        score += 3.6
    elif item.get("secondary") == emo:
        score += 1.9
    if sec and (item["emotion"] == sec or item.get("secondary") == sec):
        score += 1.3

    score -= 2.2 * abs(float(item.get("intensity", 0.5)) - inten)

    it_pace = item.get("pace_tier", "mid")
    if it_pace == pace:
        score += 1.7
    elif (pace == "slow" and it_pace == "fast") or (pace == "fast" and it_pace == "slow"):
        score -= 2.3
    else:
        score -= 0.25

    f0_med = float(item.get("f0_med", 340.0))
    f0_range = float(item.get("f0_range", 200.0))
    rms_mean = float(item.get("rms_mean", 0.08))
    if is_burst:
        score += 0.9 * min(1.0, max(0.0, (f0_med - 320.0) / 180.0))
        score += 0.7 * min(1.0, f0_range / 360.0)
    elif deliv in ("soft", "restrained", "tearful") or pace == "slow":
        if rms_mean < 0.095:
            score += 0.45
        if deliv == "tearful" and f0_range >= 200.0:
            score += 0.5
        elif deliv in ("soft", "restrained") and f0_med <= 360.0:
            score += 0.45
    return score


def _performance_ref_for_span(text, best, *, exclude_refs=None):
    emotion, delivery = best.get("primary"), best.get("delivery", "plain")
    if delivery not in DELIVERIES:
        delivery = "plain"
    if best.get("event") == "sob" or "<sob>" in text[best["start"]:best["end"]]:
        delivery = "tearful"

    excl = set(exclude_refs or ())
    if _REF_PROFILES_LIST:
        cands = [it for it in _REF_PROFILES_LIST if it["id"] not in excl]
        if cands:
            best_item = max(cands, key=lambda it: (_score_ref_profile(it, text, best), 1 if it.get("is_base") else 0))
            return best_item["id"], delivery

    if delivery == "restrained" and emotion in ("anger", "impatient", "resolve"):
        return "SR010392", delivery
    if delivery == "emphatic" and emotion in ("anger", "hurt", "impatient", "resolve"):
        return "SR011532", delivery
    return DELIVERY_REFS.get(delivery, PERFORMANCE_REFS.get(emotion, CLEAR_ACOUSTIC_REF)), delivery


def pick_performance_refs(text, plan, count):
    def score(span):
        delivery = span.get("delivery")
        expressive = delivery in DELIVERIES[1:]
        deliberate = span.get("event") == "sob" or "<sob>" in text[span["start"]:span["end"]]
        return float(span["intensity"]) + 0.2 * expressive + 0.35 * deliberate

    ranked = sorted(plan, key=score, reverse=True)
    main_ref, delivery = _performance_ref_for_span(text, ranked[0])
    choices = [main_ref]
    if _REF_PROFILES_LIST:
        scored = sorted(
            _REF_PROFILES_LIST,
            key=lambda it: _score_ref_profile(it, text, ranked[0]),
            reverse=True,
        )
        choices.extend(it["id"] for it in scored[: max(count * 3, 12)])
    if main_ref == "SR010446":
        choices.extend(("SR000124", "SR000150"))
    choices.extend(_performance_ref_for_span(text, span)[0] for span in ranked[1:])
    choices.extend((CLEAR_ACOUSTIC_REF, "SR000124", "SR011403"))
    deduped = list(dict.fromkeys(choices))
    try:
        from self_optimize import load_adaptive_weights

        ref_scores = load_adaptive_weights().get("ref_scores") or {}
        if ref_scores:
            # Blend multi-dimensional acoustic rank with Bayesian historical reliability score
            deduped = [
                r for _, r in sorted(
                    enumerate(deduped),
                    key=lambda idx_r: -(ref_scores.get(idx_r[1], 0.0) - idx_r[0] * 0.12),
                )
            ]
    except Exception:
        pass
    return deduped[:count], delivery


def pick_performance_ref(text, plan):
    refs, delivery = pick_performance_refs(text, plan, 1)
    return refs[0], delivery


def insert_directed_events(text, plan):
    parts, spans, cursor = [], [], 0
    for span in plan:
        start, end = span["start"], span["end"]
        fragment = text[start:end]
        event, at = span.get("event"), span.get("event_at")
        if event in DIRECTOR_EVENTS and isinstance(at, int) and start <= at < end:
            offset = at - start
            tag = f"<{event}>"
            tags = list(VOCAL_TAG_RE.finditer(fragment))
            for match in tags:
                if match.start() < offset < match.end():
                    offset = match.start()
                    break
            already_present = any(
                match.group() == tag and offset in (match.start(), match.end())
                for match in tags
            )
            if not already_present:
                fragment = fragment[:offset] + tag + fragment[offset:]
        parts.append(fragment)
        spans.append(dict(span, start=cursor, end=cursor + len(fragment)))
        cursor += len(fragment)
    return "".join(parts), spans


def refine_plan(text, plan):
    merged = []
    for sp in plan:
        seg = text[sp["start"]:sp["end"]]
        if merged and not any(c.isalnum() for c in seg):
            merged[-1]["end"] = sp["end"]
            merged[-1]["pause_s"] = sp.get("pause_s", 0.0)
        else:
            merged.append(dict(sp))
    out = []
    for idx, sp in enumerate(merged):
        seg = text[sp["start"]:sp["end"]]
        nsp = dict(sp)
        is_last = (idx == len(merged) - 1)
        p = float(nsp.get("pause_s", 0.0) or 0.0)
        if not is_last:
            if "…" in seg or "‥" in seg:
                p = max(p, 0.55)
            elif any(c in seg for c in "。！？!?"):
                p = max(p, 0.40)
            else:
                p = min(p, 0.15)
        else:
            p = 0.0
        nsp["pause_s"] = round(p, 2)
        out.append(nsp)
    return out


def split_long_monologue(text, plan, min_chars=75, max_span_chars=42):
    """Break coarse spans in long monologues (>min_chars) along sentence boundaries for pitch/pacing stability."""
    import re

    total_chars = sum(1 for c in text if c.isalnum())
    if total_chars <= min_chars or not plan:
        return plan
    refined = []
    for sp in plan:
        s0, e0 = int(sp["start"]), int(sp["end"])
        frag = text[s0:e0]
        if sum(1 for c in frag if c.isalnum()) <= max_span_chars:
            refined.append(sp)
            continue
        cuts = [m.end() for m in re.finditer(r"[。！？!?]+", frag) if 0 < m.end() < len(frag)]
        if not cuts:
            refined.append(sp)
            continue
        pts = [0, *cuts, len(frag)]
        merged_pts = [0]
        for pt in pts[1:-1]:
            if sum(1 for c in frag[merged_pts[-1]:pt] if c.isalnum()) >= 14:
                merged_pts.append(pt)
        merged_pts.append(len(frag))
        if len(merged_pts) <= 2:
            refined.append(sp)
            continue
        for idx, (a, b) in enumerate(zip(merged_pts[:-1], merged_pts[1:])):
            if not frag[a:b].strip():
                continue
            nsp = dict(sp)
            nsp["start"] = s0 + a
            nsp["end"] = s0 + b
            if idx < len(merged_pts) - 2:
                nsp["pause_s"] = 0.24
            refined.append(nsp)
    return refined or plan


_BREATH_MOAN_KANA = frozenset("あいうえおぁぃぅぇぉはひふへほやゆよゃゅょわをんっ")


def _is_pure_breath_fragment(frag: str) -> bool:
    """True when a fragment only contains ellipses/punctuation and breath/moan kana without lexical anchors."""
    cleaned = VOCAL_TAG_RE.sub("", frag or "")
    chars = [c for c in cleaned if c.isalnum()]
    if not chars:
        return True
    return len(chars) <= 16 and all(c in _BREATH_MOAN_KANA for c in chars)


def group_plan_segments(text, plan, *, enabled=True):
    """Keep short takes intact and batch long takes at complete sentences."""
    whole = [{"text": text, "plan": plan, "pause_after": 0.0}]
    if not enabled or len(plan) < 2:
        return whole

    def spoken_chars(value):
        return sum(c.isalnum() for c in VOCAL_TAG_RE.sub("", value))

    def sentence_end(value):
        clean = VOCAL_TAG_RE.sub("", value).rstrip().rstrip('」』】）)”’').rstrip()
        return bool(clean and clean[-1] in "。！？!?")

    if spoken_chars(text) <= 75:
        return whole
    groups = []
    first = 0
    for idx, span in enumerate(plan[:-1]):
        boundary = span["end"]
        fragment = text[plan[first]["start"]:boundary]
        remainder = text[boundary:]
        if not sentence_end(fragment) or spoken_chars(fragment) < 32:
            continue
        if spoken_chars(remainder) < 18 or _is_pure_breath_fragment(remainder):
            continue
        groups.append(plan[first:idx + 1])
        first = idx + 1
    if not groups:
        return whole
    groups.append(plan[first:])

    result = []
    for idx, group in enumerate(groups):
        start = 0 if idx == 0 else groups[idx - 1][-1]["end"]
        end = len(text) if idx == len(groups) - 1 else group[-1]["end"]
        shifted = [dict(span, start=span["start"] - start, end=span["end"] - start) for span in group]
        pause = 0.0
        if idx < len(groups) - 1:
            pause = max(0.24, float(shifted[-1].get("pause_s", 0.0) or 0.0))
            shifted[-1]["pause_s"] = 0.0
        result.append({"text": text[start:end], "plan": shifted, "pause_after": pause})
    return result


def _join_speech_batches(chunks, pauses, sr):
    """Keep natural tails and avoid adding a second pause over decoded silence."""
    import numpy as np

    def edge_silence(wave, limit):
        frame = max(1, sr // 100)
        end = min(len(wave), limit)
        for offset in range(0, end, frame):
            samples = wave[offset:min(offset + frame, end)]
            if np.max(np.abs(samples)) > 0.002:
                return offset
        return end

    def edge_level(wave):
        edge = np.abs(wave[:sr // 2])
        active = edge[edge > 0.008]
        return float(np.median(active)) if len(active) >= sr // 40 else 0.0

    batches = [wave.copy() for wave in chunks]
    parts = [batches[0]]
    for previous, current, pause in zip(batches, batches[1:], pauses):
        tail, head = previous[::-1], current
        planned = max(0, round(pause * sr))
        quiet_tail = edge_silence(tail, planned)
        quiet_head = edge_silence(head, planned)
        last_level = edge_level(tail[quiet_tail:])
        next_level = edge_level(head[quiet_head:])
        if last_level and next_level:
            gain = float(np.clip(last_level / next_level, 0.84, 1.19))
            length = min(len(current), quiet_head + sr // 3)
            current[:length] *= np.linspace(gain, 1.0, length, dtype=current.dtype)

        # Smooth only the waveform edges at the stitch, leaving the speech and pause intact.
        fade_length = min(round(sr * 0.006), len(previous) // 2, len(current) // 2)
        if fade_length >= 2:
            ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade_length, dtype=current.dtype))
            previous[-fade_length:] *= ramp[::-1]
            current[:fade_length] *= ramp
        padding = max(0, planned - quiet_tail - quiet_head)
        if padding:
            parts.append(np.zeros(padding, dtype=current.dtype))
        parts.append(current)
    return np.concatenate(parts)


def auto_plan(text):
    import re
    parts = [m.group(0) for m in re.finditer(r"[^。！？…、,]+[。！？…、,]*|[。！？…、,]+", text) if m.group(0).strip()]
    if not parts:
        parts = [text]
    plan, cursor = [], 0
    for idx, seg in enumerate(parts):
        end = cursor + len(seg)
        if "<sob>" in seg:
            primary, secondary, intensity = "hurt", "fear", 0.9
        elif "<gasp>" in seg:
            primary, secondary, intensity = "surprise", "fear", 0.85
        elif "<pant>" in seg or "<breath>" in seg:
            primary, secondary, intensity = "shy", "hurt", 0.85
        elif "<laugh>" in seg:
            primary, secondary, intensity = "joy", "tender", 0.8
        elif "！" in seg or "!" in seg:
            primary, secondary, intensity = "anger", "resolve", 0.9
        elif "？" in seg or "?" in seg:
            primary, secondary, intensity = "hurt", "surprise", 0.8
        elif "…" in seg:
            primary, secondary, intensity = "hurt", "shy", 0.75
        else:
            primary, secondary, intensity = "tender", None, 0.7
        pause_s = 0.0 if idx == len(parts) - 1 else (0.65 if "…" in seg else (0.45 if any(c in seg for c in "。！？!?") else 0.25))
        plan.append({"start": cursor, "end": end, "primary": primary, "secondary": secondary, "intensity": intensity, "pause_s": pause_s})
        cursor = end
    return plan

def auto_insert_tags(text, plan):
    if any(t in text for t in ("<gasp>", "<breath>", "<hold>", "<oral>", "<pant>", "<sob>", "<laugh>")):
        return text, plan
    out_parts, new_plan, cursor = [], [], 0
    for idx, sp in enumerate(plan):
        seg = text[sp["start"]:sp["end"]]
        pri = sp.get("primary") or ""
        sec = sp.get("secondary") or ""
        prefix = ""
        if idx == 0 and pri in ("surprise", "fear") and float(sp.get("intensity", 0)) >= 0.7:
            prefix = "<gasp>"
        elif pri == "hurt" and float(sp.get("intensity", 0)) >= 0.75:
            prefix = "<sob>"
        elif pri == "joy" and float(sp.get("intensity", 0)) >= 0.7:
            prefix = "<laugh>"
        elif idx > 0 and "…" in seg:
            prefix = "<breath>" if sec in ("hurt", "shy", "tender") else "<hold>"
        tagged_seg = prefix + seg
        out_parts.append(tagged_seg)
        new_sp = dict(sp)
        new_sp["start"] = cursor
        cursor += len(tagged_seg)
        new_sp["end"] = cursor
        new_plan.append(new_sp)
    return "".join(out_parts), new_plan

# Each call encodes one complete utterance and invokes the semantic and audio decoders once.
def engine(checkpoint, sovits_checkpoint=None):
    import yaml
    vits_path = Path(sovits_checkpoint).resolve() if sovits_checkpoint is not None else S2.resolve()
    if not vits_path.is_file():
        raise FileNotFoundError(f"SoVITS 权重不存在：{vits_path}")
    saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
    assert tuple(saved["emotions"])==EMOTIONS
    vits_key = _sovits_file_key(vits_path)
    header, s2_ckpt = _read_sovits_checkpoint(*vits_key)
    version = "v2ProPlus" if header == b"06" else "v2"
    s2_emb = s2_ckpt.get("weight", {}).get("enc_p.text_embedding.weight")
    needs_s2_expand = bool(s2_ckpt.get("vocal_tokens") or (s2_emb is not None and s2_emb.shape[0] > 732))
    init_vits = (PRE / "v2Pro/s2Gv2ProPlus.pth") if (needs_s2_expand and version == "v2ProPlus" and (PRE / "v2Pro/s2Gv2ProPlus.pth").is_file()) else (S2 if needs_s2_expand else vits_path)
    config=ROOT/"tts.yaml"
    config.write_text(yaml.safe_dump({"custom":{
        "version":version,"device":"cuda","is_half":True,
        "t2s_weights_path":S1.as_posix(),"vits_weights_path":init_vits.as_posix(),
        "bert_base_path":(PRE/"chinese-roberta-wwm-ext-large").as_posix(),
        "cnhuhbert_base_path":(PRE/"chinese-hubert-base").as_posix(),
    }}),encoding="utf-8")
    previous_cwd = os.getcwd()
    try:
        os.chdir(APP)
        from TTS_infer_pack.TTS import TTS

        tts = TTS(str(config))
    finally:
        os.chdir(previous_cwd)
    if needs_s2_expand:
        expand_sovits_model(tts.vits_model)
        tts.vits_model.load_state_dict(
            {k: v.to(device=tts.configs.device, dtype=tts.precision) for k, v in s2_ckpt["weight"].items()},
            strict=False,
        )
        tts.vits_model.to(device=tts.configs.device, dtype=tts.precision)
    if s2_ckpt.get("emotion_adapter"):
        s2_ch = int(s2_ckpt["emotion_adapter"].get("channels", 22))
        install_sovits_emotion(tts.vits_model, channels=s2_ch)
        load_sovits_emotion_state(tts.vits_model, s2_ckpt["emotion_adapter"])
        tts._sovits_ctrl_dim = s2_ch
    if saved.get("vocal_tokens") or any(v.shape[0] > 732 for k, v in saved.get("gpt_weight", {}).items() if "ar_text_embedding" in k):
        expand_gpt_model(tts.t2s_model.model)
    if "gpt_weight" in saved:
        tts.t2s_model.model.load_state_dict(
            {k.removeprefix("model."): v.to(device=tts.configs.device, dtype=tts.precision) for k, v in saved["gpt_weight"].items()},
            strict=False,
        )
    ctrl_dim = saved["adapter"]["weight"].shape[1]
    adapter=install(tts.t2s_model.model, channels=ctrl_dim)
    adapter.load_state_dict(saved["adapter"])
    adapter.to(device=tts.configs.device,dtype=tts.precision)
    tts._ctrl_dim = ctrl_dim
    dir_ckpt = ROOT / "colab_bundle/prosody-director.pt"
    if dir_ckpt.is_file():
        director = ProsodyDirector().to(tts.configs.device)
        director.load_state_dict(torch.load(dir_ckpt, map_location="cpu", weights_only=True)["state_dict"])
        director.eval()
        tts._director = director
    else:
        tts._director = None
    tts._ref_id = "SR011056"
    tts._default_sovits_path = str(vits_path)
    tts._loaded_sovits_path = tts._default_sovits_path
    tts._loaded_sovits_key = vits_key
    try:
        from self_optimize import load_self_optimized_models

        load_self_optimized_models(tts)
    except Exception:
        pass
    tts.set_ref_audio(str(DATA/"音频_WAV/SR011056.wav"))
    return tts

def _strip_leading_ellipsis(text, plan):
    import re

    if not text:
        return text, plan
    # Strip leading/trailing ellipses+quotes+whitespace and collapse multi-ellipsis runs (e.g. "……" "……") to single "……".
    ops: list[tuple[int, int, str]] = []
    m_lead = re.match(r"^[…\s　「」『』]+", text)
    if m_lead and m_lead.end() < len(text):
        ops.append((0, m_lead.end(), ""))
    lead_end = ops[0][1] if ops else 0
    m_tail = re.search(r"[…\s　「」『』]+$", text[lead_end:])
    tail_start = (lead_end + m_tail.start()) if (m_tail and m_tail.start() > 0) else len(text)
    if tail_start < len(text):
        ops.append((tail_start, len(text), ""))
    for m in re.finditer(r"…+(?:[\s　「」『』]*…+)+|…{3,}", text[lead_end:tail_start]):
        s0, e0 = lead_end + m.start(), lead_end + m.end()
        ops.append((s0, e0, "……"))
    if not ops:
        return text, plan
    ops.sort(key=lambda x: x[0])
    new_parts: list[str] = []
    pos_map = [0] * (len(text) + 1)
    cursor = 0
    new_len = 0
    for s0, e0, rep in ops:
        if s0 > cursor:
            seg = text[cursor:s0]
            for i in range(cursor, s0):
                pos_map[i] = new_len + (i - cursor)
            new_parts.append(seg)
            new_len += len(seg)
        for i in range(s0, e0):
            pos_map[i] = new_len
        if rep:
            new_parts.append(rep)
            new_len += len(rep)
        cursor = e0
    if cursor < len(text):
        seg = text[cursor:]
        for i in range(cursor, len(text)):
            pos_map[i] = new_len + (i - cursor)
        new_parts.append(seg)
        new_len += len(seg)
    pos_map[len(text)] = new_len
    new_text = "".join(new_parts)
    if not new_text:
        return text, plan
    new_plan = []
    for sp in plan:
        s = pos_map[max(0, min(len(text), int(sp["start"])))]
        e = pos_map[max(0, min(len(text), int(sp["end"])))]
        if e > s:
            nsp = dict(sp)
            nsp["start"] = s
            nsp["end"] = e
            new_plan.append(nsp)
    if not new_plan:
        new_plan = [dict(plan[0], start=0, end=len(new_text))]
    else:
        new_plan[0]["start"] = 0
        new_plan[-1]["end"] = len(new_text)
    return new_text, new_plan

@torch.inference_mode()
def _render_one_segment(tts, text, plan, conditioned=True, ref_id="SR011455", emo_scale=None, temperature=0.9, top_k=10, auto_tags=True, acoustic_ref_id=None, use_delivery_controls=True, whisper_mode=False, speed=1.0):
    import numpy as np
    from TTS_infer_pack.TTS import set_seed
    device=tts.configs.device
    text, plan = _strip_leading_ellipsis(text, plan)
    ref_id = pick_ref_id(plan, ref_id)
    if emo_scale is None:
        max_int = max((float(sp.get("intensity", 0.5)) for sp in plan), default=0.5)
        n_chars = sum(1 for c in text if c.isalnum())
        exclaim_cnt = text.count("！") + text.count("!")
        has_vocal = any(tok in text for tok in ("<gasp>", "<sob>", "<pant>", "<breath>", "<laugh>"))
        high_burst = exclaim_cnt >= 2 or (has_vocal and max_int >= 0.75) or any(
            sp.get("primary") in ("fear", "surprise") and exclaim_cnt >= 1 for sp in plan
        )
        if high_burst:
            cap = 1.40
        elif n_chars >= 14:
            cap = 1.68
        else:
            cap = 1.45
        emo_scale = round(min(cap, 1.0 + (cap - 1.0) * max(0.0, min(1.0, max_int))), 2)
    if acoustic_ref_id and acoustic_ref_id != ref_id and getattr(tts, "_acoustic_ref_id", None) != acoustic_ref_id:
        if getattr(tts, "_ref_id", None) != acoustic_ref_id:
            if acoustic_ref_id in ASMR_WHISPER_REFS:
                tts.set_ref_audio(str(ASMR_WHISPER_REFS[acoustic_ref_id]["wav"]))
            else:
                tts.set_ref_audio(str(DATA / f"音频_WAV/{acoustic_ref_id}.wav"))
            tts._ref_id = acoustic_ref_id
        tts._acoustic_ref_id = acoustic_ref_id
        tts._acoustic_spec = tts.prompt_cache["refer_spec"][0][0]
    if getattr(tts, "_ref_id", None) != ref_id:
        tts.set_ref_audio(str(DATA / f"音频_WAV/{ref_id}.wav"))
        tts._ref_id = ref_id
    rp = reference_phones(ref_id)
    ctrl_dim = getattr(tts, "_ctrl_dim", 22)
    with_delivery = (ctrl_dim == DELIVERY_CHANNELS) and use_delivery_controls and not whisper_mode
    target, controls = features(text, plan, with_delivery=with_delivery)
    if not whisper_mode and getattr(tts, "_director", None) is not None and ctrl_dim >= CHANNELS:
        has_manual_tags = any(pid >= 732 for pid in target)
        if auto_tags and not has_manual_tags:
            pred_ph, predicted = predict_controls_and_tags(tts._director, torch.tensor(target, dtype=torch.long), controls[:22], tag_thresh=0.78)
            if with_delivery:
                extra = controls.new_zeros(DELIVERY_CHANNELS - CHANNELS, len(pred_ph))
                extra[:, pred_ph < 732] = controls[CHANNELS:]
                predicted = torch.cat((predicted, extra))
            controls = predicted
            target = pred_ph.tolist()
        else:
            keep = [i for i, pid in enumerate(target) if pid < 732]
            if keep:
                _, pred_c26 = predict_controls_and_tags(tts._director, torch.tensor([target[i] for i in keep], dtype=torch.long), controls[:22, keep], tag_thresh=1.1)
                for k_idx, orig_i in enumerate(keep):
                    controls[22:26, orig_i] = pred_c26[22:26, k_idx]
    if not whisper_mode and controls.shape[0] >= 26:
        # Gentle variance restoration without over-driving V3 pitch/energy
        controls[22] = (controls[22] * 1.4 + (controls[20] - 0.5).clamp(min=0.0) * 0.7).clamp(-3.0, 3.0)
        controls[24] = (controls[24] * 1.3).clamp(-3.0, 3.0)
    controls = controls[:ctrl_dim]
    controls, emo_scale = adapt_controls_for_reference(controls, emo_scale, ref_id, acoustic_ref_id, whisper_mode=whisper_mode)
    x=torch.tensor([[*rp, *target]],device=device)
    bf=torch.zeros(1,1024,x.shape[-1],device=device,dtype=tts.precision)
    if conditioned:
        scaled = controls.clone()
        scaled[:CHANNELS] *= float(emo_scale)
        bf[0,:scaled.shape[0],len(rp):]=scaled.to(bf)
        sovits_controls = scaled[:getattr(tts, "_sovits_ctrl_dim", min(ctrl_dim, CHANNELS))]
        set_sovits_controls(tts.vits_model, sovits_controls.unsqueeze(0).to(device))
    else:
        set_sovits_controls(tts.vits_model, None)
    separate_acoustics = acoustic_ref_id is not None and acoustic_ref_id != ref_id
    ref_specs = [tts._acoustic_spec] if separate_acoustics else [item[0] for item in tts.prompt_cache["refer_spec"]]
    spec = [item.to(device=device, dtype=tts.precision) for item in ref_specs]
    set_seed(173)
    codes,n=next(tts.t2s_model.model.infer_panel_naive(
         x,torch.tensor([x.shape[-1]],device=device),
         tts.prompt_cache["prompt_semantic"].unsqueeze(0),bf,top_k=top_k,top_p=1.0,
         temperature=temperature,early_stop_num=-1,use_flash_attention=False))
    if n<=0:
        raise RuntimeError("GPT returned no generated tokens")
    eff_speed = max(0.5, min(2.0, float(speed if speed is not None else 1.0)))
    if getattr(tts.vits_model, "is_v2pro", False):
        sv_emb = None if separate_acoustics else tts.prompt_cache.get("sv_emb")
        if sv_emb is None:
            sv_id = acoustic_ref_id if separate_acoustics else ref_id
            if sv_id in ASMR_WHISPER_REFS:
                sv_emb = torch.load(ASMR_WHISPER_REFS[sv_id]["sv"], map_location=device, weights_only=True)
            else:
                sv_emb = torch.load(ROOT / f"prep/7-sv_cn/{sv_id}.wav.pt", map_location=device, weights_only=True)
        sv_emb = sv_emb.to(device=device, dtype=tts.precision)
        wave=tts.vits_model.decode(codes[:,-n:].unsqueeze(0),torch.tensor([target],device=device),spec,speed=eff_speed,sv_emb=sv_emb)[0,0]
    else:
        wave=tts.vits_model.decode(codes[:,-n:].unsqueeze(0),torch.tensor([target],device=device),spec,speed=eff_speed)[0,0]
    set_sovits_controls(tts.vits_model, None)
    wav=wave.float().cpu().numpy()
    if not len(wav) or not np.isfinite(wav).all() or not wav.any():
        raise RuntimeError("Invalid decoded audio")
    sha = hashlib.sha256(codes[:,-n:].cpu().numpy().tobytes()).hexdigest()
    return wav, n, sha, ref_id


@torch.inference_mode()
def render(tts, text, plan, output, conditioned=True, ref_id="SR011455", emo_scale=None, temperature=0.9, top_k=10, auto_tags=True, auto_performance=False, local_delivery=False, performance_ref=None, director_controls_events=False, whisper_mode=False, declick_deess=False, ambient_bed="none", foley_mode="none", ambient_level=0.18, ear_pan="off", long_monologue_mode=True, speed=1.0, inspect_dry=None):
    from math import isfinite

    import numpy as np
    from scipy.io.wavfile import write
    sr = tts.configs.sampling_rate
    if whisper_mode:
        required = (ASMR_SOVITS_PATH, *ASMR_WHISPER_REFS[WHISPER_ACOUSTIC_REF].values())
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("ASMR 缺少文件：" + "、".join(missing))
        load_sovits_weights(tts, ASMR_SOVITS_PATH)
    elif hasattr(tts, "_default_sovits_path"):
        load_sovits_weights(tts, Path(tts._default_sovits_path))
    suggested = plan[0].get("director_emo_scale") if plan else None
    director_scale = emo_scale is None and type(suggested) in (int, float) and isfinite(suggested) and suggested > 0
    if director_scale:
        emo_scale = suggested

    model_events: list[str] = []

    def save(wav, info):
        if not len(wav) or not np.isfinite(wav).all() or not np.any(wav):
            raise ValueError("没有生成有效音频。请检查台词和发声标签。")
        if model_events:
            info["model_events"] = model_events
        if batch_count > 1:
            info["monologue_batches"] = batch_count
        if director_scale and conditioned:
            info["director_emo_scale"] = emo_scale
        if conditioned and local_delivery:
            from acoustic_delivery import apply_delivery

            aligned_text, aligned_plan = _strip_leading_ellipsis(text, plan)
            wav, info["local_delivery"] = apply_delivery(
                wav, sr, aligned_text, aligned_plan, device=tts.configs.device,
            )
        auto_deess = declick_deess
        if not auto_deess:
            try:
                from self_optimize import load_adaptive_weights

                flagged_refs = set(load_adaptive_weights().get("auto_deess_refs") or [])
                used_refs = set(info.get("refs") or [])
                if info.get("acoustic_ref"):
                    used_refs.add(info["acoustic_ref"])
                if used_refs & flagged_refs:
                    auto_deess = True
            except Exception:
                pass
        if auto_deess:
            from acoustic_delivery import apply_light_declick_deess

            wav = apply_light_declick_deess(wav, sr)
        from acoustic_delivery import tame_loud_harshness

        wav = tame_loud_harshness(wav, sr)
        if inspect_dry is not None:
            info["_dry_check"] = inspect_dry(wav, sr)
        out_sr = sr
        if (ambient_bed and ambient_bed != "none") or (foley_mode and foley_mode != "none") or (ear_pan and ear_pan not in ("off", "none")):
            from acoustic_delivery import mix_soundscape

            wav, out_sr, info["soundscape"] = mix_soundscape(
                wav, sr, ambient_bed=ambient_bed, foley_mode=foley_mode, ambient_level=ambient_level, ear_pan=ear_pan, sfx_dir=ROOT / "asmr_sfx",
            )
        write(output, out_sr, (np.clip(wav,-1,1)*32767).astype(np.int16))
        info["seconds"] = round(len(wav) / out_sr, 3)
        return info

    directed_text, directed_plan = text, plan
    if conditioned and auto_tags and (auto_performance or director_controls_events):
        directed_text, directed_plan = insert_directed_events(text, plan)
    directed_text = directed_text.replace("<pant_clip>", "<pant>     ")
    if long_monologue_mode:
        directed_plan = split_long_monologue(directed_text, directed_plan)
    model_auto_tags = auto_tags and not director_controls_events and not whisper_mode
    directed_events = VOCAL_TAG_RE.findall(directed_text.replace("<pant_clip>", "<pant>"))
    batch_count = 1

    def _render_tts_chunk(c_text, c_plan, seg_ref, seg_scale, acoustic_ref_id=None, apply_local=False):
        nonlocal batch_count
        if not VOCAL_TAG_RE.search(c_text) and not any(c.isalnum() for c in c_text):
            return np.zeros(0, dtype=np.float32), 0, "empty", []
        groups = group_plan_segments(c_text, c_plan, enabled=long_monologue_mode)
        batch_count = max(batch_count, len(groups))

        if len(groups) == 1:
            w_chunk, n_tok, sha_str, used_ref = _render_one_segment(
                tts, c_text, c_plan, conditioned, seg_ref, seg_scale,
                temperature, top_k, auto_tags=model_auto_tags,
                acoustic_ref_id=acoustic_ref_id, use_delivery_controls=not local_delivery,
                whisper_mode=whisper_mode, speed=speed,
            )
            model_events.extend(VOCAL_TAG_RE.findall(c_text))
            used_refs = [used_ref]
        else:
            chunks, n_tok, shas, used_refs = [], 0, [], []
            for g in groups:
                w, gn, gsha, gref = _render_one_segment(
                    tts, g["text"], g["plan"], conditioned, seg_ref, seg_scale,
                    temperature, top_k, auto_tags=model_auto_tags,
                    acoustic_ref_id=acoustic_ref_id, use_delivery_controls=not local_delivery,
                    whisper_mode=whisper_mode, speed=speed,
                )
                model_events.extend(VOCAL_TAG_RE.findall(g["text"]))
                chunks.append(w)
                n_tok += gn
                shas.append(gsha[:12])
                used_refs.append(gref)
            w_chunk = _join_speech_batches(chunks, [g["pause_after"] for g in groups], sr)
            sha_str = "+".join(shas)

        return w_chunk, n_tok, sha_str, used_refs

    def _render_grouped(seg_ref, seg_scale, acoustic_ref_id=None):
        return _render_tts_chunk(directed_text, directed_plan, seg_ref, seg_scale, acoustic_ref_id=acoustic_ref_id)

    if performance_ref is not None:
        if isinstance(performance_ref, (tuple, list)) and len(performance_ref) == 2:
            _perf_ref, _perf_delivery = performance_ref[0], performance_ref[1]
        else:
            _perf_ref = str(performance_ref)
            _perf_delivery = (plan[0].get("delivery") or "plain") if plan else "plain"
    else:
        _perf_ref, _perf_delivery = None, None

    if auto_performance:
        validate(text, plan)
        gpt_ref, delivery = (_perf_ref, _perf_delivery) if _perf_ref is not None else pick_performance_ref(text, plan)
        target_acoustic = WHISPER_ACOUSTIC_REF if whisper_mode else CLEAR_ACOUSTIC_REF
        wav, n, sha, seg_refs = _render_grouped(
            gpt_ref, 1.0 if emo_scale is None else emo_scale, acoustic_ref_id=target_acoustic,
        )
        all_refs = list(dict.fromkeys([gpt_ref, *(seg_refs or ())]))
        return save(wav, {
            "tokens": n, "refs": all_refs,
            "semantic_sha256": sha, "acoustic_ref": target_acoustic,
            "delivery": delivery, "directed_events": directed_events,
        })
    effective_ref_id = _perf_ref if (_perf_ref is not None and ref_id == "SR011455") else ref_id
    target_acoustic = WHISPER_ACOUSTIC_REF if whisper_mode else None
    wav, n, sha, refs = _render_grouped(effective_ref_id, emo_scale, acoustic_ref_id=target_acoustic)
    return save(wav, {"tokens": n, "refs": refs, "semantic_sha256": sha, "directed_events": directed_events, "acoustic_ref": target_acoustic})

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--text",required=True)
    parser.add_argument("--context",default="")
    parser.add_argument("--plan",type=Path)
    parser.add_argument("--checkpoint",type=Path,default=ROOT/"v3_checkpoints/gpt_eval/phrase-adapter-epoch15-ft1.pt")
    parser.add_argument("--sovits",type=Path,default=ROOT/"v3_bs16_checkpoints/sovits_eval/sora-sovits-v2proplus-epoch25.pth")
    parser.add_argument("--output",type=Path,default=ROOT/"results/synthesis.wav")
    parser.add_argument("--compare",action="store_true")
    parser.add_argument("--ref-id",default="SR011455")
    parser.add_argument("--auto-performance", action="store_true", help="Select one GPT reference and render the full utterance with a clear acoustic reference")
    parser.add_argument("--local-delivery", action="store_true", help="Apply aligned pitch, loudness and tremble to the generated voice")
    parser.add_argument("--emo-scale",type=float,default=None,help="Override automatic per-segment emo_scale (1.0-1.4)")
    parser.add_argument("--temperature",type=float,default=0.9)
    parser.add_argument("--top-k",type=int,default=10)
    parser.add_argument("--no-auto-tags",action="store_true")
    parser.add_argument("--planner",choices=["auto","deepseek","ollama","rule"],default="auto")
    arg=parser.parse_args()
    arg.text = " ".join(arg.text.splitlines()).strip()
    arg.output.parent.mkdir(parents=True,exist_ok=True)
    director_controls_events = False
    if arg.plan:
        plan=json.loads(arg.plan.read_text(encoding="utf-8-sig"))
    else:
        planner = arg.planner
        if planner == "auto":
            planner = "deepseek" if os.environ.get("DEEPSEEK_API_KEY", "").strip() else "ollama"
        try:
            if planner == "deepseek":
                plan = plan_with_deepseek(arg.text, arg.context)
                director_controls_events = True
            elif planner == "ollama":
                plan = plan_with_ollama(arg.text, arg.context)
            else:
                plan = auto_plan(arg.text)
        except Exception:
            plan=auto_plan(arg.text)
        plan=refine_plan(arg.text, plan)
    tts=engine(arg.checkpoint, arg.sovits)
    if not director_controls_events and not arg.auto_performance and not arg.no_auto_tags and (getattr(tts, "_director", None) is None or getattr(tts, "_ctrl_dim", 22) < 26):
        arg.text, plan = auto_insert_tags(arg.text, plan)
    result={"text":arg.text,"plan":plan,"conditioned":render(tts,arg.text,plan,arg.output,True,arg.ref_id,arg.emo_scale,arg.temperature,arg.top_k,not arg.no_auto_tags,auto_performance=arg.auto_performance,local_delivery=arg.local_delivery,director_controls_events=director_controls_events)}
    if arg.compare:
        result["baseline"]=render(tts,arg.text,plan,arg.output.with_name(arg.output.stem+"-baseline.wav"),False,arg.ref_id,arg.emo_scale,arg.temperature,arg.top_k,not arg.no_auto_tags,auto_performance=arg.auto_performance,director_controls_events=director_controls_events)
    arg.output.with_suffix(".json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(result,flush=True)

if __name__=="__main__":
    main()
