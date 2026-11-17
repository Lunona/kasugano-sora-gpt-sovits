from io import BytesIO
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from performance import (
    APP, ROOT, DATA, PRE, S1, S2, CHANNELS, DELIVERY_CHANNELS, EMOTIONS, DELIVERIES, DIRECTOR_EVENTS,
    features, install, load_annotations, validate,
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
ASMR_DATA_DIR = Path(r"F:\缘之空\春日野穹_ASMR数据")
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

# Shared catalog for API selection, independent of the strongest segment.
API_PERFORMANCE_REFS = tuple(dict.fromkeys((
    *PERFORMANCE_REFS.values(), *DELIVERY_REFS.values(),
    "SR010392", "SR011532", "SR000124", "SR000150",
)))


def load_sovits_weights(tts, sovits_path: Path):
    """Switch SoVITS generator weights in-place when switching between normal and whisper modes."""
    target = str(Path(sovits_path).resolve())
    if getattr(tts, "_loaded_sovits_path", None) == target:
        return
    from io import BytesIO
    raw = Path(sovits_path).read_bytes()
    if raw[:2] == b"06":
        raw = b"PK" + raw[2:]
    ckpt = torch.load(BytesIO(raw), map_location="cpu", weights_only=False)
    tts.vits_model.load_state_dict(
        {k: v.to(device=tts.configs.device, dtype=tts.precision) for k, v in ckpt["weight"].items()},
        strict=False,
    )
    if ckpt.get("emotion_adapter"):
        load_sovits_emotion_state(tts.vits_model, ckpt["emotion_adapter"])
    tts.vits_model.eval()
    tts._loaded_sovits_path = target


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


def _performance_ref_for_span(text, best):
    emotion, delivery = best.get("primary"), best.get("delivery", "plain")
    if delivery not in DELIVERIES:
        delivery = "plain"
    if best.get("event") == "sob" or "<sob>" in text[best["start"]:best["end"]]:
        delivery = "tearful"
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
    if main_ref == "SR010446":
        # Historical automatic review suggests these as alternate explanations.
        choices.extend(("SR000124", "SR000150"))
    choices.extend(_performance_ref_for_span(text, span)[0] for span in ranked[1:])
    choices.extend((CLEAR_ACOUSTIC_REF, "SR000124", "SR011403"))
    return list(dict.fromkeys(choices))[:count], delivery


def pick_performance_ref(text, plan):
    refs, delivery = pick_performance_refs(text, plan, 1)
    return refs[0], delivery


def insert_directed_events(text, plan):
    if VOCAL_TAG_RE.search(text):
        return text, plan
    parts, spans, cursor = [], [], 0
    for span in plan:
        start, end = span["start"], span["end"]
        fragment = text[start:end]
        event, at = span.get("event"), span.get("event_at")
        if event in DIRECTOR_EVENTS and isinstance(at, int) and start <= at < end:
            offset = at - start
            fragment = fragment[:offset] + f"<{event}>" + fragment[offset:]
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


def group_plan_segments(text, plan, ref_id="auto"):
    """Split at real pause boundaries (>=0.4s) when emotion/reference shifts or the chunk grows long."""
    if not plan:
        return [{"text": text, "plan": plan, "pause_after": 0.0}]
    groups = []
    cur = [plan[0]]
    for sp in plan[1:]:
        prev = cur[-1]
        cur_chars = sum(1 for c in text[cur[0]["start"]:prev["end"]] if c.isalnum())
        remaining_chars = sum(1 for c in text[sp["start"]:] if c.isalnum())
        prev_ref = pick_ref_id([prev], "auto")
        cur_ref = pick_ref_id([sp], "auto")
        inten_diff = abs(float(prev.get("intensity", 0)) - float(sp.get("intensity", 0)))
        emo_shift = (ref_id == "auto") and (prev_ref != cur_ref or inten_diff >= 0.25)
        if (
            float(prev.get("pause_s", 0)) >= 0.40
            and cur_chars >= 4
            and remaining_chars >= 4
            and (cur_chars >= 35 or emo_shift)
        ):
            groups.append(cur)
            cur = [sp]
        else:
            cur.append(sp)
    groups.append(cur)
    result = []
    for g in groups:
        s0, e0 = g[0]["start"], g[-1]["end"]
        sub_text = text[s0:e0]
        sub_plan = []
        for idx, sp in enumerate(g):
            nsp = dict(sp)
            nsp["start"] -= s0
            nsp["end"] -= s0
            if idx == len(g) - 1:
                nsp["pause_s"] = 0.0
            sub_plan.append(nsp)
        result.append({"text": sub_text, "plan": sub_plan, "pause_after": float(g[-1].get("pause_s", 0.0))})
    return result

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
    from TTS_infer_pack.TTS import TTS
    saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
    assert tuple(saved["emotions"])==EMOTIONS
    vits_path = Path(sovits_checkpoint) if (sovits_checkpoint and Path(sovits_checkpoint).is_file()) else S2
    raw_vits = vits_path.read_bytes()
    header = raw_vits[:2]
    version = "v2ProPlus" if header == b"06" else "v2"
    if header == b"06":
        raw_vits = b"PK" + raw_vits[2:]
    s2_ckpt = torch.load(BytesIO(raw_vits), map_location="cpu", weights_only=False)
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
    tts._default_sovits_path = str(Path(sovits_checkpoint).resolve()) if sovits_checkpoint is not None else None
    tts._loaded_sovits_path = tts._default_sovits_path
    tts.set_ref_audio(str(DATA/"音频_WAV/SR011056.wav"))
    return tts

def _strip_leading_ellipsis(text, plan):
    import re
    m = re.match(r"^[…\s　]+", text)
    if not m or m.end() >= len(text):
        return text, plan
    cut = m.end()
    new_text = text[cut:]
    new_plan = []
    for sp in plan:
        s = max(0, sp["start"] - cut)
        e = max(0, sp["end"] - cut)
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
def _render_one_segment(tts, text, plan, conditioned=True, ref_id="SR011455", emo_scale=None, temperature=0.9, top_k=10, auto_tags=True, acoustic_ref_id=None, use_delivery_controls=True, whisper_mode=False):
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
    ref=load_annotations()[ref_id]
    rp,_=features(ref["text_ja"],[dict(start=0,end=len(ref["text_ja"]),primary=None,secondary=None,intensity=0,pause_s=0)])
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
    x=torch.tensor([rp+target],device=device)
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
    if getattr(tts.vits_model, "is_v2pro", False):
        sv_emb = None if separate_acoustics else tts.prompt_cache.get("sv_emb")
        if sv_emb is None:
            sv_id = acoustic_ref_id if separate_acoustics else ref_id
            if sv_id in ASMR_WHISPER_REFS:
                sv_emb = torch.load(ASMR_WHISPER_REFS[sv_id]["sv"], map_location=device, weights_only=True)
            else:
                sv_emb = torch.load(ROOT / f"prep/7-sv_cn/{sv_id}.wav.pt", map_location=device, weights_only=True)
        sv_emb = sv_emb.to(device=device, dtype=tts.precision)
        wave=tts.vits_model.decode(codes[:,-n:].unsqueeze(0),torch.tensor([target],device=device),spec,sv_emb=sv_emb)[0,0]
    else:
        wave=tts.vits_model.decode(codes[:,-n:].unsqueeze(0),torch.tensor([target],device=device),spec)[0,0]
    set_sovits_controls(tts.vits_model, None)
    wav=wave.float().cpu().numpy()
    if not len(wav) or not np.isfinite(wav).all() or not wav.any():
        raise RuntimeError("Invalid decoded audio")
    sha = hashlib.sha256(codes[:,-n:].cpu().numpy().tobytes()).hexdigest()
    return wav, n, sha, ref_id


@torch.inference_mode()
def render(tts, text, plan, output, conditioned=True, ref_id="SR011455", emo_scale=None, temperature=0.9, top_k=10, auto_tags=True, auto_performance=False, local_delivery=False, performance_ref=None, director_controls_events=False, whisper_mode=False, declick_deess=False):
    from math import isfinite

    import numpy as np
    from scipy.io.wavfile import write
    sr = tts.configs.sampling_rate
    if whisper_mode and ASMR_SOVITS_PATH.is_file():
        load_sovits_weights(tts, ASMR_SOVITS_PATH)
    elif hasattr(tts, "_default_sovits_path"):
        load_sovits_weights(tts, Path(tts._default_sovits_path))
    suggested = plan[0].get("director_emo_scale") if plan else None
    director_scale = emo_scale is None and type(suggested) in (int, float) and isfinite(suggested) and suggested > 0
    if director_scale:
        emo_scale = suggested

    def save(wav, info):
        if director_scale and conditioned:
            info["director_emo_scale"] = emo_scale
        if conditioned and local_delivery:
            from acoustic_delivery import apply_delivery

            aligned_text, aligned_plan = _strip_leading_ellipsis(text, plan)
            wav, info["local_delivery"] = apply_delivery(
                wav, sr, aligned_text, aligned_plan, device=tts.configs.device,
            )
        if declick_deess:
            from acoustic_delivery import apply_light_declick_deess

            wav = apply_light_declick_deess(wav, sr)
        write(output, sr, (np.clip(wav,-1,1)*32767).astype(np.int16))
        info["seconds"] = round(len(wav) / sr, 3)
        return info

    directed_text, directed_plan = text, plan
    if auto_tags and (auto_performance or director_controls_events) and not whisper_mode:
        directed_text, directed_plan = insert_directed_events(text, plan)
    model_auto_tags = auto_tags and not director_controls_events and not whisper_mode
    directed_events = VOCAL_TAG_RE.findall(directed_text)

    def _render_grouped(seg_ref, seg_scale, acoustic_ref_id=None):
        groups = group_plan_segments(directed_text, directed_plan, ref_id=seg_ref)
        if len(groups) == 1:
            wav, n, sha, used_ref = _render_one_segment(
                tts, directed_text, directed_plan, conditioned, seg_ref, seg_scale,
                temperature, top_k, auto_tags=model_auto_tags,
                acoustic_ref_id=acoustic_ref_id, use_delivery_controls=not local_delivery,
                whisper_mode=whisper_mode,
            )
            return wav, n, sha, [used_ref]
        chunks, n, shas, refs = [], 0, [], []
        fade = int(0.015 * sr)
        for idx, g in enumerate(groups):
            w, gn, gsha, gref = _render_one_segment(
                tts, g["text"], g["plan"], conditioned, seg_ref, seg_scale,
                temperature, top_k, auto_tags=model_auto_tags,
                acoustic_ref_id=acoustic_ref_id, use_delivery_controls=not local_delivery,
                whisper_mode=whisper_mode,
            )
            if len(w) > fade * 2:
                ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade, dtype=w.dtype)))
                w[:fade] *= ramp
                w[-fade:] *= ramp[::-1]
            chunks.append(w)
            n += gn
            shas.append(gsha[:12])
            refs.append(gref)
            if idx < len(groups) - 1 and g["pause_after"] > 0:
                chunks.append(np.zeros(int(g["pause_after"] * sr), dtype=w.dtype))
        return np.concatenate(chunks), n, "+".join(shas), refs

    if auto_performance:
        validate(text, plan)
        gpt_ref, delivery = performance_ref or pick_performance_ref(text, plan)
        target_acoustic = WHISPER_ACOUSTIC_REF if whisper_mode else CLEAR_ACOUSTIC_REF
        wav, n, sha, _ = _render_grouped(
            gpt_ref, 1.0 if emo_scale is None else emo_scale, acoustic_ref_id=target_acoustic,
        )
        return save(wav, {
            "tokens": n, "refs": [gpt_ref],
            "semantic_sha256": sha, "acoustic_ref": target_acoustic,
            "delivery": delivery, "directed_events": directed_events,
        })
    target_acoustic = WHISPER_ACOUSTIC_REF if whisper_mode else None
    wav, n, sha, refs = _render_grouped(ref_id, emo_scale, acoustic_ref_id=target_acoustic)
    return save(wav, {"tokens": n, "refs": refs, "semantic_sha256": sha, "directed_events": directed_events, "acoustic_ref": target_acoustic})

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--text",required=True)
    parser.add_argument("--context",default="")
    parser.add_argument("--plan",type=Path)
    parser.add_argument("--checkpoint",type=Path,default=ROOT/"v3_checkpoints/gpt_eval/phrase-adapter-epoch15.pt")
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
