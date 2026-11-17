"""Use DeepSeek to plan and review a controlled local tremble A/B test."""

import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.error import URLError

import numpy as np
import parselmouth
from scipy.io.wavfile import write
from scipy.signal import resample_poly

from acoustic_delivery import apply_delivery
from performance import APP, ROOT, deepseek_json, phones
from rebuild_v2_training_data import matched_character_times
from synthesize import CLEAR_ACOUSTIC_REF, _render_one_segment, pick_performance_ref
from text.symbols2 import symbols
from webui_sora import _load_config, get_tts


def make_pair(text, anchors, strength):
    if not isinstance(anchors, list) or len(anchors) != 2:
        raise ValueError("Expected two anchors from DeepSeek")
    bounds = []
    for anchor in anchors:
        if not isinstance(anchor, str) or not 3 <= len(anchor) <= 9 or not anchor.isalnum():
            raise ValueError("Invalid pronunciation anchor")
        start = text.find(anchor)
        if start < 0 or text.find(anchor, start + 1) >= 0:
            raise ValueError("Each anchor must occur exactly once")
        if sum(phone < len(symbols) and symbols[phone] in ("a", "e", "i", "o", "u", "N")
               for phone in phones(anchor)) < 2:
            raise ValueError("Each anchor needs at least two voiced targets")
        bounds.append((start, start + len(anchor)))
    if bounds[0][1] > bounds[1][0]:
        raise ValueError("Anchors must follow spoken order with no overlap")
    cuts = sorted({0, len(text), *bounds[0], *bounds[1]})
    base = [
        dict(start=a, end=b, primary="fear", secondary="hurt", intensity=0.72,
             pause_s=0.0, pitch_shift=0.0, energy_shift=0.0, delivery="plain")
        for a, b in zip(cuts, cuts[1:])
    ]
    return [
        [dict(span, delivery="trembling", tremble_strength=strength)
         if (span["start"], span["end"]) == anchor else dict(span)
         for span in base]
        for anchor in bounds
    ], bounds


def analyze_audio(asr, wav, rate, text, bounds):
    audio = wav.astype(np.float32) / 32768.0
    segments, _ = asr.transcribe(
        resample_poly(audio, 16000, rate), language="ja", beam_size=5,
        word_timestamps=True, condition_on_previous_text=False, vad_filter=False,
    )
    words = [word for segment in segments for word in segment.words]
    times = matched_character_times(
        text, [(word.start, word.end, word.word) for word in words], len(audio) / rate,
    )
    pitch = parselmouth.Sound(audio, sampling_frequency=rate).to_pitch(
        time_step=0.01, pitch_floor=85, pitch_ceiling=950,
    )
    hz = pitch.selected_array["frequency"]
    anchors = []
    for start, end in bounds:
        observed = [times[i] for i in range(start, end) if i in times]
        if len(observed) < 0.7 * (end - start):
            anchors.append({"word": text[start:end], "time": None})
            continue
        lo, hi = observed[0][1], observed[-1][2]
        local = hz[(pitch.xs() >= lo) & (pitch.xs() < hi)]
        valid = (local[:-1] > 0) & (local[1:] > 0)
        movement = 12 * np.abs(np.log2(local[1:][valid] / local[:-1][valid]))
        anchors.append({
            "word": text[start:end], "time": [round(lo, 3), round(hi, 3)],
            "voiced_frames": int((local > 0).sum()),
            "median_pitch_motion_st_per_10ms": round(float(np.median(movement)), 3) if movement.size else None,
        })
    return {"transcript": "".join(word.word for word in words), "anchors": anchors}


def pcm_difference(left, right):
    if left[0] != right[0] or left[1].shape != right[1].shape:
        return {"same_pcm": False, "same_duration": False}
    diff = (left[1].astype(np.float64) - right[1]) / 32768.0
    return {
        "same_pcm": not bool(diff.any()), "same_duration": True,
        "difference_rms": round(float(np.sqrt(np.mean(diff * diff))), 8),
        "peak_difference": round(float(np.abs(diff).max()), 8),
    }


def localized_difference(dry, edited, rate, region, headroom_db):
    start = round(region["start"] * rate)
    end = round(region["end"] * rate)
    difference = edited - dry * (10 ** (headroom_db / 20))
    local = difference[start:end]
    outside = np.concatenate((difference[:start], difference[end:]))
    return {
        "target_rms": round(float(np.sqrt(np.mean(local*local))), 7),
        "outside_rms": round(float(np.sqrt(np.mean(outside*outside))), 7) if outside.size else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="まだ遠くで音が聞こえるから、お願いだから手を離さないで。")
    parser.add_argument("--strength", type=float, default=0.68)
    parser.add_argument("--out", type=Path, default=ROOT / "results/tremble_acceptance" / time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    if not 0 < args.strength <= 1:
        parser.error("--strength must be greater than 0 and at most 1")
    text = "".join(args.text.split())
    args.out = args.out.resolve()
    config = _load_config()
    api = {k: config.get(k) for k in ("api_key", "base_url", "model")}
    selection = deepseek_json(
        "你是日语配音验收员，输出 JSON，选择两个可以分别检验发颤的位置。",
        "从同一行日文中选两个清楚可读的连续短语，分别在前半句和后半句。"
        "每个原词恰好出现一次，3至9个字符，不含标点或空格。保持顺序。"
        '只输出 {"anchors":["原文短语1","原文短语2"]}。\n台词：' + text,
        **api,
    )
    plans, bounds = make_pair(text, selection["anchors"], args.strength)
    args.out.mkdir(parents=True, exist_ok=True)
    tts = get_tts()
    gpt_refs = {pick_performance_ref(text, plan)[0] for plan in plans}
    if len(gpt_refs) != 1:
        raise RuntimeError("References changed across A/B")
    dry, tokens, semantic, gpt_ref = _render_one_segment(
        tts, text, plans[0], ref_id=gpt_refs.pop(), acoustic_ref_id=CLEAR_ACOUSTIC_REF,
        auto_tags=False, emo_scale=1.0, use_delivery_controls=False,
    )
    rate = tts.configs.sampling_rate
    versions = {}
    waveforms = {}
    write(args.out / "baseline.wav", rate, (np.clip(dry, -1, 1) * 32767).astype(np.int16))
    for label, slot in (("A", 0), ("B", 1), ("A_repeat", 0)):
        path = args.out / f"{label}.wav"
        changed, local = apply_delivery(dry, rate, text, plans[slot], device=tts.configs.device)
        pcm = (np.clip(changed, -1, 1) * 32767).astype(np.int16)
        write(path, rate, pcm)
        info = {
            "tokens": tokens, "refs":[gpt_ref], "semantic_sha256": semantic, "acoustic_ref": CLEAR_ACOUSTIC_REF,
            "anchor": selection["anchors"][slot], "seconds": round(len(pcm)/rate,3),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "local_delivery": local,
        }
        if local.get("tremble_applied") and len(local["regions"]) == 1:
            info["localized_difference"] = localized_difference(dry, changed, rate, local["regions"][0], local["headroom_db"])
        versions[label] = info
        waveforms[label] = rate, pcm
    comparison = pcm_difference(waveforms["A"], waveforms["B"])
    repeat = pcm_difference(waveforms["A"], waveforms["A_repeat"])
    located = all("localized_difference" in versions[variant]
                  and versions[variant]["local_delivery"]["regions"][0]["text"] == versions[variant]["anchor"]
                  and versions[variant]["localized_difference"]["target_rms"] >
                      10 * versions[variant]["localized_difference"]["outside_rms"] + 1e-4
                  for variant in ("A", "B"))
    if located and not comparison["same_pcm"] and repeat["same_pcm"]:
        verdict = "颤声处理已随目标短语移动，自然度待试听"
    else:
        verdict = "颤声未通过稳定定位复测"
    report = {
        "model": config.get("model"), "text": text, "versions": versions, "plans": plans,
        "same_dry_audio": True, "A_vs_B": comparison, "A_vs_A_repeat": repeat,
        "same_semantics": True,
        "local_verdict": verdict,
    }
    path = args.out / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    from faster_whisper import WhisperModel

    asr = WhisperModel(str(APP / "tools/asr/models/faster-whisper-large-v3-turbo"), device="cpu",
                       compute_type="int8", cpu_threads=6, local_files_only=True)
    rate, wav = waveforms["A"]
    versions["A"]["acoustics"] = analyze_audio(asr, wav, rate, text, bounds)
    rate, wav = waveforms["B"]
    versions["B"]["acoustics"] = versions["A"]["acoustics"] if comparison["same_pcm"] else analyze_audio(asr, wav, rate, text, bounds)
    evidence = {k: v for k, v in report.items() if k != "plans"}
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        report["deepseek_review"] = deepseek_json(
            "你根据本地 A/B 实测数据写简短 JSON 验收意见。你收到的是数值和转录。"
            "严格遵守 local_verdict。颤声调整是在同一底稿合成波形上做的，数值无法代替听感。",
            '请输出 JSON：{"conclusion":"一句话结论","evidence":"依据","listen_for":"复测要听什么"}。\n'
            + json.dumps(evidence, ensure_ascii=False),
            **api,
        )
    except (URLError, KeyError, ValueError) as exc:
        report["deepseek_review_error"] = str(getattr(exc, "code", type(exc).__name__))
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
