"""Self-optimizing weight & adapter fine-tuning engine driven by synthesis history and error tags."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
OPT_DIR = ROOT / "results" / "self_opt"
OPT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_JSON = OPT_DIR / "adaptive_weights.json"
DIRECTOR_OPT_PT = OPT_DIR / "prosody_director_opt.pt"
ADAPTER_OPT_PT = OPT_DIR / "phrase_adapter_opt.pt"
ADAPTER_BACKUP_PT = OPT_DIR / "phrase_adapter_factory_backup.pt"
DIRECTOR_BACKUP_PT = OPT_DIR / "prosody_director_factory_backup.pt"

ERROR_TAG_CHOICES = [
    "句中/首尾静音过长",
    "漏读/重复/吞字",
    "气泡音/呲呲杂音",
    "语气生硬/过猛",
]

ERROR_TAG_MAP = {
    "句中/首尾静音过长": "silence_long",
    "漏读/重复/吞字": "misread",
    "气泡音/呲呲杂音": "creak_hiss",
    "语气生硬/过猛": "overdriven",
}


def classify_error_detail(detail: str, *, is_manual: bool = False) -> list[str]:
    if not detail:
        return []
    is_misread = any(k in detail for k in ("相似度偏低", "漏读", "重复", "吞", "匹配"))
    # Automatic optimization only tracks 漏读/重复/吞字; other categories apply only when manually tagged by the user.
    if not is_manual:
        return ["misread"] if is_misread else []
    tags = []
    if is_misread:
        tags.append("misread")
    if "静音" in detail or "停顿" in detail:
        tags.append("silence_long")
    if any(k in detail for k in ("杂音", "呲", "气泡", "破音", "齿音", "嘶声")):
        tags.append("creak_hiss")
    if any(k in detail for k in ("生硬", "过猛", "太闷", "过重")):
        tags.append("overdriven")
    return tags or ["general_error"]


def _extract_ref_from_row(audio_path: str | None, detail: str) -> str | None:
    for src in (detail or "", audio_path or ""):
        m = re.search(r"(SR\d{6}|ASMR_\d{4})", src)
        if m:
            return m.group(1)
    return None


def _get_gpt_performance_module(tts):
    model = getattr(getattr(tts, "t2s_model", None), "model", None)
    if model is None:
        return None
    bert_proj = getattr(model, "bert_proj", None)
    if bert_proj is not None and hasattr(bert_proj, "performance"):
        return bert_proj.performance
    return getattr(model, "adapter", None)


def load_adaptive_weights() -> dict[str, Any]:
    default = {
        "updated_at": None,
        "total_positives": 0,
        "total_negatives": 0,
        "error_counts": {},
        "ref_scores": {},
        "ref_softness_delta": {},
        "pause_scale": 1.0,
        "silence_logit_bias": 0.0,
        "emo_scale_cap_delta": 0.0,
        "auto_deess_refs": [],
        "adapter_steps": 0,
    }
    if WEIGHTS_JSON.is_file():
        try:
            loaded = json.loads(WEIGHTS_JSON.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                default.update(loaded)
        except Exception:
            pass
    return default


def analyze_archive_and_update_weights(db_path: Path) -> dict[str, Any]:
    state = load_adaptive_weights()
    if not db_path.is_file():
        return state

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT id, kind, audio, ja, zh, detail FROM events ORDER BY id ASC"
    ).fetchall()
    conn.close()

    pos_count = 0
    neg_count = 0
    err_counter: Counter[str] = Counter()
    ref_wins: Counter[str] = Counter()
    ref_losses: Counter[str] = Counter()
    ref_creak_losses: Counter[str] = Counter()

    positive_samples: list[dict[str, Any]] = []
    negative_samples: list[dict[str, Any]] = []

    for eid, kind, audio, ja, zh, detail in rows:
        ref_id = _extract_ref_from_row(audio, detail)
        if kind in ("主选音频", "用户满意样本"):
            pos_count += 1
            if ref_id:
                ref_wins[ref_id] += 1
            if ja:
                positive_samples.append({"id": eid, "ja": ja, "ref": ref_id, "detail": detail})
        elif kind in ("成品检查异常", "用户标记错误"):
            etags = classify_error_detail(detail, is_manual=(kind == "用户标记错误"))
            if not etags:
                continue
            neg_count += 1
            for t in etags:
                err_counter[t] += 1
            if ref_id:
                ref_losses[ref_id] += 1
                if "creak_hiss" in etags or "overdriven" in etags:
                    ref_creak_losses[ref_id] += 1
            if ja:
                negative_samples.append({"id": eid, "ja": ja, "ref": ref_id, "tags": etags, "detail": detail})

    # Bayesian smoothed reference reliability scores in [-0.35, +0.25]
    all_refs = set(ref_wins.keys()) | set(ref_losses.keys())
    ref_scores: dict[str, float] = {}
    ref_softness_delta: dict[str, float] = {}
    auto_deess_refs: list[str] = []

    for r in sorted(all_refs):
        w = ref_wins[r]
        l = ref_losses[r]
        rate = (w + 2.0) / (w + l + 3.0)
        score = round(float(max(-0.35, min(0.25, (rate - 0.667) * 0.9))), 3)
        ref_scores[r] = score
        if ref_creak_losses[r] >= 1 or l >= 2:
            ref_softness_delta[r] = round(min(0.30, 0.08 * l + 0.12 * ref_creak_losses[r]), 3)
        if ref_creak_losses[r] >= 1:
            auto_deess_refs.append(r)

    sil_errs = err_counter.get("silence_long", 0)
    mis_errs = err_counter.get("misread", 0)
    over_errs = err_counter.get("overdriven", 0) + err_counter.get("creak_hiss", 0)

    pause_scale = round(max(0.72, 1.0 - 0.018 * min(sil_errs, 15)), 3)
    silence_logit_bias = round(max(-1.20, -0.08 * min(sil_errs, 15)), 3)
    emo_scale_cap_delta = round(max(-0.18, -0.025 * min(mis_errs + over_errs, 7)), 3)

    state.update({
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_positives": pos_count,
        "total_negatives": neg_count,
        "error_counts": dict(err_counter),
        "ref_scores": ref_scores,
        "ref_softness_delta": ref_softness_delta,
        "pause_scale": pause_scale,
        "silence_logit_bias": silence_logit_bias,
        "emo_scale_cap_delta": emo_scale_cap_delta,
        "auto_deess_refs": auto_deess_refs,
    })
    WEIGHTS_JSON.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    state["_pos_samples"] = positive_samples
    state["_neg_samples"] = negative_samples
    return state


def record_user_feedback(
    archive,
    *,
    is_positive: bool,
    tags: list[str] | None = None,
    note: str = "",
    audio: str | None = None,
    ja: str = "",
    zh: str = "",
    chat_id: str | None = None,
    status_text: str = "",
) -> dict[str, Any]:
    """Logs user feedback into DirectorArchive and immediately refreshes adaptive weights."""
    ref_match = re.search(r"(SR\d{6}|ASMR_\d{4})", status_text or "")
    ref_suffix = f" · 参考 {ref_match.group(1)}" if ref_match else ""
    if is_positive:
        kind = "用户满意样本"
        detail = f"人工确认满意{ref_suffix}" + (f" · {note.strip()}" if note and note.strip() else "")
    else:
        kind = "用户标记错误"
        tag_str = "、".join(tags or []) or "通用异常"
        detail = f"错误标签：{tag_str}{ref_suffix}" + (f" · {note.strip()}" if note and note.strip() else "")
    archive.record(chat_id, kind, detail, audio=audio, ja=ja, zh=zh)
    state = analyze_archive_and_update_weights(archive.directory / "director.sqlite3")
    state.pop("_pos_samples", None)
    state.pop("_neg_samples", None)
    return state


def finetune_internal_adapters(tts, db_path: Path, steps: int = 24, lr: float = 1.5e-4) -> dict[str, Any]:
    """Fine-tunes internal ProsodyDirector & GPT PerformanceAdapter on positive/negative history + anchor replay."""
    from performance import features

    state = analyze_archive_and_update_weights(db_path)
    pos_samples = state.pop("_pos_samples", [])
    neg_samples = state.pop("_neg_samples", [])

    device = tts.configs.device
    director = getattr(tts, "_director", None)
    adapter = _get_gpt_performance_module(tts)

    if director is not None and not DIRECTOR_BACKUP_PT.is_file():
        torch.save({k: v.detach().cpu() for k, v in director.state_dict().items()}, DIRECTOR_BACKUP_PT)
    if adapter is not None and not ADAPTER_BACKUP_PT.is_file():
        torch.save({k: v.detach().cpu() for k, v in adapter.state_dict().items()}, ADAPTER_BACKUP_PT)

    anchor_path = ROOT / "colab_data" / "training-data.pt"
    anchors = []
    if anchor_path.is_file():
        try:
            raw_anchors = torch.load(anchor_path, map_location="cpu", weights_only=False)
            anchors = [a for a in raw_anchors if len(a[2]) >= 8][:64]
        except Exception:
            anchors = []

    def _encode_sample(ja_text: str, intensity: float = 0.5):
        clean_ja = re.sub(r"<[a-z_]+>", "", ja_text).strip()
        if not clean_ja:
            return None, None
        plan = [{"start": 0, "end": len(clean_ja), "primary": "tender", "secondary": "shy", "intensity": intensity, "pause_s": 0.15}]
        try:
            ph, ctrl = features(clean_ja, plan, with_delivery=False)
            # ProsodyDirector takes base phoneme IDs < 732
            base_ph = [min(int(p), 731) for p in ph]
            if len(base_ph) < 4:
                return None, None
            return torch.tensor(base_ph, dtype=torch.long, device=device), ctrl[:22].to(device=device, dtype=torch.float32)
        except Exception:
            return None, None

    encoded_pos = [x for x in (_encode_sample(s["ja"], 0.50) for s in pos_samples[-24:]) if x[0] is not None]
    encoded_neg = []
    for s in neg_samples[-24:]:
        ph, ctrl = _encode_sample(s["ja"], 0.65)
        if ph is not None:
            encoded_neg.append((ph, ctrl, s["tags"]))

    director_loss_acc = 0.0
    director_loss_cnt = 0
    adapter_loss_acc = 0.0
    adapter_loss_cnt = 0

    # 1. Fine-tune ProsodyDirector on error tags (channel 3 = silence_prob, channels 0..2 = log_f0, slope, energy)
    if director is not None and encoded_neg:
        orig_dir_dtype = next(director.parameters()).dtype
        director.float().train()
        for p in director.parameters():
            p.requires_grad_(True)
        opt_d = torch.optim.AdamW(director.parameters(), lr=lr, weight_decay=1e-4)
        init_sd = {k: v.detach().float().clone() for k, v in director.state_dict().items()}

        # Feedback without measured target prosody should preserve approved behavior.
        director.eval()
        with torch.no_grad():
            positive_targets = [
                director(ph.unsqueeze(0), ctrl.unsqueeze(0))[0].detach()
                for ph, ctrl in encoded_pos
            ]
        director.train()
        for step in range(steps):
            opt_d.zero_grad()
            loss = torch.tensor(0.0, device=device, dtype=torch.float32)

            if encoded_neg:
                ph_n, ctrl_n, etags = encoded_neg[step % len(encoded_neg)]
                pros4, _ = director(ph_n.unsqueeze(0), ctrl_n.unsqueeze(0))
                if "silence_long" in etags:
                    sil_prob = torch.sigmoid(pros4[:, 3, :])
                    loss = loss + 0.35 * sil_prob.mean()
                    loss = loss + 0.25 * F.relu(-pros4[:, 2, :] - 0.6).mean()
                if "creak_hiss" in etags or "overdriven" in etags or "misread" in etags:
                    loss = loss + 0.30 * F.relu(pros4[:, :3, :].abs() - 1.1).pow(2).mean()
                    if pros4.shape[-1] > 2:
                        tv = (pros4[:, :3, 1:] - pros4[:, :3, :-1]).pow(2).mean()
                        loss = loss + 0.15 * tv

            if encoded_pos:
                pos_index = step % len(encoded_pos)
                ph_p, ctrl_p = encoded_pos[pos_index]
                pros4_p, _ = director(ph_p.unsqueeze(0), ctrl_p.unsqueeze(0))
                loss = loss + 0.10 * F.mse_loss(pros4_p, positive_targets[pos_index])

            l2_reg = sum((p - init_sd[k]).pow(2).sum() for k, p in director.named_parameters() if k in init_sd)
            loss = loss + 0.05 * l2_reg

            if loss.requires_grad and loss.item() > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(director.parameters(), 0.5)
                opt_d.step()
                director_loss_acc += float(loss.item())
                director_loss_cnt += 1

        for p in director.parameters():
            p.requires_grad_(False)
        director.to(dtype=orig_dir_dtype).eval()
        torch.save({k: v.detach().cpu() for k, v in director.state_dict().items()}, DIRECTOR_OPT_PT)

    # 2. Fine-tune GPT PerformanceAdapter (nn.Linear(ctrl_dim, 512, bias=False))
    # Perform optimization in float32 to avoid fp16 AdamW underflow, then cast back to orig dtype
    if adapter is not None and encoded_neg:
        orig_ad_dtype = next(adapter.parameters()).dtype
        adapter.float().train()
        for p in adapter.parameters():
            p.requires_grad_(True)
        opt_a = torch.optim.AdamW(adapter.parameters(), lr=lr * 0.5, weight_decay=1e-4)
        init_a = {k: v.detach().float().clone() for k, v in adapter.state_dict().items()}
        ctrl_dim = int(getattr(adapter, "in_features", getattr(tts, "_ctrl_dim", 22)))

        for step in range(steps):
            opt_a.zero_grad()
            loss_a = torch.tensor(0.0, device=device, dtype=torch.float32)

            if encoded_neg:
                _, ctrl_n, etags = encoded_neg[step % len(encoded_neg)]
                pad_n = torch.zeros(1, ctrl_dim, ctrl_n.shape[-1], device=device, dtype=torch.float32)
                use_c = min(22, ctrl_dim, ctrl_n.shape[0])
                pad_n[0, :use_c] = ctrl_n[:use_c]
                emb_n = adapter(pad_n.transpose(1, 2))  # [1, T, 512]
                if "silence_long" in etags and ctrl_dim > 21:
                    # Dampen pause-column projection weight to curb runaway mid-sentence silence
                    loss_a = loss_a + 0.15 * adapter.weight[:, 21].pow(2).mean()
                if "misread" in etags or "overdriven" in etags:
                    norm_n = emb_n.norm(dim=-1)
                    loss_a = loss_a + 0.25 * F.relu(norm_n - norm_n.mean() * 1.10).pow(2).mean()
                if emb_n.shape[1] > 2:
                    loss_a = loss_a + 0.15 * (emb_n[:, 1:, :] - emb_n[:, :-1, :]).pow(2).mean()

            if anchors:
                anc = anchors[step % len(anchors)]
                c_anc = anc[4].to(device=device, dtype=torch.float32)
                pad_a = torch.zeros(1, ctrl_dim, c_anc.shape[-1], device=device, dtype=torch.float32)
                use_a = min(c_anc.shape[0], ctrl_dim)
                pad_a[0, :use_a] = c_anc[:use_a]
                emb_a = adapter(pad_a.transpose(1, 2))
                if emb_a.shape[1] > 2:
                    baseline = F.linear(pad_a.transpose(1, 2), init_a["weight"])
                    loss_a = loss_a + 0.05 * F.mse_loss(emb_a, baseline)

            l2_a = sum((p - init_a[k]).pow(2).sum() for k, p in adapter.named_parameters() if k in init_a)
            loss_a = loss_a + 0.10 * l2_a

            if loss_a.requires_grad and loss_a.item() > 0:
                loss_a.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 0.5)
                opt_a.step()
                adapter_loss_acc += float(loss_a.item())
                adapter_loss_cnt += 1

        for p in adapter.parameters():
            p.requires_grad_(False)
        adapter.to(dtype=orig_ad_dtype).eval()
        torch.save({k: v.detach().cpu() for k, v in adapter.state_dict().items()}, ADAPTER_OPT_PT)

    director_loss_val = director_loss_acc / max(1, director_loss_cnt)
    adapter_loss_val = adapter_loss_acc / max(1, adapter_loss_cnt)

    state["adapter_steps"] = int(state.get("adapter_steps", 0)) + director_loss_cnt + adapter_loss_cnt
    state["last_director_loss"] = round(director_loss_val, 5)
    state["last_adapter_loss"] = round(adapter_loss_val, 5)
    WEIGHTS_JSON.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    state["_updated_steps"] = director_loss_cnt + adapter_loss_cnt
    return state


def load_self_optimized_models(tts) -> bool:
    loaded = False
    device = tts.configs.device
    director = getattr(tts, "_director", None)
    if director is not None and DIRECTOR_OPT_PT.is_file():
        try:
            d_dtype = next(director.parameters()).dtype
            sd = torch.load(DIRECTOR_OPT_PT, map_location=device, weights_only=True)
            sd = {k: v.to(device=device, dtype=d_dtype) for k, v in sd.items()}
            director.load_state_dict(sd, strict=False)
            loaded = True
        except Exception:
            pass
    adapter = _get_gpt_performance_module(tts)
    if adapter is not None and ADAPTER_OPT_PT.is_file():
        try:
            a_dtype = next(adapter.parameters()).dtype
            sd = torch.load(ADAPTER_OPT_PT, map_location=device, weights_only=True)
            sd = {k: v.to(device=device, dtype=a_dtype) for k, v in sd.items()}
            adapter.load_state_dict(sd, strict=False)
            loaded = True
        except Exception:
            pass
    return loaded


def rollback_self_optimization(tts) -> dict[str, Any]:
    device = tts.configs.device
    director = getattr(tts, "_director", None)
    if director is not None and DIRECTOR_BACKUP_PT.is_file():
        d_dtype = next(director.parameters()).dtype
        sd = torch.load(DIRECTOR_BACKUP_PT, map_location=device, weights_only=True)
        director.load_state_dict({k: v.to(device=device, dtype=d_dtype) for k, v in sd.items()}, strict=False)
    adapter = _get_gpt_performance_module(tts)
    if adapter is not None and ADAPTER_BACKUP_PT.is_file():
        a_dtype = next(adapter.parameters()).dtype
        sd = torch.load(ADAPTER_BACKUP_PT, map_location=device, weights_only=True)
        adapter.load_state_dict({k: v.to(device=device, dtype=a_dtype) for k, v in sd.items()}, strict=False)
    for p in (WEIGHTS_JSON, DIRECTOR_OPT_PT, ADAPTER_OPT_PT):
        if p.is_file():
            p.unlink()
    return load_adaptive_weights()


def format_self_opt_report(state: dict[str, Any]) -> str:
    if not state or not state.get("updated_at"):
        return "尚未运行自优化（当前使用出厂默认权重）。点击「立即根据错题记录微调权重」即可基于历史异常与人工标记自我升级。"
    errs = state.get("error_counts") or {}
    tag_zh = {v: k for k, v in ERROR_TAG_MAP.items()}
    err_str = "、".join(f"{tag_zh.get(k, k)}:{v}" for k, v in errs.items()) or "无"
    scores = state.get("ref_scores") or {}
    boosted = [f"{r}({s:+.2f})" for r, s in sorted(scores.items(), key=lambda x: -x[1]) if s > 0.01][:5]
    penalized = [f"{r}({s:+.2f})" for r, s in sorted(scores.items(), key=lambda x: x[1]) if s < -0.01][:5]
    lines = [
        f"🧬 最近自优化时间：{state['updated_at']}（累计微调 {state.get('adapter_steps', 0)} 步）",
        f"📊 历史样本统计：正样本 {state.get('total_positives', 0)} 条 ｜ 错题/异常 {state.get('total_negatives', 0)} 条",
        f"🏷️ 错误标签分布：{err_str}",
        f"⚙️ 自适应声学偏置：停顿缩放={state.get('pause_scale', 1.0):.3f} ｜ 静音抑制={state.get('silence_logit_bias', 0.0):+.2f} ｜ 情绪倍率修正={state.get('emo_scale_cap_delta', 0.0):+.3f}",
    ]
    if boosted:
        lines.append("⭐ 高分优选参考音：" + "、".join(boosted))
    if penalized:
        lines.append("⚠️ 自动降权问题参考音：" + "、".join(penalized))
    if "last_director_loss" in state:
        lines.append(f"🧠 内部网络微调 Loss：ProsodyDirector={state['last_director_loss']:.4f} ｜ PerformanceAdapter={state.get('last_adapter_loss', 0.0):.4f}")
    return "\n".join(lines)
