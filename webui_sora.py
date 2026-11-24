"""Local Gradio WebUI for Sora Kasugano expressive GPT-SoVITS synthesis."""
import argparse
import base64
import csv
from functools import lru_cache, partial
import html
import json
import mimetypes
import os
from pathlib import Path
import re
import threading
import time
import traceback
from uuid import uuid4
import urllib.error
from urllib.parse import urlsplit

from audio_goal import ALIYUN_PROVIDER, GEMINI_PROVIDER, JUDGE_PRESETS, OPENAI_PROVIDER

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
for _k in ("NO_PROXY", "no_proxy"):
    _cur = os.environ.get(_k, "")
    _parts = [p.strip() for p in _cur.split(",") if p.strip()]
    for _h in ("127.0.0.1", "localhost"):
        if _h not in _parts:
            _parts.append(_h)
    os.environ[_k] = ",".join(_parts)

import gradio as gr
from gradio.utils import get_upload_folder

from director_archive import DirectorArchive
from self_optimize import (
    ERROR_TAG_CHOICES,
    analyze_archive_and_update_weights,
    finetune_internal_adapters,
    format_self_opt_report,
    load_adaptive_weights,
    record_user_feedback,
    rollback_self_optimization,
)

from performance import (
    DIRECTOR_STRING, OLLAMA_DIRECTOR_URL, director_json_schema, is_local_ollama, ollama_director_chat,
    APP, EMOTIONS, deepseek_json, deepseek_json_stream, load_sora_persona, plan_with_deepseek, plan_with_ollama, rank_references_with_deepseek, report_director_usage, validate,
)
from synthesize import (
    _REF_PROFILES_BY_ID,
    _REF_PROFILES_LIST,
    API_PERFORMANCE_REFS,
    DATA,
    ROOT,
    auto_insert_tags,
    auto_plan,
    engine,
    pick_performance_refs,
    refine_plan,
    render,
)

DEFAULT_GPT = ROOT / "v3_checkpoints/gpt_eval/phrase-adapter-epoch15-ft1.pt"
DEFAULT_SOVITS = ROOT / "v3_bs16_checkpoints/sovits_eval/sora-sovits-v2proplus-epoch25.pth"
OUT_DIR = ROOT / "results/webui"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE = DirectorArchive(ROOT / "results/director_archive")

CONFIG_PATH = ROOT / "webui_config.json"
AUTO_PERFORMANCE = "自动选参考，整句合成"
DELIVERY_LABELS = {
    "plain": "自然", "restrained": "压声", "soft": "轻柔", "tearful": "哭腔",
    "trembling": "发颤", "emphatic": "重声", "bright": "轻快",
    "hesitant": "迟疑", "startled": "受惊",
}
_EMO_SHORT = {
    "neutral": "日常", "tender": "温柔", "shy": "害羞", "hurt": "委屈",
    "anger": "生气", "impatient": "不耐", "fear": "不安", "joy": "开心",
    "surprise": "惊讶", "resolve": "坚定",
}

from webui_theme import (
    UI_ASSETS_DIR,
    UI_CSS,
    UI_HERO,
    UI_THEME,
    _CHIBI_STICKER_URI,
    _CHIBI_WRITE_URI,
    _KURO_USAGI_ICON_SVG,
)

_TTS_LOCK = threading.Lock()
_CONFIG_LOCK = threading.Lock()
_TTS_ENGINE = None

OLLAMA_PROVIDER = "本地 Ollama"

PROVIDER_PRESETS = {
    OLLAMA_PROVIDER: (OLLAMA_DIRECTOR_URL, ""),
    "DeepSeek 官方": ("https://api.deepseek.com", "deepseek-chat"),
    "OpenAI 官方": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "硅基流动 (SiliconFlow)": ("https://api.siliconflow.cn/v1", "Qwen/Qwen2.5-7B-Instruct"),
    "阿里云百炼 (DashScope)": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
    "OpenRouter": ("https://openrouter.ai/api/v1", "deepseek/deepseek-chat"),
    "自定义 OpenAI 兼容接口": ("", ""),
}


def _load_config() -> dict:
    if CONFIG_PATH.is_file():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def _same_endpoint(first, second):
    return (first or "").strip().rstrip("/") == (second or "").strip().rstrip("/")


def _director_key(value, endpoint):
    if is_local_ollama(endpoint):
        return "ollama"
    if (value or "").strip():
        return value.strip()
    cfg = _load_config()
    if _same_endpoint(endpoint, cfg.get("base_url")) and cfg.get("api_key"):
        return cfg["api_key"].strip()
    if _same_endpoint(endpoint, os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")):
        return os.environ.get("DEEPSEEK_API_KEY", "").strip()
    return ""


def _write_config(data):
    temporary = CONFIG_PATH.with_name(f".{CONFIG_PATH.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(CONFIG_PATH)
    finally:
        temporary.unlink(missing_ok=True)


def _save_config(provider: str, base_url: str, model_name: str, api_key: str) -> str:
    if provider == OLLAMA_PROVIDER and not (model_name or "").strip():
        raise gr.Error("请先检测并选择一个 Ollama 模型。")
    with _CONFIG_LOCK:
        data = _load_config()
        stored_key = data.get("api_key", "") if _same_endpoint(base_url, data.get("base_url")) else ""
        data.update({
            "provider": provider,
            "base_url": (base_url or "").strip(),
            "model": (model_name or "").strip(),
            "api_key": "" if is_local_ollama(base_url) else ((api_key or "").strip() or stored_key),
        })
        _write_config(data)
    return f"已保存配置到 {CONFIG_PATH.name}（下次打开自动读取）"


def save_judge_config(provider, base_url, model, api_key):
    with _CONFIG_LOCK:
        data = _load_config()
        old = data.get("audio_judge") or {}
        stored_key = old.get("api_key", "") if old.get("provider") == provider and _same_endpoint(base_url, old.get("base_url")) else ""
        data["audio_judge"] = {
            "provider": provider,
            "base_url": base_url.strip(),
            "model": model.strip(),
            "api_key": (api_key or "").strip() or stored_key,
        }
        _write_config(data)
    return "已保存音频评审配置"


def on_judge_provider_change(provider):
    url, model = JUDGE_PRESETS[provider]
    return url, model, ""


EMOTION_LABELS = [
    ("neutral (平淡日常)", "neutral"),
    ("tender (温柔/柔声)", "tender"),
    ("shy (害羞/扭捏)", "shy"),
    ("hurt (委屈/受伤/哭诉)", "hurt"),
    ("anger (生气/怒斥)", "anger"),
    ("impatient (不耐烦/急躁)", "impatient"),
    ("fear (害怕/不安)", "fear"),
    ("joy (开心)", "joy"),
    ("surprise (惊讶)", "surprise"),
    ("resolve (坚定/认真/追问)", "resolve"),
]
SECONDARY_LABELS = [("无 (None)", "none"), *EMOTION_LABELS]


def get_tts():
    global _TTS_ENGINE
    with _TTS_LOCK:
        if _TTS_ENGINE is None:
            os.chdir(APP)
            _TTS_ENGINE = engine(DEFAULT_GPT, DEFAULT_SOVITS)
        return _TTS_ENGINE


def translate_zh_to_sora_ja(
    zh_text: str,
    context: str,
    api_key: str,
    base_url: str = "",
    model: str = "",
) -> str:
    key = _director_key(api_key, base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    if not key:
        raise gr.Error("中文台词需要先转成日语。请在底部「API 设置 → 导演模型」选择本地 Ollama 并选中模型，或填写云端 API Key。也可直接输入日语台词。")
    url = (base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).strip().rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    mdl = (model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")).strip()
    prompt = (
        "请把下面用户输入的中文意思，改写成『缘之空』春日野穹（Sora）的自然日语口语台词。\n"
        "要求：\n"
        "1. **彻底口语化、拒绝机翻书面腔**：严禁生硬堆砌汉字、严禁过度使用「あなた」或敬体「です／ます」；用穹的第一人称「私」，默认自然省略对方称呼（**只要中文原话里没写“悠”或“ハル”，严禁擅自加上「ハル」**），句尾自然带上少女口语助词（如「……なの」「……なんだから」「……でしょ？」「……だよ」「……ってば」）。\n"
        "2. **符合真实说话停顿**：根据情绪留白补全自然的日语标点（「、」「。」「？」「！」「……」），不要每两三个字就碎断一次，也不要一整句不喘气；禁止连续写「…… ……」双省略号，也避免句末单挂孤立的「……ん……」。\n"
        "3. 只输出最终的日语台词文本本身，不要解释、不要罗马音、不要引号。\n\n"
        f"情景/前情上下文：{context or '无'}\n"
        f"中文原话：{zh_text}"
    )
    persona = load_sora_persona()
    sys_msg = "你是精通『缘之空』春日野穹角色性格的日语配音台本翻译。只输出地道口语化的日语台词。"
    if persona:
        sys_msg += f"\n\n【春日野穹角色与语言规范】\n{persona}"
    payload = {
        "model": mdl,
        "temperature": 0.35,
        "messages": [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ],
    }
    if is_local_ollama(base_url):
        started = time.perf_counter()
        body = next(ollama_director_chat(sys_msg, prompt, mdl, temperature=0.35))
        report_director_usage("中译日", mdl, {"prompt_tokens": body.get("prompt_eval_count"),
                                             "completion_tokens": body.get("eval_count")}, started)
        return body["message"]["content"].strip().strip("「」" + chr(34) + chr(39))
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    for attempt in range(2):
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            report_director_usage("中译日", mdl, body.get("usage") if isinstance(body, dict) else None, started)
            return body["choices"][0]["message"]["content"].strip().strip("「」\"'")
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == 1:
                raise
            time.sleep(0.8)


_EMOTION_CHIP = {
    "neutral": ("😐", "#e2e8f0", "#334155"), "tender": ("🌸", "#fce7f3", "#9d174d"),
    "shy": ("☺️", "#ffe4e6", "#be123c"), "hurt": ("💧", "#dbeafe", "#1e40af"),
    "anger": ("💢", "#fee2e2", "#b91c1c"), "impatient": ("😤", "#ffedd5", "#c2410c"),
    "fear": ("😨", "#e0e7ff", "#4338ca"), "joy": ("✨", "#fef9c3", "#a16207"),
    "surprise": ("❗", "#ccfbf1", "#0f766e"), "resolve": ("🔥", "#ede9fe", "#6d28d9"),
}


def _emotion_chip(name: str, prefix: str = "") -> str:
    icon, bg, fg = _EMOTION_CHIP.get(name, ("🏷️", "#f1f5f9", "#334155"))
    return (
        f"<span style='background:{bg};color:{fg};padding:3px 10px;border-radius:999px;"
        f"font-size:12px;font-weight:600;'>{icon} {prefix}{html.escape(name)}</span>"
    )


def _render_locked_pills(segments: list[dict]) -> str:
    if not segments:
        return (
            "<div class='sora-pill-empty'>✦ 还没有锁定片段 · 输入台词后按 <b>Enter</b> 添加</div>"
        )
    cards = []
    for idx, seg in enumerate(segments, 1):
        sec = seg.get("secondary")
        sec_badge = _emotion_chip(sec, "+") if sec else ""
        cards.append(
            "<div class='sora-pill-row'>"
            f"<span class='sora-pill-idx'>{idx:02d}</span>"
            f"{_emotion_chip(seg['primary'])}{sec_badge}"
            f"<span class='sora-pill-meta'>强度 {seg['intensity']:.2f}</span>"
            f"<span class='sora-pill-meta'>⏸ {seg['pause_s']:.2f}s</span>"
            f"<span class='sora-pill-text'>{html.escape(seg['text'])}</span>"
            "</div>"
        )
    return "".join(cards)


def add_locked_segment(
    seg_text: str,
    primary: str,
    secondary: str,
    intensity: float,
    pause_s: float,
    segments: list[dict],
):
    seg_text = (seg_text or "").strip()
    segs = list(segments or [])
    if seg_text:
        sec = None if secondary in ("none", "", None, primary) else secondary
        segs.append(
            {
                "text": seg_text,
                "primary": primary if primary in EMOTIONS else "neutral",
                "secondary": sec,
                "intensity": float(max(0.0, min(1.0, intensity))),
                "pause_s": float(max(0.0, min(1.5, pause_s))),
            }
        )
    return "", segs, _render_locked_pills(segs)


def pop_locked_segment(segments: list[dict]):
    segs = list(segments or [])
    if segs:
        segs.pop()
    return segs, _render_locked_pills(segs)


def clear_locked_segments():
    return [], _render_locked_pills([])


def _build_manual_whole_plan(text: str, primary: str, secondary: str, intensity: float) -> list[dict]:
    sec = None if secondary in ("none", "", None, primary) else secondary
    return [
        {
            "start": 0,
            "end": len(text),
            "primary": primary if primary in EMOTIONS else "neutral",
            "secondary": sec,
            "intensity": float(max(0.0, min(1.0, intensity))),
            "pause_s": 0.0,
        }
    ]


def _make_plan(
    ja_text: str,
    context: str,
    plan_mode: str,
    api_key: str,
    base_url: str,
    model_name: str,
    manual_primary: str,
    manual_secondary: str,
    manual_intensity: float,
    custom_plan_json: str,
) -> tuple[list[dict], str]:
    if plan_mode == "使用下方分段 JSON":
        try:
            parsed = json.loads(custom_plan_json or "")
            if not isinstance(parsed, list) or not parsed:
                raise ValueError("分段 JSON 必须是非空数组")
            validate(ja_text, parsed)
        except ValueError as exc:
            raise gr.Error(f"分段 JSON 无效：{exc}") from exc
        return parsed, "使用手动编辑的分段 JSON"

    if plan_mode == "手动指定整句情绪":
        plan = _build_manual_whole_plan(ja_text, manual_primary, manual_secondary, manual_intensity)
        return refine_plan(ja_text, plan), f"手动整句: {manual_primary}"

    # 自动导演规划
    burl = (base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).strip()
    key = _director_key(api_key, burl)
    mdl = (model_name or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")).strip()
    if plan_mode == "自动导演 (API 大模型)" and key:
        try:
            plan = plan_with_deepseek(ja_text, context, api_key=key, base_url=burl, model=mdl)
            return refine_plan(ja_text, plan), f"API 导演规划 ({mdl})"
        except Exception as exc:
            note = f"API 失败({exc})，已回退"
    elif plan_mode == "自动导演 (本地 Ollama)":
        try:
            plan = plan_with_ollama(ja_text, context)
            return refine_plan(ja_text, plan), "Ollama 本地导演规划"
        except Exception as exc:
            note = f"Ollama 失败({exc})，已回退"
    else:
        note = "未填 API Key，尝试本地 Ollama / 规则"

    if plan_mode == "自动导演 (API 大模型)":
        try:
            plan = plan_with_ollama(ja_text, context)
            return refine_plan(ja_text, plan), f"{note} -> Ollama"
        except Exception:
            pass

    return refine_plan(ja_text, auto_plan(ja_text)), f"{note} -> 规则分段"


def _line_language(mode, text):
    if mode in ("中文", "日语"):
        return mode
    if re.fullmatch(r"(?:[\s…。！？、!?,.]*<(?:breath|gasp|hold|oral|pant|pant_clip|sob|laugh)>)+[\s…。！？、!?,.]*", text):
        return "日语"
    return "日语" if re.search(r"[ぁ-ゖァ-ヺㇰ-ㇿｦ-ﾟ]", text) else "中文"


IMAGE_ATTACHMENTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
TEXT_ATTACHMENTS = {".txt", ".md", ".csv", ".json", ".srt", ".ass", ".vtt"}
DOCUMENT_ATTACHMENTS = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}
ATTACHMENT_TYPES = IMAGE_ATTACHMENTS | TEXT_ATTACHMENTS | DOCUMENT_ATTACHMENTS


def _uploaded_attachment(path):
    file = Path(path).resolve()
    folders = (Path(get_upload_folder()), ARCHIVE.attachments)
    if not any(file.is_relative_to(folder.resolve()) for folder in folders) or not file.is_file():
        raise gr.Error("附件已失效，请重新上传。")
    if file.suffix.lower() not in ATTACHMENT_TYPES:
        raise gr.Error(f"暂不支持 {file.suffix} 附件。")
    return file


@lru_cache(maxsize=32)
def _deepseek_document(file, modified_ns, size, key, purpose):
    import requests

    with file.open("rb") as stream:
        response = requests.post(
            "https://api.deepseek.com/files",
            headers={"Authorization": f"Bearer {key}"},
            data={"purpose": purpose},
            files={"file": (file.name, stream)},
            timeout=150,
        )
    response.raise_for_status()
    return response.json()["id"]


def _discussion_content(prompt, history, key, base_url, model):
    paths = list(dict.fromkeys(
        turn["content"][0] for turn in history[-24:]
        if turn.get("role") == "user" and isinstance(turn.get("content"), (tuple, list))
        and turn["content"]
    ))
    if not paths:
        return prompt
    files = [_uploaded_attachment(path) for path in paths]
    if len(files) > 8 or sum(file.stat().st_size for file in files) > 24 * 1024 * 1024:
        raise gr.Error("每段讨论可带最多 8 个附件，总大小不超过 24 MB。")
    content = [{"type": "text", "text": prompt}]
    for file in files:
        suffix = file.suffix.lower()
        content.append({"type": "text", "text": f"附件：{file.name}"})
        if suffix in IMAGE_ATTACHMENTS:
            encoded = base64.b64encode(file.read_bytes()).decode("ascii")
            media_type = mimetypes.guess_type(file.name)[0]
            content.append({"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{encoded}"}})
        elif suffix in TEXT_ATTACHMENTS:
            raw = file.read_bytes()
            for encoding in ("utf-8-sig", "gb18030", "cp932"):
                try:
                    decoded = raw.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise gr.Error(f"无法读取 {file.name}，请另存为 UTF-8 后重传。")
            if len(decoded) > 100_000:
                raise gr.Error(f"{file.name} 超过 10 万字，请拆分后提交。")
            content.append({"type": "text", "text": decoded})
        else:
            url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            mdl = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
            if urlsplit(url).hostname != "api.deepseek.com" or mdl not in {
                "deepseek-flash", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp",
            }:
                raise gr.Error("PDF 和 Office 附件需要使用 DeepSeek 官方 API 的 deepseek-flash。")
            api_key = _director_key(key, base_url)
            if not api_key:
                raise gr.Error("请先在 API 设置中填写密钥。")
            stat = file.stat()
            purpose = "user_data" if suffix in {".xls", ".xlsx"} else "rich_layout_recognition"
            file_id = _deepseek_document(file, stat.st_mtime_ns, stat.st_size, api_key, purpose)
            content.append({"type": "file", "file_id": file_id})
    return content


def _discussion_payload(input_mode, line, context, history):
    return json.dumps({
        "台词语言": _line_language(input_mode, line) if line else "未定",
        "台词": line,
        "场景与语气": context,
        "讨论记录": [
            {"role": turn["role"], "content": turn["content"]}
            for turn in history[-24:]
            if turn.get("role") in ("user", "assistant") and isinstance(turn.get("content"), str)
        ],
    }, ensure_ascii=False)


def _confirms_production(message, history):
    if not any(turn.get("role") == "assistant" for turn in (history or [])):
        return False
    return bool(re.fullmatch(
        r"(?:好[的了]?|行|可以|就这样|就这个|按(?:刚才|上面|这个)(?:的)?方案)?[，,、\s]*"
        r"(?:那就|那|就)?(?:直接|现在|马上)?(?:开始)?"
        r"(?:做|制作|生成|合成|录制|开做|动手做|按这个做)(?:吧|了|呗)?[。！!\s]*",
        (message or "").strip(),
    ))


def discuss_performance_stream(message, history, input_mode, line, context, api_key, base_url, model, *, production_task=False):
    composer = message or {}
    message = (composer.get("text", "") or "").strip()
    paths = composer.get("files") or []
    if not message and not paths:
        raise gr.Error("先输入消息或选择附件。")
    production_task = production_task or _confirms_production(message, history)
    turns = [*(history or []), *({"role": "user", "content": (path,)} for path in paths)]
    request_text = message or "请查看我附上的内容，并结合当前讨论给出回应。"
    turns.append({"role": "user", "content": request_text})
    yield [*turns, {"role": "assistant", "content": "…"}], None
    url = (base_url or "").strip()
    key = _director_key(api_key, url)
    if not key:
        raise gr.Error("请先配置导演 API 密钥。")
    mdl = (model or "").strip()
    try:
        local = is_local_ollama(url)
        payload = json.loads(_discussion_payload(input_mode, line or "", context, turns))
        payload["用户已明确下达制作任务"] = production_task
        payload["可用表演参考素材"] = _reference_notes(API_PERFORMANCE_REFS)
        discussion_only = local and not production_task and bool(re.search(
            r"(?:^|[，。；,;])\s*(?:不要|别|不用|先不|暂不)(?:生成|合成|制作|录制)(?=\s*(?:[。！？!?，,；;]|$))", message,
        ))
        if local:
            payload["最新要求"] = request_text
        persona = load_sora_persona()
        discuss_sys = (
            "你是春日野穹的日语配音导演，用简洁中文与用户讨论台词和表演。"
            "附件是用户指定的参考素材，按用户在对话中的要求使用。结合输入区和讨论记录回应最新一条用户消息。可以提议日语台词，说明潜台词、语速、停顿和是否需要独立拟声。"
            "这个页面能读取提供的候选目录并自动选参考、生成试听。你可以推荐目录中的具体编号，说明是按文本与标注判断。"
            "页面执行本地合成，你没有收到原声或生成音频，不能声称听过。"
            "用户否决的方案不要再当作共识。当用户下达制作任务时，action 填 generate。"
            "“给我做一段雨夜重逢”“录一段嘴硬但其实在关心对方的语音”都是制作任务，即使没用‘生成’这个词。"
            "结合上一轮的问题和已商定的方案判断用户现在是否授权制作。只要语义明确就填 generate，无需出现任何关键词。比如上一轮问是否开做，用户答‘行’或‘就它了’，可以立即制作；上一轮仅询问喜欢哪版时，同样的答复只是选定方案，继续 discuss。‘按刚才那个来，剩下的你定’等间接下达的制作任务也填 generate，并沿用已确认的台词和安排。还在提问、比较、要求先看修改稿、暂缓或取消时填 discuss。明确下达制作任务时，按制作任务处理。"
            "收到制作任务后，自行补足未指定的场景、人物关系、事件和潜台词。自行写自然日语台词，决定语速、停顿、情绪变化和拟声。"
            "台词一律保持穹本人对当前听众说话，除非用户在消息里明确提到“悠/ハル”，否则不要擅自在台词里加「ハル」或任何具体人名。"
            "方案以用户的明确要求为准，普通创作细节自行决定。回复简短展示场景、拟采用的台词和演法。"
            "询问是否能选参考、试音建议或讨论演法都填 discuss。生成动作执行前，回复不要声称已经完成。"
            "只输出 JSON，reply 为给用户看的回复，action 为 discuss 或 generate。"
        )
        if local:
            discuss_sys += "请直接回答最新要求，用中文给出一两句表演建议。台词示例用日语。制作阶段会从参考库选角。"
        if persona:
            discuss_sys += f"\n\n【春日野穹角色与配音表演基准档案】\n{persona}"
        final_result = None
        final_reply = ""
        for partial_reply, parsed in deepseek_json_stream(
            discuss_sys,
            _discussion_content(json.dumps(payload, ensure_ascii=False), turns, key, url, mdl),
            api_key=key, base_url=url, model=mdl,
            schema=director_json_schema(reply=DIRECTOR_STRING, action={"enum": ["discuss"] if discussion_only else ["discuss", "generate"]}),
        ):
            if partial_reply:
                final_reply = partial_reply
                yield [*turns, {"role": "assistant", "content": partial_reply}], None
            if parsed is not None:
                final_result = parsed
        reply = (final_result or {}).get("reply", final_reply).strip()
        if not reply or final_result is None:
            raise ValueError("empty reply")
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as exc:
        detail = str(exc) if str(exc).startswith("API 连续两次") else type(exc).__name__
        raise gr.Error(f"导演未能回复（{detail}）。") from exc
    action = "generate" if production_task else final_result.get("action", "discuss")
    if discussion_only:
        action = "discuss"
    yield [*turns, {"role": "assistant", "content": reply}], action


def _render_progress(pct: float, desc: str) -> str:
    p = max(2, min(100, int(round(float(pct) * 100))))
    safe_desc = html.escape(desc or "正在合成…")
    return (
        "<div class='sora-prog-card'>"
        f"<div class='sora-prog-head'><span class='sora-prog-title'>{_KURO_USAGI_ICON_SVG}<span>{safe_desc}</span></span><span>{p}%</span></div>"
        f"<div class='sora-prog-track'><div class='sora-prog-fill' style='width:{p}%'></div></div>"
        "</div>"
    )


def _render_generation_error(detail: str) -> str:
    return (
        "<div class='sora-prog-card' role='alert'>"
        "<div class='sora-prog-head'><strong>生成失败</strong></div>"
        f"<div>{html.escape(detail)}</div></div>"
    )


def _generation_error_detail(exc: Exception) -> str:
    if isinstance(exc, gr.Error):
        return exc.message
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in (401, 403):
            return f"导演接口拒绝了请求（HTTP {exc.code}）。请检查 API Key 和模型权限。"
        return f"导演接口返回 HTTP {exc.code}。请核对 API 地址和模型名后重试。"
    if isinstance(exc, (urllib.error.URLError, TimeoutError)):
        return "未能连接导演接口或等待超时。使用本地 Ollama 时，请确认服务已启动、模型已选中，再重试。"
    traceback.print_exception(exc)
    return f"合成中断（{type(exc).__name__}）。详细原因见启动窗口。"


def _run_synth_with_progress(*args, **kwargs):
    """Run synthesize_action in a background thread while streaming dedicated progress bar updates."""
    import queue

    q: queue.Queue = queue.Queue()
    box = {"res": None, "err": None}

    def _cb(val, desc):
        q.put((float(val), str(desc)))

    def _worker():
        try:
            box["res"] = synthesize_action(*args, **kwargs, progress_cb=_cb)
        except BaseException as exc:
            box["err"] = exc
        finally:
            q.put(None)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    last_state = (0.03, "📝 正在准备台词与导演配置…")
    yield gr.update(value=_render_progress(*last_state), visible=True), None
    while True:
        try:
            item = q.get(timeout=0.18)
        except queue.Empty:
            continue
        if item is None:
            break
        while not q.empty():
            nxt = q.get_nowait()
            if nxt is None:
                item = None
                break
            item = nxt
        if item is None:
            break
        last_state = item
        yield gr.update(value=_render_progress(*last_state), visible=True), None
    t.join()
    if box["err"] is not None:
        raise box["err"]
    yield gr.update(value="", visible=False), box["res"]


def _conversation_goal(history, goal_settings, synthesis_args, conversation_id=None):
    judge_key, judge_model, judge_provider, judge_url, rounds, use_current, audio, ja, zh = goal_settings
    goal = "\n".join([
        turn["content"].strip() for turn in history
        if turn.get("role") == "user" and isinstance(turn.get("content"), str) and turn["content"].strip()
    ][-6:])
    if not goal:
        raise gr.Error("请在对话输入框里写下试听目标。")
    if use_current and not audio:
        raise gr.Error("试听区还没有可用于对比的成品。")
    for *result, baseline, status in iterate_to_goal(
        goal, judge_key, judge_model, judge_provider, judge_url, rounds, audio if use_current else None, ja, zh, history, *synthesis_args, conversation_id=conversation_id,
    ):
        yield *result, gr.update(value=baseline, visible=bool(baseline)), gr.update(value=status, visible=True)


def director_chat_action(message, history, execution_mode, goal_mode, judge_key, judge_model, judge_provider, judge_url, rounds,
                         use_current, audio, ja, zh, input_mode, line, context, provider, key, url,
                         model, *settings, production_task=False, conversation_id=None):
    turns = list(history or [])
    action = None
    for stream_turns, stream_action in discuss_performance_stream(
        message, history, input_mode, line, context, key, url, model, production_task=production_task,
    ):
        turns = stream_turns
        action = stream_action
        if action is None:
            yield turns, *(gr.skip() for _ in range(10))

    generate = action == "generate"
    if generate and execution_mode != "全自动":
        turns = [*turns, {"role": "assistant", "content": "已收到生成请求。请点「按右侧讨论生成」批准。"}]
        generate = False

    yield turns, *(gr.skip() for _ in range(10))
    if not generate:
        return

    synthesis_args = (input_mode, line, context, provider, key, url, model, *settings)
    try:
        if goal_mode:
            goal_settings = (judge_key, judge_model, judge_provider, judge_url, rounds, use_current, audio, ja, zh)
            for output in _conversation_goal(turns, goal_settings, synthesis_args, conversation_id):
                yield turns, *output, gr.update(value="", visible=False)
        else:
            for prog_upd, output in _run_synth_with_progress(
                *synthesis_args, discussion_history=turns, conversation_id=conversation_id,
            ):
                if output is None:
                    yield gr.skip(), *(gr.skip() for _ in range(9)), prog_upd
                else:
                    yield [*turns, {"role": "assistant", "content": "已生成，可以在下方试听。"}], *output, gr.skip(), gr.skip(), prog_upd
    except Exception as exc:
        err_detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        reply = str(exc) if isinstance(exc, gr.Error) else f"生成失败（{err_detail}）。"
        ARCHIVE.record(conversation_id, "生成问题", reply)
        yield [*turns, {"role": "assistant", "content": reply}], *(gr.skip() for _ in range(9)), gr.update(value="", visible=False)


def generate_from_conversation(message, history, goal_mode, judge_key, judge_model, judge_provider, judge_url, rounds,
                               use_current, audio, ja, zh, *settings, conversation_id=None):
    if not goal_mode:
        for prog_upd, output in _run_synth_with_progress(
            *settings, discussion_history=history, conversation_id=conversation_id,
        ):
            if output is None:
                yield gr.skip(), *(gr.skip() for _ in range(9)), prog_upd
            else:
                yield history, *output, gr.skip(), gr.skip(), prog_upd
        return

    draft = message or {}
    turns = list(history or [])
    turns.extend({"role": "user", "content": (path,)} for path in draft.get("files", []))
    text = (draft.get("text") or "").strip()
    if text:
        turns.append({"role": "user", "content": text})
    goal_settings = (judge_key, judge_model, judge_provider, judge_url, rounds, use_current, audio, ja, zh)
    for output in _conversation_goal(turns, goal_settings, settings, conversation_id):
        yield turns, *output, gr.update(value="", visible=False)


def _finish_discussion(input_mode, line, context, history, key, base_url, model):
    if not history:
        raise gr.Error("先发一条消息和导演讨论，再按讨论生成。")
    try:
        persona = load_sora_persona()
        finish_sys = (
            "你是春日野穹的日语配音导演。把输入区和讨论定稿为一条可直接合成的日语台词与表演说明。"
            "附件作为参考素材，按对话里的用户要求使用。以最近的用户要求为准，保留原场景与已确认的细节，排除用户否决的提议。"
            "按输入方式认定原台词。如果已有日语原句且用户只要求调整演法，逐字保留该句；"
            "如果以中文为准，翻成自然日语；如果用户讨论了新台词，采用最新确认的内容。"
            "制作任务只给主题或情境时，自行构思发生了什么、角色想掩饰什么，再写出适合该情境的日语台词。"
            "台词一律只以穹的口吻对当前听众说话，除非用户在对话或输入里明确提到“悠/ハル”，否则不要擅自在台词里添加「ハル」或任何角色名字。"
            "禁止在台词中连续写「…… ……」双省略号或堆砌多余省略号，也避免句末单挂孤立的「……ん……」，改用正常逗号「、」、单组「……」或贴近实词的收尾。"
            "需要短场景时可用连贯的几句，按任务要求控制长度。批准过的剧情与台词优先。别把动作旁白写入要念的台词。"
            "把场景和潜台词简述放进 context，让后续规划决定情绪变化和选参考。"
            "表演说明包含有共识的语气、节奏和拟声意图。保留用户明确确认的参考编号或选取偏好，交给后续参考排序。"
            "只输出 JSON。ja_text 为最终日语台词，zh_text 为这份日语台词的自然中文译文，context 为中文表演说明，三个字段均为字符串。"
        )
        local = is_local_ollama(base_url)
        if local:
            finish_sys += "ja_text 只填要念的日语句子，例如「もう……遅いよ。」。把解释放在 context，禁止把用户指令、中文或 Markdown 放入 ja_text。"
        if persona:
            finish_sys += f"\n\n【春日野穹角色与配音表演基准档案】\n{persona}"
        result = deepseek_json(
            finish_sys,
            _discussion_content(_discussion_payload(input_mode, line, context, history), history, key, base_url, model),
            api_key=key, base_url=base_url, model=model, operation="讨论定稿",
            schema=director_json_schema(ja_text=DIRECTOR_STRING, zh_text=DIRECTOR_STRING, context=DIRECTOR_STRING),
        )
        ja_line = " ".join(result["ja_text"].splitlines()).strip()
        brief = result["context"].strip()
        translation = result.get("zh_text")
        translation = translation.strip() if isinstance(translation, str) else ""
        if not ja_line:
            raise ValueError("empty line")
        if local and not re.search(r"[ぁ-ゖァ-ヺ]", ja_line) and ja_line != (line or "").strip():
            raise gr.Error("本地导演没有给出可合成的日语台词，请要求它重写为日语后再生成。")
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as exc:
        raise gr.Error(f"导演未能定稿，请检查 API 配置后重试（{type(exc).__name__}）。") from exc
    return ja_line, "\n".join(part for part in (context, brief) if part), translation


def _reference_notes(ref_ids):
    from soundfile import info

    notes = {}
    pending = []
    for ref_id in ref_ids:
        prof = _REF_PROFILES_BY_ID.get(ref_id)
        if prof:
            notes[ref_id] = {
                "id": ref_id,
                "line": prof["text_ja"],
                "acoustic_profile": prof["acoustic_summary"],
                "historical_note": prof["historical_note"],
                "historical_label": prof["historical_label"],
            }
        else:
            pending.append(ref_id)

    if pending:
        pend_set = set(pending)
        csv_path = ROOT / "colab_data/逐条表演复核.csv"
        if csv_path.is_file():
            with csv_path.open(encoding="utf-8-sig", newline="") as file:
                for row in csv.DictReader(file):
                    ref_id = row["语音编号"]
                    if ref_id in pend_set:
                        pend_set.remove(ref_id)
                        wav = DATA / "音频_WAV" / f"{ref_id}.wav"
                        try:
                            duration = info(wav).duration
                        except (RuntimeError, OSError):
                            continue
                        if not 2.5 <= duration <= 10.0 or not row["日文"].strip():
                            continue
                        notes[ref_id] = {
                            "id": ref_id,
                            "line": row["日文"],
                            "historical_note": row["情绪依据"],
                            "historical_label": row["整句多标签"],
                        }
                        if not pend_set:
                            break
    return [notes[ref_id] for ref_id in ref_ids if ref_id in notes]


def build_candidate_pool_for_api(text: str, plan: list[dict], ref_ids=None, max_pool: int = 24) -> list[dict]:
    """Pre-shortlist top multi-dimensional pace/emotion/pitch matches so DeepSeek sees rich diverse ground truth."""
    allowed = list(ref_ids or API_PERFORMANCE_REFS)
    if len(allowed) <= max_pool:
        return _reference_notes(allowed)
    # 1. Top rule-matched anchors for this sentence (covers target pace & emotion)
    top_ids, _ = pick_performance_refs(text, plan, min(12, max_pool // 2))
    chosen = list(dict.fromkeys(cid for cid in top_ids if cid in allowed))
    allowed_set = set(allowed)
    # 2. Add diverse high/slow/soft representatives from ref_profiles covering emotions in plan
    plan_emos = {sp.get("primary") for sp in plan if sp.get("primary")} | {sp.get("secondary") for sp in plan if sp.get("secondary")}
    for prof in _REF_PROFILES_LIST:
        rid = prof["id"]
        if rid in allowed_set and rid not in chosen and (prof["emotion"] in plan_emos or prof.get("is_base")):
            chosen.append(rid)
            if len(chosen) >= max_pool:
                break
    for rid in allowed:
        if len(chosen) >= max_pool:
            break
        if rid not in chosen:
            chosen.append(rid)
    return _reference_notes(chosen[:max_pool])


def _chinese_transcript(ja_text, key, base_url, model):
    key = _director_key(key, base_url)
    if not key:
        return "未配置翻译 API，日文台词见下方。"
    try:
        result = deepseek_json(
            "把日语台词译成自然中文，保留原意和语气。只输出 JSON，zh_text 为中文译文，不添加剧情或说明。",
            json.dumps({"ja_text": ja_text}, ensure_ascii=False),
            api_key=key, base_url=base_url, model=model,
        )
        text = result["zh_text"].strip()
        if text:
            return text
    except (OSError, ValueError, KeyError, AttributeError, TypeError, RuntimeError):
        pass
    return "未取得中文译文，日文台词见下方。"


_BED_LABELS = {"rain": "窗外雨声"}
_FOLEY_LABELS = {"rustle": "衣料摩擦靠近"}
_PAN_LABELS = {"center": "正中贴耳", "left": "左耳贴耳", "right": "右耳贴耳", "alternate": "按句左右耳交替"}


def _resolve_director_controls(
    plan,
    whisper_mode: bool,
    declick_deess: bool,
    local_delivery: bool,
    director_auto_fx: bool,
    ambient_bed: str = "none",
    foley_mode: str = "none",
    ear_pan: str = "auto",
    text: str = "",
    context: str = "",
):
    """Let DeepSeek / rule planner auto-enable whisper, de-click/de-ess, local tremble/pitch, and soundscape."""
    notes = []
    bed_out = (ambient_bed or "none").strip().lower()
    foley_out = (foley_mode or "none").strip().lower()
    pan_out = (ear_pan or "off").strip().lower()
    if not director_auto_fx or not plan:
        return (
            whisper_mode,
            declick_deess,
            local_delivery,
            "none" if bed_out == "auto" else bed_out,
            "none" if foley_out == "auto" else foley_out,
            "off" if pan_out == "auto" else pan_out,
            notes,
        )
    explicit_whisper = any(bool(sp.get("director_whisper_mode")) for sp in plan)
    all_soft_whisper = all(
        sp.get("delivery") == "soft"
        and float(sp.get("intensity", 0.5)) <= 0.55
        and (sp.get("primary") or "neutral") in ("tender", "shy", "neutral")
        for sp in plan
    )
    if not whisper_mode and (explicit_whisper or all_soft_whisper):
        whisper_mode = True
        notes.append("贴耳轻语模式")
    explicit_deess = [sp.get("director_declick_deess") for sp in plan if "director_declick_deess" in sp]
    heavy_breath_text = sum((text or "").count(k) for k in ("はぁ", "ふぅ", "んっ", "あっ", "はっ")) >= 3
    if not declick_deess:
        if any(bool(v) for v in explicit_deess):
            declick_deess = True
            notes.append("轻度去刺耳(导演)")
        elif (not explicit_deess or explicit_deess[0] is not False) and (whisper_mode or heavy_breath_text):
            declick_deess = True
            notes.append("轻度去刺耳")
    has_local_fx = any(
        float(sp.get("tremble_strength") or 0.0) > 0.0
        or bool(sp.get("delivery_text"))
        or (not whisper_mode and (abs(float(sp.get("pitch_shift") or 0.0)) >= 0.05 or abs(float(sp.get("energy_shift") or 0.0)) >= 0.05))
        for sp in plan
    )
    if not local_delivery and has_local_fx:
        local_delivery = True
        notes.append("局部音高/响度/颤音")

    combined_txt = f"{text} {context}"
    if pan_out == "auto":
        if "左耳" in combined_txt or "左边" in combined_txt:
            pan_out = "left"
        elif "右耳" in combined_txt or "右边" in combined_txt:
            pan_out = "right"
        elif whisper_mode:
            pan_out = "center"
        else:
            pan_out = "off"
        if pan_out in _PAN_LABELS:
            notes.append(f"声像({_PAN_LABELS[pan_out]})")

    if bed_out == "auto":
        dir_bed = next((sp.get("director_ambient_bed") for sp in plan if sp.get("director_ambient_bed") == "rain"), None)
        if dir_bed:
            bed_out = dir_bed
        elif "雨" in combined_txt:
            bed_out = "rain"
        else:
            bed_out = "none"
        if bed_out in _BED_LABELS:
            notes.append(f"环境音({_BED_LABELS[bed_out]})")

    if foley_out == "auto":
        foley_out = "none"

    return whisper_mode, declick_deess, local_delivery, bed_out, "none", pan_out, notes


def _resolve_ambient_level(plan, bed_out: str, ambient_level: float, director_auto_fx: bool) -> tuple[float, str | None]:
    lvl = float(ambient_level)
    if director_auto_fx and bed_out == "rain" and abs(lvl - 0.18) < 1e-4 and plan:
        dir_lvl = next(
            (sp.get("director_ambient_level") for sp in plan if isinstance(sp.get("director_ambient_level"), (int, float))),
            None,
        )
        if dir_lvl is not None:
            resolved = round(max(0.06, min(0.35, float(dir_lvl))), 2)
            return resolved, f"雨声比例({resolved:.2f})"
    return lvl, None


def synthesize_action(
    input_mode: str,
    line: str,
    context: str,
    provider_name: str,
    api_key: str,
    base_url: str,
    model_name: str,
    plan_mode: str,
    manual_primary: str,
    manual_secondary: str,
    manual_intensity: float,
    emo_scale_mode: str,
    emo_scale_val: float,
    ref_mode: str,
    temperature: float,
    top_k: int,
    custom_plan_json: str,
    locked_segments: list[dict],
    local_delivery: bool = False,
    candidate_count: int = 2,
    api_reference_ranking: bool = True,
    whisper_mode: bool = False,
    declick_deess: bool = False,
    director_auto_fx: bool = True,
    ambient_bed: str = "none",
    foley_mode: str = "none",
    ambient_level: float = 0.18,
    ear_pan: str = "auto",
    long_monologue_mode: bool = True,
    speed_factor: float = 1.0,
    discussion_history=None,
    chinese_transcript="",
    conversation_id=None,
    progress=None,
    progress_cb=None,
):
    def _prog(val: float, desc: str):
        if progress_cb is not None:
            try:
                progress_cb(val, desc)
            except Exception:
                pass
        if progress is not None:
            try:
                progress(val, desc=desc)
            except Exception:
                pass

    _prog(0.03, "📝 正在准备台词与导演配置…")
    line = " ".join((line or "").splitlines()).strip()
    context = (context or "").strip()
    burl = (base_url or "").strip()
    mdl = (model_name or "").strip()
    key = _director_key(api_key, burl)

    from_discussion = discussion_history is not None
    if from_discussion:
        line, context, chinese_transcript = _finish_discussion(
            input_mode, line, context, discussion_history, key, burl, mdl,
        )
        input_mode = "日语"
        plan_mode = "自动导演 (API 大模型)"
        ref_mode = AUTO_PERFORMANCE
        api_reference_ranking = True

    effective_context = context
    locked = list(locked_segments or [])
    if plan_mode in ("使用下方锁定标签片段", "使用上方锁定标签片段"):
        if not locked:
            raise gr.Error("还没有锁定任何片段：请在「按回车锁定标签片段」框里输入台词并按回车添加。")
        ja_parts = []
        built_plan = []
        cursor = 0
        full_raw = "".join(s["text"] for s in locked)
        needs_translation = _line_language(input_mode, full_raw) == "中文"
        if needs_translation:
            chinese_transcript = full_raw
        for idx, seg in enumerate(locked):
            raw_part = seg["text"]
            if needs_translation:
                seg_ctx = (
                    f"整句完整意思：{full_raw}\n"
                    f"当前是第{idx + 1}/{len(locked)}段，情绪提示：{seg['primary']}"
                    + (f"+{seg['secondary']}" if seg.get("secondary") else "")
                    + (f"\n附加情景：{context}" if context else "")
                )
                ja_part = translate_zh_to_sora_ja(raw_part, seg_ctx, key, burl, mdl)
            else:
                ja_part = raw_part
            ja_parts.append(ja_part)
            start = cursor
            end = cursor + len(ja_part)
            cursor = end
            built_plan.append(
                {
                    "start": start,
                    "end": end,
                    "primary": seg["primary"],
                    "secondary": seg.get("secondary"),
                    "intensity": seg["intensity"],
                    "pause_s": 0.0 if idx == len(locked) - 1 else seg["pause_s"],
                }
            )
        ja_text = "".join(ja_parts)
        plan = refine_plan(ja_text, built_plan)
        planner_note = f"已锁定标签片段 ({len(locked)} 段，标签未参与翻译)"
    else:
        if not line:
            raise gr.Error("请输入中文或日语台词。")
        ja_text = line
        if _line_language(input_mode, line) == "中文":
            _prog(0.08, "🌐 正在将中文台词改写为穹口吻日语…")
            ja_text = translate_zh_to_sora_ja(line, context, key, burl, mdl)
            chinese_transcript = line
            effective_context = f"中文原话：{line}" + (f"\n附加情景：{context}" if context else "")

        _prog(0.16, "🎬 AI 导演正在规划分段情绪与演法…")
        plan, planner_note = _make_plan(
            ja_text,
            effective_context,
            plan_mode,
            key,
            burl,
            mdl,
            manual_primary,
            manual_secondary,
            manual_intensity,
            custom_plan_json,
        )

    if from_discussion and not planner_note.startswith("API 导演规划 ("):
        raise gr.Error(f"讨论已定稿，但 API 未能完成表演规划（{planner_note}），请检查后重试。")

    _prog(0.24, "🔊 正在加载语音合成引擎…")
    tts = get_tts()
    auto_performance = ref_mode == AUTO_PERFORMANCE
    director_controls_events = planner_note.startswith("API 导演规划 (")
    whisper_mode, declick_deess, local_delivery, resolved_bed, resolved_foley, resolved_pan, auto_fx_notes = _resolve_director_controls(
        plan, whisper_mode, declick_deess, local_delivery, director_auto_fx,
        ambient_bed=ambient_bed, foley_mode=foley_mode, ear_pan=ear_pan, text=ja_text, context=effective_context,
    )
    resolved_ambient_level, lvl_note = _resolve_ambient_level(plan, resolved_bed, ambient_level, director_auto_fx)
    if lvl_note:
        auto_fx_notes.append(lvl_note)
    synth_text, synth_plan = ja_text, plan
    if not director_controls_events and not auto_performance and (getattr(tts, "_director", None) is None or getattr(tts, "_ctrl_dim", 22) < 26):
        synth_text, synth_plan = auto_insert_tags(ja_text, plan)

    emo_scale = None if emo_scale_mode == "自动动态上限" else float(emo_scale_val)
    ref_id = "auto" if ref_mode.startswith("按情绪自动匹配") else "SR011455"
    _prim_emo = (synth_plan[0].get("primary") or "neutral") if synth_plan else "neutral"
    _emo_tag = "耳语" if whisper_mode else _EMO_SHORT.get(_prim_emo, "日常")
    _raw_label = (chinese_transcript or line or ja_text or "").strip()
    _clean_chars = "".join(re.findall(r"[\u4e00-\u9fff\u3040-\u30ffA-Za-z0-9]+", re.sub(r"<[a-z_]+>", "", _raw_label)))
    _snip = (_clean_chars[:14] or "台词")
    job_name = f"穹_{_emo_tag}_{_snip}_{time.strftime('%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    candidates = [None]
    reference_notes = ""
    api_overview = ""
    count = 1
    if auto_fx_notes:
        reference_notes += "\n导演自动启用：" + "、".join(auto_fx_notes)
    if auto_performance:
        count = int(candidate_count)
        refs = []
        delivery = None
        api_key_for_rank = key
        if api_reference_ranking and plan_mode == "自动导演 (API 大模型)":
            if api_key_for_rank:
                try:
                    _prog(0.30, "🎯 正在由导演从 63 档声学参考中选角…")
                    if is_local_ollama(burl):
                        pool = build_candidate_pool_for_api(synth_text, synth_plan, API_PERFORMANCE_REFS, max_pool=len(API_PERFORMANCE_REFS))
                    else:
                        pool = build_candidate_pool_for_api(synth_text, synth_plan, API_PERFORMANCE_REFS)
                    if len(pool) < count:
                        raise ValueError("可用参考数量不足")
                    api_overview, reasons = rank_references_with_deepseek(
                        synth_text, effective_context, synth_plan, pool,
                        count=min(count + 2, len(pool)), api_key=api_key_for_rank, base_url=burl, model=mdl,
                    )
                    refs = list(reasons)
                    reference_notes = "\nAPI 配参考（含重试备用，依据原声文本及历史标注）"
                    for ref in refs:
                        reference_notes += f"\n{ref} · {reasons[ref]}"
                except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                    api_overview = ""
                    refs = []
                    reference_notes = f"\nAPI 配参考失败，已用本地规则（{type(exc).__name__}）"
            else:
                reference_notes = "\n未配置 API，已用本地规则配参考"
        if not refs:
            refs, delivery = pick_performance_refs(synth_text, synth_plan, count + 2)
        candidates = [(candidate, delivery) for candidate in refs]

    results = []
    failed = []
    flagged = []
    checks = []
    inspect_dry = None
    if auto_performance:
        from acoustic_delivery import check_dry_take, score_dry_expressiveness

        def inspect_dry(wav, rate):
            issues, match = check_dry_take(
                wav, rate, synth_text, synth_plan, device=tts.configs.device, whisper_mode=whisper_mode,
            )
            return issues, match, score_dry_expressiveness(wav, rate, synth_plan, whisper_mode=whisper_mode)

    with _TTS_LOCK:
        total_cands = max(1, len(candidates))
        for idx_c, candidate in enumerate(candidates):
            ref_lbl = candidate[0] if candidate else ref_id
            base_p = 0.36 + 0.54 * (idx_c / total_cands)
            _prog(base_p, f"🎙️ 正在合成第 {idx_c + 1}/{min(count + 1, total_cands)} 候选（参考 {ref_lbl}）…")
            suffix = f"_{candidate[0]}" if candidate else ""
            output = OUT_DIR / f"{job_name}{suffix}.wav"
            try:
                result = render(
                    tts,
                    synth_text,
                    synth_plan,
                    output,
                    conditioned=True,
                    ref_id=ref_id,
                    emo_scale=emo_scale,
                    temperature=float(temperature),
                    top_k=int(top_k),
                    auto_tags=True,
                    auto_performance=auto_performance,
                    local_delivery=local_delivery,
                    performance_ref=candidate,
                    director_controls_events=director_controls_events,
                    whisper_mode=whisper_mode,
                    declick_deess=declick_deess,
                    ambient_bed=resolved_bed,
                    foley_mode=resolved_foley,
                    ambient_level=float(resolved_ambient_level),
                    ear_pan=resolved_pan,
                    long_monologue_mode=bool(long_monologue_mode),
                    speed=float(speed_factor),
                    inspect_dry=inspect_dry,
                )
            except (RuntimeError, OSError, ValueError) as exc:
                label = candidate[0] if candidate else ref_id
                ARCHIVE.record(conversation_id, "生成问题", f"{label} · {type(exc).__name__} · {exc}", ja=ja_text)
                if len(candidates) == 1:
                    raise
                failed.append((candidate[0], exc))
                continue
            if auto_performance:
                _prog(min(0.95, base_p + 0.18), f"🎧 正在校验第 {idx_c + 1} 候选台词匹配与情绪张力…")
                issues, match, expr = result.pop("_dry_check")
                align_m = float(match) if match is not None else 0.75
                expr_s = float(expr["expr_score"])
                comp = (0.65 * align_m + 0.35 * expr_s) if align_m >= 0.86 else (align_m * 0.65 + expr_s * 0.15)
                result["_align_match"] = float(match) if match is not None else 0.0
                result["_expr_score"] = expr_s
                result["_composite_score"] = round(comp, 4)
                detail = "、".join(issues) if issues else "通过可用检查"
                if match is not None:
                    detail += f"（台词匹配 {match:.1%} · 情绪张力 {expr_s:.0%} · F0 {expr['f0_med']:.0f}Hz）"
                elif not issues:
                    detail += f"（未取得对齐分 · 情绪张力 {expr_s:.0%} · F0 {expr['f0_med']:.0f}Hz）"
                checks.append(f"{candidate[0]} · {detail}")
                if issues:
                    ARCHIVE.record(conversation_id, "成品检查异常", detail, audio=output, ja=ja_text)
                    flagged.append((output, result))
                    continue
            results.append((output, result))
            best_comp = max((t[1].get("_composite_score", 0.0) for t in results), default=0.0)
            if len(results) >= count and (not auto_performance or best_comp >= 0.94 or len(results) >= count + 1):
                break

    if not results and flagged:
        results = flagged[:count]
        checks.append("未找到通过检查的版本，已保留产出供人工试听。")

    if not results:
        raise gr.Error(f"所有整句候选生成失败：{failed[0][1]}")

    if auto_performance and len(results) > 1:
        # Promote best combined phoneme accuracy + emotional expressiveness (while respecting local_delivery alignment).
        results.sort(
            key=lambda take: (
                1 if (local_delivery and take[1].get("local_delivery", {}).get("status") == "unaligned") else 0,
                -(take[1].get("_composite_score") or take[1].get("_align_match") or 0.0),
            )
        )
        results = results[:count]
    elif local_delivery and len(results) > 1 and not api_overview:
        results.sort(key=lambda take: take[1].get("local_delivery", {}).get("status") == "unaligned")

    out_wav, info = results[0]
    clean_main = OUT_DIR / f"{job_name}_主选.wav"
    if out_wav != clean_main and out_wav.is_file():
        out_wav.replace(clean_main)
        out_wav = clean_main
    alternates = []
    for alt_idx, (alt_p, _) in enumerate(results[1:3], 1):
        clean_alt = OUT_DIR / f"{job_name}_备选{alt_idx}.wav"
        if alt_p != clean_alt and alt_p.is_file():
            alt_p.replace(clean_alt)
            alt_p = clean_alt
        alternates.append(str(alt_p))
    alternates.extend([None] * (2 - len(alternates)))

    plan_pretty = json.dumps(plan, ensure_ascii=False, indent=2)
    references = f"参考原声：{', '.join(info.get('refs', [ref_id]))}"
    if auto_performance:
        gpt_route = " → ".join(info.get("refs", [ref_id]))
        references = f"语义参考：{gpt_route}，SoVITS 音色锁定：{info['acoustic_ref']}"
        if api_overview:
            references += f"\nAPI 导演整句演法：{api_overview}"
        else:
            delivery = DELIVERY_LABELS.get(info["delivery"], info["delivery"])
            references += f"\n本地配参考的侧重语气：{delivery}"
    if director_controls_events or info["directed_events"]:
        events = "、".join(info["directed_events"]) or "未安排"
        label = "API 导演计划的发声事件" if director_controls_events else "计划的发声事件"
        references += f"\n{label}：{events}"
    if "director_emo_scale" in info:
        references += f"\nDeepSeek 情绪倍率：{info['director_emo_scale']:g}"
    if info.get("model_events"):
        references += "\n交给模型合成的发声标签：" + "、".join(info["model_events"])
    if info.get("monologue_batches"):
        references += f"\n长独白稳态分句：共 {info['monologue_batches']} 段接续合成"
    if info.get("silence_trimmed_s"):
        references += f"\n长静音智能压缩：已自动缩减 {info['silence_trimmed_s']:.2f}s 死静音"
    if info.get("soundscape"):
        sc = info["soundscape"]
        sc_parts = []
        if sc.get("ear_pan"):
            sc_parts.append(_PAN_LABELS.get(sc["ear_pan"], sc["ear_pan"]))
        if sc.get("bed"):
            sc_parts.append(_BED_LABELS.get(sc["bed"], sc["bed"]))
        if sc.get("foley"):
            sc_parts.append(f"{_FOLEY_LABELS.get(sc['foley'], sc['foley'])}×{sc.get('foley_count', 1)}")
        if sc_parts:
            references += "\n48kHz 立体声音景：" + " + ".join(sc_parts)
    if local_delivery and info.get("local_delivery"):
        acoustic = info["local_delivery"]
        if acoustic["status"] == "applied":
            applied = f"{len(acoustic['regions'])}/{acoustic['requested']} 段已应用"
            if any(r["pitch_st"] for r in acoustic["regions"]) and not acoustic["pitch_applied"]:
                applied += "，未检测到可调整的有声音高"
            if acoustic["tremble_applied"]:
                words = "、".join(r["text"] for r in acoustic["regions"] if r["tremble_applied"])
                applied += f"（颤声：{words}）"
        elif acoustic["status"] == "unaligned":
            applied = "发音位置未可靠对齐，已保留合成原声"
        else:
            applied = "这句无需额外调整"
        if acoustic.get("loudness_adapter"):
            applied += "（响度适配器第 15 轮）"
        references += f"\n局部音高、响度与颤声：{applied}"
    if checks:
        references += "\n成品检查：\n" + "\n".join(checks)
    if auto_performance:
        trial_refs = "、".join(result["refs"][0] for _, result in results)
        references += f"\n整句试音：{trial_refs}（主选在前）"
        if failed:
            references += "\n生成失败：" + "、".join(ref for ref, _ in failed)
    discussion_note = "已按导演讨论定稿 · " if from_discussion else ""
    status = (
        f"规划来源：{discussion_note}{planner_note}\n"
        f"{references}{reference_notes}\n"
        f"音频时长：{info['seconds']}s ({info['tokens']} tokens)\n"
        f"保存路径：{out_wav}"
    )
    chinese_transcript = chinese_transcript or _chinese_transcript(ja_text, key, burl, mdl)
    ARCHIVE.record(conversation_id, "主选音频", f"{status}\n\n情绪规划\n{plan_pretty}",
                   audio=out_wav, ja=ja_text, zh=chinese_transcript)
    for alternate in filter(None, alternates):
        ARCHIVE.record(conversation_id, "备选音频", Path(alternate).name, audio=alternate, ja=ja_text, zh=chinese_transcript)
    return str(out_wav), ja_text, plan_pretty, status, *alternates, chinese_transcript


def iterate_to_goal(
    goal, judge_key, judge_model, judge_provider, judge_url, max_rounds, current_take, current_ja, current_zh, history,
    input_mode, line, context, provider, key, url, model, _plan_mode, primary, secondary,
    intensity, scale_mode, scale, _ref_mode, temperature, top_k, _plan_json, _locked,
    local_delivery, candidate_count, _api_reference_ranking, whisper_mode=False, declick_deess=False, director_auto_fx=True, ambient_bed="none", foley_mode="none", ambient_level=0.18, ear_pan="auto", long_monologue_mode=True, speed_factor=1.0, *, conversation_id=None,
):
    from audio_goal import review_takes

    goal = (goal or "").strip()
    judge_url = (judge_url or JUDGE_PRESETS[judge_provider][0]).strip().rstrip("/")
    env_names = {GEMINI_PROVIDER: "GEMINI_API_KEY", OPENAI_PROVIDER: "OPENAI_API_KEY", ALIYUN_PROVIDER: "DASHSCOPE_API_KEY"}
    env_key = os.environ.get(env_names.get(judge_provider, ""), "") if judge_url == JUDGE_PRESETS[judge_provider][0] else ""
    saved_judge = _load_config().get("audio_judge") or {}
    saved_key = saved_judge.get("api_key", "") if saved_judge.get("provider") == judge_provider and _same_endpoint(judge_url, saved_judge.get("base_url")) else ""
    judge_key = (judge_key or saved_key or env_key).strip()
    key = _director_key(key, url)
    judge_model = (judge_model or "").strip()
    if not goal:
        raise gr.Error("请填写能通过试听判断的目标。")
    if not judge_key or not judge_model or not judge_url:
        raise gr.Error("请在底部「API 设置」中配置音频评审的厂商、API 地址、Key 和模型。")
    if not key:
        raise gr.Error("请先配置导演 API 密钥。")
    max_rounds = int(max_rounds)
    if not 1 <= max_rounds <= 8:
        raise gr.Error("生成轮数必须在 1 到 8 之间。")

    task = {"role": "user", "content": f"按以下试听目标制作语音：{goal}"}
    history = [*(history or []), task]
    base_context = (context or "").strip()
    if current_take:
        # Gradio supplies a cached copy; the generated filename identifies the original.
        baseline = (OUT_DIR / Path(current_take).name).resolve()
        if not baseline.is_relative_to(OUT_DIR.resolve()) or not baseline.is_file() or baseline.suffix != ".wav":
            raise gr.Error("当前试听文件不可用，请重新生成后再做对比。")
        ja_text = (current_ja or "").strip()
        if not ja_text:
            raise gr.Error("当前试听缺少对应日语台词。")
        baseline = str(baseline)
        chinese_transcript = (current_zh or "").strip()
    else:
        baseline = None
        ja_text, base_context, chinese_transcript = _finish_discussion(
            input_mode, (line or "").strip(), base_context, history, key, url, model,
        )

    chinese_transcript = chinese_transcript or _chinese_transcript(ja_text, key, url, model)
    starting_round = 0 if baseline else 1
    notes = []
    best = None
    fallback = None
    feedback = ""

    def output(take, status):
        if take is None:
            return *(gr.skip() for _ in range(7)), baseline, status
        return *take, baseline, status

    for round_no in range(starting_round, max_rounds + 1):
        label = "当前成品" if round_no == 0 else f"第 {round_no}/{max_rounds} 轮"
        yield output(best[1] if best else fallback, "\n".join([*notes, f"{label}：正在生成并试听评审…"]))
        try:
            if round_no == 0:
                result = (baseline, ja_text, gr.skip(), gr.skip(), None, None, chinese_transcript)
            else:
                direction = "\n".join(filter(None, (base_context, f"本次试听目标：{goal}", feedback)))
                result = synthesize_action(
                    "日语", ja_text, direction, provider, key, url, model, "自动导演 (API 大模型)",
                    primary, secondary, intensity, scale_mode, scale, AUTO_PERFORMANCE,
                    temperature, top_k, "", [], local_delivery, candidate_count, True,
                    whisper_mode, declick_deess, director_auto_fx,
                    ambient_bed, foley_mode, ambient_level, ear_pan, long_monologue_mode, speed_factor,
                    chinese_transcript=chinese_transcript, conversation_id=conversation_id,
                )
                if not result[3].startswith("规划来源：API 导演规划"):
                    raise RuntimeError("导演未返回 API 表演规划")
            paths = [path for path in (result[0], *result[4:6]) if path]
            fallback = result
            reviews = review_takes(paths, ja_text, goal, api_key=judge_key, model=judge_model,
                                   provider=judge_provider, base_url=judge_url)
        except Exception as exc:
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    err_body = exc.read().decode("utf-8", errors="ignore").strip()
                except Exception:
                    err_body = ""
                detail = f"HTTP {exc.code}" + (f": {err_body[:160]}" if err_body else "")
            else:
                detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            message = f"{label}中断（{detail}），已保留可试听版本。"
            ARCHIVE.record(conversation_id, "目标评审问题", message, ja=ja_text)
            yield output(best[1] if best else fallback, "\n".join([*notes, message]))
            return

        ordered = sorted(range(len(paths)), key=lambda i: (reviews[i][1], reviews[i][0]), reverse=True)
        index = ordered[0]
        score, met, suggestion = reviews[index]
        alternates = ([paths[i] for i in ordered[1:]] + [None, None])[:2]
        take_status = result[3]
        if index != 0:
            take_status += f"\n目标评审选中的试听文件：{paths[index]}"
        selected = (paths[index], ja_text, result[2], take_status, *alternates, chinese_transcript)
        if not best or (met, score) > best[0]:
            best = ((met, score), selected)
        if baseline is None:
            baseline = result[0]
        notes.append(f"{label} · {score:g}/100 · {'达到目标' if met else '待改进'} · {suggestion}")
        ARCHIVE.record(conversation_id, "目标评审", notes[-1], ja=ja_text, zh=chinese_transcript)
        if met:
            yield output(best[1], "\n".join([*notes, "已达到评审目标，迭代结束。"]))
            return
        feedback = (
            f"上一轮评分最高的录音为 {Path(paths[index]).name}。试听反馈：{suggestion}。"
            "请保持日语台词不变，调整演法及参考选择以解决指出的问题。"
        )

    yield output(best[1], "\n".join([*notes, "已到轮次上限，试听区保留评分最高的一版。"]))


def _ollama_models(current_model):
    try:
        with urllib.request.urlopen(f"{OLLAMA_DIRECTOR_URL}/models", timeout=2) as response:
            models = sorted({item["id"] for item in json.load(response)["data"]})
    except (OSError, ValueError, KeyError, TypeError):
        choices = [current_model] if current_model else []
        return gr.update(choices=choices, value=current_model), "未连到 Ollama。请打开 Ollama 后重新检测。"
    if current_model and current_model not in models:
        return gr.update(choices=[current_model, *models], value=current_model), "未检测到已保存的模型，请下载该模型或从列表另选。"
    if not models:
        return gr.update(choices=[], value=None), "Ollama 已连接。请先在 Ollama 下载一个本地模型，再重新检测。"
    return gr.update(choices=models, value=current_model or models[0]), f"Ollama 已连接，找到 {len(models)} 个模型。"


def _director_fields(provider, url, model):
    local = provider == OLLAMA_PROVIDER
    choices = [model] if model else []
    model_update, status = _ollama_models(model) if local else (gr.update(choices=choices, value=model), "")
    return (gr.update(value=url, interactive=not local), model_update,
            gr.update(value="", visible=not local), gr.update(visible=local),
            gr.update(value=status, visible=local))


def _saved_director_values(cfg):
    provider = cfg.get("provider")
    if provider == OLLAMA_PROVIDER:
        return provider, OLLAMA_DIRECTOR_URL, cfg.get("model") or None
    if provider in PROVIDER_PRESETS:
        default_url, default_model = PROVIDER_PRESETS[provider]
        return provider, cfg.get("base_url") or default_url, cfg.get("model") or default_model
    default_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
    default_model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip()
    return "DeepSeek 官方", default_url, default_model


def _restore_director_config():
    provider, url, model = _saved_director_values(_load_config())
    return provider, *_director_fields(provider, url, model)


def on_provider_change(provider_name: str):
    url, model = PROVIDER_PRESETS.get(provider_name, ("", ""))
    return _director_fields(provider_name, url, model or None)


def _refresh_ollama_models(provider_name, current_model):
    if provider_name != OLLAMA_PROVIDER:
        return gr.skip(), gr.skip()
    return _ollama_models(current_model)


def _archive_menu(active):
    choices = ARCHIVE.list_chats()
    if active and not any(chat_id == active for _, chat_id in choices):
        choices.insert(0, ("新对话", active))
    return gr.update(choices=choices, value=active)


def _save_director_history(chat_id, turns):
    safe_turns = []
    for turn in turns:
        content = turn.get("content")
        if isinstance(content, (tuple, list)) and content:
            try:
                content = [str(_uploaded_attachment(content[0]))]
            except gr.Error:
                content = "附件已不可用，请重新上传。"
        safe_turns.append({"role": turn.get("role"), "content": content})
    return ARCHIVE.save_chat(chat_id, safe_turns)


def send_director_message(callback, chat_id, draft, *args):
    clear_composer = bool(draft and ((draft.get("text") or "").strip() or draft.get("files")))
    for output in archive_conversation(callback, chat_id, draft, *args):
        yield *output, None if clear_composer else gr.skip()
        clear_composer = False


def archive_conversation(callback, chat_id, *args):
    try:
        for turns, *outputs in callback(*args, conversation_id=chat_id):
            if turns is gr.skip() or not isinstance(turns, list):
                yield gr.skip(), *outputs, gr.skip()
            else:
                saved = _save_director_history(chat_id, turns)
                yield saved, *outputs, _archive_menu(chat_id)
    except Exception as exc:
        detail = str(exc) if isinstance(exc, gr.Error) else type(exc).__name__
        ARCHIVE.record(chat_id, "对话或生成问题", detail)
        draft, history = args[:2]
        draft = draft or {}
        text = (draft.get("text") or "").strip()
        turns = list(history or [])
        paths = draft.get("files") or []
        turns.extend({"role": "user", "content": (path,)} for path in paths)
        if text:
            turns.append({"role": "user", "content": text})
        if text or paths:
            turns.append({"role": "assistant", "content": detail})
        saved = _save_director_history(chat_id, turns)
        yield saved, *(gr.skip() for _ in range(9)), gr.update(value="", visible=False), _archive_menu(chat_id)


def archived_synthesis(chat_id, *settings):
    try:
        for prog_upd, out in _run_synth_with_progress(*settings, conversation_id=chat_id):
            if out is None:
                yield *(gr.skip() for _ in range(7)), prog_upd
            else:
                yield *out, prog_upd
    except Exception as exc:
        detail = _generation_error_detail(exc)
        yield *(gr.skip() for _ in range(7)), gr.update(value=_render_generation_error(detail), visible=True)
        ARCHIVE.record(chat_id, "生成问题", detail)


def _library_choices(scope, conversation_id):
    by_chat = conversation_id if scope == "当前对话" else None
    main_only = scope == "⭐ 主选成品"
    audio_only = main_only or scope in ("全部音频（含备选）", "全部音频", "当前对话")
    choices = ARCHIVE.list_events(audio_only=audio_only, main_only=main_only, chat_id=by_chat)
    table_rows = ARCHIVE.list_table_rows(audio_only=audio_only, main_only=main_only, chat_id=by_chat)
    first_val = choices[0][1] if choices else None
    first_audio, first_detail = _library_entry(first_val) if first_val else (None, "")
    return gr.update(choices=choices[:250], value=first_val), table_rows[:120], first_audio, first_detail


def _library_entry(value):
    if not value:
        return None, ""
    if str(value).startswith("wav:"):
        name = str(value).removeprefix("wav:")
        path = (OUT_DIR / Path(name).name).resolve()
        detail = f"本地音频 · {path.name}"
    else:
        row = ARCHIVE.event(str(value))
        if not row:
            return None, "记录不可用。"
        path = Path(row["audio"]).resolve() if row["audio"] else None
        fname = path.name if path else "无音频文件"
        detail = "\n".join(filter(None, [
            f"📄 文件名：{fname}",
            f"🕒 时间：{row['created'][0:19].replace('T', ' ')} · {row['kind']}",
            f"💬 对话：{row['title']}" if row["title"] else "",
            f"🇨🇳 中文台词：{row['zh']}" if row["zh"] else "",
            f"🇯🇵 日文台词：{row['ja']}" if row["ja"] else "",
            "────────────────────────",
            row["detail"],
        ]))
    if path and (not path.is_relative_to(OUT_DIR.resolve()) or path.suffix.lower() != ".wav" or not path.is_file()):
        return None, detail + "\n音频文件不可用。"
    return str(path) if path else None, detail


def _on_library_table_select(evt: gr.SelectData, table_rows):
    if evt is None or not table_rows:
        return gr.skip(), gr.skip(), gr.skip()
    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else int(evt.index)
    if 0 <= row_idx < len(table_rows):
        rec_id = str(table_rows[row_idx][0])
        audio_p, detail_txt = _library_entry(rec_id)
        return rec_id, audio_p, detail_txt
    return gr.skip(), gr.skip(), gr.skip()


def _feedback_action(is_positive: bool, tags, note, audio, ja, zh, status_txt, chat_id):
    if not audio and not (ja or "").strip():
        return "⚠️ 请先生成一条语音再标注。", gr.skip()
    state = record_user_feedback(
        ARCHIVE,
        is_positive=is_positive,
        tags=tags or [],
        note=note or "",
        audio=audio,
        ja=ja or "",
        zh=zh or "",
        chat_id=chat_id,
        status_text=status_txt or "",
    )
    msg = "✅ 已计入满意正样本并提升对应参考权重。" if is_positive else f"🏷️ 已写入错题库（{'、'.join(tags or ['通用异常'])}）并更新惩罚权重。"
    return msg, format_self_opt_report(state)


def _recompute_self_opt_action():
    state = analyze_archive_and_update_weights(ARCHIVE.directory / "director.sqlite3")
    state.pop("_pos_samples", None)
    state.pop("_neg_samples", None)
    return format_self_opt_report(state)


def _run_self_opt_finetune_action():
    _prog(0.24, "🔊 正在加载语音合成引擎…")
    tts = get_tts()
    with _TTS_LOCK:
        state = finetune_internal_adapters(tts, ARCHIVE.directory / "director.sqlite3", steps=24, lr=1.5e-4)
    if not state.get("_updated_steps"):
        return "没有产生有效的权重更新。\n\n" + format_self_opt_report(state)
    return "✅ 已完成一轮错题对比与内部权重微调（ProsodyDirector + PerformanceAdapter）！\n\n" + format_self_opt_report(state)


def _reset_self_opt_action():
    _prog(0.24, "🔊 正在加载语音合成引擎…")
    tts = get_tts()
    with _TTS_LOCK:
        state = rollback_self_optimization(tts)
    return "↩️ 已恢复出厂默认内部权重并清空自适应惩罚。\n\n" + format_self_opt_report(state)


def _lib_load_to_editor(entry_id):
    if not entry_id or str(entry_id).startswith("wav:"):
        return gr.skip(), "⚠️ 该早期音频无台词文本记录。"
    row = ARCHIVE.event(str(entry_id))
    if not row:
        return gr.skip(), "⚠️ 记录不存在。"
    txt = (row["zh"] or "").strip() or (row["ja"] or "").strip()
    return txt, f"📥 已载入记录 #{entry_id} 的台词到上方编辑器。"


def _lib_mark_good(entry_id, chat_id):
    if not entry_id or str(entry_id).startswith("wav:"):
        return "⚠️ 请先选择一条历史记录。", gr.skip()
    row = ARCHIVE.event(str(entry_id))
    if not row:
        return "⚠️ 记录不存在。", gr.skip()
    state = record_user_feedback(
        ARCHIVE,
        is_positive=True,
        tags=[],
        note=f"来自音频库记录 #{entry_id}",
        audio=row["audio"],
        ja=row["ja"],
        zh=row["zh"],
        chat_id=chat_id,
        status_text=row["detail"],
    )
    return f"✅ 记录 #{entry_id} 已计入正样本并刷新自优化权重。", format_self_opt_report(state)


def _lib_delete_record(entry_id):
    if not entry_id or str(entry_id).startswith("wav:"):
        return "⚠️ 请先选择一条有效记录。"
    ARCHIVE.delete_event(str(entry_id))
    return f"🗑️ 已删除历史记录 #{entry_id}。"


_STYLE_PRESETS = {
    "daily": ("tender", "neutral", 0.45, 0.35, False, False, False, "off", "none", "🌿 已切换「日常自然」：平缓日常对话，标准距离无环境底噪。"),
    "clingy": ("shy", "tender", 0.62, 0.35, False, False, False, "center", "none", "🌸 已切换「撒娇粘人」：羞涩温柔双叠＋正中近距人头麦包覆。"),
    "tsundere": ("impatient", "anger", 0.65, 0.30, False, False, False, "off", "none", "😒 已切换「冷淡不耐」：不耐烦＋微怒压声，干脆无耳语混响。"),
    "asmr_sleep": ("tender", "shy", 0.42, 0.50, True, True, False, "left", "none", "🌙 已切换「近耳哄睡 ASMR」：左耳贴耳轻语 ep5＋自动去齿音（纯净近耳人声，无背景噪音）。"),
}


def _apply_style_preset(preset_key: str):
    vals = _STYLE_PRESETS[preset_key]
    return vals


def build_ui() -> gr.Blocks:
    cfg = _load_config()
    default_provider, default_url, default_mdl = _saved_director_values(cfg)
    judge_cfg = cfg.get("audio_judge") or {}
    judge_provider = judge_cfg.get("provider")
    if judge_provider not in JUDGE_PRESETS:
        judge_provider = GEMINI_PROVIDER
    judge_default_url, judge_default_model = JUDGE_PRESETS[judge_provider]
    page_head = """
        <meta name="google" content="notranslate">
        <script>
            document.documentElement.lang = "zh-CN";
            document.documentElement.setAttribute("translate", "no");
            document.documentElement.classList.add("notranslate");
        </script>
    """
    with gr.Blocks(title="春日野穹 · 表演向语音合成", theme=UI_THEME, css=UI_CSS, head=page_head) as demo:
        gr.HTML(UI_HERO)
        with gr.Row(equal_height=False):
            # ───────────── 左侧：台词剧本 + 苍穹声场回放 ─────────────
            with gr.Column(scale=7, min_width=480):
                with gr.Group(elem_classes="sora-card"):
                    _chibi_write_img = (
                        f"<img src='{_CHIBI_WRITE_URI}' class='sora-chibi-float' alt='' />"
                        if _CHIBI_WRITE_URI
                        else ""
                    )
                    gr.HTML(f"<div class='sora-section sora-section-chibi'>✦ 台词脚本 <span class='sec-sub'>SCRIPT</span>{_chibi_write_img}</div>")
                    line_box = gr.Textbox(
                        label="台词（中文或日语）", show_label=False,
                        placeholder="输入台词，例如：你怎么才回来啊…… ／ 別に、待ってたわけじゃないよ。",
                        lines=4, max_lines=12,
                    )
                    ctx_box = gr.Textbox(
                        label="剧情 / 语气上下文（可选，帮导演判断潜台词）",
                        placeholder="例如：等了很久终于等到对方回来，嘴上先责怪，后半句松一口气。",
                        lines=2, max_lines=5,
                    )
                    with gr.Row(elem_classes="sora-inline-bar"):
                        input_mode = gr.Radio(
                            choices=["自动识别", "中文", "日语"], value="自动识别",
                            label="台词语言",
                            info="有假名按日语合成；纯汉字日语请选「日语」。",
                            scale=3,
                        )
                        long_monologue_mode = gr.Checkbox(
                            label="📜 长独白分句稳态接续（>75字自动拆批）",
                            value=True,
                            scale=2,
                        )
                gen_btn = gr.Button("🎀 唤起穹的声音 · 按左栏台词生成", variant="primary", size="lg", elem_id="sora-gen-btn")
                synth_progress = gr.HTML(visible=False)

                with gr.Group(elem_classes="sora-card"):
                    _chibi_img = (
                        f"<img src='{_CHIBI_STICKER_URI}' class='sora-chibi-float' alt='' />"
                        if _CHIBI_STICKER_URI
                        else ""
                    )
                    gr.HTML(f"<div class='sora-section sora-section-chibi'>✦ 声场试听 <span class='sec-sub'>PLAYBACK</span>{_chibi_img}</div>")
                    audio_out = gr.Audio(
                        label="生成结果试听", type="filepath", interactive=False,
                        autoplay=True, elem_classes="sora-audio-box",
                    )
                    with gr.Row():
                        generated_zh = gr.Textbox(label="双语字幕 · 中文", lines=2, interactive=False)
                        generated_ja = gr.Textbox(label="双语字幕 · 日文", lines=2, interactive=False)
                    status_out = gr.Textbox(label="声场与运行详情", lines=4, max_lines=12, interactive=False)
                    with gr.Accordion("🎚️ 整句备选试听（展开对比备选与起始版本）", open=False, elem_classes="sora-acc"):
                        with gr.Row():
                            alternate_one = gr.Audio(label="备选 1", type="filepath", interactive=False, elem_classes="sora-audio-box")
                            alternate_two = gr.Audio(label="备选 2", type="filepath", interactive=False, elem_classes="sora-audio-box")
                            goal_baseline = gr.Audio(label="起始版本对比", type="filepath", interactive=False, visible=False, elem_classes="sora-audio-box")
                    with gr.Accordion("🏷️ 标记效果与错题反馈（用于模型自优化升级）", open=False, elem_classes="sora-acc"):
                        feedback_tags = gr.CheckboxGroup(
                            choices=ERROR_TAG_CHOICES,
                            label="如有问题，请勾选错误标签",
                        )
                        feedback_note = gr.Textbox(
                            label="补充说明（可选）",
                            placeholder="例如：后半句停顿太长 / 尾音有轻微嘶声",
                            lines=1,
                        )
                        with gr.Row():
                            fb_good_btn = gr.Button("👍 这条不错（正样本）", size="sm", elem_classes="sora-fb-good")
                            fb_bad_btn = gr.Button("🏷️ 记入错题并降权", size="sm", variant="stop", elem_classes="sora-fb-bad")
                        feedback_status = gr.Textbox(label="反馈状态", lines=1, interactive=False)

            # ───────────── 右侧：导演与调音控制台 ─────────────
            with gr.Column(scale=5, min_width=390):
                with gr.Group(elem_classes="sora-studio-panel"):
                    with gr.Tabs(elem_classes="sora-tabs"):
                        with gr.Tab("✦ 导演对话"):
                            saved_chats = ARCHIVE.list_chats()
                            active_chat = saved_chats[0][1] if saved_chats else ARCHIVE.new_chat()
                            conversation_id = gr.State(active_chat)
                            with gr.Row(equal_height=True):
                                conversation_menu = gr.Dropdown(
                                    choices=saved_chats or [("新对话", active_chat)], value=active_chat,
                                    label="聊天历史", show_label=False, interactive=True, scale=4, container=False,
                                )
                                new_conversation = gr.Button("＋ 新对话", size="sm", scale=1, min_width=90)
                            director_chat = gr.Chatbot(
                                label="导演对话", show_label=False, type="messages", height=400,
                                value=ARCHIVE.load_chat(active_chat), elem_classes="sora-chat",
                            )
                            director_message = gr.MultimodalTextbox(
                                label="给导演的话", show_label=False, lines=2, file_types=sorted(ATTACHMENT_TYPES),
                                file_count="multiple", submit_btn=False,
                                placeholder="例如：给我做一段雨夜重逢的语音，剧情和日语台词你来定。（Enter 发送）",
                            )
                            with gr.Row(equal_height=True):
                                director_execution = gr.Dropdown(
                                    choices=[("🔒 审批后生成", "审批后生成"), ("⚡ 全自动", "全自动")],
                                    value="审批后生成", label="导演执行方式", show_label=False,
                                    container=False, allow_custom_value=False, scale=2, min_width=130,
                                )
                                director_goal_mode = gr.Checkbox(label="🎯 目标模式", container=False, scale=1, min_width=100)
                            with gr.Group(visible=False, elem_classes="sora-subcard") as goal_opts_group:
                                with gr.Row():
                                    goal_rounds = gr.Slider(1, 8, value=3, step=1, label="目标模式最多生成轮数")
                                    goal_use_current = gr.Checkbox(label="先评审当前成品", value=False)
                            with gr.Row():
                                director_send = gr.Button("💬 发送", size="sm")
                                director_task = gr.Button("📋 下达制作任务", size="sm")
                                director_generate = gr.Button("🎙️ 按讨论生成", variant="primary", size="sm", elem_classes="sora-art-primary")
                            goal_status = gr.Textbox(label="目标评审", lines=3, interactive=False, visible=False)

                        with gr.Tab("🎀 情绪控制"):
                            with gr.Group(elem_classes="sora-subcard"):
                                gr.HTML("<div class='sora-section'>✦ 穹式风格预设 <span class='sec-sub'>PRESETS</span></div>")
                                with gr.Row(elem_classes="sora-preset-row"):
                                    preset_daily_btn = gr.Button("🌿 日常自然", size="sm", elem_classes="sora-preset-daily")
                                    preset_clingy_btn = gr.Button("🎀 撒娇粘人", size="sm", elem_classes="sora-preset-clingy")
                                with gr.Row(elem_classes="sora-preset-row"):
                                    preset_tsun_btn = gr.Button("🖤 冷淡不耐", size="sm", elem_classes="sora-preset-tsun")
                                    preset_asmr_btn = gr.Button("🌙 近耳哄睡 ASMR", size="sm", variant="primary", elem_classes="sora-preset-asmr")
                                preset_hint = gr.Textbox(label="预设状态", lines=2, interactive=False, value="提示：点击上方预设可一键配好整套情绪与耳语/音景参数，或继续微调。")
                            plan_mode = gr.Radio(
                                choices=[
                                    ("完整导演（使用已配置模型）", "自动导演 (API 大模型)"),
                                    ("仅情绪规划（本地 Ollama）", "自动导演 (本地 Ollama)"),
                                    "手动指定整句情绪",
                                    "使用下方锁定标签片段",
                                    "使用下方分段 JSON",
                                ],
                                value="自动导演 (API 大模型)",
                                label="情绪控制方式",
                                elem_classes="sora-radio-stack",
                            )
                            with gr.Group(visible=False, elem_classes="sora-subcard") as manual_group:
                                with gr.Row():
                                    manual_primary = gr.Dropdown(
                                        choices=EMOTION_LABELS, value="tender", label="主情绪"
                                    )
                                    manual_secondary = gr.Dropdown(
                                        choices=SECONDARY_LABELS, value="none", label="次情绪"
                                    )
                                with gr.Row():
                                    manual_intensity = gr.Slider(
                                        0.1, 0.95, value=0.65, step=0.05, label="情绪强度"
                                    )
                                    manual_pause = gr.Slider(
                                        0.0, 1.2, value=0.4, step=0.05, label="锁定段后停顿 (秒)"
                                    )
                            with gr.Group(visible=False, elem_classes="sora-subcard") as locked_group:
                                gr.HTML("<div class='sora-hint'>🏷️ 输入一小句台词后按回车，自动带入上方情绪、强度与停顿，像标签一样锁定（翻译不会改动标签）。</div>")
                                locked_state = gr.State([])
                                seg_input = gr.Textbox(
                                    label="锁定片段", show_label=False,
                                    placeholder="例如：你怎么才回来啊……（输完直接按 Enter）",
                                )
                                with gr.Row():
                                    add_seg_btn = gr.Button("＋ 锁定", size="sm", variant="primary")
                                    pop_seg_btn = gr.Button("↶ 撤销", size="sm")
                                    clear_seg_btn = gr.Button("🗑 清空", size="sm", variant="stop")
                                locked_preview = gr.HTML(value=_render_locked_pills([]))
                            with gr.Accordion("📝 情绪分段 JSON（微调后选「使用下方分段 JSON」生效）", open=False, elem_classes="sora-acc") as json_acc:
                                plan_json_box = gr.Code(
                                    label="情绪分段 JSON",
                                    show_label=False,
                                    language="json",
                                    lines=10,
                                )

                        with gr.Tab("🌙 声音效果"):
                            gr.HTML("<div class='sora-tab-banner sora-banner-fx'><div class='tb-title'>🌙 苍穹声学与贴耳音景</div><div class='tb-sub'>轻语 ASMR · KU100 近讲方位 · 雨夜环境音场</div></div>")
                            director_auto_fx = gr.Checkbox(
                                label="🤖 由导演自动启用轻语、去刺耳与局部颤音",
                                value=True,
                                info="DeepSeek 规划出贴耳轻语或局部颤音/重音时，自动打开下面对应开关。",
                            )
                            with gr.Group(elem_classes="sora-subcard"):
                                whisper_mode = gr.Checkbox(
                                    label="🌙 贴耳轻语模式（ASMR）",
                                    value=False,
                                    info="切换 ASMR 洗净权重 ep5 + 低气泡音参考 ASMR_0290，并自动限制过重的负向控制。",
                                )
                                declick_deess = gr.Checkbox(
                                    label="🪶 轻度去刺耳（De-click + De-ess）",
                                    value=False,
                                    info="柔化清辅音尖刺与高频嘶声，保留元音和气声。",
                                )
                                local_delivery = gr.Checkbox(
                                    label="〰️ 局部音高、响度与颤声（试验）",
                                    value=False,
                                    info="按导演规划对指定字词做局部音高/响度/颤声后处理。",
                                )
                            with gr.Accordion("🏞️ 48kHz 双声道贴耳方位与环境音景（ASMR 可选）", open=False, elem_classes="sora-acc"):
                                ear_pan = gr.Dropdown(
                                    choices=[
                                        ("🤖 跟随导演自动", "auto"),
                                        ("🔇 标准普通距离", "off"),
                                        ("🎧 正中贴耳吹气", "center"),
                                        ("👈 左耳贴耳吹气", "left"),
                                        ("👉 右耳贴耳吹气", "right"),
                                        ("🔄 按句左右耳交替", "alternate"),
                                    ],
                                    value="auto",
                                    label="人头麦贴耳方位与近讲吹气感（KU100 近讲气流共鸣）",
                                )
                                ambient_bed = gr.Dropdown(
                                    choices=[
                                        ("🤖 跟随导演自动", "auto"),
                                        ("🔇 无环境音", "none"),
                                        ("🌧️ 窗外雨声", "rain"),
                                    ],
                                    value="auto",
                                    label="持续环境背景音",
                                )
                                foley_mode = gr.State("none")
                                ambient_level = gr.Slider(
                                    0.05, 0.45, value=0.18, step=0.02,
                                    label="环境音音量比例",
                                    info="保持默认 0.18 且开启导演自动时，由导演按雨势/轻语自动调节；手动拖动后优先按你的数值。",
                                )
                        with gr.Tab("◇ 高级调音"):
                            gr.HTML("<div class='sora-tab-banner sora-banner-adv'><div class='tb-title'>◇ 声线参考与律动微调</div><div class='tb-sub'>63 档原声参考匹配 · 情绪倍率与呼吸语速细调</div></div>")
                            ref_mode = gr.Radio(
                                choices=[
                                    AUTO_PERFORMANCE,
                                    "固定平静参考 SR011455 (最稳)",
                                    "按情绪自动匹配原声参考 (张力更强)",
                                ],
                                value=AUTO_PERFORMANCE,
                                label="参考音频策略",
                                elem_classes="sora-radio-stack",
                            )
                            with gr.Row():
                                candidate_count = gr.Slider(
                                    minimum=1, maximum=3, step=1, value=2,
                                    label="整句试音数量",
                                    info="每条 GPT 参考各生成一版，备选在左侧试听。",
                                )
                                api_reference_ranking = gr.Checkbox(
                                    value=True, label="API 导演辅助选参考",
                                    info="结合原声台词与历史标注给候选排序。",
                                )
                            with gr.Row():
                                emo_scale_mode = gr.Radio(
                                    choices=["自动动态上限", "手动固定倍率"],
                                    value="自动动态上限",
                                    label="情绪倍率 emo_scale",
                                )
                                emo_scale_val = gr.Slider(
                                    1.0, 1.7, value=1.35, step=0.05, label="手动倍率", interactive=False,
                                )
                            with gr.Row():
                                temperature = gr.Slider(0.5, 1.2, value=0.9, step=0.05, label="temperature")
                                top_k = gr.Slider(5, 30, value=10, step=1, label="top_k")
                                speed_factor = gr.Slider(0.90, 1.12, value=1.00, step=0.02, label="语速微调", info="仅建议微调，防破坏原生呼吸节奏")

        # ───────────── 底部：音频库 / 自优化 / API 设置 ─────────────
        with gr.Group(elem_classes="sora-dock"):
            with gr.Tabs(elem_classes="sora-tabs"):
                with gr.Tab("✦ 音频库与记录"):
                    with gr.Row(equal_height=False, elem_classes="sora-align-end"):
                        library_scope = gr.Dropdown(
                            choices=["⭐ 主选成品", "当前对话", "全部音频（含备选）", "全部记录"],
                            value="⭐ 主选成品",
                            label="筛选范围", scale=2,
                        )
                        library_entry = gr.Dropdown(label="快速检索（按台词与时间展示）", interactive=True, scale=5)
                        library_refresh = gr.Button("↻ 刷新列表", size="sm", scale=1, min_width=96)
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=7):
                            library_table = gr.Dataframe(
                                headers=["ID", "时间", "类型", "中文台词", "日文台词", "保存文件名"],
                                datatype=["str", "str", "str", "str", "str", "str"],
                                interactive=False,
                                wrap=True,
                                label="生成记录列表（点击任意一行即可右侧试听）",
                                elem_classes="sora-table",
                            )
                        with gr.Column(scale=5):
                            library_audio = gr.Audio(label="库内试听", type="filepath", interactive=False, elem_classes="sora-audio-box")
                            library_detail = gr.Textbox(label="完整记录详情与文件名", lines=6, interactive=False)
                            with gr.Row():
                                lib_load_btn = gr.Button("📥 载入台词重制", size="sm")
                                lib_good_btn = gr.Button("👍 标为正样本", size="sm", elem_classes="sora-fb-good")
                                lib_del_btn = gr.Button("🗑️ 删除此记录", size="sm", variant="stop")
                            lib_action_msg = gr.Textbox(label="操作提示", lines=1, interactive=False)

                with gr.Tab("◇ 自优化升级"):
                    gr.HTML("<div class='sora-tab-banner sora-banner-selfopt'><div class='tb-title'>◇ 穹声线自优化笔记 · SELF-OPTIMIZATION</div><div class='tb-sub'>结合历史生成记录与人工正负样本反馈，动态优化参考音优先级并支持本地 GPU 小步对比微调（自带出厂备份可一键回滚）。</div></div>")
                    self_opt_report = gr.Textbox(
                        label="当前自优化与错误标签权重状态",
                        value=lambda: format_self_opt_report(load_adaptive_weights()),
                        lines=7,
                        interactive=False,
                    )
                    with gr.Row():
                        opt_recompute_btn = gr.Button("📊 汇总记录并更新参考音评分", size="sm")
                        opt_train_btn = gr.Button("🧬 立即微调内部权重 (24步)", variant="primary", size="sm", elem_classes="sora-art-primary")
                        opt_reset_btn = gr.Button("↩️ 恢复出厂默认权重", variant="stop", size="sm")

                with gr.Tab("✦ API 设置"):
                    gr.HTML("<div class='sora-tab-banner sora-banner-api'><div class='tb-title'>✦ 导演与评审接口配置 · API SETTINGS</div><div class='tb-sub'>配置潜台词编排大模型（DeepSeek / OpenAI 兼容）与目标模式 WAV 音频评审接口，密钥仅保存在本地配置中。</div></div>")
                    gr.HTML("<div class='sora-section'>✦ 导演模型 <span class='sec-sub'>DIRECTOR LLM</span></div>")
                    with gr.Row():
                        provider_box = gr.Dropdown(
                            choices=list(PROVIDER_PRESETS.keys()),
                            value=default_provider,
                            label="模型厂商预设",
                        )
                        model_box = gr.Dropdown(
                            choices=[default_mdl] if default_mdl else [],
                            allow_custom_value=True,
                            label="模型名 (Model)",
                            value=default_mdl,

                        )
                        base_url_box = gr.Textbox(
                            label="API Base URL",
                            value=default_url,
                            placeholder="https://api.deepseek.com",
                            interactive=default_provider != OLLAMA_PROVIDER,
                        )
                    with gr.Row(equal_height=False, elem_classes="sora-align-end"):
                        api_key_box = gr.Textbox(
                            label="API Key（留空使用当前地址已保存的密钥）",
                            type="password",
                            placeholder="sk-...",
                            visible=default_provider != OLLAMA_PROVIDER,
                            scale=4,
                        )
                        refresh_ollama_btn = gr.Button("检测 Ollama 模型", visible=default_provider == OLLAMA_PROVIDER)
                        save_cfg_btn = gr.Button("💾 保存配置", variant="primary", scale=1, min_width=120, elem_classes="sora-art-primary")

                    ollama_status = gr.Markdown(visible=default_provider == OLLAMA_PROVIDER)
                    gr.HTML("<div class='sora-section' style='margin-top:14px'>✦ 音频评审模型 <span class='sec-sub'>AUDIO JUDGE</span></div>")
                    with gr.Row():
                        goal_judge_provider = gr.Dropdown(
                            choices=list(JUDGE_PRESETS), value=judge_provider, label="评审厂商预设",
                        )
                        goal_judge_model = gr.Textbox(
                            label="评审模型（须支持 WAV 输入）", value=judge_cfg.get("model", judge_default_model),
                            lines=1, max_lines=1,
                        )
                        goal_judge_url = gr.Textbox(
                            label="评审 API Base URL", value=judge_cfg.get("base_url", judge_default_url),
                            lines=1, max_lines=1,
                        )
                    with gr.Row(equal_height=False, elem_classes="sora-align-end"):
                        goal_judge_key = gr.Textbox(
                            label="评审 API Key（留空使用当前地址已保存的密钥）", type="password",
                            info="评审会把生成的 WAV 发送到此处配置的接口。", scale=4,
                        )
                        save_judge_btn = gr.Button("💾 保存评审配置", variant="primary", scale=1, min_width=130, elem_classes="sora-art-primary")
                    api_save_status = gr.Textbox(label="配置保存状态", lines=1, interactive=False, placeholder="点击上方保存按钮后，状态会显示在这里。")

        def _toggle_plan_mode(mode):
            manual = mode in ("手动指定整句情绪", "使用下方锁定标签片段", "使用上方锁定标签片段")
            locked_vis = mode in ("使用下方锁定标签片段", "使用上方锁定标签片段")
            return gr.update(visible=manual), gr.update(visible=locked_vis), gr.update(open=(mode == "使用下方分段 JSON"))

        plan_mode.change(_toggle_plan_mode, inputs=[plan_mode], outputs=[manual_group, locked_group, json_acc])
        director_goal_mode.change(
            lambda on: gr.update(visible=bool(on)), inputs=[director_goal_mode], outputs=[goal_opts_group],
        )
        emo_scale_mode.change(
            lambda m: gr.update(interactive=m == "手动固定倍率"), inputs=[emo_scale_mode], outputs=[emo_scale_val],
        )

        director_fields = [base_url_box, model_box, api_key_box, refresh_ollama_btn, ollama_status]
        provider_box.input(on_provider_change, inputs=[provider_box], outputs=director_fields)
        demo.load(_restore_director_config, outputs=[provider_box, *director_fields])
        refresh_ollama_btn.click(
            _refresh_ollama_models, inputs=[provider_box, model_box], outputs=[model_box, ollama_status],
        )

        goal_judge_provider.input(
            fn=on_judge_provider_change, inputs=[goal_judge_provider],
            outputs=[goal_judge_url, goal_judge_model, goal_judge_key],
        )
        save_judge_btn.click(
            fn=save_judge_config,
            inputs=[goal_judge_provider, goal_judge_url, goal_judge_model, goal_judge_key],
            outputs=[api_save_status],
        )

        save_cfg_btn.click(
            fn=_save_config,
            inputs=[provider_box, base_url_box, model_box, api_key_box],
            outputs=[api_save_status],
        )

        seg_input.submit(
            fn=add_locked_segment,
            inputs=[seg_input, manual_primary, manual_secondary, manual_intensity, manual_pause, locked_state],
            outputs=[seg_input, locked_state, locked_preview],
        )
        add_seg_btn.click(
            fn=add_locked_segment,
            inputs=[seg_input, manual_primary, manual_secondary, manual_intensity, manual_pause, locked_state],
            outputs=[seg_input, locked_state, locked_preview],
        )
        pop_seg_btn.click(
            fn=pop_locked_segment,
            inputs=[locked_state],
            outputs=[locked_state, locked_preview],
        )
        clear_seg_btn.click(
            fn=clear_locked_segments,
            inputs=[],
            outputs=[locked_state, locked_preview],
        )

        synthesis_inputs = [
            input_mode, line_box, ctx_box, provider_box, api_key_box,
            base_url_box, model_box, plan_mode, manual_primary, manual_secondary,
            manual_intensity, emo_scale_mode, emo_scale_val, ref_mode, temperature,
            top_k, plan_json_box, locked_state, local_delivery, candidate_count,
            api_reference_ranking, whisper_mode, declick_deess, director_auto_fx,
            ambient_bed, foley_mode, ambient_level, ear_pan, long_monologue_mode, speed_factor,
        ]
        synthesis_outputs = [
            audio_out, generated_ja, plan_json_box, status_out, alternate_one, alternate_two, generated_zh,
        ]
        goal_inputs = [
            director_goal_mode, goal_judge_key, goal_judge_model, goal_judge_provider, goal_judge_url, goal_rounds, goal_use_current,
            audio_out, generated_ja, generated_zh,
        ]
        conversation_inputs = [conversation_id, director_message, director_chat, *goal_inputs, *synthesis_inputs]
        chat_inputs = [conversation_id, director_message, director_chat, director_execution, *goal_inputs, *synthesis_inputs]
        conversation_outputs = [
            director_chat, *synthesis_outputs, goal_baseline, goal_status, synth_progress, conversation_menu,
        ]
        chat_outputs = [*conversation_outputs, director_message]
        gr.on(
            triggers=[director_send.click, director_message.submit],
            fn=partial(send_director_message, director_chat_action),
            inputs=chat_inputs, outputs=chat_outputs,
            show_progress="hidden", concurrency_id="director_chat_send", concurrency_limit=1,
        )
        director_task.click(
            partial(send_director_message, partial(director_chat_action, production_task=True)),
            inputs=chat_inputs, outputs=chat_outputs, api_name="dispatch_director_task",
            show_progress="hidden", concurrency_id="director_chat_send", concurrency_limit=1,
        )
        library_inputs = [library_scope, conversation_id]
        library_outputs = [library_entry, library_table, library_audio, library_detail]
        gen_btn.click(
            archived_synthesis, inputs=[conversation_id, *synthesis_inputs],
            outputs=[*synthesis_outputs, synth_progress], show_progress="hidden",
        ).then(_library_choices, inputs=library_inputs, outputs=library_outputs, show_progress="hidden")
        director_generate.click(
            partial(archive_conversation, generate_from_conversation), inputs=conversation_inputs,
            outputs=conversation_outputs, api_name="generate_from_discussion",
            show_progress="hidden",
        ).then(_library_choices, inputs=library_inputs, outputs=library_outputs, show_progress="hidden")

        def show_conversation(chat_id):
            chat_id = chat_id or ARCHIVE.new_chat()
            return (chat_id, None, ARCHIVE.load_chat(chat_id), _archive_menu(chat_id),
                    None, "", "", "", "", None, None,
                    gr.update(value=None, visible=False), gr.update(value="", visible=False))

        chat_switch_outputs = [
            conversation_id, director_message, director_chat, conversation_menu,
            audio_out, generated_ja, generated_zh, plan_json_box, status_out, alternate_one, alternate_two,
            goal_baseline, goal_status,
        ]
        conversation_menu.input(show_conversation, inputs=[conversation_menu], outputs=chat_switch_outputs)
        new_conversation.click(lambda: show_conversation(None), outputs=chat_switch_outputs)

        demo.load(_library_choices, inputs=library_inputs, outputs=library_outputs)
        library_refresh.click(_library_choices, inputs=library_inputs, outputs=library_outputs)
        library_scope.change(_library_choices, inputs=library_inputs, outputs=library_outputs)
        library_entry.input(_library_entry, inputs=[library_entry], outputs=[library_audio, library_detail])
        library_table.select(
            _on_library_table_select, inputs=[library_table], outputs=[library_entry, library_audio, library_detail],
        )

        fb_inputs = [feedback_tags, feedback_note, audio_out, generated_ja, generated_zh, status_out, conversation_id]
        fb_outputs = [feedback_status, self_opt_report]
        fb_good_btn.click(
            partial(_feedback_action, True), inputs=fb_inputs, outputs=fb_outputs,
        ).then(_library_choices, inputs=library_inputs, outputs=library_outputs)
        fb_bad_btn.click(
            partial(_feedback_action, False), inputs=fb_inputs, outputs=fb_outputs,
        ).then(_library_choices, inputs=library_inputs, outputs=library_outputs)
        opt_recompute_btn.click(_recompute_self_opt_action, outputs=[self_opt_report])
        opt_train_btn.click(_run_self_opt_finetune_action, outputs=[self_opt_report])
        opt_reset_btn.click(_reset_self_opt_action, outputs=[self_opt_report])

        lib_load_btn.click(_lib_load_to_editor, inputs=[library_entry], outputs=[line_box, lib_action_msg])
        lib_good_btn.click(
            _lib_mark_good, inputs=[library_entry, conversation_id], outputs=[lib_action_msg, self_opt_report],
        ).then(_library_choices, inputs=library_inputs, outputs=library_outputs)
        lib_del_btn.click(
            _lib_delete_record, inputs=[library_entry], outputs=[lib_action_msg],
        ).then(_library_choices, inputs=library_inputs, outputs=library_outputs)

        preset_outputs = [
            manual_primary, manual_secondary, manual_intensity, manual_pause,
            whisper_mode, declick_deess, director_auto_fx,
            ear_pan, ambient_bed, preset_hint,
        ]
        preset_daily_btn.click(partial(_apply_style_preset, "daily"), outputs=preset_outputs)
        preset_clingy_btn.click(partial(_apply_style_preset, "clingy"), outputs=preset_outputs)
        preset_tsun_btn.click(partial(_apply_style_preset, "tsundere"), outputs=preset_outputs)
        preset_asmr_btn.click(partial(_apply_style_preset, "asmr_sleep"), outputs=preset_outputs)
    return demo


def _pick_free_port(host: str, start_port: int) -> int:
    import socket
    for p in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    return start_port


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7865)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()

    print("正在预加载春日野穹 V3 模型 (old_ep15 GPT + bs16_s2_ep25 SoVITS)...", flush=True)
    get_tts()
    print("模型加载完成，正在启动本地网页...", flush=True)

    port = _pick_free_port(args.host, args.port)
    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=port,
        inbrowser=args.open,
        share=False,
        allowed_paths=[str(ARCHIVE.attachments.resolve()), str(UI_ASSETS_DIR.resolve())],
    )


if __name__ == "__main__":
    main()
