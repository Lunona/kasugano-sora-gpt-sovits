from audio_goal import GEMINI_PROVIDER, JUDGE_PRESETS, OPENAI_PROVIDER
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
from uuid import uuid4
import urllib.request
from urllib.parse import urlsplit

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

from performance import (
    APP, EMOTIONS, deepseek_json, plan_with_deepseek, plan_with_ollama, rank_references_with_deepseek,
)
from synthesize import (
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

DEFAULT_GPT = ROOT / "v3_checkpoints/gpt_eval/phrase-adapter-epoch15.pt"
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

_TTS_LOCK = threading.Lock()
_TTS_ENGINE = None

PROVIDER_PRESETS = {
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


def _save_config(provider: str, base_url: str, model_name: str, api_key: str) -> str:
    data = _load_config()
    data.update({
        "provider": provider,
        "base_url": (base_url or "").strip(),
        "model": (model_name or "").strip(),
        "api_key": (api_key or "").strip(),
    })
    CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"已保存配置到 {CONFIG_PATH.name}（下次打开自动读取）"

def save_judge_config(provider, base_url, model, api_key):
    data = _load_config()
    data["audio_judge"] = {
        "provider": provider,
        "base_url": base_url.strip(),
        "model": model.strip(),
        "api_key": api_key.strip(),
    }
    CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
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
    key = (api_key or os.environ.get("DEEPSEEK_API_KEY", "")).strip()
    if not key:
        raise RuntimeError("请先填写 API Key，才能将中文自动转换为穹口吻的日文台词。")
    url = (base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).strip().rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    mdl = (model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")).strip()
    prompt = (
        "请把下面用户输入的中文意思，改写成『缘之空』春日野穹（Sora）的自然日语口语台词。\n"
        "要求：\n"
        "1. 符合穹的说话习惯（略带傲娇、依赖感、直率或轻声低语）；即使中文原话没有标点，也要根据语气自动补全自然的日语标点（如「、」「。」「？」「！」「……」）。\n"
        "2. 只输出最终的日语台词文本本身，不要解释、不要罗马音、不要引号。\n\n"
        f"情景/前情上下文：{context or '无'}\n"
        f"中文原话：{zh_text}"
    )
    payload = {
        "model": mdl,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": "你是精通『缘之空』春日野穹角色语气的日语台词翻译。只输出日语台词。"},
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"].strip().strip("「」\"'")


def _render_locked_pills(segments: list[dict]) -> str:
    if not segments:
        return (
            "<div style='padding:10px 12px;border:1px dashed #cbd5e1;border-radius:10px;color:#64748b;font-size:13px;'>"
            "暂无锁定片段。在上方输入一小句台词、选好情绪后<b>直接按回车</b>，就会像话题标签一样锁定在这里（标签与文字分离，翻译绝不会改掉标签）。"
            "</div>"
        )
    cards = []
    for idx, seg in enumerate(segments, 1):
        pri = html.escape(seg["primary"])
        sec = seg.get("secondary")
        sec_badge = (
            f"<span style='background:#ede9fe;color:#5b21b6;padding:2px 8px;border-radius:999px;font-size:12px;margin-right:4px;'>#{html.escape(sec)}</span>"
            if sec
            else ""
        )
        text_esc = html.escape(seg["text"])
        cards.append(
            "<div style='display:flex;align-items:center;flex-wrap:wrap;gap:6px;padding:8px 10px;"
            "margin-bottom:6px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;'>"
            f"<span style='color:#64748b;font-weight:600;font-size:12px;'>{idx}.</span>"
            f"<span style='background:#dbeafe;color:#1d4ed8;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:600;'>#{pri}</span>"
            f"{sec_badge}"
            f"<span style='background:#f1f5f9;color:#334155;padding:2px 7px;border-radius:999px;font-size:12px;'>强度 {seg['intensity']:.2f}</span>"
            f"<span style='background:#f1f5f9;color:#334155;padding:2px 7px;border-radius:999px;font-size:12px;'>停 {seg['pause_s']:.2f}s</span>"
            f"<span style='color:#0f172a;font-size:14px;margin-left:4px;'>{text_esc}</span>"
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
        parsed = json.loads(custom_plan_json)
        if not isinstance(parsed, list) or not parsed:
            raise ValueError("分段 JSON 必须是非空数组。")
        return parsed, "使用手动编辑的分段 JSON"

    if plan_mode == "手动指定整句情绪":
        plan = _build_manual_whole_plan(ja_text, manual_primary, manual_secondary, manual_intensity)
        return refine_plan(ja_text, plan), f"手动整句: {manual_primary}"

    # 自动导演规划
    key = (api_key or os.environ.get("DEEPSEEK_API_KEY", "")).strip()
    burl = (base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).strip()
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
            api_key = key or os.environ.get("DEEPSEEK_API_KEY", "").strip()
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


def discuss_performance(message, history, input_mode, line, context, api_key, base_url, model, *, production_task=False):
    composer = message or {}
    message = (composer.get("text", "") or "").strip()
    paths = composer.get("files") or []
    if not message and not paths:
        raise gr.Error("先输入消息或选择附件。")
    turns = [*(history or []), *({"role": "user", "content": (path,)} for path in paths)]
    request_text = message or "请查看我附上的内容，并结合当前讨论给出回应。"
    turns.append({"role": "user", "content": request_text})
    key = (api_key or "").strip()
    url = (base_url or "").strip()
    mdl = (model or "").strip()
    try:
        payload = json.loads(_discussion_payload(input_mode, line or "", context, turns))
        payload["已通过下达制作任务按钮提交"] = production_task
        payload["可用表演参考素材"] = _reference_notes(API_PERFORMANCE_REFS)
        result = deepseek_json(
            "你是春日野穹的日语配音导演，用简洁中文与用户讨论台词和表演。"
            "附件是用户指定的参考素材，按用户在对话中的要求使用。结合输入区和讨论记录回应最新一条用户消息。可以提议日语台词，说明潜台词、语速、停顿和是否需要独立拟声。"
            "这个页面能读取提供的候选目录并自动选参考、生成试听。你可以推荐目录中的具体编号，说明是按文本与标注判断。"
            "页面执行本地合成，你没有收到原声或生成音频，不能声称听过。"
            "用户否决的方案不要再当作共识。当用户下达制作任务时，action 填 generate。"
            "“给我做一段雨夜重逢”“录一段嘴硬但其实在关心对方的语音”都是制作任务，即使没用‘生成’这个词。"
            "通过下达制作任务按钮提交时，按制作任务处理。"
            "收到制作任务后，自行补足未指定的场景、人物关系、事件和潜台词。自行写自然日语台词，决定语速、停顿、情绪变化和拟声。"
            "方案以用户的明确要求为准，普通创作细节自行决定。回复简短展示场景、拟采用的台词和演法。"
            "询问是否能选参考、试音建议或讨论演法都填 discuss。生成动作执行前，回复不要声称已经完成。"
            "只输出 JSON，reply 为给用户看的回复，action 为 discuss 或 generate。",
            _discussion_content(json.dumps(payload, ensure_ascii=False), turns, key, url, mdl),
            api_key=key, base_url=url, model=mdl,
        )
        reply = result["reply"].strip()
        if not reply:
            raise ValueError("empty reply")
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as exc:
        detail = str(exc) if str(exc).startswith("API 连续两次") else type(exc).__name__
        raise gr.Error(f"导演未能回复（{detail}）。") from exc
    action = "generate" if production_task else result.get("action", "discuss")
    return [*turns, {"role": "assistant", "content": reply}], action


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
    turns, action = discuss_performance(
        message, history, input_mode, line, context, key, url, model, production_task=production_task,
    )
    generate = action == "generate"
    if generate and execution_mode != "全自动":
        turns = [*turns, {"role": "assistant", "content": "已收到生成请求。请点「按讨论生成语音」批准。"}]
        generate = False

    yield None, turns, *(gr.skip() for _ in range(9))
    if not generate:
        return

    synthesis_args = (input_mode, line, context, provider, key, url, model, *settings)
    try:
        if goal_mode:
            goal_settings = (judge_key, judge_model, judge_provider, judge_url, rounds, use_current, audio, ja, zh)
            for output in _conversation_goal(turns, goal_settings, synthesis_args, conversation_id):
                yield None, turns, *output
        else:
            output = synthesize_action(*synthesis_args, discussion_history=turns, conversation_id=conversation_id)
            yield None, [*turns, {"role": "assistant", "content": "已生成，可以在下方试听。"}], *output, gr.skip(), gr.skip()
    except Exception as exc:
        reply = str(exc) if isinstance(exc, gr.Error) else f"生成失败（{type(exc).__name__}）。"
        ARCHIVE.record(conversation_id, "生成问题", reply)
        yield None, [*turns, {"role": "assistant", "content": reply}], *(gr.skip() for _ in range(9))


def generate_from_conversation(message, history, goal_mode, judge_key, judge_model, judge_provider, judge_url, rounds,
                               use_current, audio, ja, zh, *settings, conversation_id=None):
    if not goal_mode:
        yield message, history, *synthesize_action(*settings, discussion_history=history, conversation_id=conversation_id), gr.skip(), gr.skip()
        return

    draft = message or {}
    turns = list(history or [])
    turns.extend({"role": "user", "content": (path,)} for path in draft.get("files", []))
    text = (draft.get("text") or "").strip()
    if text:
        turns.append({"role": "user", "content": text})
    goal_settings = (judge_key, judge_model, judge_provider, judge_url, rounds, use_current, audio, ja, zh)
    for output in _conversation_goal(turns, goal_settings, settings, conversation_id):
        yield None, turns, *output


def _finish_discussion(input_mode, line, context, history, key, base_url, model):
    if not history:
        raise gr.Error("先发一条消息和导演讨论，再按讨论生成。")
    try:
        result = deepseek_json(
            "你是春日野穹的日语配音导演。把输入区和讨论定稿为一条可直接合成的日语台词与表演说明。"
            "附件作为参考素材，按对话里的用户要求使用。以最近的用户要求为准，保留原场景与已确认的细节，排除用户否决的提议。"
            "按输入方式认定原台词。如果已有日语原句且用户只要求调整演法，逐字保留该句；"
            "如果以中文为准，翻成自然日语；如果用户讨论了新台词，采用最新确认的内容。"
            "制作任务只给主题或情境时，自行构思发生了什么、角色想掩饰什么，再写出适合该情境的日语台词。"
            "需要短场景时可用连贯的几句，按任务要求控制长度。批准过的剧情与台词优先。别把动作旁白写入要念的台词。"
            "把场景和潜台词简述放进 context，让后续规划决定情绪变化和选参考。"
            "表演说明包含有共识的语气、节奏和拟声意图。保留用户明确确认的参考编号或选取偏好，交给后续参考排序。"
            "只输出 JSON。ja_text 为最终日语台词，zh_text 为这份日语台词的自然中文译文，context 为中文表演说明，三个字段均为字符串。",
            _discussion_content(_discussion_payload(input_mode, line, context, history), history, key, base_url, model),
            api_key=key, base_url=base_url, model=model,
        )
        ja_line = " ".join(result["ja_text"].splitlines()).strip()
        brief = result["context"].strip()
        translation = result.get("zh_text")
        translation = translation.strip() if isinstance(translation, str) else ""
        if not ja_line:
            raise ValueError("empty line")
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as exc:
        raise gr.Error(f"导演未能定稿，请检查 API 配置后重试（{type(exc).__name__}）。") from exc
    return ja_line, "\n".join(part for part in (context, brief) if part), translation


def _reference_notes(ref_ids):
    from soundfile import info

    pending = set(ref_ids)
    notes = {}
    with (ROOT / "colab_data/逐条表演复核.csv").open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            ref_id = row["语音编号"]
            if ref_id in pending:
                pending.remove(ref_id)
                wav = DATA / "音频_WAV" / f"{ref_id}.wav"
                try:
                    duration = info(wav).duration
                except (RuntimeError, OSError):
                    continue
                if not 3.0 <= duration <= 10.0 or not row["日文"].strip():
                    continue
                notes[ref_id] = {
                    "id": ref_id,
                    "line": row["日文"],
                    "historical_note": row["情绪依据"],
                    "historical_label": row["整句多标签"],
                }
                if not pending:
                    break
    return [notes[ref_id] for ref_id in ref_ids if ref_id in notes]


def _chinese_transcript(ja_text, key, base_url, model):
    if not (key or os.environ.get("DEEPSEEK_API_KEY", "")).strip():
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


def _resolve_director_controls(plan, whisper_mode: bool, declick_deess: bool, local_delivery: bool, director_auto_fx: bool):
    """Let DeepSeek / rule planner auto-enable whisper, de-click/de-ess, and local tremble/pitch."""
    notes = []
    if not director_auto_fx or not plan:
        return whisper_mode, declick_deess, local_delivery, notes
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
        if not declick_deess:
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
    return whisper_mode, declick_deess, local_delivery, notes


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
    discussion_history=None,
    chinese_transcript="",
    conversation_id=None,
):
    line = " ".join((line or "").splitlines()).strip()
    context = (context or "").strip()
    key = (api_key or "").strip()
    burl = (base_url or "").strip()
    mdl = (model_name or "").strip()
    _save_config(provider_name, burl, mdl, key)

    from_discussion = discussion_history is not None
    if from_discussion:
        line, context, chinese_transcript = _finish_discussion(
            input_mode, line, context, discussion_history, key, burl, mdl,
        )
        input_mode = "日语"
        plan_mode = "自动导演 (API 大模型)"
        ref_mode = AUTO_PERFORMANCE
        api_reference_ranking = True

    locked = list(locked_segments or [])
    if plan_mode == "使用上方锁定标签片段":
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
        effective_context = context
        ja_text = line
        if _line_language(input_mode, line) == "中文":
            ja_text = translate_zh_to_sora_ja(line, context, key, burl, mdl)
            chinese_transcript = line
            effective_context = f"中文原话：{line}" + (f"\n附加情景：{context}" if context else "")

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

    tts = get_tts()
    auto_performance = ref_mode == AUTO_PERFORMANCE
    director_controls_events = planner_note.startswith("API 导演规划 (")
    whisper_mode, declick_deess, local_delivery, auto_fx_notes = _resolve_director_controls(
        plan, whisper_mode, declick_deess, local_delivery, director_auto_fx,
    )
    synth_text, synth_plan = ja_text, plan
    if not director_controls_events and not auto_performance and (getattr(tts, "_director", None) is None or getattr(tts, "_ctrl_dim", 22) < 26):
        synth_text, synth_plan = auto_insert_tags(ja_text, plan)

    emo_scale = None if emo_scale_mode == "自动动态上限" else float(emo_scale_val)
    ref_id = "auto" if ref_mode.startswith("按情绪自动匹配") else "SR011455"
    job_name = f"sora_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
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
        api_key_for_rank = key or os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if api_reference_ranking and plan_mode == "自动导演 (API 大模型)":
            if api_key_for_rank:
                try:
                    pool = _reference_notes(API_PERFORMANCE_REFS)
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
                except (OSError, ValueError, KeyError, TypeError) as exc:
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
    if auto_performance:
        from acoustic_delivery import check_take

    with _TTS_LOCK:
        for candidate in candidates:
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
                )
            except (RuntimeError, OSError) as exc:
                label = candidate[0] if candidate else ref_id
                ARCHIVE.record(conversation_id, "生成问题", f"{label} · {type(exc).__name__} · {exc}", ja=ja_text)
                if len(candidates) == 1:
                    raise
                failed.append((candidate[0], exc))
                continue
            if auto_performance:
                issues, match = check_take(output, synth_text, synth_plan, device=tts.configs.device, whisper_mode=whisper_mode)
                detail = "、".join(issues) if issues else "通过可用检查"
                if match is not None:
                    detail += f"（台词匹配 {match:.0%}）"
                elif not issues:
                    detail += "（未取得发音对齐分数）"
                checks.append(f"{candidate[0]} · {detail}")
                if issues:
                    ARCHIVE.record(conversation_id, "成品检查异常", detail, audio=output, ja=ja_text)
                    flagged.append((output, result))
                    continue
            results.append((output, result))
            if len(results) >= count:
                break

    if not results and flagged:
        results = flagged[:count]
        checks.append("未找到通过检查的版本，已保留产出供人工试听。")

    if not results:
        raise gr.Error(f"所有整句候选生成失败：{failed[0][1]}")

    if local_delivery and len(results) > 1 and not api_overview:
        # Prefer a reliably aligned take when applying requested local controls.
        results.sort(key=lambda take: take[1]["local_delivery"]["status"] == "unaligned")

    out_wav, info = results[0]
    alternates = [str(output) for output, _ in results[1:]]
    alternates.extend([None] * (2 - len(alternates)))

    plan_pretty = json.dumps(plan, ensure_ascii=False, indent=2)
    references = f"参考原声：{', '.join(info.get('refs', [ref_id]))}"
    if auto_performance:
        references = f"整句参考：GPT {info['refs'][0]}，SoVITS {info['acoustic_ref']}"
        if api_overview:
            references += f"\nAPI 导演整句演法：{api_overview}"
        else:
            delivery = DELIVERY_LABELS.get(info["delivery"], info["delivery"])
            references += f"\n本地配参考的侧重语气：{delivery}"
    if director_controls_events or info["directed_events"]:
        events = "、".join(info["directed_events"]) or "未安排"
        label = "API 导演发声事件" if director_controls_events else "发声事件"
        references += f"\n{label}：{events}"
    if "director_emo_scale" in info:
        references += f"\nDeepSeek 情绪倍率：{info['director_emo_scale']:g}"
    if local_delivery:
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
    local_delivery, candidate_count, _api_reference_ranking, whisper_mode=False, declick_deess=False, director_auto_fx=True, *, conversation_id=None,
):
    from audio_goal import review_takes

    goal = (goal or "").strip()
    judge_url = (judge_url or JUDGE_PRESETS[judge_provider][0]).strip().rstrip("/")
    env_names = {GEMINI_PROVIDER: "GEMINI_API_KEY", OPENAI_PROVIDER: "OPENAI_API_KEY"}
    env_key = os.environ.get(env_names.get(judge_provider, ""), "") if judge_url == JUDGE_PRESETS[judge_provider][0] else ""
    judge_key = (judge_key or env_key).strip()
    judge_model = (judge_model or "").strip()
    if not goal:
        raise gr.Error("请填写能通过试听判断的目标。")
    if not judge_key or not judge_model or not judge_url:
        raise gr.Error("请在底部「API 设置」中配置音频评审的厂商、API 地址、Key 和模型。")
    if not (key or os.environ.get("DEEPSEEK_API_KEY", "")).strip():
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
                    chinese_transcript=chinese_transcript, conversation_id=conversation_id,
                )
                if not result[3].startswith("规划来源：API 导演规划"):
                    raise RuntimeError("导演未返回 API 表演规划")
            paths = [path for path in (result[0], *result[4:6]) if path]
            fallback = result
            reviews = review_takes(paths, ja_text, goal, api_key=judge_key, model=judge_model,
                                   provider=judge_provider, base_url=judge_url)
        except Exception as exc:
            detail = f"HTTP {exc.code}" if isinstance(exc, urllib.error.HTTPError) else type(exc).__name__
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


def on_provider_change(provider_name: str):
    url, mdl = PROVIDER_PRESETS.get(provider_name, ("", ""))
    return url, mdl


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


def archive_conversation(callback, chat_id, *args):
    try:
        for message, turns, *outputs in callback(*args, conversation_id=chat_id):
            saved = _save_director_history(chat_id, turns)
            yield message, saved, *outputs, _archive_menu(chat_id)
    except Exception as exc:
        detail = str(exc) if isinstance(exc, gr.Error) else type(exc).__name__
        ARCHIVE.record(chat_id, "对话或生成问题", detail)
        draft, history = args[:2]
        draft = draft or {}
        text = (draft.get("text") or "").strip()
        turns = list(history or [])
        if text:
            turns.extend((
                {"role": "user", "content": text},
                {"role": "assistant", "content": detail},
            ))
        saved = _save_director_history(chat_id, turns)
        yield draft, saved, *(gr.skip() for _ in range(9)), _archive_menu(chat_id)


def archived_synthesis(chat_id, *settings):
    try:
        return synthesize_action(*settings, conversation_id=chat_id)
    except Exception as exc:
        detail = str(exc) if isinstance(exc, gr.Error) else type(exc).__name__
        ARCHIVE.record(chat_id, "生成问题", detail)
        raise


def _library_choices(scope, conversation_id):
    by_chat = conversation_id if scope == "当前对话" else None
    choices = ARCHIVE.list_events(audio_only=scope == "全部音频", chat_id=by_chat)
    if scope == "全部音频":
        recorded = ARCHIVE.audio_paths()
        existing = sorted(OUT_DIR.glob("sora_*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        choices.extend((f"早期音频 · {path.name}", f"wav:{path.name}")
                       for path in existing if str(path) not in recorded)
    return gr.update(choices=choices[:250], value=None), None, ""


def _library_entry(value):
    if not value:
        return None, ""
    if value.startswith("wav:"):
        name = value.removeprefix("wav:")
        row = None
        path = (OUT_DIR / Path(name).name).resolve()
        detail = f"早期音频 · {path.name}"
    else:
        row = ARCHIVE.event(value)
        if not row:
            return None, "记录不可用。"
        path = Path(row["audio"]).resolve() if row["audio"] else None
        detail = "\n".join(filter(None, [
            f"{row['created'][0:19].replace('T', ' ')} · {row['kind']}",
            f"对话：{row['title']}" if row["title"] else "",
            f"中文：{row['zh']}" if row["zh"] else "",
            f"日文：{row['ja']}" if row["ja"] else "",
            row["detail"],
        ]))
    if path and (not path.is_relative_to(OUT_DIR.resolve()) or path.suffix.lower() != ".wav" or not path.is_file()):
        return None, detail + "\n音频文件不可用。"
    return str(path) if path else None, detail


def build_ui() -> gr.Blocks:
    cfg = _load_config()
    default_provider = cfg.get("provider") if cfg.get("provider") in PROVIDER_PRESETS else "DeepSeek 官方"
    env_key = cfg.get("api_key") or os.environ.get("DEEPSEEK_API_KEY", "").strip()
    default_url = cfg.get("base_url") or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
    default_mdl = cfg.get("model") or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip()
    judge_cfg = cfg.get("audio_judge") or {}
    judge_provider = judge_cfg.get("provider")
    if judge_provider not in JUDGE_PRESETS:
        judge_provider = GEMINI_PROVIDER
    judge_default_url, judge_default_model = JUDGE_PRESETS[judge_provider]
    with gr.Blocks(title="春日野穹 GPT-SoVITS V3 情绪表演") as demo:
        gr.Markdown(
            "## 春日野穹 GPT-SoVITS V3 本地语音生成\n"
            "默认权重：`old_ep15` GPT + `bs16_s2_ep25` SoVITS（支持任意 OpenAI 兼容 API 厂商/模型、本地 Ollama 与手动情绪面板）"
        )
        with gr.Row():
            with gr.Column(scale=5):
                line_box = gr.Textbox(
                    label="台词（中文或日语）",
                    placeholder="例如：你怎么才回来啊…… ／ 別に、待ってたわけじゃないよ。",
                    lines=3,
                )
                input_mode = gr.Radio(
                    choices=["自动识别", "中文", "日语"], value="自动识别",
                    label="台词语言",
                    info="有假名按日语直接合成。只有汉字的日语请选「日语」。",
                )
                ctx_box = gr.Textbox(
                    label="剧情 / 语气上下文（可选，帮导演判断潜台词）",
                    placeholder="例如：等了很久终于等到对方回来，嘴上先责怪，后半句松一口气。",
                    lines=2,
                )
                with gr.Accordion("和 DeepSeek 讨论", open=True):
                    saved_chats = ARCHIVE.list_chats()
                    active_chat = saved_chats[0][1] if saved_chats else ARCHIVE.new_chat()
                    conversation_id = gr.State(active_chat)
                    with gr.Row():
                        conversation_menu = gr.Dropdown(
                            choices=saved_chats or [("新对话", active_chat)], value=active_chat,
                            label="聊天历史", interactive=True, scale=4,
                        )
                        new_conversation = gr.Button("＋ 新对话", scale=1)
                    director_chat = gr.Chatbot(
                        label="导演对话", type="messages", height=225, value=ARCHIVE.load_chat(active_chat),
                    )
                    director_message = gr.MultimodalTextbox(
                        label="给导演的话", lines=2, file_types=sorted(ATTACHMENT_TYPES),
                        file_count="multiple", submit_btn=False,
                        placeholder="例如：给我做一段雨夜重逢的语音，剧情和日语台词你来定。",
                    )
                    with gr.Row():
                        director_execution = gr.Dropdown(
                            choices=[("权限 · 审批后生成", "审批后生成"), ("权限 · 全自动", "全自动")],
                            value="审批后生成", label="导演执行方式", show_label=False,
                            container=False, allow_custom_value=False, scale=1, min_width=145,
                        )
                        director_goal_mode = gr.Checkbox(label="目标模式", container=False, scale=0)
                        director_send = gr.Button("发送")
                        director_task = gr.Button("下达制作任务")
                        director_generate = gr.Button("按讨论生成语音", variant="primary")
                    goal_status = gr.Textbox(label="目标评审", lines=3, interactive=False, visible=False)


            with gr.Column(scale=4):
                plan_mode = gr.Radio(
                    choices=[
                        "自动导演 (API 大模型)",
                        "自动导演 (本地 Ollama)",
                        "手动指定整句情绪",
                        "使用上方锁定标签片段",
                        "使用下方分段 JSON",
                    ],
                    value="自动导演 (API 大模型)",
                    label="情绪控制方式",
                )
                with gr.Row():
                    manual_primary = gr.Dropdown(
                        choices=EMOTION_LABELS, value="tender", label="主情绪"
                    )
                    manual_secondary = gr.Dropdown(
                        choices=SECONDARY_LABELS, value="none", label="次情绪"
                    )
                with gr.Row():
                    manual_intensity = gr.Slider(
                        0.1, 0.95, value=0.65, step=0.05, label="情绪强度 (intensity)"
                    )
                    manual_pause = gr.Slider(
                        0.0, 1.2, value=0.4, step=0.05, label="锁定段后停顿(s)"
                    )
                with gr.Row():
                    emo_scale_mode = gr.Radio(
                        choices=["自动动态上限", "手动固定倍率"],
                        value="自动动态上限",
                        label="emo_scale 模式",
                    )
                    emo_scale_val = gr.Slider(
                        1.0, 1.7, value=1.35, step=0.05, label="手动 emo_scale"
                    )
                ref_mode = gr.Radio(
                    choices=[
                        AUTO_PERFORMANCE,
                        "固定平静参考 SR011455 (最稳)",
                        "按情绪自动匹配原声参考 (张力更强)",
                    ],
                    value=AUTO_PERFORMANCE,
                    label="参考音频策略",
                )
                candidate_count = gr.Slider(
                    minimum=1, maximum=3, step=1, value=2,
                    label="整句模式试音数量",
                    info="每条 GPT 参考各生成一版，备选可在下方试听。",
                )
                api_reference_ranking = gr.Checkbox(
                    value=True, label="API 导演辅助选参考",
                    info="在 API 导演的整句模式下，结合原声台词与历史标注给候选排序。",
                )
                director_auto_fx = gr.Checkbox(
                    label="由导演自动启用轻语、去刺耳与局部颤音",
                    value=True,
                    info="开启后，DeepSeek 导演规划出贴耳轻语或局部颤音/重音时会自动启用下方对应功能。",
                )
                local_delivery = gr.Checkbox(label="局部音高、响度与颤声（试验）", value=False)
                with gr.Row():
                    whisper_mode = gr.Checkbox(
                        label="贴耳轻语模式 (ASMR ep5 + 低气泡音参考)",
                        value=False,
                        info="自动切换 ASMR 洗净权重与 ASMR_0290，并限制过重负向控制防劈音。",
                    )
                    declick_deess = gr.Checkbox(
                        label="轻度去刺耳 (De-click + De-ess)",
                        value=False,
                        info="轻微柔化清辅音尖刺与高频嘶声，保留元音和气声。",
                    )
                with gr.Row():
                    temperature = gr.Slider(0.5, 1.2, value=0.9, step=0.05, label="temperature")
                    top_k = gr.Slider(5, 30, value=10, step=1, label="top_k")

        with gr.Accordion("按回车锁定标签片段（像话题胶囊一样把情绪锁在片段上，翻译不改标签）", open=True):
            locked_state = gr.State([])
            with gr.Row():
                seg_input = gr.Textbox(
                    label="输入一小句台词后按回车锁定（将自动带入上方选中的主/次情绪、强度与停顿）",
                    placeholder="例如：你怎么才回来啊……（输完直接按 Enter）",
                    scale=5,
                )
                add_seg_btn = gr.Button("+ 锁定片段", scale=1)
                pop_seg_btn = gr.Button("撤销最后一段", scale=1)
                clear_seg_btn = gr.Button("清空", scale=1)
            locked_preview = gr.HTML(value=_render_locked_pills([]))

        gen_btn = gr.Button("生成穹的语音", variant="primary")

        generated_zh = gr.Textbox(label="生成台词 · 中文", lines=2, interactive=False)
        generated_ja = gr.Textbox(label="生成台词 · 日文", lines=2, interactive=False)

        with gr.Row():
            audio_out = gr.Audio(label="生成结果试听", type="filepath", interactive=False, autoplay=True)
            status_out = gr.Textbox(label="运行状态", lines=4, interactive=False)

        with gr.Accordion("整句备选试听", open=True):
            with gr.Row():
                alternate_one = gr.Audio(label="备选 1", type="filepath", interactive=False)
                alternate_two = gr.Audio(label="备选 2", type="filepath", interactive=False)
                goal_baseline = gr.Audio(label="起始版本对比", type="filepath", interactive=False, visible=False)

        plan_json_box = gr.Code(
            label="当前情绪分段 JSON（可微调后选「使用下方分段 JSON」重新生成）",
            language="json",
            lines=8,
        )

        with gr.Accordion("音频库与生成记录", open=False):
            with gr.Row():
                library_scope = gr.Dropdown(
                    choices=["全部音频", "全部记录", "当前对话"], value="全部音频",
                    label="范围", scale=3,
                )
                library_refresh = gr.Button("刷新", scale=1)
            library_entry = gr.Dropdown(label="选择记录", interactive=True)
            library_audio = gr.Audio(label="库内试听", type="filepath", interactive=False)
            library_detail = gr.Textbox(label="记录详情", lines=5, interactive=False)

        with gr.Accordion("API 设置", open=False):
            with gr.Row():
                provider_box = gr.Dropdown(
                    choices=list(PROVIDER_PRESETS.keys()),
                    value=default_provider,
                    label="模型厂商预设",
                )
                model_box = gr.Textbox(
                    label="模型名 (Model)",
                    value=default_mdl,
                    placeholder="deepseek-chat / gpt-4o-mini / qwen-plus",
                )
            base_url_box = gr.Textbox(
                label="API Base URL（OpenAI 兼容接口地址，可自定义中转站）",
                value=default_url,
                placeholder="https://api.deepseek.com",
            )
            with gr.Row():
                api_key_box = gr.Textbox(
                    label="API Key（生成或点保存后自动写入本地 webui_config.json，下次免重填）",
                    value=env_key,
                    type="password",
                    placeholder="sk-...",
                    scale=4,
                )
                save_cfg_btn = gr.Button("保存配置到本地", scale=1)

            with gr.Row():
                goal_judge_provider = gr.Dropdown(
                    choices=list(JUDGE_PRESETS), value=judge_provider, label="音频评审厂商预设",
                )
                goal_judge_model = gr.Textbox(
                    label="音频评审模型", value=judge_cfg.get("model", judge_default_model),
                    info="模型和接口须支持 WAV 音频输入。",
                )
            goal_judge_url = gr.Textbox(
                label="音频评审 API Base URL", value=judge_cfg.get("base_url", judge_default_url),
            )
            with gr.Row():
                goal_judge_key = gr.Textbox(
                    label="音频评审 API Key", type="password", value=judge_cfg.get("api_key", ""),
                    info="评审会把生成的 WAV 发送到此处配置的接口。",
                )
                save_judge_btn = gr.Button("保存音频评审配置", scale=1)

            with gr.Row():
                goal_rounds = gr.Slider(1, 8, value=3, step=1, label="目标模式最多生成轮数")
                goal_use_current = gr.Checkbox(label="目标模式先评审当前成品", value=False)

        provider_box.change(
            fn=on_provider_change,
            inputs=[provider_box],
            outputs=[base_url_box, model_box],
        )

        goal_judge_provider.input(
            fn=on_judge_provider_change, inputs=[goal_judge_provider],
            outputs=[goal_judge_url, goal_judge_model, goal_judge_key],
        )
        save_judge_btn.click(
            fn=save_judge_config,
            inputs=[goal_judge_provider, goal_judge_url, goal_judge_model, goal_judge_key],
            outputs=[status_out],
        )

        save_cfg_btn.click(
            fn=_save_config,
            inputs=[provider_box, base_url_box, model_box, api_key_box],
            outputs=[status_out],
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
            director_message, director_chat, *synthesis_outputs, goal_baseline, goal_status, conversation_menu,
        ]
        send_message = partial(archive_conversation, director_chat_action)
        director_send.click(send_message, inputs=chat_inputs, outputs=conversation_outputs)
        director_message.submit(send_message, inputs=chat_inputs, outputs=conversation_outputs)
        director_task.click(
            partial(archive_conversation, partial(director_chat_action, production_task=True)),
            inputs=chat_inputs, outputs=conversation_outputs, api_name="dispatch_director_task",
        )
        gen_btn.click(archived_synthesis, inputs=[conversation_id, *synthesis_inputs], outputs=synthesis_outputs)
        director_generate.click(
            partial(archive_conversation, generate_from_conversation), inputs=conversation_inputs,
            outputs=conversation_outputs, api_name="generate_from_discussion",
        )

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

        library_inputs = [library_scope, conversation_id]
        library_outputs = [library_entry, library_audio, library_detail]
        demo.load(_library_choices, inputs=library_inputs, outputs=library_outputs)
        library_refresh.click(_library_choices, inputs=library_inputs, outputs=library_outputs)
        library_scope.change(_library_choices, inputs=library_inputs, outputs=library_outputs)
        library_entry.input(_library_entry, inputs=[library_entry], outputs=[library_audio, library_detail])
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
        allowed_paths=[str(ARCHIVE.attachments.resolve())],
    )


if __name__ == "__main__":
    main()
