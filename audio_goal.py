"""Evaluate generated takes against a voice direction with an audio-capable API."""

import base64
import json
import re
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit


GEMINI_PROVIDER = "Google Gemini 官方"
OPENAI_PROVIDER = "OpenAI 官方"
JUDGE_PRESETS = {
    GEMINI_PROVIDER: ("https://generativelanguage.googleapis.com/v1beta", "gemini-3.8-flash"),
    OPENAI_PROVIDER: ("https://api.openai.com/v1", "gpt-audio-1.5"),
    "自定义 OpenAI 兼容接口": ("", ""),
    "自定义 Gemini 兼容接口": ("", ""),
}


def review_takes(paths, line, goal, *, api_key, model, provider=GEMINI_PROVIDER, base_url=None):
    if provider not in JUDGE_PRESETS:
        raise ValueError("请选择音频评审厂商")
    gemini = provider in (GEMINI_PROVIDER, "自定义 Gemini 兼容接口")
    if not model.strip() or (gemini and not re.fullmatch(r"[a-zA-Z0-9._-]+", model)):
        raise ValueError("评审模型名无效")
    base = (base_url or JUDGE_PRESETS[provider][0]).strip().rstrip("/")
    parsed = urlsplit(base)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.query or parsed.fragment:
        raise ValueError("请填写有效的音频评审 API Base URL")
    files = [Path(path) for path in paths]
    prompt = (
        "你是日语配音质检。实际听完每条录音，依据同一目标独立评分。"
        "检查台词是否完整、发音、停顿、响度、情绪变化和目标中特别要求的演法。"
        "只依据录音里听得见的内容。feedback 用中文指出具体问题和下一轮可调的方向，尽量定位到词。"
        "只在台词完整且目标关键要求都达到时标 met=true。"
        "输出 JSON 对象 reviews，其中每条包含 take（从1开始的编号）、score（0到100）、met（布尔值）、feedback（字符串）。"
        "每条录音恰好评一次。\n"
        + json.dumps({"验收目标": goal, "日语台词": line}, ensure_ascii=False)
    )
    recordings = [base64.b64encode(file.read_bytes()).decode("ascii") for file in files]
    headers = {"Content-Type": "application/json"}
    if gemini:
        parts = [{"text": prompt}]
        for index, data in enumerate(recordings, 1):
            parts.extend([
                {"text": f"录音 {index}"},
                {"inlineData": {"mimeType": "audio/wav", "data": data}},
            ])
        endpoint = f"{base}/models/{model}:generateContent"
        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseMimeType": "application/json"},
        }
        headers["x-goog-api-key"] = api_key
    else:
        parts = [{"type": "text", "text": prompt}]
        for index, data in enumerate(recordings, 1):
            parts.extend([
                {"type": "text", "text": f"录音 {index}"},
                {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}},
            ])
        endpoint = f"{base}/chat/completions"
        body = {"model": model, "modalities": ["text"], "messages": [{"role": "user", "content": parts}]}
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(endpoint, data=json.dumps(body).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(request, timeout=90) as response:
        result = json.load(response)
    if gemini:
        response_parts = result["candidates"][0]["content"]["parts"]
        answer_text = "".join(part.get("text", "") for part in response_parts if not part.get("thought"))
    else:
        answer_text = result["choices"][0]["message"]["content"]
    answer_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer_text.strip(), flags=re.IGNORECASE)
    answer = json.loads(answer_text)
    reviews = {}
    for review in answer["reviews"]:
        index, score, met, feedback = (review[key] for key in ("take", "score", "met", "feedback"))
        if (
            type(index) is not int or index not in range(1, len(files) + 1) or index in reviews
            or type(score) not in (int, float) or not 0 <= score <= 100
            or type(met) is not bool or not isinstance(feedback, str) or not feedback.strip()
        ):
            raise ValueError("音频评审返回了无效结果")
        reviews[index] = (score, met, " ".join(feedback.split())[:500])
    if len(reviews) != len(files):
        raise ValueError("音频评审缺少候选结果")
    return [reviews[index] for index in range(1, len(files) + 1)]
