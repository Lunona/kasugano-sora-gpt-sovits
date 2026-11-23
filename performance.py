"""Japanese phrase conditioning for the existing GPT-SoVITS v2 semantic decoder."""
import difflib
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
import urllib.request
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
APP = Path(os.environ.get("SORAVOICE_APP", "F:/GPT-SoVITS/app"))
DATA = Path(os.environ.get("SORAVOICE_DATA", ROOT.parent / "春日野穹_语音数据"))
PRE = APP / "GPT_SoVITS/pretrained_models"
S1 = PRE / "s1v3.ckpt"
S2 = PRE / "v2Pro/s2Gv2ProPlus.pth"
tmp = ROOT / "tmp"
tmp.mkdir(exist_ok=True)
os.environ.update(TEMP=str(tmp), TMP=str(tmp), HF_HOME=str(ROOT/"cache"), HF_HUB_OFFLINE="1", USE_LIBUV="0")
os.environ["PATH"] = str(APP/"runtime") + os.pathsep + os.environ["PATH"]
sys.path[:0] = [str(APP/"GPT_SoVITS"), str(APP/"GPT_SoVITS/eres2net"), str(APP)]
import torch
from torch import nn

EMOTIONS = ("neutral","anger","hurt","fear","joy","shy","tender","surprise","resolve","impatient")
DELIVERIES = ("plain", "restrained", "soft", "tearful", "trembling", "emphatic", "bright", "hesitant", "startled")
DIRECTOR_EVENTS = ("breath", "gasp", "hold", "oral", "pant", "sob", "laugh")
CHANNELS = len(EMOTIONS)*2 + 6
DELIVERY_CHANNELS = CHANNELS + 2
DELIVERY_PRESETS = {
    "restrained": (-0.2, -0.6),
    "soft": (0.0, -0.65),
    "emphatic": (0.45, 0.65),
}
LABELS = {
    "平静":"neutral","日常":"neutral","问答":"neutral","否定":"neutral","冷淡":"neutral",
    "生气":"anger","抗拒":"anger","愤怒":"anger","不满":"anger","责怪":"anger","质问":"anger",
    "委屈":"hurt","悲伤":"hurt","痛苦":"hurt","自责":"hurt","遗憾":"hurt","失落":"hurt","惋惜":"hurt",
    "害怕":"fear","紧张":"fear","后怕":"fear","慌乱":"fear","担心":"fear","不安":"fear","戒备":"fear",
    "开心":"joy","轻快":"joy","欣喜":"joy","笑意":"joy","期待":"joy","满意":"joy",
    "害羞":"shy","犹豫":"shy","迟疑":"shy",
    "亲昵":"tender","温柔":"tender","关心":"tender","依恋":"tender","安抚":"tender","撒娇":"tender",
    "惊讶":"surprise","震惊":"surprise","疑惑":"surprise","意外":"surprise","困惑":"surprise",
    "坚定":"resolve","坚持":"resolve","笃定":"resolve","决绝":"resolve",
    "不耐":"impatient","催促":"impatient","急切":"impatient","埋怨":"impatient",
}

def normalize_label(label):
    return next((value for part,value in LABELS.items() if part in (label or "")), None)

def phones(text):
    import sys
    sys.path.insert(0, str(ROOT / "colab_bundle"))
    from vocal_tokens import VOCAL_TAG_RE, VOCAL_TOKEN_IDS
    from text.cleaner import clean_text
    from text import cleaned_text_to_sequence

    out = []
    for part in VOCAL_TAG_RE.split(text):
        if not part:
            continue
        if part in VOCAL_TOKEN_IDS:
            out.append(VOCAL_TOKEN_IDS[part])
        elif part.strip():
            symbols, _, _ = clean_text(part, "ja", "v2")
            out.extend(cleaned_text_to_sequence(symbols, "v2"))
    return out

def validate(text, plan):
    if not text.strip() or "\n" in text or len(text)>10000 or len(plan)<1:
        raise ValueError("Expected non-empty text without newlines and at least 1 contiguous span")
    cursor = 0
    for number, span in enumerate(plan, 1):
        if not isinstance(span, dict):
            raise ValueError(f"第 {number} 段必须是 JSON 对象")
        required = ("start", "end", "primary", "secondary", "intensity", "pause_s")
        missing = [key for key in required if key not in span]
        if missing:
            raise ValueError(f"第 {number} 段缺少字段：{', '.join(missing)}")
        start, end = span["start"], span["end"]
        if (type(start) is not int or type(end) is not int or start != cursor
                or not start < end <= len(text) or not text[start:end].strip()):
            raise ValueError(f"第 {number} 段的 start/end 无效：需要从字符 {cursor} 开始，且不能超出台词")
        for key in ("primary", "secondary"):
            if span[key] is not None and span[key] not in EMOTIONS:
                raise ValueError(f"第 {number} 段的 {key} 情绪无效：{span[key]!r}")
        for key in ("intensity", "pause_s"):
            val = span[key]
            if type(val) not in (int, float) or not 0 <= val <= 1:
                raise ValueError(f"第 {number} 段的 {key} 必须是 0 到 1 之间的数字")
        cursor = end
    if cursor != len(text):
        raise ValueError(f"分段只覆盖到字符 {cursor}，请输入完整台词的分段")

def phone_span_owners(text, plan):
    validate(text,plan)
    full=phones(text)
    blocks=[phones(text[s["start"]:s["end"]]) for s in plan]
    joined=[ph for block in blocks for ph in block]
    if not full or not joined:
        raise ValueError("No usable Japanese phonemes")
    owners=[i for i,block in enumerate(blocks) for _ in block]
    matcher=difflib.SequenceMatcher(a=joined,b=full,autojunk=False)
    assignment=[]
    for opcode,a,b,c,d in matcher.get_opcodes():
        if opcode=="equal":
            assignment.extend(owners[a:b])
        elif d>c:
            assignment.extend(owners[min(len(owners)-1,a+int(j*max(1,b-a)/(d-c)))] for j in range(d-c))
    return full, assignment


def delivery_values(span):
    pitch, energy = DELIVERY_PRESETS.get(str(span.get("delivery")), (0.0, 0.0))
    values = []
    for key, default in (("pitch_shift", pitch), ("energy_shift", energy)):
        raw = span.get(key)
        value = float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else default
        values.append(max(-1.0, min(1.0, value)) if value == value else default)
    return tuple(values)


def delivery_tremble(span):
    value = span.get("tremble_strength")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value == value:
        return max(0.0, min(1.0, float(value)))
    return 0.68 if span.get("delivery") == "trembling" else 0.0


def features(text, plan, *, with_delivery=False):
    full, assignment = phone_span_owners(text, plan)
    cond=torch.zeros(DELIVERY_CHANNELS if with_delivery else CHANNELS,len(full))
    if with_delivery:
        from text import symbols2
        vowels = {i for i, symbol in enumerate(symbols2.symbols) if symbol in frozenset("aeiouN")}
    for i,span in enumerate(plan):
        cols=[j for j,owner in enumerate(assignment) if owner==i]
        if not cols:
            continue
        weights=span.get("emotion_weights") or {}
        for key,w_key,start in (("primary","primary_weight",0),("secondary","secondary_weight",len(EMOTIONS))):
            if span[key]:
                w=float(span.get(w_key, weights.get(span[key], 1.0)))
                cond[start+EMOTIONS.index(span[key]),cols]=w
        for extra in span.get("additional_emotions", []):
            if extra not in (span["primary"],span["secondary"]):
                cond[len(EMOTIONS)+EMOTIONS.index(extra),cols]=float(weights.get(extra, 1.0))
        cond[len(EMOTIONS)*2,cols]=span["intensity"]
        cond[len(EMOTIONS)*2+1,cols[-1]]=span["pause_s"]
        if with_delivery:
            selected = [j for j in cols if full[j] in vowels]
            if selected:
                for channel, value in enumerate(delivery_values(span), CHANNELS):
                    cond[channel, selected] = value
    return full,cond

class PerformanceAdapter(nn.Module):
    def __init__(self, projection, channels=CHANNELS):
        super().__init__()
        self.projection = projection
        self.performance = nn.Linear(channels, projection.out_features, bias=False)
        nn.init.zeros_(self.performance.weight)

    def forward(self, tensor):
        return self.projection(tensor) + self.performance(tensor[..., :self.performance.in_features])


def install(decoder, channels=CHANNELS):
    decoder.requires_grad_(False)
    decoder.bert_proj = PerformanceAdapter(decoder.bert_proj, channels)
    dev = decoder.bert_proj.projection.weight.device
    dtype = decoder.bert_proj.projection.weight.dtype
    decoder.bert_proj.performance.to(device=dev, dtype=dtype)
    return decoder.bert_proj.performance

def load_annotations():
    p=DATA/"整句表演标注"/"整句表演标注.jsonl"
    return {r["id"]:r for r in map(json.loads,p.read_text(encoding="utf-8").splitlines())}

@lru_cache(maxsize=1)
def _reference_texts(path, signature):
    rows = map(json.loads, path.read_text(encoding="utf-8").splitlines())
    return {row["id"]: row["text_ja"] for row in rows}


@lru_cache(maxsize=256)
def _reference_phones(text):
    return tuple(phones(text))


def reference_phones(ref_id):
    path = DATA / "整句表演标注" / "整句表演标注.jsonl"
    stat = path.stat()
    return _reference_phones(_reference_texts(path, (stat.st_mtime_ns, stat.st_size, stat.st_ino))[ref_id])


PHRASE_OVERRIDES=json.loads((ROOT/"phrase-overrides.json").read_text(encoding="utf-8-sig"))

def annotated_plan(row):
    text=row["text_ja"]
    emotion=row["emotion"]
    primary=normalize_label(emotion["primary"])
    secondary=normalize_label(emotion["secondary"])
    if secondary==primary:
        secondary=None
    # Candidate, positional labels are used only when both sides have distinct lexical evidence.
    switches={}
    for turn in row["transition_candidates"]:
        a=normalize_label(next(iter(turn["before"]),None))
        b=normalize_label(next(iter(turn["after"]),None))
        if a and b and a!=b:
            switches[turn["offset"]]=(a,b)
    if row["id"] in PHRASE_OVERRIDES:
        anchor,before,after=PHRASE_OVERRIDES[row["id"]]
        offset=text.index(anchor)
        assert offset>0 and text.count(anchor)==1
        switches[offset]=(before,after)
    stops={p["offset"]:p["mark"] for p in row["text_pause_candidates"]}
    breaks=sorted(set(switches)|{offset for offset,mark in stops.items() if mark!="、"})
    breaks=[p for p in breaks if 0<p<len(text) and text[:p].strip() and text[p:].strip()][:7]
    origin=0
    current=primary
    spans=[]
    # Duration values are weak text-based targets, independent of unaligned waveform intervals.
    for end in breaks+[len(text)]:
        if not text[origin:end].strip():
            continue
        upcoming=switches.get(end)
        region=upcoming[0] if upcoming else current
        marks=stops.get(end,"")
        pause=.35 if re.search(r"[…。‥！？!?]",marks) else .12 if marks else 0
        intensity=.8 if re.search(r"[！!]",text[origin:end]) else .45
        spans.append(dict(start=origin,end=end,primary=region,secondary=secondary if region!=secondary else None,intensity=intensity,pause_s=pause))
        origin=end
        if upcoming:
            current=upcoming[1]
    return spans

def _split_director_chunks(text):
    cuts=[0,*[m.end() for m in re.finditer(r"[。！？!?、，]+|[…‥]{2,}",text) if m.end()<len(text)],len(text)]
    chunks=[text[a:b] for a,b in zip(cuts,cuts[1:])]
    if len(chunks)>16:
        cuts=[0,*[m.end() for m in re.finditer(r"[。！？!?]+|[…‥]{2,}",text) if m.end()<len(text)],len(text)]
        chunks=[text[a:b] for a,b in zip(cuts,cuts[1:])]
    return cuts, chunks

def _director_prompt(chunks, context):
    return ("你是日语配音导演，为春日野穹的台词逐段分配真实情绪。可选情绪："
            "neutral(平淡日常), anger(怒斥/生气), hurt(委屈/受伤/哭诉/质问), fear(害怕/不安), joy(开心), "
            "shy(害羞/扭捏), tender(温柔/柔声), surprise(惊讶), resolve(决意/坚定认真/追问), impatient(不耐烦/急躁)。\n"
            "规则：1. 结合context(记忆库与前情对话)判断潜台词，标点和语气词只是线索，可以自然说。傲娇可能是轻微嘴硬、平静掩饰或局促辩解，结合前情区分。没有明确要求时避免自动加结巴和颤声；要准确区分 impatient、hurt、resolve 与 anger；"
            "2. pause_s按实际演法安排，省略号可以是短顿、拖尾或吞回话。连贯辩解可短顿或不断开，普通逗号通常0~0.15秒。最后一段pause_s必须为0；"
            "3. 情绪强度intensity在0.35~0.90之间；可额外给出primary_weight(0.4~1.0)和secondary_weight(0.0~0.8)控制主次情绪混合比例。\n"
            "4. delivery描述这段的说话方式，选 plain(普通), restrained(克制压声), soft(轻柔), "
            "tearful(带哭腔说话), trembling(声音发颤), emphatic(突然提高音量或重声), bright(轻快带笑意), "
            "hesitant(迟疑结巴), startled(受惊)。委屈可以平静说，害怕也可以压住声音，根据前情选实际听得出的表现。\n"
            "5. 结合前情决定是否插入独立拟声或发声事件。event可选 breath(短换气), gasp(短促惊呼), "
            "hold(持续哼鸣), oral(口部拟声，如咂嘴), pant(急促喘息), sob(抽泣), laugh(独立笑声)。"
            "说话拖尾和延长元音用语气及停顿表达，不用hold；普通傲娇或省略号不自动加气声、咂嘴或笑声。"
            "台词已经写出的「ふん」「えっ」是原文的一部分，按原文读，不叠加同一声。"
            "每段至多一个事件。event_before必须是在该段只出现一次、含实际字词的原文片段，事件放在它前面。"
            "不需要独立发声时，event和event_before都填null。保留台词原文。\n"
            "6. 如本段明确需要另外提降音高或加减响度，分别填pitch_shift、energy_shift，范围-1~1。"
            "0表示这段无额外调整；null表示沿用delivery的自然设定。轻声不一定要降音高，突然提高音高也不一定要大声。"
            "如果这项说话方式只作用在本段几个字上，delivery_text填要调整的原文连续片段，必须在本段只出现一次；作用于整段时填null。"
            "确实需要发颤时可以填 tremble_strength（0 至 1），其余情况填 null。\n"
            f"输出严格JSON：{{\"segments\": [共{len(chunks)}个对象，每项含 primary, secondary(可为null), primary_weight, secondary_weight, intensity, pause_s, delivery, event, event_before, pitch_shift, energy_shift, delivery_text, tremble_strength]}}。JSON only.\n"
            +json.dumps({"context":context,"chunks":dict(enumerate(chunks,1))},ensure_ascii=False))

def _scope_delivery(text, span, anchor):
    start, end = span["start"], span["end"]
    if not isinstance(anchor, str) or not anchor.strip():
        return [span]
    chunk = text[start:end]
    index = chunk.find(anchor)
    if index < 0 or chunk.find(anchor, index + 1) >= 0:
        return [span]
    left, right = start + index, start + index + len(anchor)
    if not text[start:left].strip():
        left = start
    if not text[right:end].strip():
        right = end
    scoped = []
    for a, b in ((start, left), (left, right), (right, end)):
        if a == b:
            continue
        segment = dict(span, start=a, end=b, pause_s=span["pause_s"] if b == end else 0.0)
        if (a, b) != (left, right):
            segment.update(delivery="plain", pitch_shift=0.0, energy_shift=0.0, tremble_strength=0.0)
        if not a <= segment.get("event_at", -1) < b:
            segment.pop("event", None)
            segment.pop("event_at", None)
        scoped.append(segment)
    return scoped


def _finalize_director_plan(text, cuts, plan):
    valid_emo = set(EMOTIONS)
    result = []
    for span,a,b in zip(plan,cuts[:-1],cuts[1:],strict=True):
        if span.get("primary") not in valid_emo:
            span["primary"] = "neutral"
        if span.get("secondary") not in valid_emo or span.get("secondary") == span["primary"]:
            span["secondary"] = None
        span["intensity"] = float(max(0.0, min(1.0, span.get("intensity", 0.5))))
        span["pause_s"] = float(max(0.0, min(1.0, span.get("pause_s", 0.0))))
        if span.get("delivery") not in DELIVERIES:
            span["delivery"] = "plain"
        event, anchor = span.pop("event", None), span.pop("event_before", None)
        span.pop("event_at", None)
        chunk = text[a:b]
        if event in DIRECTOR_EVENTS and isinstance(anchor, str) and any(c.isalnum() for c in anchor):
            match = chunk.find(anchor)
            if match >= 0 and chunk.find(anchor, match + 1) == -1:
                span.update(event=event, event_at=a + match)
        span.update(start=a,end=b)
        result.extend(_scope_delivery(text, span, span.pop("delivery_text", None)))
    result[-1]["pause_s"]=0
    validate(text,result)
    return result

def _director_plan_schema(count):
    spec={"type":"object","properties":{
        "primary":{"enum":[None,*EMOTIONS]},"secondary":{"enum":[None,*EMOTIONS]},
        "intensity":{"type":"number","minimum":0,"maximum":1},
        "pause_s":{"type":"number","minimum":0,"maximum":1},
        "delivery":{"enum":list(DELIVERIES)},
        "event":{"enum":[None,*DIRECTOR_EVENTS]},
        "pitch_shift":{"type":["number","null"],"minimum":-1,"maximum":1},
        "energy_shift":{"type":["number","null"],"minimum":-1,"maximum":1},
        "tremble_strength":{"type":["number","null"],"minimum":0,"maximum":1},
        "delivery_text":{"type":["string","null"]},
        "event_before":{"type":["string","null"]}},
        "required":["primary","secondary","intensity","pause_s","delivery","event","event_before"]}
    return {"type":"object","properties":{"segments":{"type":"array","minItems":count,
             "maxItems":count,"items":spec},
             "emo_scale":{"type":"number"}, "whisper_mode":{"type":"boolean"},
             "declick_deess":{"type":"boolean"}, "ambient_bed":{"enum":["none","rain"]},
             "ambient_level":{"type":"number"}},"required":["segments"]}


def plan_with_ollama(text, context):
    cuts, chunks = _split_director_chunks(text)
    schema = _director_plan_schema(len(chunks))
    payload=dict(model=os.environ.get("SORAVOICE_PLANNER","qwen2.5:3b"),prompt=_director_prompt(chunks, context),format=schema,
                 stream=False,keep_alive=0,options={"temperature":0,"num_gpu":0,"num_predict":min(2400, 180 * len(chunks) + 80)})
    request=urllib.request.Request("http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload,ensure_ascii=False).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(request,timeout=95) as result:
        plan=json.loads(json.load(result)["response"])["segments"]
    return _finalize_director_plan(text, cuts, plan)

def report_director_usage(operation, model, usage, started, *, first_content=None):
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("prompt_tokens_details")
    details = details if isinstance(details, dict) else {}
    hit = usage.get("prompt_cache_hit_tokens", details.get("cached_tokens"))
    miss = usage.get("prompt_cache_miss_tokens")
    prompt = usage.get("prompt_tokens")
    if type(miss) is not int and type(prompt) is int and type(hit) is int:
        miss = max(0, prompt - hit)
    rate = None
    if type(hit) is int and type(miss) is int and hit + miss > 0:
        rate = round(hit / (hit + miss), 4)
    metrics = {
        "operation": operation, "model": model, "prompt_tokens": prompt,
        "cache_hit_tokens": hit, "cache_miss_tokens": miss, "cache_hit_rate": rate,
        "completion_tokens": usage.get("completion_tokens"),
        "seconds": round(perf_counter() - started, 3),
    }
    if first_content is not None:
        metrics["first_content_seconds"] = round(first_content - started, 3)
    try:
        print("[导演 API] " + json.dumps(metrics, ensure_ascii=False), flush=True)
    except (OSError, UnicodeError):
        pass


OLLAMA_DIRECTOR_URL = "http://127.0.0.1:11434/v1"
DIRECTOR_STRING = {"type": "string", "minLength": 1}


def is_local_ollama(url):
    return (url or "").strip().rstrip("/").removesuffix("/chat/completions") == OLLAMA_DIRECTOR_URL


def director_json_schema(**properties):
    return {"type": "object", "properties": properties, "required": list(properties)}


def _director_response_format(url, schema):
    if schema and is_local_ollama(url):
        return {"type": "json_schema", "json_schema": {"name": "director_result", "schema": schema}}
    return {"type": "json_object"}


def deepseek_json(system, prompt, *, api_key=None, base_url=None, model=None, timeout=120, operation="导演请求", schema=None):
    import time
    import urllib.error

    key = api_key or os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    url = (base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    mdl = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    payload = {
        "model": mdl,
        "temperature": 0.1,
        "response_format": _director_response_format(url, schema),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "Connection": "close",
        },
    )
    last_exc = None
    for attempt in range(3):
        started = perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            report_director_usage(operation, mdl, body.get("usage") if isinstance(body, dict) else None, started)
            content = body["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("Missing JSON content")
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
            result = json.loads(content)
            if not isinstance(result, dict):
                raise ValueError("Expected a JSON object")
            return result
        except (urllib.error.URLError, OSError, TimeoutError, ValueError, KeyError, IndexError, TypeError) as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(0.8 * (attempt + 1))
                continue
            if isinstance(exc, (ValueError, KeyError, IndexError, TypeError)):
                raise RuntimeError("API 连续多次未返回可解析的 JSON") from exc
            raise last_exc


def _extract_partial_json_reply(buf: str) -> str | None:
    """Incrementally unescape the string value of `"reply"` from a partial JSON buffer."""
    m = re.search(r'"reply"\s*:\s*"', buf)
    if not m:
        return None
    i = m.end()
    out = []
    n = len(buf)
    while i < n:
        ch = buf[i]
        if ch == '"':
            break
        if ch == "\\":
            if i + 1 >= n:
                break
            nxt = buf[i + 1]
            esc = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
            if nxt in esc:
                out.append(esc[nxt])
                i += 2
            elif nxt == "u":
                if i + 6 > n:
                    break
                try:
                    out.append(chr(int(buf[i + 2 : i + 6], 16)))
                except ValueError:
                    break
                i += 6
            else:
                out.append(nxt)
                i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def deepseek_json_stream(system, prompt, *, api_key=None, base_url=None, model=None, timeout=120, schema=None):
    """Stream SSE chunks from DeepSeek JSON mode, yielding (partial_reply, None) then (full_reply, parsed_dict)."""
    key = api_key or os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    url = (base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    mdl = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    payload = {
        "model": mdl,
        "temperature": 0.1,
        "stream": True,
        "response_format": _director_response_format(url, schema),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    if urlsplit(url).hostname == "api.deepseek.com":
        payload["stream_options"] = {"include_usage": True}
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "Connection": "close",
        },
    )
    started = perf_counter()
    first_content = None
    usage = None
    buf = []
    last_partial = ""
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    if isinstance(chunk, dict) and chunk.get("usage") is not None:
                        usage = chunk["usage"]
                    delta = chunk["choices"][0]["delta"].get("content") or ""
                except (ValueError, KeyError, IndexError, TypeError):
                    continue
                if delta:
                    if first_content is None:
                        first_content = perf_counter()
                    buf.append(delta)
                    partial = _extract_partial_json_reply("".join(buf))
                    if partial is not None and partial != last_partial:
                        last_partial = partial
                        yield partial, None
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", "".join(buf).strip(), flags=re.IGNORECASE)
        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("reply"), str):
            raise ValueError("Missing reply in streamed JSON")
        report_director_usage("导演讨论", mdl, usage, started, first_content=first_content)
        yield parsed["reply"].strip(), parsed
    except Exception:
        parsed = deepseek_json(system, prompt, api_key=api_key, base_url=base_url, model=model, timeout=timeout, operation="导演讨论回退", schema=schema)
        reply = parsed["reply"].strip() if isinstance(parsed.get("reply"), str) else ""
        yield reply, parsed


def load_sora_persona() -> str:
    """Load the live Kasugano Sora character & voice-acting guide from sora_persona.md if present."""
    path = ROOT / "sora_persona.md"
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def plan_with_deepseek(text, context="", api_key=None, base_url=None, model=None):
    from math import isfinite

    cuts, chunks = _split_director_chunks(text)
    persona = load_sora_persona()
    system_prompt = (
        "你是只输出合法JSON的日语配音导演，严禁输出任何多余解释文字。"
        "在 segments 同级额外给出 emo_scale（正数，1.0为正常倍率，轻声或克制可降低）、whisper_mode（布尔值）、"
        "declick_deess（布尔值，是否开启轻度去刺耳柔化高频气沙与尖刺）、"
        "ambient_bed（环境底景，可选 \"none\"、\"rain\" 窗外雨声）。"
        "当 ambient_bed 为 \"rain\" 时，可额外给出 ambient_level（0.08~0.32 浮点数，轻声耳语配微雨用 0.08~0.14，普通雨夜用 0.16~0.22，大雨渲染用 0.24~0.30）。"
        "如果是贴耳轻语、睡前低语、撒娇私语等近麦 ASMR 场景，将 whisper_mode 设为 true，否则设为 false。"
        "当台词包含较多喘息叹气（「はぁ」「ふぅ」「あっ」「んっ」）、密集气口、哭腔或贴耳轻语，需要柔化气噪与清辅音尖刺时，将 declick_deess 设为 true，否则设为 false。"
    )
    local = is_local_ollama(base_url)
    if persona and not local:
        system_prompt += f"\n\n【春日野穹角色与配音表演基准档案】\n{persona}"
    batch_size = 6 if local else 20
    if len(chunks) <= (batch_size if local else 25):
        batch_ranges = [(0, len(chunks))]
    else:
        batch_ranges = [(i, min(i + batch_size, len(chunks))) for i in range(0, len(chunks), batch_size)]

    all_segments = []
    scales = []
    whisper_votes = []
    deess_votes = []
    bed_votes = []
    bed_levels = []
    prev_tail = None
    for start_idx, end_idx in batch_ranges:
        sub_chunks = chunks[start_idx:end_idx]
        batch_ctx = context
        if len(batch_ranges) > 1:
            tail_note = f"，上一段结尾情绪：{prev_tail}" if prev_tail else ""
            batch_ctx = f"{context}\n[长台词分段规划：共 {len(chunks)} 段，当前第 {start_idx + 1}..{end_idx} 段{tail_note}]".strip()
        response = None
        for attempt in range(2):
            resp = deepseek_json(
                system_prompt,
                _director_prompt(sub_chunks, batch_ctx),
                api_key=api_key, base_url=base_url, model=model, operation="表演规划",
                schema=_director_plan_schema(len(sub_chunks)),
            )
            segs = resp.get("segments")
            if isinstance(segs, list) and len(segs) == len(sub_chunks):
                response = resp
                break
            if attempt == 1:
                got = len(segs) if isinstance(segs, list) else type(segs).__name__
                raise RuntimeError(f"导演规划分段数不匹配：期望 {len(sub_chunks)}，返回 {got}")
        segs = response["segments"]
        all_segments.extend(segs)
        last = segs[-1]
        prev_tail = f"{last.get('primary')}/{last.get('secondary')} intensity={last.get('intensity')} delivery={last.get('delivery')}"
        sc = response.get("emo_scale")
        if type(sc) in (int, float) and isfinite(sc) and sc > 0:
            scales.append(float(sc))
        if isinstance(response.get("whisper_mode"), bool):
            whisper_votes.append(response["whisper_mode"])
        if isinstance(response.get("declick_deess"), bool):
            deess_votes.append(response["declick_deess"])
        if response.get("ambient_bed") == "rain":
            bed_votes.append(response["ambient_bed"])
            al = response.get("ambient_level")
            if type(al) in (int, float) and isfinite(al):
                bed_levels.append(max(0.06, min(0.35, float(al))))

    plan = _finalize_director_plan(text, cuts, all_segments)
    if scales:
        plan[0]["director_emo_scale"] = round(sum(scales) / len(scales), 2)
    if any(whisper_votes):
        plan[0]["director_whisper_mode"] = True
    if deess_votes:
        plan[0]["director_declick_deess"] = any(deess_votes)
    if bed_votes:
        plan[0]["director_ambient_bed"] = bed_votes[0]
    if bed_levels:
        plan[0]["director_ambient_level"] = round(sum(bed_levels) / len(bed_levels), 2)
    return plan



def rank_references_with_deepseek(text, context, plan, candidates, *, count, api_key, base_url, model):
    data = {"text": text, "context": context, "plan": plan, "references": candidates, "count": count}
    persona = load_sora_persona()
    sys_prompt = (
        "你是日语配音选角导演。根据完整台词、前情和所有片段的主次情绪及权重，决定整句的表演方向和参考音频。"
        "写一句话概括整句情绪如何混合和变化。参考要能支撑整句话，避免围绕单一高强度片段选取。"
        "例如扭捏可以是开头克制嘴硬、中途故作自然、句尾露出害羞，不必概括为迟疑。"
        "素材只有文字、声学画像(acoustic_profile)和历史标注，你没有听到原声。历史语气标注可能有误，只能作为弱线索；"
        "请重点参考 acoustic_profile 里的实测语速(快/中/慢字/s)与 F0 音高跨度：慢速低落/委屈长句务必选语速慢或中且情绪相符的参考，切忌用快嘴吵架参考去配慢悲剧。"
        "优先考虑台词在做什么（例如否认、找借口、掩饰关心）、情绪强度及节奏适配。"
        "不用迎合旧标签，不要从标点断言音频有结巴、颤声或具体停顿。"
        '只输出 JSON：{"overview":"中文概括整句演法","ranking":[{"id":"候选编号","reason":"中文简述适用理由或风险"}]}。'
        "按适配度列出恰好 count 个不同候选，禁止添加未提供的编号。先给整句首选，再给有明确区别的备选。"
    )
    if persona and not is_local_ollama(base_url):
        sys_prompt += f"\n\n【春日野穹角色与配音表演基准档案】\n{persona}"
    entry_schema = director_json_schema(id={"enum": [candidate["id"] for candidate in candidates]},
                                        reason=DIRECTOR_STRING)
    result = deepseek_json(
        sys_prompt,
        json.dumps(data, ensure_ascii=False),
        api_key=api_key, base_url=base_url, model=model, operation="参考音排序",
        schema=director_json_schema(overview=DIRECTOR_STRING, ranking={
            "type": "array", "minItems": count, "maxItems": count, "items": entry_schema,
        }),
    )
    overview = result["overview"]
    if not isinstance(overview, str) or not overview.strip():
        raise ValueError("API 未提供整句演法")
    reasons = {}
    allowed = {candidate["id"] for candidate in candidates}
    for item in result["ranking"]:
        ref_id = item["id"]
        reason = item["reason"]
        if ref_id not in allowed or ref_id in reasons or not isinstance(reason, str):
            raise ValueError("API 返回无效或重复的参考编号")
        reasons[ref_id] = " ".join(reason.split())[:160]
    if len(reasons) != count or not all(reasons.values()):
        raise ValueError("API 返回的参考数量或理由无效")
    return " ".join(overview.split())[:200], reasons
