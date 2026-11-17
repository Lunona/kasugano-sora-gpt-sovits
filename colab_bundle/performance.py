"""Japanese phrase conditioning for the existing GPT-SoVITS v2 semantic decoder."""
import difflib
import json
import os
from pathlib import Path
import re
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent
APP = Path(os.environ.get("SORAVOICE_APP", "F:/GPT-SoVITS/app"))
DATA = Path(os.environ.get("SORAVOICE_DATA", ROOT.parent / "春日野穹_语音数据"))
PRE = APP / "GPT_SoVITS/pretrained_models"
S1 = PRE / "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
S2 = PRE / "gsv-v2final-pretrained/s2G2333k.pth"
tmp = ROOT / "tmp"
tmp.mkdir(exist_ok=True)
os.environ.update(TEMP=str(tmp), TMP=str(tmp), HF_HOME=str(ROOT/"cache"), HF_HUB_OFFLINE="1", USE_LIBUV="0")
os.environ["PATH"] = str(APP/"runtime") + os.pathsep + os.environ["PATH"]
sys.path[:0] = [str(APP/"GPT_SoVITS"), str(APP/"GPT_SoVITS/eres2net"), str(APP)]
import torch
from torch import nn

EMOTIONS = ("neutral","anger","hurt","fear","joy","shy","tender","surprise","resolve","impatient")
CHANNELS = len(EMOTIONS)*2 + 6
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
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
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
    if not text.strip() or "\n" in text or len(text)>300 or not 1<=len(plan)<=8:
        raise ValueError("Expected one Japanese utterance and 1–8 contiguous spans")
    cursor = 0
    for span in plan:
        if (span["start"]!=cursor or not cursor<span["end"]<=len(text)
                or not text[cursor:span["end"]].strip()):
            raise ValueError("Spans must cover the original text exactly")
        for key in ("primary","secondary"):
            if span[key] is not None and span[key] not in EMOTIONS:
                raise ValueError(f"Unknown emotion {span[key]!r}")
        if not 0<=span["intensity"]<=1 or not 0<=span["pause_s"]<=1:
            raise ValueError("Invalid intensity or pause")
        cursor=span["end"]
    if cursor!=len(text):
        raise ValueError("Missing text at end of plan")

def features(text, plan):
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
    cond=torch.zeros(CHANNELS,len(full))
    for i,span in enumerate(plan):
        cols=[j for j,owner in enumerate(assignment) if owner==i]
        if not cols:
            continue
        for key,start in (("primary",0),("secondary",len(EMOTIONS))):
            if span[key]:
                cond[start+EMOTIONS.index(span[key]),cols]=1
        for extra in span.get("additional_emotions", []):
            if extra not in (span["primary"],span["secondary"]):
                cond[len(EMOTIONS)+EMOTIONS.index(extra),cols]=1
        cond[len(EMOTIONS)*2,cols]=span["intensity"]
        cond[len(EMOTIONS)*2+1,cols[-1]]=span["pause_s"]
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

def plan_with_ollama(text, context):
    cuts=[0,*[m.end() for m in re.finditer(r"[。！？!?、，]+|[…‥]{2,}",text) if m.end()<len(text)],len(text)]
    chunks=[text[a:b] for a,b in zip(cuts,cuts[1:])]
    if len(chunks)>8:
        cuts=cuts[:8]+[len(text)]
        chunks=[text[a:b] for a,b in zip(cuts,cuts[1:])]
    spec={"type":"object","properties":{
        "primary":{"enum":[None,*EMOTIONS]},"secondary":{"enum":[None,*EMOTIONS]},
        "intensity":{"type":"number","minimum":0,"maximum":1},
        "pause_s":{"type":"number","minimum":0,"maximum":1}},
        "required":["primary","secondary","intensity","pause_s"]}
    schema={"type":"object","properties":{"segments":{"type":"array","minItems":len(chunks),
             "maxItems":len(chunks),"items":spec}},"required":["segments"]}
    prompt=("你是日语配音导演。按序给每个片段一个表演计划，优先采纳上下文中明确的演法。表情通常延续到下一片段，遇到明确转折才换。 "
            "The chunks only locate possible prosody changes, the utterance will be synthesized as one continuous take. "
            "例：先责怪但委屈，后松一口气。前段 primary=anger secondary=hurt intensity=0.7，转折前 pause_s=0.3，后段 primary=tender secondary=null intensity=0.35。禁止一律最高强度。 "
            "Use zero if the boundary should be spoken continuously. Last pause is zero. JSON only.\n"
            +json.dumps({"context":context,"chunks":dict(enumerate(chunks,1))},ensure_ascii=False))
    payload=dict(model=os.environ.get("SORAVOICE_PLANNER","qwen2.5:3b"),prompt=prompt,format=schema,
                 stream=False,keep_alive=0,options={"temperature":0,"num_gpu":0,"num_predict":700})
    request=urllib.request.Request("http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload,ensure_ascii=False).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(request,timeout=95) as result:
        plan=json.loads(json.load(result)["response"])["segments"]
    for span,a,b in zip(plan,cuts[:-1],cuts[1:],strict=True):
        span.update(start=a,end=b)
    plan[-1]["pause_s"]=0
    validate(text,plan)
    return plan
