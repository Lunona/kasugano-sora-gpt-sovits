import csv
import json
import re
from collections import Counter
from pathlib import Path

from performance import ROOT, DATA, CHANNELS, annotated_plan, features, load_annotations, torch
from train import HOLDOUT
from context_labels import contextual_affect, acts, local_affect

EVENTS = {
    "气声／短呼吸": r"(?<![ぁ-ん])(?:は[ぁあー]{1,}|ふ[ぅうー]{1,}|ハ[ァアー]{1,}|ほぅ)[っッ]?(?=[…‥、。！？!?　 ]|$)",
    "短促呼声": r"(?<![ぁ-ん])(?:あっ|ひゃっ|きゃっ|うっ|んっ|わっ|あぅっ)(?=[…‥、。！？!?　 ]|$)",
    "持续发声": r"([ぁあぅうぉお])\1{2,}|[んン][ぅうー]{3,}",
    "口部拟声": r"ちゅ[っッ]?|チュ[ッっ]?|ぺろ[っッ]?|パク[ッっ]?|ぱく[っッ]?|んぐ[っッ]?",

    "咳嗽": r"ゴホ[ッっ]?|ゲホ|[けケこコ][ほホ][ッっ]|咳|咳き",
    "喘息或急促呼吸": r"(?:は[ぁあー]+[っッ]?|ふ[ぅうー]+[っッ]?|ハ[ァアー]+[ッっ]?){2,}|息切れ",
    "笑声": r"(?:あはは|ふふ[っッ]?|えへへ|うふふ)",
    "哭声": r"うっく|う[っッ]?う[ぅう]{2,}|ひっく|すすり泣",
}
SECONDARY_CUES = {
    "anger":r"やめて|近づかないで|邪魔|許さない|うるさい",
    "hurt":r"ごめん|さびし|寂し|悲し|つらい|苦し",
    "fear":r"怖い|こわい|助けて|どうしよう",
    "joy":r"うれしい|嬉しい|やった[！!]|楽し",
    "shy":r"恥ずかし|照れくさ",
    "tender":r"大好き|ありがとう|そばにいて|優し",
    "surprise":r"まさか|本当に|嘘でしょ|どうして",
}
LABEL = dict(neutral="日常／平静", anger="生气／抗拒",hurt="悲伤／委屈",fear="害怕／紧张",
             joy="开心",shy="害羞／迟疑",tender="温柔／关心",surprise="惊讶／疑惑",resolve="坚定",impatient="不耐／催促")
REACTION = re.compile(r'^[ぁあぃいぅうぇえぉおっッんンはハふフぅー…‥！？!?、。\s]+$')

def make_plan(row, story):
    text=row["text_ja"]
    plan=annotated_plan(row)
    events=[]
    for kind,pattern in EVENTS.items():
        for m in re.finditer(pattern,text):
            events.append(dict(kind=kind,start=m.start(),end=m.end(),text=m.group(),
                               audio_time=None,status="文本候选，需听校"))
    events.sort(key=lambda e:(e["start"],e["end"]))
    original = [span.copy() for span in plan]
    context = contextual_affect(row["context"])
    notes=[]
    for span in plan:
        fragment=text[span["start"]:span["end"]]
        cues=[label for label,pattern in SECONDARY_CUES.items() if re.search(pattern,fragment)]
        if not span["primary"] and len(cues)==1:
            span["primary"]=cues[0]
            notes.append("局部词汇")
        if span["primary"] and not span["secondary"] and len(cues)==1 and cues[0]!=span["primary"]:
            span["secondary"]=cues[0]
            notes.append("局部次情绪候选")
        if not span["primary"] and len(context)==1:
            span["primary"]=next(iter(context))
            notes.extend(context[span["primary"]])
        elif span["primary"] in context:
            notes.extend(context[span["primary"]])
        if not span["primary"] and not events:
            label,evidence=local_affect(fragment,row["context"],row["delivery_cues"])
            if label:
                span["primary"]=label
                notes.append(evidence)
        candidates = cues + (list(context) if len(context)==1 else [])
        span["additional_emotions"]=list(dict.fromkeys(
            c for c in candidates if c not in (span["primary"],span["secondary"])))
        span["labels"]=list(dict.fromkeys(c for c in
            [span["primary"],span["secondary"],*span["additional_emotions"]] if c))
        if not span["labels"]:
            span["emotion_status"]="证据不足"
        else:
            span["emotion_status"]="待听校"
    if story.get("label") and not any(span["labels"] for span in plan):
        for span in plan:
            span["primary"]=story["label"]
            span["labels"]=[story["label"]]
            span["emotion_status"]="场景预判，待听校"
        notes.append(story["basis"])
    # Ambiguous lines retain explicit unknown labels. Physical reactions can coexist with emotion.
    kind="常规台词"
    if events:
        kind="有非言语事件线索"
    elif REACTION.fullmatch(text) or "反应音／拟声候选" in row["delivery_cues"]:
        kind="疑似反应音"
    return plan,events,kind,notes,original

def main():
    source=load_annotations()
    story=json.loads((ROOT/"colab_data/场景语境审查.json").read_text(encoding="utf-8"))
    compiled=[]
    manifest=[]
    table=[]
    for line in (ROOT/"prep/6-name2semantic-0.tsv").read_text(encoding="utf-8").splitlines():
        name,codes=line.split("\t")
        row=source[name.removesuffix(".wav")]
        if not row["initial_training_candidate"] or not codes or not row["text_ja"]:
            continue
        text=row["text_ja"]
        scene_review=story[row["id"]]
        plan,events,kind,notes,original=make_plan(row,scene_review)
        phone,condition=features(text,plan)
        semantic=list(map(int,codes.split()))
        assert semantic and (DATA/row["audio"]).is_file()
        split="holdout" if row["script"] in HOLDOUT else "train"
        transitions=[]
        for a,b in zip(plan,plan[1:]):
            if a["primary"] and b["primary"] and a["primary"]!=b["primary"]:
                transitions.append({"offset":b["start"],"before":a["primary"],"after":b["primary"],
                                    "anchor":text[b["start"]:b["end"]],"status":"文本推断，待听校"})
        value=dict(row)
        value.update(split=split,spans=plan,events=events,utterance_type=kind,positioned_transitions=transitions,
                     span_status="逐条自动预标注，待听校",span_basis=sorted(set(notes)),
                     preexisting_span_plan=original,scene_review=scene_review,
                     speech_acts=acts(text,row["context"],kind),
                     emotion_labels=list(dict.fromkeys(e for span in plan for e in span["labels"])))
        manifest.append(value)
        compiled.append((row["id"],split,torch.tensor(phone),torch.tensor(semantic),condition))
        table.append({
            "语音编号":row["id"],"用途":split,"场景判断":row["scene"],"场景依据":row["scene_basis"],
            "整句源主标签":row["emotion"]["primary"] or "待判断","整句源次标签":row["emotion"]["secondary"] or "",
            "日文":text,"中文":row["text_zh"],"完整WAV":row["audio"],"脚本":row["script"],
            "对话功能":"、".join(value["speech_acts"]),"情绪依据":"、".join(value["span_basis"]),
            "发声类型":kind,"发声线索":"、".join(row["delivery_cues"]),
            "整句多标签":"＋".join(LABEL[e] for e in value["emotion_labels"]),
            "分段情绪":" ｜ ".join(f'{text[s["start"]:s["end"]]} [{"＋".join(LABEL[e] for e in s["labels"]) or "待判断"}｜强度{s["intensity"]}｜停顿提示{s["pause_s"]}秒]' for s in plan),
            "转折文字位置":"、".join(str(x["offset"]) for x in transitions),
            "发声事件候选":json.dumps(events,ensure_ascii=False),
            "波形短声候选秒":json.dumps(scene_review["acoustic"].get("sound_candidates",[])),
            "场景主题":"、".join(topic for topic,_ in scene_review["topics"]),
            "情绪仍待听校原因":scene_review["unresolved"],
            "波形低能量时间段":json.dumps(row["audio_low_energy_intervals"],ensure_ascii=False),
            "复核状态":"自动预标注，待听校" if value["emotion_labels"] else "已标语用或发声，情绪待听校",
        })
    counts=Counter(x["split"] for x in manifest)
    assert counts=={"train":2575,"holdout":93}
    out=ROOT/"colab_data"
    out.mkdir(exist_ok=True)
    with (out/"逐条表演标注.jsonl").open("w",encoding="utf-8",newline="\n") as f:
        for item in manifest:
            f.write(json.dumps(item,ensure_ascii=False)+"\n")
    with (out/"逐条表演复核.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    torch.save(compiled,out/"training-data.pt")
    loaded=torch.load(out/"training-data.pt",weights_only=True)
    assert len({x[0] for x in loaded})==2668 and all(x[4].shape==(CHANNELS,len(x[2])) for x in loaded)
    stats={"split":dict(counts)}
    for split in counts:
        rs=[r for r in manifest if r["split"]==split]
        stats[split]={"spans":sum(len(r["spans"]) for r in rs),
          "all_unknown":sum(not r["emotion_labels"] for r in rs),
          "context_evidence":sum(any("旁白" in n for n in r["span_basis"]) for r in rs),
          "transitions":sum(bool(r["positioned_transitions"]) for r in rs),
          "event_text":sum(bool(r["events"]) for r in rs),
          "event_waveform":sum(bool(r["scene_review"]["acoustic"].get("sound_candidates")) for r in rs),
          "event_union":sum(bool(r["events"] or r["scene_review"]["acoustic"].get("sound_candidates")) for r in rs),
          "three_or_more":sum(len(r["emotion_labels"])>=3 for r in rs),
          "three_in_one_span":sum(any(len(s["labels"])>=3 for s in r["spans"]) for r in rs),
          "scene":dict(Counter(r["scene"] for r in rs)),
          "events":dict(Counter(e["kind"] for r in rs for e in r["events"]))}
    (out/"标注统计.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding="utf-8")
    print(stats)

if __name__=="__main__":
    main()
