"""Contextual evidence for dialogue annotation. Physical delivery and affect stay separate."""
import re
from collections import defaultdict

AFFECT = {
    "anger": r"怒鳴|怒りを|苛立|いらだ|不機嫌|睨みつけ|にらみつけ|声を荒げ",
    "hurt": r"泣き出|すすり泣|涙を(?:流|浮かべ)|泣きそう|悲しそう|寂しそう",
    "fear": r"怯え|おびえ|怖が|震えた声|恐る恐る|背中に隠れ",
    "joy": r"笑顔|笑みを浮かべ|嬉しそう|うれしそう|楽しそう",
    "shy": r"照れ隠し|照れくさそう|恥ずかしそう|頬を赤らめ|顔を真っ赤|顔は、?真っ赤",
    "tender": r"優しい声|やさしい声|穏やかに|愛おしそう",
    "surprise": r"目を丸く|驚いたよう|驚きの声|びっくりしたよう",
}
SPEECH_ACTS = {
    "道歉": r"ごめん|すみません|悪かった",
    "致谢": r"ありがと|感謝",
    "拒绝／制止": r"やめて|いらない|いやだ|嫌だ|近づかないで|しないで|^嫌[。…]*$",
    "提议／邀约": r"しよう|行こう|食べよ|ないか[？?]|どう[？?]",
    "请求": r"てください|お願い|てくれる|てくれない|ってよ[。…]?$",
    "疑问／确认": r"[？?]|(?:なの|ですか|でしょ|どこ|かしら)[…。‥]*$",
    "应答": r"^[っ…‥。「　\s]*(?:うん|はい|そう|いいえ|ええ|わかった|違う|別に)[。…‥」　\s]*$",
    "呼唤": r"^[っ…‥。「　\s]*(?:ハル|悠)[っッ！!？?。…‥」　\s]*$",
}
SORA = r"穹|そら"
NEGATED = r"わけではな|わけじゃな|とは思えな|のではな|かもしれ|のだろう"
PARTICIPANT = r"悠|僕|奈緒|瑛|亮平|初佳|一葉|委員長"

def contextual_affect(context):
    evidence=defaultdict(list)
    for turn in context:
        if abs(turn["offset"])>2 or turn["speaker"]!="心の声":
            continue
        for sentence in re.split(r"[。！？]",turn["text"]):
            if not re.search(SORA,sentence) or re.search(NEGATED,sentence):
                continue
            # Only attribute a cue when Sora is an explicit subject or possessive.
            for emotion,pattern in AFFECT.items():
                direct=rf"(?:穹[はがのも])(?:(?!{PARTICIPANT}).){{0,18}}(?:{pattern})"
                trailing=rf"(?:{pattern})(?:(?!{PARTICIPANT}).){{0,15}}(?:穹)(?:$|[、…])"
                if re.search(direct,sentence) or re.search(trailing,sentence):
                    evidence[emotion].append(f'旁白 {turn["offset"]:+} 明示穹的反应')
    return evidence

def acts(text, context, kind):
    result=[label for label,pattern in SPEECH_ACTS.items() if re.search(pattern,text)]
    if not result and kind!="常规台词":
        result=["反应音／带发声事件"]
    if not result:
        previous=next((x for x in context if x["offset"]==-1),None)
        if previous and previous["speaker"] not in {"穹","心の声"} and re.search(r"[？?]$",previous["text"]):
            result=["回应上一句提问"]
    return result or ["陈述／具体用途待确认"]


def local_affect(text, context, style):
    text=text.strip("…‥。　 ")
    previous=next((x for x in context if x["offset"]==-1),None)
    if re.search(r"^(?:ごめん(?:なさい)?|すみません|私が悪かった)",text):
        return "hurt","台词在道歉或自责"
    if re.search(r"^(?:え[っッぇー]*|うそ|嘘)(?:[、…‥！!?？]|$)|^(?:本当|ほんと)[？?]",text):
        return "surprise","以惊讶式确认开头"
    if re.search(r"楽しみ|嬉しい|うれしい|やった[！!]|わぁ[！!]",text):
        return "joy","台词表达期待或喜悦"
    if re.search(r"恥ずかし|照れくさ",text):
        return "shy","台词明确表达难为情"
    if re.search(r"^(?:やめて|近づかないで|邪魔しないで|帰って)[！!]|うるさい|許さない",text):
        return "anger","强制止或斥责的文字线索"
    if re.search(r"怖[いく]|こわ[いく]|助けて|どうしよう",text):
        return "fear","台词明示恐惧或求助"
    if re.search(r"^(?:ありがとう|ありがとね|大丈夫だよ|そばにいて)",text):
        return "tender","明确致谢、安慰或寻求陪伴"
    if previous and previous["speaker"] not in {"穹","心の声"}:
        if re.fullmatch(r"(?:うん|はい|そう|分かった|わかった|別に|そうなの[？?]?)",text):
            if not any(x in style for x in ("激动／强调","感叹／强调","语调起伏明显")):
                return "neutral","回应对方的简短应答，暂按日常语气"
    return None,""