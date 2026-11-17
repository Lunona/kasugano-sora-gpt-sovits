"""Reconstruct complete script scenes, and associate contextual and waveform evidence."""
import csv
import json
import re
import sys
import wave
from collections import Counter
from pathlib import Path

import numpy as np
from context_labels import AFFECT, NEGATED, acts, local_affect

BASE = Path('../春日野穹_语音数据')
OUT = Path('colab_data')
sys.path.insert(0,str(Path('../提取工作').resolve()))
from xp3 import index, extract
import zlib

def read_scenes():
    path=next(Path('..').glob('*/data.xp3'))
    with path.open('rb') as stream:
        for name,_,parts in index(stream):
            if not name.endswith('.ks'):
                continue
            blob=extract(stream,parts)
            clear=blob.translate(bytes(i^parts[b'adlr'][0] for i in range(256)))
            assert zlib.adler32(clear)==int.from_bytes(parts[b'adlr'],'little')
            lines=clear.decode('cp932').splitlines()
            matches=list(i for i,t in enumerate(lines) if t.startswith('@Talk '))
            turns=[]
            for i in matches:
                end=i+1
                while end<len(lines) and not lines[end].startswith(('@','*',';')):
                    end+=1
                turns.append((i+1,re.search(r'name=(\S+)',lines[i])[1],''.join(lines[i+1:end]).strip()))
            yield name,turns

TOPICS={
    '就餐／做饭':r'ご飯|昼ご飯|昼休み|弁当|料理|食べ|台所|お湯|冷蔵庫',
    '上学／学校':r'学校|授業|制服|教室|宿題|先生|登校',
    '日常安排／出行':r'買い物|出かけ|出掛け|帰る|電車|駅|支度|洗濯|出発',
    '家中设备／生活':r'エアコン|ネット|電話|お風呂|テレビ|洗面所|布団',
    '争执／关系':r'邪魔|近づ|許せ|無かったこと|終わりにし|関係|傷つけ|家族',
    '事故／健康':r'溺れ|怪我|大丈夫|病院|具合|死んじゃ|心配|危な',
    '失落／悼念':r'亡くな|葬式|お墓|死んだ|事故で|父さん|母さん',
}
ADULT=re.compile(r'自慰|クリトリス|恥丘|陰茎|精液|射精|膣|性器|愛撫')
OBJECTIVE=re.compile(r'^.{0,38}(?:ある|ない|あった|だった|してる|する|できる|出来る|だよ|だね|だもの|です|だったよ|だったから|だけ)[…。‥]*$')
ACTIONS={
 'anger':r'怒鳴|不機嫌|睨みつけ|苛立|怒ったよう',
 'hurt':r'涙を|泣き|うなだれ|声を詰ま',
 'fear':r'震え|怯え|おびえ|背中に隠',
 'joy':r'笑顔|笑い|嬉しそう|楽しそう',
 'shy':r'照れ|顔を赤|頬を赤|顔は、?真っ赤',
 'tender':r'穏やか|優しい声|やさしい声',
 'surprise':r'目を丸|驚いたよう',
}

def power(path):
    with wave.open(str(path),'rb') as f:
        rate=f.getframerate()
        sig=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(np.float32)/32768
    n=rate//100
    frames=sig[:len(sig)//n*n].reshape(-1,n)
    if not len(frames):
        return {}
    rms=np.sqrt(np.mean(frames**2,axis=1))
    lvl=20*np.log10(np.maximum(rms,1e-5))
    active=lvl>max(-58.,float(lvl.max())-30.)
    if not active.any():
        return {}
    active_ids=np.flatnonzero(active)
    edges=np.flatnonzero(np.diff(np.r_[False,active,False]))
    # Short isolated quiet sounds are review candidates only, not detected breaths.
    median=float(np.median(lvl[active]))
    isolated=[]
    for i,(a,b) in enumerate(zip(edges[::2],edges[1::2])):
        if i==0 or i==len(edges)//2-1 or not 8<=b-a<=45:
            continue
        if a-edges[2*i-1]>=9 and edges[2*i+2]-b>=9 and np.median(lvl[a:b])<median-7:
            isolated.append([round(a*n/rate,3),round(b*n/rate,3)])
    return {'active_dbfs':round(median,1),'dynamic_db':round(float(np.ptp(np.percentile(lvl[active],[15,85]))),1),
            'sound_candidates':isolated}

def context_for(scene,i,ac,row):
    line,speaker,text=scene[i]
    surrounding=scene[max(0,i-14):i+15]
    nearby=scene[max(0,i-4):i+5]
    theme=Counter()
    for l,sp,t in surrounding:
        for topic,pattern in TOPICS.items():
            if re.search(pattern,t):
                theme[topic]+=2 if (l,sp,t) in nearby else 1
    adult_positions=[l for l,_,t in surrounding if ADULT.search(t)]
    label,basis=local_affect(text,row['context'],row['delivery_cues'])
    previous=scene[i-1] if i else None
    following=scene[i+1] if i+1<len(scene) else None
    mentions=[]
    for j in range(max(0,i-3),min(len(scene),i+4)):
        if j==i:
            continue
        l,sp,t=scene[j]
        if sp!='心の声' or re.search(NEGATED,t):
            continue
        for affect,pattern in ACTIONS.items():
            if re.search(r'穹[はがのも、][^。！？]{0,24}(?:'+pattern+r')',t) or re.search(r'(?:'+pattern+r')[^。！？]{0,12}穹',t):
                mentions.append((affect,l))
    inferred=list(dict.fromkeys(x[0] for x in mentions))
    if label is None and len(inferred)==1:
        label=inferred[0]
        basis='场景旁白明确描写她的反应'
    has_events=bool(row['events']) or row['utterance_type']!='常规台词'
    daily=[t for t,n in theme.most_common() if t in list(TOPICS)[:4] and n>=4]
    strong=re.search(r'[！!]|っ{2,}',text) or any(x in row['delivery_cues'] for x in ('激动／强调','语调起伏明显','感叹／强调'))
    speech=acts(text,row['context'],row['utterance_type'])
    if label is None and daily and not has_events and not strong and not adult_positions and not any(theme[k]>=5 for k in list(TOPICS)[4:]):
        plain=text.strip('…。‥　 ')
        plain_act=any(a in speech for a in ('疑问／确认','回应上一句提问','应答','提议／邀约','请求'))
        if plain_act or OBJECTIVE.search(plain) and ac.get('dynamic_db',40)<24:
            label='neutral'
            basis='日常场景中的问答、安排或事实陈述，暂定日常语气'
    if label is None and following and previous and not has_events and not adult_positions and not strong:
        if following[1] not in ('穹','心の声') and re.search(r'(?:ですか|ますか|どこ|いつ|どうやって).{0,18}[？?]$',text):
            label='neutral'
            basis='询问信息，下一轮由其他角色作答，暂定日常语气'
    return dict(label=label,basis=basis,topics=theme.most_common(3),
                cue_lines=mentions,adult_context_lines=adult_positions,acoustic=ac,
                unresolved='' if label or row['emotion_labels'] else ('以发声反应为主，具体情绪待试听' if has_events else '语境可确定说话用途，情绪方向仍需听音'))

def main():
    rows=list(map(json.loads,(OUT/'逐条表演标注.jsonl').read_text(encoding='utf8').splitlines()))
    scenes=dict(read_scenes())
    indexes={s:{line:i for i,(line,_,_) in enumerate(ts)} for s,ts in scenes.items()}
    results={}
    for r in rows:
        scene=scenes[r['script']]
        i=indexes[r['script']][r['script_line']]
        assert scene[i][2]==r['text_ja']
        ac=power(BASE/r['audio'])
        results[r['id']]=context_for(scene,i,ac,r)
    summary=dict(scripts=len(scenes),turns=sum(map(len,scenes.values())),
        used_scenes=len({r['script'] for r in rows}),records=len(results))
    (OUT/'场景语境审查.json').write_text(json.dumps(results,ensure_ascii=False),encoding='utf8')
    (OUT/'场景语境统计.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(summary)
    print(Counter(results[r['id']]['label'] for r in rows if r['split']=='train' and not r['emotion_labels']))
    print('waveform candidates',sum(bool(v['acoustic'].get('sound_candidates')) for v in results.values()))

if __name__=='__main__':
    main()
