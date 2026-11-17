import csv
import json
import subprocess
import sys
import wave
from collections import Counter
from performance import ROOT, DATA, EMOTIONS, features, torch

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()]

def main():
    data = ROOT / "colab_data"
    rows = read_jsonl(data / "逐条表演标注.jsonl")
    before = read_jsonl(data / "backup_before_auto/逐条表演标注.jsonl")
    assert [r["id"] for r in rows] == [r["id"] for r in before]
    assert len({r["id"] for r in rows}) == len(rows) == 2668
    for new, old in zip(rows, before, strict=True):
        for key in ("id", "audio", "text_ja", "split", "duration_s", "scene"):
            assert new[key] == old[key], (new["id"], key)
    train = [r for r in rows if r["split"] == "train"]
    assert len(train) == 2575
    outs = data / "auto_review_batches/outputs"
    batch_paths = sorted(outs.glob("batch_*.json"))
    assert [p.stem for p in batch_paths] == [f"batch_{i:03d}" for i in range(129)]
    reviewed = [item for p in batch_paths for item in json.loads(p.read_text(encoding="utf-8-sig"))]
    assert len(reviewed) == 2575 and {r["id"] for r in reviewed} == {r["id"] for r in train}
    raw = torch.load(data / "training-data.pt", weights_only=True)
    prior = {item[0]: item for item in raw}
    assert set(prior) == {r["id"] for r in rows}
    semantic = {
        name.removesuffix(".wav"): list(map(int, tokens.split()))
        for line in (ROOT/"prep/6-name2semantic-0.tsv").read_text(encoding="utf-8").splitlines()
        for name, tokens in [line.split("\t")]
    }
    issues = []
    counts = Counter()
    final = []
    for row in rows:
        id = row["id"]
        old = prior[id]
        assert old[1] == row["split"] and old[3].tolist() == semantic[id]
        if row["split"] != "train":
            final.append(old)
            continue
        text = row["text_ja"]
        assert row["emotion_labels"] and set(row["emotion_labels"]) <= set(EMOTIONS)
        plan = [dict(s) for s in row["spans"]]
        assert all(set(s["labels"]) <= set(EMOTIONS) for s in plan)
        with wave.open(str(DATA / row["audio"]), "rb") as f:
            duration = f.getnframes() / f.getframerate()
        assert abs(duration - row["duration_s"]) < .001
        for span in plan:
            if span["end"] == len(text) and span["pause_s"]:
                counts["句末暂停未作为句内提示"] += 1
                span["pause_s"] = 0
            elif span["pause_s"] > 1:
                counts["句内停顿按接口上限截为1秒"] += 1
                span["pause_s"] = 1
        phone, cond = features(text, plan)
        assert phone == old[2].tolist() and torch.isfinite(cond).all(), id
        counts["训练张量已变化"] += int(not torch.equal(cond, old[4]))
        final.append((id, old[1], old[2], old[3], cond))
        for event in row["events"]:
            a,b = event["audio_time"]
            assert 0 <= event["start"] <= event["end"] <= len(text)
            assert 0 <= a < b <= duration + .01, id
            if b > duration:
                issues.append((id,"事件结束秒略超原音频",str(b),f"{duration:.6f}"))
            anchor = text[event["start"]:event["end"]]
            if anchor != event["text"]:
                issues.append((id,"事件描述与字符区间原文不同",event["text"],anchor))
        by_start = {s["start"]: s for s in row["spans"][1:]}
        for event in row["positioned_transitions"]:
            assert event["offset"] in by_start, id
            if not text[event["offset"]:].startswith(event["anchor"]):
                issues.append((id,"转折锚点描述与该位置原文不同",event["anchor"],text[event["offset"]:]))
    counts.update({"训练语音":len(train),"留出语音":len(rows)-len(train),
                   "三个及以上情绪":sum(len(r["emotion_labels"])>=3 for r in train),
                   "含事件语音":sum(bool(r["events"]) for r in train),
                   "事件":sum(len(r["events"]) for r in train),
                   "含转折语音":sum(bool(r["positioned_transitions"]) for r in train)})
    counts.update(issue[1] for issue in issues)
    assert not any(issue[1] in {"转折锚点描述与该位置原文不同", "事件结束秒略超原音频"} for issue in issues)
    target = data / "training-data.pt"
    backup = data / "training-data_before_gemini.pt"
    if not backup.exists():
        import shutil
        shutil.copyfile(target, backup)
    temp = target.with_suffix(".pending")
    torch.save(final,temp)
    check = torch.load(temp,weights_only=True)
    assert len(check)==2668 and all(torch.equal(x[4], y[4]) for x,y in zip(final,check,strict=True))
    temp.replace(target)
    with (data/"Gemini标注核验明细.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["语音编号","差异","标注值","音频或脚本核对值"])
        writer.writerows(issues)
    (data/"Gemini标注核验统计.json").write_text(json.dumps(counts,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(counts,ensure_ascii=False))

if __name__ == "__main__":
    main()
