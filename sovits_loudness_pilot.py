"""Train and test a local loudness condition before the SoVITS output nonlinearity."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io.wavfile import read, write
import torch

from acoustic_delivery import apply_delivery
from performance import ROOT
from synthesize import _render_one_segment
from webui_sora import get_tts
from sovits_loudness import LoudnessCondition


def pair_loss(model, wave, db):
    want = (wave * 10 ** (db / 20)).clamp(-0.98, 0.98)
    predicted = model(wave.clamp(-0.998, 0.998).atanh(), db).tanh()
    return (predicted - want).square().mean()


def train(model, device, epochs=1, steps=None, out=None):
    data = torch.load(ROOT / "colab_data/training-data-delivery.pt", weights_only=True)
    rng = np.random.default_rng(72)
    groups = {"train": [], "holdout": []}
    for uid, split, *_ in data["items"]:
        frames = data["frame_targets"][uid][1].nonzero().flatten().tolist()
        if not frames:
            continue
        sr, raw = read(ROOT / "prep/5-wav32k" / f"{uid}.wav")
        assert sr == 32000 and raw.dtype == np.int16 and raw.ndim == 1
        start = max(0, frames[len(frames)//2] * 1280 - 3200)
        if start + 6400 <= len(raw):
            groups[split].append(raw[start:start+6400].astype(np.float32) / 32768)
    held = torch.tensor(np.stack(groups["holdout"]), device=device)
    db = torch.tensor([-6.0, 4.0], device=device)[None, :, None]
    with torch.no_grad():
        before = pair_loss(model, held[:, None], db).item()
    opt = torch.optim.AdamW(model.parameters(), lr=0.012)
    count = len(groups["train"])
    steps = steps or (count+7)//8
    history = []
    for epoch in range(1, epochs+1):
        items = np.concatenate([rng.permutation(count) for _ in range((steps*8+count-1)//count)])[:steps*8]
        for pos in range(0, len(items), 8):
            wave = torch.tensor(np.stack([groups["train"][i] for i in items[pos:pos+8]]), device=device)
            loss = pair_loss(model, wave[:, None], db)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            after = pair_loss(model, held[:, None], db).item()
        metric = {"epoch":epoch,"step":epoch*steps,"holdout_mse":after}
        history.append(metric)
        if out:
            torch.save(model.state_dict(), out/f"loudness-epoch{epoch:02d}.pt")
            (out/"training.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
        print(json.dumps(metric),flush=True)
    return {"train":count,"holdout":len(held),"steps":epochs*steps,"epochs":epochs,
            "paired_mse_before":before,"paired_mse_after":after,"history":history}


@torch.inference_mode()
def compare(model, out):
    text = "今日はもう休んでいいよ。私はここにいるから、ゆっくり目を閉じて。"
    cuts = (0, text.index("。")+1, text.index("、")+1, len(text))
    plan = [dict(start=a, end=b, primary="tender", secondary=None, intensity=.55,
                 pause_s=pause, pitch_shift=0, energy_shift=shift)
            for a,b,pause,shift in zip(cuts[:-1], cuts[1:], (.4,.2,0), (0,.5,.5), strict=True)]
    tts = get_tts()
    post = tts.vits_model.dec.conv_post
    activations = []
    capture = post.register_forward_pre_hook(lambda _mod, args: activations.append(args[0]))
    try:
        dry, tokens, digest, _ = _render_one_segment(tts, text, plan, ref_id="SR011403",
            acoustic_ref_id="SR011455", auto_tags=False, emo_scale=1)
    finally:
        capture.remove()
    rate = tts.configs.sampling_rate
    assert len(activations) == 1
    _, alignment = apply_delivery(dry, rate, text, plan, device=tts.configs.device)
    regions = alignment["regions"]
    assert len(regions) == 2, alignment
    measures = {}
    for label, signs in (("front_soft", [-1,1]), ("back_soft", [1,-1])):
        db = torch.zeros(len(dry), device=tts.configs.device)
        for region, sign in zip(regions, signs, strict=True):
            lo, hi = (round(region[k]*rate) for k in ("start","end"))
            fade = min(round(.035*rate), (hi-lo)//2)
            db[lo:hi] = sign*4
            ramp = torch.linspace(0,1,fade,device=db.device)
            db[lo:lo+fade] *= ramp
            db[hi-fade:hi] *= ramp.flip(0)
        hook = post.register_forward_hook(lambda _m,_args,logits: model(logits.float(),db).to(logits))
        try:
            result = post(activations[0]).tanh()[0,0].float().cpu().numpy()
        finally:
            hook.remove()
        assert result.shape == dry.shape
        changes = []
        for region in regions:
            lo,hi = (round(region[k]*rate) for k in ("start","end"))
            changes.append(round(float(10*np.log10(np.mean(result[lo:hi]**2)/np.mean(dry[lo:hi]**2))),2))
        assert all(change * sign > 1 for change,sign in zip(changes,signs,strict=True))
        measures[label] = changes
        write(out/f"{label}.wav",rate,(result*32767).astype(np.int16))
    write(out/"baseline.wav",rate,(dry*32767).astype(np.int16))
    return {"text":text,"seconds":len(dry)/rate,"tokens":tokens,"semantic_sha256":digest,
            "regions":regions,"measured_gain_db":measures}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=1)
    parser.add_argument("--steps-per-epoch",type=int)
    parser.add_argument("--out",type=Path,default=ROOT/"results/sovits_loudness_pilot")
    parser.add_argument("--init",type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or (args.steps_per_epoch is not None and args.steps_per_epoch < 1):
        parser.error("epochs and steps must be positive")
    out = args.out
    out.mkdir(exist_ok=True)
    model = LoudnessCondition().cuda()
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=True))
    training = train(model, "cuda", args.epochs, args.steps_per_epoch, out)
    print(training, flush=True)
    torch.save(model.state_dict(),out/"loudness-condition.pt")
    report = {"training":training,"comparison":compare(model,out)}
    (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(report,flush=True)
