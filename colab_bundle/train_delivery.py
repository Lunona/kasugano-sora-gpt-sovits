"""Continue GPT training for local pitch and loudness controls."""

import argparse
import json
from pathlib import Path
import random

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from frame_prosody import FrameProsody, frame_batch
from performance import CHANNELS, EMOTIONS, install
from train_condition import Text2SemanticDecoder, batch, export_gpt_weights, objective, write_checkpoint
from vocal_tokens import VOCAL_TOKENS


def shift_controls(example):
    phones, semantic, controls, *targets = example
    changed = controls.clone()
    delivery = changed[CHANNELS:]
    voiced = delivery.abs().sum(dim=0) > 0
    count = int(voiced.sum())
    if count >= 3:
        delivery[:, voiced] = delivery[:, voiced].roll(max(1, count // 2), dims=1)
    return phones, semantic, changed, *targets


def remove_controls(example):
    phones, semantic, controls, *targets = example
    changed = controls.clone()
    changed[CHANNELS:] = 0
    return phones, semantic, changed, *targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--focus-vowels", action="store_true")
    parser.add_argument("--frame-prosody", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not torch.cuda.is_available():
        parser.error("Select a CUDA runtime, with positive epochs and batch size.")
    torch.manual_seed(72)
    device = torch.device("cuda")

    saved = torch.load(args.init, map_location="cpu", weights_only=True)
    config = saved["config"]
    channels = CHANNELS + 2
    model = Text2SemanticDecoder(config, top_k=3)
    model.load_state_dict({k.removeprefix("model."): v for k, v in saved["gpt_weight"].items()})
    model.to(device).eval()
    adapter = install(model, channels)
    with torch.no_grad():
        weights = saved["adapter"]["weight"]
        if weights.shape[1] not in (CHANNELS, channels):
            raise ValueError("Starting checkpoint has incompatible condition channels")
        adapter.weight[:, :weights.shape[1]].copy_(weights)
    del saved
    grad_mask = torch.zeros_like(adapter.weight)
    grad_mask[:, CHANNELS:] = 1
    adapter.weight.register_hook(lambda grad: grad * grad_mask)
    prediction_params = list(model.h.layers[-1].requires_grad_(True).parameters())
    prediction_params.extend(model.ar_predict_layer.requires_grad_(True).parameters())
    trainable = [adapter.weight, *prediction_params]
    optimizer = torch.optim.AdamW([
        {"params": [adapter.weight], "lr": 8e-3 if args.focus_vowels or args.frame_prosody else 2e-3, "weight_decay": 0},
        {"params": prediction_params, "lr": 2e-5, "weight_decay": 0},
    ])

    data = torch.load(args.data, map_location="cpu", weights_only=True)
    refs = data["references"]
    focus = data["vowel_frames"]
    rows = [(row[1], (*row[2:], focus[row[0]], *data["frame_targets"][row[0]])) for row in data["items"]]
    train = [row for split, row in rows if split == "train"]
    valid = [row for split, row in rows if split == "holdout"]
    assert train and valid
    print(f"Delivery training: {len(train)} targets, {len(valid)} held out, {len(refs)} references.", flush=True)

    probe = None
    acoustic_weights = None
    if args.frame_prosody:
        probe = FrameProsody().to(device).eval().requires_grad_(False)
        state = torch.load(args.frame_prosody, map_location="cpu", weights_only=True)
        probe.load_state_dict(state["state_dict"])
        print(f"Frame supervision R² (pitch, loudness): {state['scores']}", flush=True)
        acoustic_weights = torch.tensor([float(score >= 0.10) for score in state["scores"]], device=device)
        if not acoustic_weights.any():
            raise ValueError("Acoustic frame targets did not generalize on held-out tokens")

    def loss_for(examples, ref, local=False, logits=False):
        token_weights = None
        if local:
            max_len = max(row[1].numel() for row in examples)
            token_weights = torch.zeros(len(examples), max_len + 1)
            for i, row in enumerate(examples):
                token_weights[i, :row[3].numel()] = row[3]
            token_weights = token_weights.to(device)
        return objective(
            model, batch([row[:3] for row in examples], device, True, fixed_ref=ref), device,
            ref_sem_len=len(ref[1]), mask_padding=True, token_weights=token_weights, return_token_logits=logits,
        )

    def acoustic_supervision(examples, ref, logits):
        shifted = list(map(shift_controls, examples))
        _, counterfactual_logits = loss_for(shifted, ref, logits=True)
        _, target, voiced, valid_frames = frame_batch(
            [(row[1], row[5], row[6]) for row in examples], device
        )
        desired_deltas = []
        for row, alternative in zip(examples, shifted):
            owners = row[4]
            differences = alternative[2][CHANNELS:] - row[2][CHANNELS:]
            desired_deltas.append(differences[:, owners.clamp_min(0)].T)
        delta = pad_sequence(desired_deltas, batch_first=True).transpose(1, 2).to(device)
        target_shifted = (target + delta).clamp(-1, 1)
        actual = probe.from_logits(logits, valid_frames)
        counterfactual = probe.from_logits(counterfactual_logits, valid_frames)
        actual_error = F.smooth_l1_loss(actual, target, reduction="none")
        revised_error = F.smooth_l1_loss(counterfactual, target_shifted, reduction="none")
        delta_error = F.smooth_l1_loss(counterfactual-actual, target_shifted-target, reduction="none")
        weight = voiced[:, None] * acoustic_weights[None, :, None]
        return ((actual_error + revised_error + delta_error) * weight).sum() / weight.sum().clamp_min(1)

    @torch.no_grad()
    def validate():
        result = {}
        variants = (("true", valid), ("shifted", list(map(shift_controls, valid))),
                    ("zero", list(map(remove_controls, valid))))
        for local in (False, True) if args.focus_vowels else (False,):
            for variant, rows in variants:
                vals = [loss_for(rows[i:i+args.batch_size], refs["SR010709"], local=local).item()
                        for i in range(0, len(rows), args.batch_size)]
                key = f"local_{variant}" if local else variant
                result[key] = round(sum(vals) / len(vals), 5)
        return result

    if args.smoke:
        sub = [min(train, key=lambda x: len(x[1]))]
        loss = loss_for(sub, refs["SR010709"])
        loss.backward()
        assert torch.isfinite(loss) and bool(adapter.weight.grad[:, CHANNELS:].abs().sum() > 0)
        optimizer.step()
        print(f"Delivery CUDA smoke passed: loss={loss.item():.4f}", flush=True)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    before = validate()
    print(json.dumps({"epoch": 0, "validation": before}), flush=True)
    ref_pool = tuple(refs.values())
    step = 0
    for epoch in range(1, args.epochs + 1):
        order = list(train)
        rng = random.Random(72 + epoch)
        rng.shuffle(order)
        for i in range(0, len(order), 60):
            order[i:i+60] = sorted(order[i:i+60], key=lambda row: len(row[1]))
        for offset in range(0, len(order), args.batch_size):
            subset = order[offset:offset+args.batch_size]
            ref = rng.choice(ref_pool)
            optimizer.zero_grad(set_to_none=True)
            if probe is not None:
                loss, logits = loss_for(subset, ref, logits=True)
                loss = loss + 0.85 * acoustic_supervision(subset, ref, logits)
            else:
                loss = loss_for(subset, ref)
            if args.focus_vowels and probe is None:
                compare = sorted(subset, key=lambda row: int(row[3].sum()), reverse=True)[:2]
                true_loss = loss_for(compare, ref, local=True)
                wrong_loss = loss_for(list(map(shift_controls, compare)), ref, local=True)
                loss = loss + 0.25 * true_loss + 0.5 * (true_loss - wrong_loss + 0.12).relu()
            elif probe is None and step % 4 == 0:
                example = rng.choice(subset)
                shifted = shift_controls(example)
                if not torch.equal(example[2], shifted[2]):
                    true_loss = loss_for([example], ref)
                    wrong_loss = loss_for([shifted], ref)
                    loss = loss + 0.15 * (true_loss - wrong_loss + 0.05).relu()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite delivery loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            step += 1
            if step % 25 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f}", flush=True)
        after = validate()
        print(json.dumps({"epoch": epoch, "step": step, "validation": after}), flush=True)
        write_checkpoint(args.out / f"phrase-adapter-epoch{epoch}.pt", {
            "gpt_weight": export_gpt_weights(model), "adapter": {"weight": adapter.weight.detach().cpu()},
            "config": config, "emotions": EMOTIONS, "vocal_tokens": VOCAL_TOKENS,
            "steps": step, "train_items": len(train), "delivery_controls": ["pitch", "energy"],
            "frame_prosody_supervision": acoustic_weights.tolist() if acoustic_weights is not None else None,
            "loss": {"before": before, "validation": after},
        })


if __name__ == "__main__":
    main()
