"""Train both the phrase-emotion condition adapter AND the GPT semantic prediction backbone."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random

import torch

from performance import CHANNELS, EMOTIONS, install
from AR.models.t2s_model import Text2SemanticDecoder
from vocal_tokens import EXTENDED_VOCAB_SIZE, VOCAL_TOKENS, expand_gpt_model


def write_checkpoint(path, state):
    tmp = path.with_suffix(".pending")
    torch.save(state, tmp)
    tmp.replace(path)


def batch(examples, device, enabled, fixed_ref=None):
    channels = examples[0][2].size(0)
    if fixed_ref is not None:
        ref_p, ref_y = fixed_ref
        zero_c = torch.zeros(channels, len(ref_p))
        examples = [
            (torch.cat([ref_p, p]), torch.cat([ref_y, y]), torch.cat([zero_c, c], dim=1))
            for p, y, c in examples
        ]
    n = len(examples)
    pl = [len(p) for p, _, _ in examples]
    yl = [len(y) for _, y, _ in examples]
    phones = torch.zeros(n, max(pl), dtype=torch.long)
    tokens = torch.zeros(n, max(yl), dtype=torch.long)
    features = torch.zeros(n, 1024, max(pl))
    for i, (p, y, c) in enumerate(examples):
        phones[i, -len(p):] = p
        tokens[i, :len(y)] = y
        if enabled:
            features[i, :channels, -len(p):] = c
    return [x.to(device) for x in (phones, torch.tensor(pl), tokens, torch.tensor(yl), features)]


def objective(model, args, device, ref_sem_len=0, mask_padding=False, token_weights=None, return_token_logits=False):
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if device.type == "cuda" and torch.cuda.is_bf16_supported() else nullcontext())
    with amp:
        if ref_sem_len <= 0:
            loss, _ = model.forward_old(*args)
            return loss / args[3].sum()
        from AR.models.utils import make_pad_mask, make_pad_mask_left
        x, x_lens, y, y_lens, bert_feature = args
        x_emb = model.ar_text_embedding(x)
        x_emb = x_emb + model.bert_proj(bert_feature.transpose(1, 2))
        x_emb = model.ar_text_position(x_emb)
        x_mask = make_pad_mask_left(x_lens)
        y_mask = make_pad_mask(y_lens)
        y_mask_int = y_mask.type(torch.int64)
        codes = y.type(torch.int64) * (1 - y_mask_int)
        y_in, targets = model.pad_y_eos(codes, y_mask_int, eos_id=model.EOS)
        x_len, y_len = x_lens.max(), y_lens.max()
        y_pos = model.ar_audio_position(model.ar_audio_embedding(y_in))
        xy_padding_mask = torch.concat([x_mask, y_mask], dim=1)
        x_attn_mask = torch.nn.functional.pad(
            torch.zeros((x_len, x_len), dtype=torch.bool, device=x.device), (0, y_len), value=True
        )
        y_attn_mask = torch.nn.functional.pad(
            torch.triu(torch.ones(y_len, y_len, dtype=torch.bool, device=x.device), diagonal=1), (x_len, 0), value=False
        )
        xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)
        bsz, src_len = x_emb.shape[0], x_len + y_len
        _xy_pad = xy_padding_mask.view(bsz, 1, 1, src_len).expand(-1, model.num_head, -1, -1).reshape(bsz * model.num_head, 1, src_len)
        xy_attn_mask = xy_attn_mask.logical_or(_xy_pad)
        attn_bias = torch.zeros_like(xy_attn_mask, dtype=x_emb.dtype)
        attn_bias.masked_fill_(xy_attn_mask, float("-inf"))
        xy_dec, _ = model.h((torch.concat([x_emb, y_pos], dim=1), None), mask=attn_bias)
        logits = model.ar_predict_layer(xy_dec[:, x_len - 1:]).permute(0, 2, 1)
        if mask_padding:
            valid = torch.arange(targets.shape[1], device=y.device)[None, :] <= y_lens[:, None]
            targets = targets.masked_fill(~valid, -100)[:, ref_sem_len:]
            if token_weights is not None:
                losses = torch.nn.functional.cross_entropy(logits[:, :, ref_sem_len:], targets, reduction="none")
                return (losses * token_weights).sum() / token_weights.sum().clamp_min(1)
            loss = torch.nn.functional.cross_entropy(logits[:, :, ref_sem_len:], targets)
            if return_token_logits:
                return loss, logits[:, :, ref_sem_len:-1]
            return loss
        loss = torch.nn.functional.cross_entropy(logits[:, :, ref_sem_len:], targets[:, ref_sem_len:], reduction="sum")
        return loss / (y_lens - ref_sem_len).clamp(min=1).sum()


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def set_backbone_trainable(model, adapter, trainable: bool):
    model.requires_grad_(trainable)
    adapter.requires_grad_(True)


def export_gpt_weights(model):
    out = {}
    for k, v in model.state_dict().items():
        if "bert_proj.head." in k or "bert_proj.performance." in k:
            continue
        clean_k = k.replace("bert_proj.projection.", "bert_proj.")
        out[f"model.{clean_k}"] = v.detach().half().cpu()
    return out


def main():
    p = argparse.ArgumentParser()
    for name in ("data", "base", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--adapter-lr", type=float, default=1e-3)
    p.add_argument("--gpt-lr", type=float, default=1e-4)
    p.add_argument("--fixed-ref-id", default="")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.save_every) < 1:
        p.error("epochs, batch size, save interval must be positive")
    if not args.smoke and not torch.cuda.is_available():
        p.error("Select a GPU runtime")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke and device.type == "cpu":
        torch.set_num_threads(2)

    baseline = torch.load(args.base, map_location="cpu", weights_only=False)
    config = baseline["config"]
    model = Text2SemanticDecoder(config, top_k=3)
    model.load_state_dict({k.removeprefix("model."): v for k, v in baseline["weight"].items()})
    del baseline
    expand_gpt_model(model)
    config["model"]["phoneme_vocab_size"] = EXTENDED_VOCAB_SIZE
    model.to(device)
    adapter = install(model)

    adapter_ids = {id(param) for param in adapter.parameters()}
    backbone_params = [param for param in model.parameters() if id(param) not in adapter_ids]

    # Enable gradients on both adapter and GPT semantic prediction backbone
    set_backbone_trainable(model, adapter, True)
    model.train()

    optimizer = torch.optim.AdamW(
        [
            {"params": list(adapter.parameters()), "lr": args.adapter_lr, "weight_decay": 0.0},
            {"params": backbone_params, "lr": args.gpt_lr, "weight_decay": 0.01},
        ]
    )

    raw = torch.load(args.data, map_location="cpu", weights_only=True)
    train = [(ph, y, c) for _, s, ph, y, c in raw if s == "train"]
    valid = [(ph, y, c) for uid, s, ph, y, c in raw if s == "holdout" and uid != args.fixed_ref_id]
    assert len(train) == 2575 and len(valid) in (92, 93)
    assert all(c.shape == (CHANNELS, len(ph)) and torch.isfinite(c).all() for _, _, ph, _, c in raw)
    fixed_ref = None
    ref_sem_len = 0
    if args.fixed_ref_id:
        match = [(ph, y) for uid, _, ph, y, _ in raw if uid == args.fixed_ref_id]
        assert match, f"fixed-ref-id {args.fixed_ref_id} not found in {args.data}"
        fixed_ref = match[0]
        ref_sem_len = len(fixed_ref[1])
        print(f"Using fixed reference prefix: {args.fixed_ref_id} (ph={len(fixed_ref[0])}, sem={ref_sem_len})", flush=True)

    if args.smoke:
        sample = min(train, key=lambda x: len(x[1]))
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model, batch([sample], device, True, fixed_ref=fixed_ref), device, ref_sem_len=ref_sem_len)
        loss.backward()
        assert torch.isfinite(loss)
        assert any(param.grad is not None and param.grad.abs().sum() > 0 for param in backbone_params)
        optimizer.step()
        print(f"GPT+Adapter smoke test passed on {device}: loss={float(loss.detach()):.4f}")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "resume.pt"
    identity = {
        "data_sha256": digest(args.data),
        "base_sha256": digest(args.base),
        "batch_size": args.batch_size,
        "emotions": EMOTIONS,
        "vocal_tokens": VOCAL_TOKENS,
        "fixed_ref_id": args.fixed_ref_id,
        "mode": "joint_gpt_and_adapter_vocal_v2_fixedref",
    }
    epoch, step, cursor = 0, 0, 0
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("identity") == identity:
            if "weight" in state["adapter"]:
                adapter.performance.load_state_dict(state["adapter"])
            else:
                adapter.load_state_dict(state["adapter"], strict=False)
            if "gpt_weight" in state:
                model.load_state_dict(
                    {
                        k.removeprefix("model.").replace("bert_proj.", "bert_proj.projection."): v.to(device)
                        for k, v in state["gpt_weight"].items()
                    },
                    strict=False,
                )
            optimizer.load_state_dict(state["optimizer"])
            epoch, cursor, step = state["epoch"], state["cursor"], state["step"]
            print(f"Resuming GPT+Adapter epoch={epoch+1} cursor={cursor} step={step}", flush=True)
        else:
            print("Existing resume.pt is from an older run/mode; starting fresh joint GPT+Adapter training.", flush=True)
            torch.manual_seed(72)
    else:
        torch.manual_seed(72)

    def snapshot():
        return {
            "adapter": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
            "gpt_weight": export_gpt_weights(model),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "cursor": cursor,
            "step": step,
            "identity": identity,
        }

    while epoch < args.epochs:
        model.train()
        order = list(range(len(train)))
        random.Random(72 + epoch).shuffle(order)
        for i in range(0, len(order), 80):
            order[i:i+80] = sorted(order[i:i+80], key=lambda j: len(train[j][1]))
        while cursor < len(order):
            # Warmup adapter first so zero-initialized adapter aligns before updating GPT backbone
            optimizer.param_groups[1]["lr"] = 0.0 if step < args.warmup_steps else args.gpt_lr
            subset = [train[j] for j in order[cursor:cursor+args.batch_size]]
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model, batch(subset, device, True, fixed_ref=fixed_ref), device, ref_sem_len=ref_sem_len)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch+1}, offset {cursor}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            cursor += len(subset)
            step += 1
            if step % args.save_every == 0:
                write_checkpoint(path, snapshot())
                print(
                    f"[GPT+Adapter] epoch={epoch+1} step={step} loss={float(loss.detach()):.4f} "
                    f"gpt_lr={optimizer.param_groups[1]['lr']:.1e}",
                    flush=True,
                )
        model.eval()
        with torch.no_grad():
            metrics = {}
            for enabled, key in ((True, "conditioned"), (False, "baseline")):
                vals = [
                    objective(model, batch(valid[i:i+args.batch_size], device, enabled, fixed_ref=fixed_ref), device, ref_sem_len=ref_sem_len).item()
                    for i in range(0, len(valid), args.batch_size)
                ]
                metrics[key] = sum(vals) / len(vals)
        epoch += 1
        cursor = 0
        gpt_w = export_gpt_weights(model)
        write_checkpoint(
            args.out / f"phrase-adapter-epoch{epoch}.pt",
            {
                "adapter": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                "gpt_weight": gpt_w,
                "config": config,
                "emotions": EMOTIONS,
                "vocal_tokens": VOCAL_TOKENS,
                "steps": step,
                "train_items": len(train),
                "loss": metrics,
            },
        )
        write_checkpoint(
            args.out / f"sora-gpt-v2-epoch{epoch}.ckpt",
            {
                "weight": gpt_w,
                "config": config,
                "info": f"{epoch}epoch_{step}iteration",
            },
        )
        write_checkpoint(path, snapshot())
        print(json.dumps({"stage": "gpt", "epoch": epoch, "step": step, **metrics}), flush=True)


if __name__ == "__main__":
    main()
