"""Lightweight BiGRU Prosody & Auto-Vocal-Tag Director trained on Sora's 2,668 aligned utterances."""
import argparse
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from vocal_tokens import VOCAL_TOKENS  # 7 tokens: 732..738


class ProsodyDirector(nn.Module):
    def __init__(self, vocab_size: int = 732, emo_dim: int = 22, hidden: int = 192):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 64)
        self.in_proj = nn.Linear(64 + emo_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.15)
        self.prosody_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 4),  # [log_f0_norm, f0_slope, energy_norm, silence_prob]
        )
        self.tag_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, len(VOCAL_TOKENS) + 1),  # 0=none, 1..7=<gasp>..<laugh>
        )

    def forward(self, base_phones: torch.Tensor, emo22: torch.Tensor):
        # base_phones: [B, T], emo22: [B, 22, T]
        x = torch.cat([self.emb(base_phones.clamp(0, 731)), emo22.transpose(1, 2)], dim=-1)
        h, _ = self.gru(F.silu(self.in_proj(x)))
        return self.prosody_head(h).transpose(1, 2), self.tag_head(h)  # [B,4,T], [B,T,8]


def strip_vocal_tags(phones: torch.Tensor, cond26: torch.Tensor):
    """Strip 732..738 tags from training sequence into (base_phones, emo22, prosody4, tag_targets)."""
    base_p, emo_cols, pros_cols, tag_lbls = [], [], [], []
    pending_tag = 0
    for i, pid in enumerate(phones.tolist()):
        if pid >= 732:
            pending_tag = (pid - 732) + 1  # 1..7
        else:
            base_p.append(pid)
            emo_cols.append(cond26[:22, i])
            pros_cols.append(cond26[22:26, i])
            tag_lbls.append(pending_tag)
            pending_tag = 0
    if not base_p:
        return None
    return (
        torch.tensor(base_p, dtype=torch.long),
        torch.stack(emo_cols, dim=1).float(),
        torch.stack(pros_cols, dim=1).float(),
        torch.tensor(tag_lbls, dtype=torch.long),
    )


class DirectorDataset(Dataset):
    def __init__(self, items, split: str, pitch_masks=None):
        self.samples = []
        for uid, sp, ph, _, c26 in items:
            if sp != split:
                continue
            out = strip_vocal_tags(ph, c26)
            if out is not None:
                mask = pitch_masks[uid][:, ph < 732] if pitch_masks is not None else torch.ones(2, len(out[0]), dtype=torch.bool)
                self.samples.append((*out, mask))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate(batch):
    B = len(batch)
    T = max(x[0].shape[0] for x in batch)
    ph = torch.zeros(B, T, dtype=torch.long)
    emo = torch.zeros(B, 22, T)
    pros = torch.zeros(B, 4, T)
    tags = torch.full((B, T), -100, dtype=torch.long)
    mask = torch.zeros(B, T)
    pitch = torch.zeros(B, 2, T, dtype=torch.bool)
    for i, (p, e, pr, tg, pm) in enumerate(batch):
        L = p.shape[0]
        ph[i, :L] = p
        emo[i, :, :L] = e
        pros[i, :, :L] = pr
        tags[i, :L] = tg
        mask[i, :L] = 1.0
        pitch[i, :, :L] = pm
    return ph, emo, pros, tags, mask, pitch


def prosody_loss(pred, target, padding_mask, pitch_mask):
    active = torch.cat([pitch_mask, padding_mask[:, None].bool()], dim=1)
    loss = F.smooth_l1_loss(pred[:, :3], target[:, :3], reduction="none")
    return (loss * active).sum() / active.sum().clamp_min(1)


@torch.inference_mode()
def predict_controls_and_tags(model: ProsodyDirector, base_phones: torch.Tensor, emo22: torch.Tensor, tag_thresh: float = 0.45):
    """Predict 26-ch controls and auto-insert vocal tokens (732..738) from plain base_phones + emo22."""
    device = next(model.parameters()).device
    ph = base_phones.unsqueeze(0).to(device)
    e22 = emo22.unsqueeze(0).to(device).float()
    pros4, tag_logits = model(ph, e22)
    pros4 = pros4[0].cpu()
    pros4[3] = torch.sigmoid(pros4[3])
    probs = F.softmax(tag_logits[0], dim=-1).cpu()

    out_phones, out_cols = [], []
    # Allow at most 1 vocal tag every 6 phonemes to keep pacing natural
    last_tag_pos = -6
    for i in range(base_phones.shape[0]):
        non_none = probs[i, 1:]
        best_cls = int(non_none.argmax()) + 1
        best_p = float(non_none[best_cls - 1])
        if best_p >= tag_thresh and (i - last_tag_pos) >= 6:
            out_phones.append(732 + (best_cls - 1))
            col = torch.cat([emo22[:, i].float(), pros4[:, i]], dim=0)
            out_cols.append(col)
            last_tag_pos = i
        out_phones.append(int(base_phones[i]))
        out_cols.append(torch.cat([emo22[:, i].float(), pros4[:, i]], dim=0))
    return torch.tensor(out_phones, dtype=torch.long), torch.stack(out_cols, dim=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT.parent / "colab_data/training-data-v2.pt")
    p.add_argument("--out", type=Path, default=ROOT / "prosody-director.pt")
    p.add_argument("--pitch-mask", type=Path)
    p.add_argument("--epochs", type=int, default=40)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = torch.load(args.data, map_location="cpu", weights_only=False)
    mask_path = args.pitch_mask or args.data.with_name(args.data.stem + "-pitch-mask.pt")
    named_alignment = args.data.stem.endswith(("-word-aligned", "-phoneme-aligned"))
    if (args.pitch_mask or named_alignment) and not mask_path.is_file():
        p.error(f"Missing pitch mask: {mask_path}")
    masks = torch.load(mask_path, map_location="cpu", weights_only=True) if mask_path.is_file() else None
    train_ds = DirectorDataset(items, "train", masks)
    val_ds = DirectorDataset(items, "holdout", masks)

    # Compute tag class weights (downweight class 0 = none)
    counts = torch.ones(8)
    for _, _, _, tg, _ in train_ds:
        for c in range(8):
            counts[c] += int((tg == c).sum())
    weights = (counts.sum() / (8.0 * counts)).clamp(max=15.0)
    weights[0] = 0.25
    weights = weights.to(device)

    model = ProsodyDirector().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate)

    best_val = 1e9
    for ep in range(1, args.epochs + 1):
        model.train()
        tot_loss = 0.0
        for ph, emo, pros, tags, mask, pitch in train_loader:
            ph, emo, pros, tags, mask = ph.to(device), emo.to(device), pros.to(device), tags.to(device), mask.to(device)
            pred_p, tag_logits = model(ph, emo)
            loss_prosody = prosody_loss(pred_p, pros, mask, pitch.to(device))
            loss_sil = (F.binary_cross_entropy_with_logits(pred_p[:, 3], pros[:, 3].clamp(0, 1), reduction="none") * mask).sum() / mask.sum()
            loss_tag = F.cross_entropy(tag_logits.view(-1, 8), tags.view(-1), weight=weights, ignore_index=-100)
            loss = loss_prosody + 0.5 * loss_sil + 0.35 * loss_tag
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += float(loss.detach())
        sched.step()

        if ep % 10 == 0 or ep == args.epochs:
            model.eval()
            val_f0, n_val = 0.0, 0
            with torch.inference_mode():
                for ph, emo, pros, tags, mask, pitch in val_loader:
                    ph, emo, pros, mask = ph.to(device), emo.to(device), pros.to(device), mask.to(device)
                    pred_p, _ = model(ph, emo)
                    val_f0 += float(prosody_loss(pred_p, pros, mask, pitch.to(device)))
                    n_val += 1
            val_f0 /= max(1, n_val)
            print(f"Epoch {ep}/{args.epochs} | train_loss={tot_loss/len(train_loader):.4f} | val_prosody_l1={val_f0:.4f}", flush=True)
            if val_f0 < best_val:
                best_val = val_f0
                torch.save({"state_dict": model.state_dict(), "val_prosody_l1": best_val}, args.out)
    print(f"Saved best ProsodyDirector to {args.out} (val_prosody_l1={best_val:.4f})")


if __name__ == "__main__":
    main()
