"""Fine-tune the GPT-SoVITS v2ProPlus SoVITS acoustic/timbre model (S2) on all 2,575 Sora Kasugano voice lines."""
import argparse
from collections import OrderedDict
from io import BytesIO
import json
import os
from pathlib import Path
import random
import sys
import wave

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
APP = Path(os.environ.get("SORAVOICE_APP", "F:/GPT-SoVITS/app"))
sys.path[:0] = [str(APP / "GPT_SoVITS"), str(APP / "GPT_SoVITS/eres2net"), str(APP)]

from module import commons
from module.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from module.mel_processing import mel_spectrogram_torch, spec_to_mel_torch, spectrogram_torch
from module.models import MultiPeriodDiscriminator, SynthesizerTrn
import utils
from performance import CHANNELS
from sovits_emotion import export_sovits_emotion_state, install_sovits_emotion, load_sovits_emotion_state, set_sovits_controls
from vocal_tokens import VOCAL_TOKENS, expand_sovits_model


S2_CONFIG = {
    "train": {
        "log_interval": 50,
        "seed": 1234,
        "epochs": 4,
        "learning_rate": 0.0001,
        "betas": [0.8, 0.99],
        "eps": 1e-09,
        "batch_size": 8,
        "fp16_run": True,
        "lr_decay": 0.999875,
        "segment_size": 20480,
        "c_mel": 45,
        "c_kl": 1.0,
        "text_low_lr_rate": 0.4,
        "grad_ckpt": False,
    },
    "data": {
        "max_wav_value": 32768.0,
        "sampling_rate": 32000,
        "filter_length": 2048,
        "hop_length": 640,
        "win_length": 2048,
        "n_mel_channels": 128,
        "mel_fmin": 0.0,
        "mel_fmax": None,
        "add_blank": True,
        "n_speakers": 300,
        "cleaned_text": True,
    },
    "model": {
        "inter_channels": 192,
        "hidden_channels": 192,
        "filter_channels": 768,
        "n_heads": 2,
        "n_layers": 6,
        "kernel_size": 3,
        "p_dropout": 0.0,
        "resblock": "1",
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "upsample_rates": [10, 8, 2, 2, 2],
        "upsample_initial_channel": 768,
        "upsample_kernel_sizes": [20, 16, 8, 2, 2],
        "n_layers_q": 3,
        "use_spectral_norm": False,
        "gin_channels": 1024,
        "semantic_frame_rate": "25hz",
        "freeze_quantizer": True,
        "version": "v2ProPlus",
    },
    "content_module": "cnhubert",
}


class SoraSoVITSDataset(Dataset):
    """Loads 32kHz WAV + HuBERT SSL + phoneme IDs + SV embedding for v2ProPlus."""

    def __init__(self, data_pt: Path, prep_dir: Path, hps_data):
        raw = torch.load(data_pt, map_location="cpu", weights_only=True)
        self.wav_dir = prep_dir / "5-wav32k"
        self.ssl_dir = prep_dir / "4-cnhubert"
        self.sv_dir = prep_dir / "7-sv_cn"
        self.sampling_rate = hps_data.sampling_rate
        self.filter_length = hps_data.filter_length
        self.hop_length = hps_data.hop_length
        self.win_length = hps_data.win_length

        self.items = []
        self.lengths = []
        for uid, split, phone_tensor, _, controls in raw:
            if split != "train":
                continue
            wav_p = self.wav_dir / f"{uid}.wav"
            ssl_p = self.ssl_dir / f"{uid}.wav.pt"
            sv_p = self.sv_dir / f"{uid}.wav.pt"
            if not wav_p.is_file() or not ssl_p.is_file() or not sv_p.is_file():
                continue
            size = wav_p.stat().st_size
            duration = max(0, size - 44) / (self.sampling_rate * 2)
            if 0.5 <= duration < 54.0:
                self.items.append((uid, phone_tensor.long(), controls.float()))
                self.lengths.append(size // (2 * self.hop_length))

        assert len(self.items) > 100, f"Too few valid audio items found: {len(self.items)}"

    def __len__(self):
        return len(self.items)

    def _read_wav_spec(self, wav_p: Path):
        with wave.open(str(wav_p), "rb") as wf:
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        audio_norm = torch.from_numpy(pcm).unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm, self.filter_length, self.sampling_rate, self.hop_length, self.win_length, center=False
        )
        return torch.squeeze(spec, 0), audio_norm

    def __getitem__(self, idx):
        uid, text, controls = self.items[idx]
        spec, wav = self._read_wav_spec(self.wav_dir / f"{uid}.wav")
        ssl = torch.load(self.ssl_dir / f"{uid}.wav.pt", map_location="cpu", weights_only=True).detach()
        if ssl.shape[-1] != spec.shape[-1]:
            dtype = ssl.dtype
            diff = spec.shape[-1] - ssl.shape[-1]
            if diff > 0:
                ssl = F.pad(ssl.float(), (0, diff), mode="replicate").to(dtype)
            elif diff < 0:
                ssl = ssl[..., : spec.shape[-1]]
        sv_emb = torch.load(self.sv_dir / f"{uid}.wav.pt", map_location="cpu", weights_only=True).detach()
        return ssl, spec, wav, text, sv_emb, controls, uid


def collate_sovits(batch):
    _, ids_sorted = torch.sort(torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True)
    max_ssl_len = int(2 * ((max(x[0].size(2) for x in batch) // 2) + 1))
    max_spec_len = int(2 * ((max(x[1].size(1) for x in batch) // 2) + 1))
    max_wav_len = max(x[2].size(1) for x in batch)
    max_text_len = max(x[3].size(0) for x in batch)

    n = len(batch)
    ssl_lengths = torch.LongTensor(n)
    spec_lengths = torch.LongTensor(n)
    wav_lengths = torch.LongTensor(n)
    text_lengths = torch.LongTensor(n)

    spec_padded = torch.zeros(n, batch[0][1].size(0), max_spec_len, dtype=torch.float32)
    wav_padded = torch.zeros(n, 1, max_wav_len, dtype=torch.float32)
    ssl_padded = torch.zeros(n, batch[0][0].size(1), max_ssl_len, dtype=torch.float32)
    text_padded = torch.zeros(n, max_text_len, dtype=torch.long)
    sv_embs = torch.zeros(n, 20480, dtype=torch.float32)
    controls_padded = torch.zeros(n, CHANNELS, max_text_len, dtype=torch.float32)

    for i, idx in enumerate(ids_sorted):
        ssl, spec, wav, text, sv_emb, ctrl, _ = batch[idx]
        ssl_padded[i, :, : ssl.size(2)] = ssl[0, :, :].float()
        ssl_lengths[i] = ssl.size(2)
        spec_padded[i, :, : spec.size(1)] = spec
        spec_lengths[i] = spec.size(1)
        wav_padded[i, :, : wav.size(1)] = wav
        wav_lengths[i] = wav.size(1)
        text_padded[i, : text.size(0)] = text
        text_lengths[i] = text.size(0)
        sv_embs[i] = sv_emb.reshape(-1).float()
        controls_padded[i, :, : ctrl.size(1)] = ctrl

    uids = [batch[idx][-1] for idx in ids_sorted]
    return ssl_padded, ssl_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, text_padded, text_lengths, sv_embs, controls_padded, uids


def make_length_batches(lengths, batch_size: int, seed: int):
    indices = list(range(len(lengths)))
    random.Random(seed).shuffle(indices)
    window = batch_size * 20
    for i in range(0, len(indices), window):
        indices[i : i + window] = sorted(indices[i : i + window], key=lambda j: lengths[j])
    batches = [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]
    random.Random(seed + 999).shuffle(batches)
    return batches


def write_checkpoint(path: Path, state):
    tmp = path.with_suffix(".pending")
    torch.save(state, tmp)
    tmp.replace(path)


def export_sovits_inference_pth(net_g, hps, epoch: int, step: int, out_path: Path):
    """Saves v2ProPlus inference weights with the 06 magic header expected by GPT-SoVITS."""
    opt = OrderedDict()
    opt["weight"] = OrderedDict()
    for key, val in net_g.state_dict().items():
        if (
            "enc_q" in key
            or "enc_p.text_embedding.proj." in key
            or "enc_p.text_embedding.film_" in key
            or "sv_emb.proj." in key
            or "sv_emb.film_" in key
        ):
            continue
        clean_key = key.replace("enc_p.text_embedding.base.", "enc_p.text_embedding.").replace("sv_emb.base.", "sv_emb.")
        opt["weight"][clean_key] = val.detach().half().cpu()
    opt["emotion_adapter"] = export_sovits_emotion_state(net_g)
    opt["vocal_tokens"] = list(VOCAL_TOKENS)
    opt["config"] = hps
    opt["info"] = f"{epoch}epoch_{step}iteration"
    bio = BytesIO()
    torch.save(opt, bio)
    data = b"06" + bio.getvalue()[2:]
    tmp = out_path.with_suffix(".pending")
    tmp.write_bytes(data)
    tmp.replace(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True, help="Path to training-data.pt")
    p.add_argument("--prep", type=Path, required=True, help="Path to prep dir containing 4-cnhubert, 5-wav32k, 7-sv_cn")
    p.add_argument("--base-g", type=Path, required=True, help="Path to pretrained s2Gv2ProPlus.pth")
    p.add_argument("--init-g-export", type=Path, help="Overlay a project inference checkpoint on the full base generator")
    p.add_argument("--base-d", type=Path, help="Path to pretrained s2Dv2ProPlus.pth when base-g has no discriminator")
    p.add_argument("--out", type=Path, required=True, help="Output directory for checkpoints")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--freeze-backbone", action="store_true", help="Freeze SoVITS backbone and train only emotion adapters + new vocal tokens")
    p.add_argument("--unfreeze-flow", action="store_true", help="When used with --freeze-backbone, also unfreeze flow and enc_p (keeping dec and enc_q frozen)")
    p.add_argument("--unfreeze-dec", action="store_true", help="When used with --freeze-backbone, gently unfreeze dec and net_d at 0.15x LR")
    p.add_argument("--dec-only", action="store_true", help="Freeze flow/enc_p/emotion_adapter and polish only dec + net_d at 0.15x LR")
    refs = p.add_mutually_exclusive_group()
    refs.add_argument("--fixed-ref-id", default="", help="Override refer_spec and sv_emb with a single fixed neutral reference utterance")
    refs.add_argument("--cross-ref-manifest", type=Path, help="Sample cross-utterance references using this manifest")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    cfg = json.loads(json.dumps(S2_CONFIG))
    cfg["train"]["epochs"] = args.epochs
    cfg["train"]["batch_size"] = args.batch_size
    hps = utils.HParams(**cfg)
    hps.data = utils.HParams(**cfg["data"])
    hps.train = utils.HParams(**cfg["train"])
    hps.model = utils.HParams(**cfg["model"])

    if not args.smoke and not torch.cuda.is_available():
        p.error("Select a GPU runtime")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(hps.train.seed)
    dataset = SoraSoVITSDataset(args.data, args.prep, hps.data)
    print(f"SoVITS v2ProPlus dataset loaded: {len(dataset)} Sora training utterances", flush=True)
    cross_ref = None
    if args.cross_ref_manifest:
        from cross_reference import CrossReference

        cross_ref = CrossReference(dataset, args.cross_ref_manifest, hps.train.seed)

    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)

    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm, version=hps.model.version).to(device)

    with args.base_g.open("rb") as stream:
        header = stream.read(2)
        stream.seek(0)
        if header == b"06":
            stream = BytesIO(b"PK" + stream.read()[2:])
        loaded_g = torch.load(stream, map_location="cpu", weights_only=False)
    if "net_g" in loaded_g:
        expand_sovits_model(net_g)
        emotion_params = install_sovits_emotion(net_g)
        missing, unexpected = net_g.load_state_dict(loaded_g["net_g"], strict=False)
        non_film_missing = [k for k in missing if ".film_" not in k]
        assert not non_film_missing and not unexpected, f"resume net_g mismatch: missing={non_film_missing[:5]}, unexpected={unexpected[:5]}"
        if "net_d" in loaded_g:
            net_d.load_state_dict(loaded_g["net_d"], strict=False)
        elif args.base_d:
            net_d.load_state_dict(torch.load(args.base_d, map_location="cpu", weights_only=False)["weight"], strict=False)
        elif not args.freeze_backbone or args.unfreeze_dec or args.dec_only:
            p.error("--base-d is required when training a discriminator and base-g has no discriminator weights")
    else:
        if not args.base_d:
            p.error("--base-d is required with a pretrained generator")
        g_weights = loaded_g["weight"]
        assert any("enc_q" in k for k in g_weights), "base-g must contain enc_q (use s2Gv2ProPlus.pth or a resume .pt)"
        net_g.load_state_dict(g_weights, strict=False)
        net_d.load_state_dict(torch.load(args.base_d, map_location="cpu", weights_only=False)["weight"], strict=False)
        expand_sovits_model(net_g)
        emotion_params = install_sovits_emotion(net_g)
        del g_weights
    del loaded_g

    if args.init_g_export:
        raw_export = args.init_g_export.read_bytes()
        if raw_export[:2] == b"06":
            raw_export = b"PK" + raw_export[2:]
        exported = torch.load(BytesIO(raw_export), map_location="cpu", weights_only=False)
        state = {
            {"enc_p.text_embedding.weight": "enc_p.text_embedding.base.weight",
             "sv_emb.weight": "sv_emb.base.weight",
             "sv_emb.bias": "sv_emb.base.bias"}.get(key, key): value
            for key, value in exported["weight"].items()
        }
        missing, unexpected = net_g.load_state_dict(state, strict=False)
        unexpected_missing = [
            key for key in missing
            if not key.startswith(("enc_q.", "enc_p.text_embedding.proj.", "sv_emb.proj."))
        ]
        if unexpected_missing or unexpected:
            raise ValueError(f"Generator export mismatch: missing={unexpected_missing[:5]}, unexpected={unexpected[:5]}")
        emotion = exported["emotion_adapter"]
        if emotion["channels"] != CHANNELS or not all(key in emotion for key in ("text_proj", "sv_proj")):
            raise ValueError("Generator export has incompatible SoVITS emotion weights")
        load_sovits_emotion_state(net_g, emotion)
        print(f"Initialized generator from {args.init_g_export.name} ({exported.get('info')})", flush=True)
        del state, exported, raw_export, emotion
    if args.dec_only:
        net_g.requires_grad_(False)
        net_g.dec.requires_grad_(True)
        net_d.requires_grad_(True)
        optim_g = torch.optim.AdamW(
            [{"params": list(net_g.dec.parameters()), "lr": hps.train.learning_rate * 0.15}],
            hps.train.learning_rate * 0.15,
            betas=tuple(hps.train.betas),
            eps=hps.train.eps,
        )
        optim_d = torch.optim.AdamW(
            net_d.parameters(),
            hps.train.learning_rate * 0.15,
            betas=tuple(hps.train.betas),
            eps=hps.train.eps,
        )
    elif args.freeze_backbone:
        net_g.requires_grad_(False)
        net_d.requires_grad_(False)
        for p_em in emotion_params:
            p_em.requires_grad_(True)
        emb_w = net_g.enc_p.text_embedding.base.weight
        emb_w.requires_grad_(True)
        row_mask = torch.zeros_like(emb_w)
        if args.unfreeze_flow:
            row_mask[:732] = 0.05  # 5.0x * 0.05 = 0.25x gentle LR on base vocab
        row_mask[732:] = 1.0
        emb_w.register_hook(lambda grad, m=row_mask.to(device): grad * m)
        param_groups = [
            {"params": emotion_params, "lr": hps.train.learning_rate * 5.0},
            {"params": [emb_w], "lr": hps.train.learning_rate * 5.0},
        ]
        if args.unfreeze_flow:
            net_g.flow.requires_grad_(True)
            net_g.enc_p.encoder_text.requires_grad_(True)
            net_g.enc_p.mrte.requires_grad_(True)
            param_groups.append({"params": list(net_g.flow.parameters()), "lr": hps.train.learning_rate * 0.35})
            param_groups.append({"params": list(net_g.enc_p.encoder_text.parameters()) + list(net_g.enc_p.mrte.parameters()), "lr": hps.train.learning_rate * 0.25})
        if args.unfreeze_dec:
            net_g.dec.requires_grad_(True)
            net_d.requires_grad_(True)
            param_groups.append({"params": list(net_g.dec.parameters()), "lr": hps.train.learning_rate * 0.15})
        optim_g = torch.optim.AdamW(
            param_groups,
            hps.train.learning_rate * 5.0,
            betas=tuple(hps.train.betas),
            eps=hps.train.eps,
        )
        if args.unfreeze_dec:
            optim_d = torch.optim.AdamW(
                net_d.parameters(),
                hps.train.learning_rate * 0.15,
                betas=tuple(hps.train.betas),
                eps=hps.train.eps,
            )
        else:
            optim_d = None
    else:
        em_p = list(map(id, emotion_params))
        te_p = list(map(id, net_g.enc_p.text_embedding.base.parameters()))
        et_p = list(map(id, net_g.enc_p.encoder_text.parameters()))
        mrte_p = list(map(id, net_g.enc_p.mrte.parameters()))
        base_params = [param for param in net_g.parameters() if id(param) not in te_p + et_p + mrte_p + em_p and param.requires_grad]

        optim_g = torch.optim.AdamW(
            [
                {"params": base_params, "lr": hps.train.learning_rate},
                {"params": list(net_g.enc_p.text_embedding.base.parameters()), "lr": hps.train.learning_rate * hps.train.text_low_lr_rate},
                {"params": list(net_g.enc_p.encoder_text.parameters()), "lr": hps.train.learning_rate * hps.train.text_low_lr_rate},
                {"params": list(net_g.enc_p.mrte.parameters()), "lr": hps.train.learning_rate * hps.train.text_low_lr_rate},
                {"params": emotion_params, "lr": hps.train.learning_rate * 5.0},
            ],
            hps.train.learning_rate,
            betas=tuple(hps.train.betas),
            eps=hps.train.eps,
        )
        optim_d = torch.optim.AdamW(
            net_d.parameters(),
            hps.train.learning_rate,
            betas=tuple(hps.train.betas),
            eps=hps.train.eps,
        )

    use_fp16 = hps.train.fp16_run and device.type == "cuda"
    scaler = GradScaler(enabled=use_fp16)

    fixed_sv_emb = None
    cross_ref_input = None
    if cross_ref:
        net_g.ref_enc.register_forward_pre_hook(lambda _m, a: (cross_ref_input[0].to(a[0]), cross_ref_input[1].to(a[1])))
    if args.fixed_ref_id:
        fixed_spec, _ = dataset._read_wav_spec(dataset.wav_dir / f"{args.fixed_ref_id}.wav")
        fixed_ref_spec = fixed_spec[:704, :].unsqueeze(0).to(device)
        fixed_ref_mask = torch.ones(1, 1, fixed_spec.size(1), device=device)
        fixed_sv_emb = torch.load(dataset.sv_dir / f"{args.fixed_ref_id}.wav.pt", map_location="cpu", weights_only=True).detach().reshape(1, 20480).float().to(device)
        net_g.ref_enc.register_forward_pre_hook(
            lambda _m, a: (fixed_ref_spec.expand(a[0].size(0), -1, -1).to(a[0]), fixed_ref_mask.expand(a[0].size(0), -1, -1).to(a[1]))
        )
        print(f"Using fixed reference spec/sv_emb: {args.fixed_ref_id} (spec_frames={fixed_spec.size(1)})", flush=True)

    def step_batch(batch_data):
        nonlocal cross_ref_input
        *tensors, uids = batch_data
        ssl, ssl_lengths, spec, spec_lengths, y, y_lengths, text, text_lengths, sv_emb, controls = [x.to(device) for x in tensors]
        if fixed_sv_emb is not None:
            sv_emb = fixed_sv_emb.expand(sv_emb.size(0), -1).to(sv_emb)
        if cross_ref:
            cross_ref_input, sv_emb, ref_ids = cross_ref.batch(uids, device)
            if args.smoke:
                print(f"Cross-reference smoke pairs: {list(zip(uids, ref_ids))}", flush=True)
        ssl.requires_grad = False
        set_sovits_controls(net_g, controls, text_lengths)
        with autocast(enabled=use_fp16):
            y_hat, kl_ssl, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), stats_ssl = net_g(
                ssl, spec, spec_lengths, text, text_lengths, sv_emb
            )
            mel = spec_to_mel_torch(
                spec, hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax
            )
            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_slice = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)
            if optim_d is not None:
                y_d_hat_r, y_d_hat_g, _, _ = net_d(y_slice, y_hat.detach())
                with autocast(enabled=False):
                    loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
            else:
                loss_disc = torch.zeros((), device=device)

        if optim_d is not None:
            optim_d.zero_grad(set_to_none=True)
            scaler.scale(loss_disc).backward()
            scaler.unscale_(optim_d)
            commons.clip_grad_value_(net_d.parameters(), None)
            scaler.step(optim_d)

        with autocast(enabled=use_fp16):
            if optim_d is not None:
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y_slice, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                if optim_d is not None:
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, _ = generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel + kl_ssl + loss_kl
                else:
                    loss_gen_all = loss_mel + kl_ssl + loss_kl

        optim_g.zero_grad(set_to_none=True)
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()
        set_sovits_controls(net_g, None)
        return float(loss_disc.detach()), float(loss_gen_all.detach()), float(loss_mel.detach())

    if args.smoke:
        if cross_ref:
            cross_ref.begin_epoch(1)
        net_g.train()
        net_d.train()
        sample_batch = collate_sovits([dataset[0], dataset[1]])
        ld, lg, lmel = step_batch(sample_batch)
        assert np.isfinite([ld, lg, lmel]).all()
        if not args.dec_only:
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in emotion_params), "SoVITS emotion adapters received no gradients"
        print(f"SoVITS v2ProPlus+Emotion smoke test passed on {device}: loss_d={ld:.4f} loss_g={lg:.4f} loss_mel={lmel:.4f}", flush=True)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    resume_path = args.out / "s2_v2proplus_emotion_resume.pt"
    mode = "sovits_v2proplus_cross_ref_v1" if cross_ref else "sovits_v2proplus_emotion_v1"
    run_settings = {
        "refs": cross_ref.fingerprint,
        "batch_size": args.batch_size,
        "frozen_backbone": args.freeze_backbone,
        "unfrozen_flow": args.unfreeze_flow,
        "unfrozen_dec": args.unfreeze_dec,
        "dec_only": args.dec_only,
    } if cross_ref else None
    start_epoch = 1
    global_step = 0
    scheduler_state = None
    if resume_path.exists():
        st = torch.load(resume_path, map_location="cpu", weights_only=False)
        if st.get("mode") != mode and (cross_ref or st.get("mode") == "sovits_v2proplus_cross_ref_v1"):
            raise ValueError("Reference modes differ. Use a new --out directory to start a new run.")
        if st.get("mode") == mode:
            if cross_ref and st.get("run_settings") != run_settings:
                raise ValueError("Cross-reference run settings differ from the saved checkpoint.")
            net_g.load_state_dict(st["net_g"])
            net_d.load_state_dict(st["net_d"])
            optim_g.load_state_dict(st["optim_g"])
            if optim_d is not None and st.get("optim_d") is not None:
                optim_d.load_state_dict(st["optim_d"])
            start_epoch = st["epoch"] + 1
            global_step = st["step"]
            scheduler_state = st.get("schedulers")
            if cross_ref:
                cross_ref.last_refs.update(st.get("last_refs", {}))
            print(f"Resuming SoVITS v2ProPlus+Emotion from epoch={start_epoch} step={global_step}", flush=True)
        del st

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=-1)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=-1) if optim_d is not None else None
    if scheduler_state:
        scheduler_g.load_state_dict(scheduler_state["g"])
        if scheduler_d is not None and scheduler_state["d"] is not None:
            scheduler_d.load_state_dict(scheduler_state["d"])

    for epoch in range(start_epoch, hps.train.epochs + 1):
        if cross_ref:
            cross_ref.begin_epoch(epoch)
        net_g.train()
        if optim_d is not None:
            net_d.train()
        else:
            net_d.eval()
        batches = make_length_batches(dataset.lengths, hps.train.batch_size, hps.train.seed + epoch)
        loader = DataLoader(dataset, num_workers=2, pin_memory=True, collate_fn=collate_sovits, batch_sampler=batches)
        mel_losses = []
        for batch_data in loader:
            ld, lg, lmel = step_batch(batch_data)
            mel_losses.append(lmel)
            global_step += 1
            if global_step % hps.train.log_interval == 0:
                print(
                    f"[SoVITS v2ProPlus] epoch={epoch}/{hps.train.epochs} step={global_step} "
                    f"loss_d={ld:.3f} loss_g={lg:.3f} loss_mel={lmel:.3f}",
                    flush=True,
                )
        scheduler_g.step()
        if scheduler_d is not None:
            scheduler_d.step()
        avg_mel = sum(mel_losses) / max(1, len(mel_losses))
        export_sovits_inference_pth(net_g, hps, epoch, global_step, args.out / f"sora-sovits-v2proplus-epoch{epoch}.pth")
        export_sovits_inference_pth(net_g, hps, epoch, global_step, args.out / "sora-sovits-latest.pth")
        write_checkpoint(
            resume_path,
            {
                "mode": mode,
                "run_settings": run_settings,
                "last_refs": cross_ref.last_refs if cross_ref else None,
                "schedulers": {"g": scheduler_g.state_dict(), "d": scheduler_d.state_dict() if scheduler_d else None},
                "net_g": {k: v.detach().cpu() for k, v in net_g.state_dict().items()},
                "net_d": {k: v.detach().cpu() for k, v in net_d.state_dict().items()},
                "optim_g": optim_g.state_dict(),
                "optim_d": optim_d.state_dict() if optim_d is not None else None,
                "epoch": epoch,
                "step": global_step,
            },
        )
        if cross_ref:
            print(f"Cross-reference draws: {dict(cross_ref.counts)}", flush=True)
        print(json.dumps({"stage": "sovits_v2proplus", "epoch": epoch, "step": global_step, "avg_mel_loss": round(avg_mel, 4)}), flush=True)


if __name__ == "__main__":
    main()
