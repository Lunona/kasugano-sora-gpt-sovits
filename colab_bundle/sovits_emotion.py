"""Intra-sentence emotion conditioning adapters for GPT-SoVITS v2ProPlus (SoVITS net_g)."""
import torch
from torch import nn

from performance import CHANNELS


class SoVITSEmotionTextEmbedding(nn.Module):
    """Adds zero-initialized per-phoneme emotion embeddings onto TextEncoder.text_embedding."""

    def __init__(self, base: nn.Module, channels: int = CHANNELS):
        super().__init__()
        self.base = base
        self.proj = nn.Linear(channels, base.embedding_dim, bias=False)
        nn.init.zeros_(self.proj.weight)
        self._controls = None

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        out = self.base(text)
        if self._controls is not None:
            controls = self._controls.to(device=out.device, dtype=out.dtype).transpose(1, 2)
            out = out + self.proj(controls)
        return out


class SoVITSSvEmotionAdapter(nn.Module):
    """Adds zero-initialized utterance emotion summary onto global style sv_emb."""

    def __init__(self, base: nn.Module, channels: int = CHANNELS):
        super().__init__()
        self.base = base
        self.proj = nn.Linear(channels * 2, base.out_features, bias=False)
        nn.init.zeros_(self.proj.weight)
        self._summary = None

    def forward(self, sv_emb: torch.Tensor) -> torch.Tensor:
        out = self.base(sv_emb)
        if self._summary is not None:
            summary = self._summary.to(device=out.device, dtype=out.dtype)
            out = out + self.proj(summary)
        return out


def install_sovits_emotion(net_g: nn.Module, channels: int = CHANNELS) -> list[nn.Parameter]:
    """Wraps net_g.enc_p.text_embedding and net_g.sv_emb with emotion adapters."""
    if not isinstance(net_g.enc_p.text_embedding, SoVITSEmotionTextEmbedding):
        dev = net_g.enc_p.text_embedding.weight.device
        dtype = net_g.enc_p.text_embedding.weight.dtype
        net_g.enc_p.text_embedding = SoVITSEmotionTextEmbedding(net_g.enc_p.text_embedding, channels).to(
            device=dev, dtype=dtype
        )
    if getattr(net_g, "is_v2pro", False) and not isinstance(net_g.sv_emb, SoVITSSvEmotionAdapter):
        dev = net_g.sv_emb.weight.device
        dtype = net_g.sv_emb.weight.dtype
        net_g.sv_emb = SoVITSSvEmotionAdapter(net_g.sv_emb, channels).to(device=dev, dtype=dtype)

    params = list(net_g.enc_p.text_embedding.proj.parameters())
    if getattr(net_g, "is_v2pro", False):
        params.extend(net_g.sv_emb.proj.parameters())
    return params


def set_sovits_controls(net_g: nn.Module, controls: torch.Tensor | None, text_lengths: torch.Tensor | None = None):
    """Sets per-phoneme controls [B, C, T] and utterance summary [B, 2C] on net_g adapters."""
    if controls is None:
        if isinstance(net_g.enc_p.text_embedding, SoVITSEmotionTextEmbedding):
            net_g.enc_p.text_embedding._controls = None
        if getattr(net_g, "is_v2pro", False) and isinstance(net_g.sv_emb, SoVITSSvEmotionAdapter):
            net_g.sv_emb._summary = None
        return

    if isinstance(net_g.enc_p.text_embedding, SoVITSEmotionTextEmbedding):
        net_g.enc_p.text_embedding._controls = controls

    if getattr(net_g, "is_v2pro", False) and isinstance(net_g.sv_emb, SoVITSSvEmotionAdapter):
        if text_lengths is not None:
            t_max = controls.size(2)
            mask = (torch.arange(t_max, device=controls.device).unsqueeze(0) < text_lengths.unsqueeze(1)).unsqueeze(1)
            mask_f = mask.to(controls.dtype)
            mean_c = (controls * mask_f).sum(dim=2) / mask_f.sum(dim=2).clamp(min=1.0)
            masked_for_max = controls.masked_fill(~mask, -1e4)
            peak_c = masked_for_max.max(dim=2).values.clamp(min=0.0)
        else:
            mean_c = controls.mean(dim=2)
            peak_c = controls.max(dim=2).values
        net_g.sv_emb._summary = torch.cat([mean_c, peak_c], dim=1)


def export_sovits_emotion_state(net_g: nn.Module) -> dict:
    """Extracts emotion adapter weights for saving inside inference .pth checkpoints."""
    state = {"channels": CHANNELS}
    if isinstance(net_g.enc_p.text_embedding, SoVITSEmotionTextEmbedding):
        state["text_proj"] = net_g.enc_p.text_embedding.proj.state_dict()
    if getattr(net_g, "is_v2pro", False) and isinstance(net_g.sv_emb, SoVITSSvEmotionAdapter):
        state["sv_proj"] = net_g.sv_emb.proj.state_dict()
    return state


def load_sovits_emotion_state(net_g: nn.Module, state: dict | None):
    """Loads saved emotion adapter weights into an already-installed net_g."""
    if not state:
        return
    if "text_proj" in state and isinstance(net_g.enc_p.text_embedding, SoVITSEmotionTextEmbedding):
        net_g.enc_p.text_embedding.proj.load_state_dict(state["text_proj"])
    if "sv_proj" in state and getattr(net_g, "is_v2pro", False) and isinstance(net_g.sv_emb, SoVITSSvEmotionAdapter):
        net_g.sv_emb.proj.load_state_dict(state["sv_proj"])
