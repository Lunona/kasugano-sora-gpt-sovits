"""Non-verbal vocalization token vocabulary expansion (IDs 732..738) for GPT-SoVITS v2/v2ProPlus."""
import re
import torch
from torch import nn

BASE_VOCAB_SIZE = 732

VOCAL_TOKENS = (
    "<gasp>",    # 732: 短促呼声
    "<breath>",  # 733: 气声／短呼吸、咳嗽
    "<hold>",    # 734: 持续发声
    "<oral>",    # 735: 口部拟声
    "<pant>",    # 736: 喘息或急促呼吸
    "<sob>",     # 737: 哭声
    "<laugh>",   # 738: 笑声
)

KIND_TO_TOKEN = {
    "短促呼声": "<gasp>",
    "气声／短呼吸": "<breath>",
    "咳嗽": "<breath>",
    "持续发声": "<hold>",
    "口部拟声": "<oral>",
    "喘息或急促呼吸": "<pant>",
    "哭声": "<sob>",
    "笑声": "<laugh>",
}

VOCAL_TOKEN_IDS = {tok: BASE_VOCAB_SIZE + i for i, tok in enumerate(VOCAL_TOKENS)}
EXTENDED_VOCAB_SIZE = BASE_VOCAB_SIZE + len(VOCAL_TOKENS)
VOCAL_TAG_RE = re.compile("(" + "|".join(re.escape(t) for t in VOCAL_TOKENS) + ")")

# Seed phoneme symbols (breath/pause/vowel) used to initialize new embedding rows smoothly
SEED_PHONEMES = ("h", "a", "N", "-", "cl", "…", ",")


def get_seed_ids() -> list[int]:
    from text import symbols2

    sym_to_id = {s: i for i, s in enumerate(symbols2.symbols)}
    ids = [sym_to_id[s] for s in SEED_PHONEMES if s in sym_to_id]
    return ids or [0]


def expand_embedding_weight(weight: torch.Tensor, seed_ids: list[int] | None = None) -> torch.Tensor:
    """Expands [732, D] embedding matrix to [739, D], preserving rows 0..731 verbatim."""
    if weight.shape[0] >= EXTENDED_VOCAB_SIZE:
        return weight
    if seed_ids is None:
        seed_ids = get_seed_ids()
    valid = [i for i in seed_ids if i < weight.shape[0]] or [0]
    base_vec = weight[valid].float().mean(dim=0, keepdim=True)
    gen = torch.Generator(device="cpu").manual_seed(2026)
    noise = torch.randn((len(VOCAL_TOKENS), weight.shape[1]), generator=gen, dtype=torch.float32) * 1e-3
    new_rows = (base_vec.cpu() + noise).to(device=weight.device, dtype=weight.dtype)
    return torch.cat([weight, new_rows], dim=0)


def expand_gpt_model(model: nn.Module):
    """Expands Text2SemanticDecoder.ar_text_embedding from 732 to 739 in-place."""
    emb = model.ar_text_embedding.word_embeddings
    if emb.num_embeddings >= EXTENDED_VOCAB_SIZE:
        return
    new_w = expand_embedding_weight(emb.weight.data)
    new_emb = nn.Embedding(EXTENDED_VOCAB_SIZE, emb.embedding_dim, device=emb.weight.device, dtype=emb.weight.dtype)
    new_emb.weight.data.copy_(new_w)
    model.ar_text_embedding.word_embeddings = new_emb
    if hasattr(model, "phoneme_vocab_size"):
        model.phoneme_vocab_size = EXTENDED_VOCAB_SIZE


def expand_sovits_model(net_g: nn.Module):
    """Expands SynthesizerTrn.enc_p.text_embedding from 732 to 739 in-place."""
    emb = net_g.enc_p.text_embedding
    if emb.num_embeddings >= EXTENDED_VOCAB_SIZE:
        return
    new_w = expand_embedding_weight(emb.weight.data)
    new_emb = nn.Embedding(EXTENDED_VOCAB_SIZE, emb.embedding_dim, device=emb.weight.device, dtype=emb.weight.dtype)
    new_emb.weight.data.copy_(new_w)
    net_g.enc_p.text_embedding = new_emb


def insert_events_into_text_and_plan(text: str, plan: list[dict], events: list[dict]) -> tuple[str, list[dict]]:
    """Inserts <tag> tokens at event start offsets and shifts span boundaries accordingly."""
    inserts = []
    for ev in events or []:
        tok = KIND_TO_TOKEN.get(ev.get("kind"))
        if not tok:
            continue
        pos = max(0, min(len(text), int(ev.get("start", 0))))
        inserts.append((pos, tok))
    if not inserts:
        return text, plan

    inserts.sort(key=lambda x: x[0])
    out_parts = []
    shifts = [0] * (len(text) + 1)
    cursor = 0
    prev_pos = 0
    added = 0
    for pos, tok in inserts:
        while cursor < pos:
            shifts[cursor] = added
            cursor += 1
        out_parts.append(text[prev_pos:pos])
        out_parts.append(tok)
        prev_pos = pos
        added += len(tok)
        shifts[pos] = added
    out_parts.append(text[prev_pos:])
    while cursor <= len(text):
        shifts[cursor] = added
        cursor += 1

    new_text = "".join(out_parts)
    new_plan = []
    for sp in plan:
        s = sp["start"] + shifts[sp["start"]]
        e = sp["end"] + shifts[sp["end"]]
        # Keep span 0 starting at 0 if a tag was inserted at index 0
        if not new_plan:
            s = 0
        new_sp = dict(sp)
        new_sp["start"] = s
        new_sp["end"] = e
        new_plan.append(new_sp)
    if new_plan:
        new_plan[-1]["end"] = len(new_text)
    return new_text, new_plan
