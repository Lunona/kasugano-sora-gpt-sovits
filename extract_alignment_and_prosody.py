"""Extract GPU word-level timestamps and 25 Hz prosody (F0, slope, energy, pauses) for all 2,668 utterances."""
import argparse
import difflib
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
TORCH_LIB = Path(r"F:\GPT-SoVITS\venv\Lib\site-packages\torch\lib")
if TORCH_LIB.is_dir():
    os.environ["PATH"] = str(TORCH_LIB) + os.pathsep + os.environ.get("PATH", "")

sys.path.insert(0, str(ROOT))
import librosa
import numpy as np
import scipy.io.wavfile as wav
import torch
from faster_whisper import WhisperModel
from performance import DATA, load_annotations

PUNCT_RE = re.compile(r"[\s\u3000、。！？…・，「」『』（）()!?.,~\-]+")
TAG_RE = re.compile(r"<(?:gasp|breath|hold|oral|pant|sob|laugh)>")
ASR_MODEL = Path(r"F:\GPT-SoVITS\app\tools\asr\models\faster-whisper-large-v3-turbo")


def load_wav_32k(path: Path) -> np.ndarray:
    sr, y = wav.read(str(path))
    if y.dtype == np.int16:
        y = y.astype(np.float32) / 32768.0
    elif y.dtype == np.int32:
        y = y.astype(np.float32) / 2147483648.0
    else:
        y = y.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 32000:
        y = librosa.resample(y, orig_sr=sr, target_sr=32000)
    return y


def align_chars_to_words(text_ja: str, words: list, dur: float):
    """Map each character of text_ja to (start_s, end_s) monotonically using Whisper word timestamps."""
    n = len(text_ja)
    if n == 0:
        return []
    # Build ASR character stream with per-char timestamps
    asr_chars, asr_times = [], []
    for w in words:
        wt = PUNCT_RE.sub("", w.word or "")
        if not wt:
            continue
        ws, we = float(w.start), max(float(w.end), float(w.start) + 0.02)
        for idx, ch in enumerate(wt):
            t0 = ws + (we - ws) * (idx / len(wt))
            t1 = ws + (we - ws) * ((idx + 1) / len(wt))
            asr_chars.append(ch)
            asr_times.append((t0, t1))

    # Build clean character indices of text_ja
    ref_indices = [i for i, ch in enumerate(text_ja) if not PUNCT_RE.match(ch)]
    ref_chars = [text_ja[i] for i in ref_indices]

    char_starts = [None] * n
    char_ends = [None] * n

    if ref_chars and asr_chars:
        sm = difflib.SequenceMatcher(None, ref_chars, asr_chars, autojunk=False)
        for block in sm.get_matching_blocks():
            for k in range(block.size):
                ri = ref_indices[block.a + k]
                t0, t1 = asr_times[block.b + k]
                char_starts[ri] = t0
                char_ends[ri] = t1
        # If very few matched (e.g. kanji vs kana difference), fall back to proportional word span
        matched = sum(1 for x in char_starts if x is not None)
        if matched < max(1, len(ref_chars) // 4):
            w_start = asr_times[0][0]
            w_end = asr_times[-1][1]
            for pos, ri in enumerate(ref_indices):
                char_starts[ri] = w_start + (w_end - w_start) * (pos / len(ref_indices))
                char_ends[ri] = w_start + (w_end - w_start) * ((pos + 1) / len(ref_indices))
    else:
        w_start, w_end = 0.0, dur
        for pos, ri in enumerate(ref_indices or list(range(n))):
            denom = len(ref_indices) or n
            char_starts[ri] = w_start + (w_end - w_start) * (pos / denom)
            char_ends[ri] = w_start + (w_end - w_start) * ((pos + 1) / denom)

    # Monotonic linear interpolation for unmatched chars & punctuation
    known = [(i, char_starts[i], char_ends[i]) for i in range(n) if char_starts[i] is not None]
    if not known:
        return [(dur * i / n, dur * (i + 1) / n) for i in range(n)]

    # Fill leading, middle gaps, and trailing
    first_i, first_s, _ = known[0]
    for i in range(first_i):
        char_starts[i] = first_s * (i / max(1, first_i))
        char_ends[i] = first_s * ((i + 1) / max(1, first_i))

    for (i0, _, e0), (i1, s1, _) in zip(known, known[1:]):
        gap = i1 - i0 - 1
        if gap > 0:
            for g in range(1, gap + 1):
                char_starts[i0 + g] = e0 + (s1 - e0) * ((g - 1) / gap)
                char_ends[i0 + g] = e0 + (s1 - e0) * (g / gap)

    last_i, _, last_e = known[-1]
    tail = n - 1 - last_i
    if tail > 0:
        for g in range(1, tail + 1):
            char_starts[last_i + g] = last_e + (dur - last_e) * ((g - 1) / tail)
            char_ends[last_i + g] = last_e + (dur - last_e) * (g / tail)

    # Enforce strict monotonicity
    out = []
    prev_t = 0.0
    for s, e in zip(char_starts, char_ends):
        s = max(prev_t, min(dur, float(s)))
        e = max(s, min(dur, float(e)))
        out.append((s, e))
        prev_t = e
    return out


def extract_prosody_25hz(y32: np.ndarray, t_sem: int):
    """Extract 25 Hz (hop=1280 at 32kHz) F0, voiced mask, RMS dB, and silence mask."""
    hop = 1280
    rms = librosa.feature.rms(y=y32, frame_length=hop * 2, hop_length=hop, center=True)[0]
    f0 = librosa.yin(y32, fmin=80, fmax=950, sr=32000, frame_length=hop * 2, hop_length=hop, center=True)

    # Fit exact T_sem frames
    def fit_len(arr):
        if len(arr) >= t_sem:
            return arr[:t_sem]
        return np.pad(arr, (0, t_sem - len(arr)), mode="edge")

    rms = fit_len(rms).astype(np.float32)
    f0 = fit_len(f0).astype(np.float32)

    peak_rms = max(float(np.max(rms)), 1e-5)
    voiced = (rms > max(0.008, 0.12 * peak_rms)) & (f0 >= 85.0) & (f0 <= 920.0)
    silence = rms < max(0.005, 0.07 * peak_rms)

    # Smooth log2(F0) across unvoiced frames
    if voiced.any():
        idx = np.arange(t_sem)
        log_f0 = np.interp(idx, idx[voiced], np.log2(f0[voiced])).astype(np.float32)
    else:
        log_f0 = np.full(t_sem, np.log2(350.0), dtype=np.float32)

    energy_db = (20.0 * np.log10(np.maximum(rms, 1e-5))).astype(np.float32)
    return {
        "f0_hz": f0,
        "log_f0": log_f0,
        "voiced": voiced,
        "energy_db": energy_db,
        "silence": silence,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "colab_data/training-data.pt")
    p.add_argument("--out", type=Path, default=ROOT / "aligned_prosody.pt")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    items = torch.load(args.data, map_location="cpu", weights_only=False)
    anns = load_annotations()
    if args.limit > 0:
        # Ensure SR011444 is included in limit check
        subset = [x for x in items if x[0] == "SR011444"] + [x for x in items if x[0] != "SR011444"][: max(0, args.limit - 1)]
        items = subset

    print(f"Loading faster-whisper from {ASR_MODEL} on CUDA...", flush=True)
    model = WhisperModel(str(ASR_MODEL), device="cuda", compute_type="float16")

    results = {}
    if args.out.is_file() and args.limit == 0:
        results = torch.load(args.out, map_location="cpu", weights_only=False)
        print(f"Resumed {len(results)} existing entries from {args.out}", flush=True)

    for idx, (uid, split, ph, sem, pl) in enumerate(items, 1):
        if uid in results:
            continue
        ann = anns[uid]
        text_ja = TAG_RE.sub("", ann["text_ja"])
        wav_path = ROOT / f"prep/5-wav32k/{uid}.wav"
        if not wav_path.is_file():
            wav_path = DATA / f"音频_WAV/{uid}.wav"
        y32 = load_wav_32k(wav_path)
        dur = len(y32) / 32000.0
        y16 = librosa.resample(y32, orig_sr=32000, target_sr=16000)

        segs, _ = model.transcribe(
            y16, language="ja", word_timestamps=True, initial_prompt=text_ja,
            vad_filter=False, condition_on_previous_text=False, beam_size=1, no_speech_threshold=0.9,
        )
        words = [w for s in segs for w in (s.words or [])]
        char_spans = align_chars_to_words(text_ja, words, dur)
        prosody = extract_prosody_25hz(y32, len(sem))

        results[uid] = {
            "text_ja": text_ja,
            "duration_s": dur,
            "char_spans": char_spans,
            "words": [(round(float(w.start), 3), round(float(w.end), 3), w.word) for w in words],
            **prosody,
        }

        if args.limit > 0:
            vf0 = prosody["f0_hz"][prosody["voiced"]]
            f0_med = float(np.median(vf0)) if len(vf0) else 0.0
            sil_pct = float(prosody["silence"].mean() * 100.0)
            print(f"[{idx}/{len(items)}] {uid}: dur={dur:.2f}s words={len(words)} F0_med={f0_med:.1f}Hz silence={sil_pct:.1f}% text={text_ja[:28]}")
        elif idx % 100 == 0 or idx == len(items):
            torch.save(results, args.out)
            print(f"Progress: {len(results)}/{len(items)} saved to {args.out}", flush=True)

    if args.limit == 0:
        torch.save(results, args.out)
        print(f"Done: {len(results)} utterances saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
