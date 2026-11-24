"""Apply local pitch, loudness and tremble to generated SoVITS speech."""

from functools import lru_cache
import json
from math import gcd

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import resample_poly
import torch

from performance import APP, ROOT, delivery_tremble, delivery_values, phone_span_owners
from sovits_loudness import LoudnessCondition


@lru_cache(maxsize=1)
def _loudness_adapter():
    path = ROOT / "results/sovits_loudness_15epochs/loudness-epoch15.pt"
    if not path.is_file():
        return None
    model = LoudnessCondition().eval()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


def _apply_loudness(audio, db):
    active = db != 0
    model = _loudness_adapter() if active.any() else None
    if model is None:
        return audio * np.power(10.0, db / 20.0), False
    signal = torch.from_numpy(audio[active].astype(np.float32))
    gain = torch.from_numpy(db[active].astype(np.float32))
    output = audio.copy()
    output[active] = model(signal.clamp(-0.998, 0.998).atanh(), gain).tanh().numpy()
    return output, True


@lru_cache(maxsize=1)
def _aligner(device):
    from transformers import HubertForCTC

    path = ROOT / "tmp/japanese-hubert-base-phoneme-ctc-v4"
    model = HubertForCTC.from_pretrained(path, local_files_only=True, use_safetensors=True)
    vocabulary = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
    return model.to(device).eval(), vocabulary


def _pitch_resynthesis(audio, rate, semitones, wet_weight=None):
    import parselmouth
    from parselmouth.praat import call

    sound = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=rate)
    manipulation = call(sound, "To Manipulation", 0.01, 85, 950)
    original_tier = call(manipulation, "Extract pitch tier")
    count = int(call(original_tier, "Get number of points"))
    if not count:
        return audio, False
    tier = call("Create PitchTier", "local_delivery", sound.xmin, sound.xmax)
    sample_index = np.flatnonzero(np.abs(semitones) >= 0.08)
    if not sample_index.size:
        return audio, False
    changed = False
    for index in range(1, count + 1):
        t = float(call(original_tier, "Get time from index", index))
        hz = float(call(original_tier, "Get value at index", index))
        amount = float(semitones[min(len(audio)-1, max(0, round(rate*t)))])
        changed |= abs(amount) >= 0.08
        call(tier, "Add point", t, hz * 2 ** (amount / 12))
    if not changed:
        return audio, False
    call([tier, manipulation], "Replace pitch tier")
    shifted = call(manipulation, "Get resynthesis (overlap-add)").values[0].astype(np.float32)
    if shifted.shape != audio.shape or not np.isfinite(shifted).all():
        raise ValueError("Pitch resynthesis did not preserve the sample count")
    if wet_weight is None:
        # Full-wet inside active regions with short ~12ms edge crossfade to avoid dry/wet comb-filter flanging.
        active = (np.abs(semitones) >= 0.08).astype(np.float32)
        weight = np.clip(gaussian_filter1d(active, max(1.0, rate * 0.006), mode="constant"), 0.0, 1.0)
    else:
        weight = wet_weight
    return audio + (shifted - audio) * weight, True


def _voiced_envelope(audio, rate):
    import parselmouth

    sound = parselmouth.Sound(audio, sampling_frequency=rate)
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=85, pitch_ceiling=950)
    voiced = (pitch.selected_array["frequency"] > 0).astype(np.float32)
    gate = np.interp(np.arange(len(audio)), pitch.xs() * rate, voiced, left=0, right=0)
    return gaussian_filter1d(gate.astype(np.float32), rate * 0.007, mode="constant")


def _tremble_curves(voiced, rate, strength):
    sample_times = np.arange(len(voiced), dtype=np.float32) / rate
    edge_time = np.minimum(sample_times, sample_times[-1] - sample_times)
    edge = np.clip(edge_time / 0.035, 0.0, 1.0)
    edge = 0.5 - 0.5 * np.cos(np.pi * edge)
    gate = voiced * edge
    cycles_per_second = 5.05 + 0.9 * np.sin(2 * np.pi * 1.8 * sample_times + 0.4)
    phase = 2 * np.pi * np.cumsum(cycles_per_second / rate)
    depth = (0.76 + 0.24 * np.sin(2 * np.pi * 2.1 * sample_times + 0.8)) * strength
    pitch = 1.12 * depth * gate * (np.sin(phase) + 0.18 * np.sin(2 * phase + 0.6))
    loudness = 0.85 * depth * gate * np.sin(phase - 0.55)
    return pitch, loudness, gate


def _word_regions(samples, text, plan, values, duration):
    from faster_whisper import WhisperModel
    from rebuild_v2_training_data import matched_character_times
    from vocal_tokens import VOCAL_TAG_RE

    path = APP / "tools/asr/models/faster-whisper-large-v3-turbo"
    if not (path / "model.bin").is_file():
        return [], 0.0
    whisper = WhisperModel(str(path), device="cpu", compute_type="int8", cpu_threads=4, local_files_only=True)
    segments, _ = whisper.transcribe(
        samples, language="ja", beam_size=5, word_timestamps=True,
        condition_on_previous_text=False, vad_filter=False,
    )
    words = [(w.start, w.end, w.word) for seg in segments for w in seg.words if w.probability >= 0.42]
    # Keep offsets stable when removing the event markup from the transcript.
    spoken = VOCAL_TAG_RE.sub(lambda m: " " * len(m.group()), text)
    times = matched_character_times(spoken, words, duration)
    total = sum(c.isalnum() for c in spoken)
    match = len(times) / max(total, 1)
    if match < 0.78:
        return [], match
    regions = []
    for i, (span, control) in enumerate(zip(plan, values, strict=True)):
        if not any(control):
            continue
        expected = [j for j in range(span["start"], span["end"]) if spoken[j].isalnum()]
        found = [times[j] for j in expected if j in times]
        if len(found) >= 2 and len(found) >= 0.7 * len(expected):
            regions.append((i, found[0][1], found[-1][2]))
    return regions, match


def _align_audio(audio, rate, text, plan, device):
    from align_v2_phoneme_pitch import align_phones
    from text.symbols2 import symbols

    model, vocabulary = _aligner(str(device))
    phones, owners = phone_span_owners(text, plan)
    names = [symbols[phone] if phone < len(symbols) else "" for phone in phones]
    ratio = gcd(16000, rate)
    samples = np.ascontiguousarray(resample_poly(audio, 16000 // ratio, rate // ratio), dtype=np.float32)
    logits = model(torch.from_numpy(samples).to(device).unsqueeze(0)).logits[0]
    windows, match = align_phones(logits, names, vocabulary, vowels_only=False)
    return samples, owners, names, vocabulary, windows, match


@torch.inference_mode()
def check_take(path, text, plan, *, device, whisper_mode=False, has_soundscape=False):
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32")
    return check_dry_take(audio, rate, text, plan, device=device, whisper_mode=whisper_mode)


@torch.inference_mode()
def check_dry_take(audio, rate, text, plan, *, device, whisper_mode=False):
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if not audio.size or not np.isfinite(audio).all():
        return ["音频为空或损坏"], None

    hop = max(1, rate // 50)
    if len(audio) < hop:
        return ["音频过短"], None
    frames = audio[:len(audio) // hop * hop].reshape(-1, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    peak = float(rms.max())
    if peak < 0.001:
        return ["音频近乎静音"], None

    issues = []
    active = np.flatnonzero(rms >= max(0.0008, peak * 0.035))
    if len(active) >= 2 and (np.max(np.diff(active)) - 1) * hop / rate >= 1.5:
        issues.append("句中连续静音超过 1.5 秒")
    if whisper_mode:
        irreg, subh, n_voiced = creak_score(audio, rate)
        # Breathy whisper naturally has some low-periodicity frames; only flag heavy creak.
        if n_voiced >= 40 and (irreg >= 35.0 or subh >= 35.0):
            issues.append(f"轻语声带卡顿偏多 ({irreg:.0f}%/{subh:.0f}%)")
    if issues:
        return issues, None

    try:
        _, _, names, vocabulary, _, match = _align_audio(audio, rate, text, plan, device)
    except (OSError, RuntimeError, ValueError, ImportError):
        return issues, None
    # Short clips and explicit vocal effects can produce noisy phoneme estimates.
    if sum(name in vocabulary for name in names) >= 12 and match < 0.45:
        issues.append("发音序列与台词相似度偏低，可能漏读或重复")
    return issues, match


@torch.inference_mode()
def apply_delivery(audio, rate, text, plan, *, device="cpu"):
    """Measure positions on the generated speech and keep its original duration."""
    values = [(*delivery_values(span), delivery_tremble(span)) for span in plan]
    requested = sum(bool(any(control)) for control in values)
    if not requested:
        return audio, {"status": "none", "requested": 0, "regions": []}

    samples, owners, names, vocabulary, windows, match = _align_audio(audio, rate, text, plan, device)
    matched = []
    method = "phoneme"
    for i, control in enumerate(values):
        if not any(control):
            continue
        expected = [j for j, owner in enumerate(owners) if owner == i and names[j] in vocabulary]
        found = [windows[j] for j in expected if windows[j] is not None]
        if not found or len(found) < min(2, len(expected)) or len(found) / len(expected) < 0.5:
            continue
        matched.append((i, found[0][0], found[-1][1]))
    if not matched:
        matched, match = _word_regions(samples, text, plan, values, len(audio)/rate)
        method = "word"

    semitones = np.zeros_like(audio, dtype=np.float32)
    decibels = np.zeros_like(semitones)
    wobble = np.zeros_like(semitones)
    wobble_db = np.zeros_like(semitones)
    wobble_gate = np.zeros_like(semitones)
    voiced = _voiced_envelope(audio, rate) if any(values[i][2] for i, _, _ in matched) else None
    regions = []
    for i, lo, hi in matched:
        pitch, energy, tremble = values[i]
        start = max(0, round(rate * lo))
        end = min(len(audio), round(rate * hi))
        if end - start < 0.045 * rate:
            continue
        semitones[start:end] = pitch * 3.0
        decibels[start:end] = energy * 6.0
        voiced_seconds = float(np.count_nonzero(voiced[start:end] > 0.6) / rate) if tremble else 0.0
        if tremble and voiced_seconds >= 0.11:
            wobble[start:end], wobble_db[start:end], wobble_gate[start:end] = _tremble_curves(
                voiced[start:end], rate, tremble,
            )
        elif not (pitch or energy):
            continue
        regions.append({
            "text": text[plan[i]["start"]:plan[i]["end"]],
            "start": round(start / rate, 3), "end": round(end / rate, 3),
            "pitch_st": round(pitch * 3.0, 2), "gain_db": round(energy * 6.0, 2),
            "tremble_strength": round(tremble, 2),
            "tremble_voiced_seconds": round(voiced_seconds, 3),
            "tremble_applied": bool(tremble and voiced_seconds >= 0.11),
        })
    if not regions:
        return audio, {"status": "unaligned", "match": round(match, 3), "method": method,
                       "requested": requested, "regions": []}

    curves = [gaussian_filter1d(control, rate * 0.012, mode="constant")
              for control in (semitones, decibels)]
    pitched, pitch_applied = audio, False
    if semitones.any() or wobble.any():
        combined = curves[0] + wobble
        # Use 100% wet inside modified spans with ~12ms boundary crossfade (never partial dry+wet flanging).
        active = ((np.abs(semitones) >= 0.08) | (wobble_gate > 0.05)).astype(np.float32)
        if active.any():
            wet = np.clip(gaussian_filter1d(active, max(1.0, rate * 0.006), mode="constant"), 0.0, 1.0)
            pitched, pitch_applied = _pitch_resynthesis(audio, rate, combined, wet)
    tremble_applied = pitch_applied and bool(wobble.any())
    pre_peak = float(np.abs(pitched).max())
    eff_db = curves[1]
    # Only gently back off positive boost when already near full-scale (>0.78) to preserve shouting power.
    if pre_peak > 0.78 and eff_db.max() > 0.0:
        safe_pos_max = max(1.5, 3.9 * (0.80 / pre_peak))
        eff_db = np.where(eff_db > 0.0, np.minimum(eff_db, safe_pos_max), eff_db)
    result, adapted = _apply_loudness(pitched, eff_db)
    result *= np.power(10.0, wobble_db * tremble_applied / 20.0)
    peak = float(np.abs(result).max())
    headroom = min(1.0, 0.94 / max(peak, 1e-6))
    result *= headroom
    return result.astype(np.float32), {
        "status": "applied", "match": round(match, 3), "method": method, "requested": requested,
        "regions": regions, "pitch_applied": pitch_applied, "tremble_applied": tremble_applied,
        "loudness_adapter": "epoch15" if adapted else None,
        "headroom_db": round(float(20 * np.log10(headroom)), 2),
    }


def creak_score(audio: np.ndarray, rate: int) -> tuple[float, float, int]:
    """Measure irregular-phonation % and subharmonic-heavy % over low-band voiced frames."""
    y = np.asarray(audio, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    win, hop = int(0.03 * rate), int(0.01 * rate)
    lo, hi = max(1, int(rate / 500)), max(2, int(rate / 120))
    if len(y) < win or hi <= lo:
        return 0.0, 0.0, 0
    voiced = bad = sub = 0
    hann = np.hanning(win)
    freqs = np.fft.rfftfreq(8192, 1.0 / rate)
    low_mask = freqs < 1500
    for i in range(0, len(y) - win, hop):
        seg = y[i:i + win]
        if np.sqrt(np.mean(seg ** 2)) < 0.03:
            continue
        seg = seg - seg.mean()
        spec = np.abs(np.fft.rfft(seg * hann, 8192))
        pow_spec = spec ** 2
        if pow_spec[low_mask].sum() / (pow_spec.sum() + 1e-12) < 0.6:
            continue
        ac = np.correlate(seg, seg, "full")[win - 1:]
        lag = lo + int(np.argmax(ac[lo:hi]))
        r = ac[lag] / (ac[0] + 1e-9)
        f0 = rate / lag
        voiced += 1
        if r < 0.6:
            bad += 1
        h = np.mean([spec[(freqs > f0 * k * 0.93) & (freqs < f0 * k * 1.07)].max(initial=0.0) for k in (1, 2, 3)])
        s = np.mean([spec[(freqs > f0 * k * 0.93) & (freqs < f0 * k * 1.07)].max(initial=0.0) for k in (0.5, 1.5, 2.5)])
        if 20.0 * np.log10(s / (h + 1e-12) + 1e-12) > -15.0:
            sub += 1
    if voiced == 0:
        return 0.0, 0.0, 0
    return 100.0 * bad / voiced, 100.0 * sub / voiced, voiced


def apply_light_declick_deess(audio: np.ndarray, rate: int) -> np.ndarray:
    """Gently tame 1-3ms HF click spikes and harsh 4-12kHz sibilance while preserving vowels."""
    from scipy import signal

    y = np.asarray(audio, dtype=np.float64)
    if len(y) < int(0.05 * rate):
        return audio
    # 1. Light De-clicker (>1kHz band, preserving <1kHz vocal body)
    hp = signal.sosfiltfilt(signal.butter(2, 1800, btype="highpass", fs=rate, output="sos"), y)
    win = max(3, int(0.030 * rate))
    local_rms = np.sqrt(np.convolve(hp ** 2, np.ones(win) / win, mode="same") + 1e-10)
    excess = np.maximum(0.0, np.abs(hp) / np.maximum(local_rms, 0.002) - 3.0)
    k_sm = max(3, int(0.002 * rate) | 1)
    sm_win = np.hanning(k_sm)
    sm_win /= sm_win.sum()
    gain_click = 10.0 ** (-np.minimum(9.0, np.convolve(excess, sm_win, mode="same") * 3.2) / 20.0)
    y = (
        signal.sosfiltfilt(signal.butter(2, 1000, btype="lowpass", fs=rate, output="sos"), y)
        + signal.sosfiltfilt(signal.butter(2, 1000, btype="highpass", fs=rate, output="sos"), y) * gain_click
    )
    # 2. Light dynamic De-esser (4-12kHz)
    high_hz = min(12000, int(rate * 0.45))
    if high_hz > 4500:
        sib = signal.sosfiltfilt(signal.butter(3, [4000, high_hz], btype="bandpass", fs=rate, output="sos"), y)
        body = signal.sosfiltfilt(signal.butter(2, 3500, btype="lowpass", fs=rate, output="sos"), y)
        w_env = max(3, int(0.008 * rate))
        k_env = np.hanning(w_env)
        k_env /= k_env.sum()
        sib_rms = np.sqrt(np.convolve(sib ** 2, k_env, mode="same") + 1e-10)
        body_rms = np.sqrt(np.convolve(body ** 2, k_env, mode="same") + 1e-10)
        drive = np.minimum(np.maximum(0.0, sib_rms / np.maximum(body_rms, 0.005) - 0.35) * 1.8, np.maximum(0.0, (sib_rms - 0.005) / 0.010) * 1.5)
        gain_sib = 10.0 ** (-np.minimum(7.5, drive * 5.0) / 20.0)
        y = (y - sib) + sib * gain_sib
    return np.clip(y, -0.95, 0.95).astype(np.float32)


def tame_loud_harshness(audio: np.ndarray, rate: int, ceiling: float = 0.94) -> np.ndarray:
    """Dynamically tame >4.5kHz piercing sibilance on loud bursts while keeping 2.5-4.5kHz shout power intact."""
    from scipy import signal
    from scipy.ndimage import maximum_filter1d

    y = np.asarray(audio, dtype=np.float64)
    if y.ndim > 1 or len(y) < int(0.05 * rate):
        return audio

    # 1. Level-triggered high-sibilance smoothing (>4.5kHz only, leaving 2.5-4.5kHz shout presence full & soaring)
    high_hz = min(10500, int(rate * 0.45))
    if high_hz > 5000:
        harsh = signal.sosfiltfilt(signal.butter(3, [4500, high_hz], btype="bandpass", fs=rate, output="sos"), y)
        body = signal.sosfiltfilt(signal.butter(2, 3500, btype="lowpass", fs=rate, output="sos"), y)
        w_env = max(5, int(0.010 * rate) | 1)
        k_env = np.hanning(w_env)
        k_env /= k_env.sum()
        tot_rms = np.sqrt(np.convolve(y ** 2, k_env, mode="same") + 1e-10)
        harsh_rms = np.sqrt(np.convolve(harsh ** 2, k_env, mode="same") + 1e-10)
        body_rms = np.sqrt(np.convolve(body ** 2, k_env, mode="same") + 1e-10)

        loud_gate = np.clip((tot_rms - 0.10) / 0.10, 0.0, 1.0)
        excess_ratio = np.maximum(0.0, harsh_rms / np.maximum(body_rms, 0.025) - 0.22)
        att_db = np.minimum(3.8, loud_gate * excess_ratio * 12.0)
        gain_h = 10.0 ** (-att_db / 20.0)
        y = (y - harsh) + harsh * gain_h

    # 2. High-threshold soft-knee true-peak limiter (only touches extreme transients >0.84, keeping loud punch)
    peak_env = maximum_filter1d(np.abs(y), size=max(3, int(0.004 * rate)))
    k_lim = max(5, int(0.008 * rate) | 1)
    lim_win = np.hanning(k_lim)
    lim_win /= lim_win.sum()
    smooth_env = np.maximum(peak_env, np.convolve(peak_env, lim_win, mode="same"))

    thresh = 0.84
    over = np.maximum(0.0, smooth_env - thresh)
    head = max(0.04, float(ceiling) - thresh)
    limited_env = np.where(smooth_env > thresh, thresh + (over * head) / (over + head), smooth_env)
    gain_lim = np.divide(limited_env, smooth_env, out=np.ones_like(smooth_env), where=smooth_env > 1e-6)
    y = y * gain_lim
    max_abs = float(np.max(np.abs(y)))
    if max_abs > ceiling:
        y = y * (ceiling / max_abs)
    return y.astype(np.float32)


def trim_excess_silence(
    audio: np.ndarray,
    rate: int,
    max_internal_s: float = 0.85,
    target_internal_s: float = 0.45,
    max_edge_s: float = 0.25,
) -> tuple[np.ndarray, float]:
    """Compress abnormal TTS dead-silence gaps (>max_internal_s) and long edge silence without cutting breath."""
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim > 1 or y.size < int(0.4 * rate):
        return y, 0.0
    hop = max(1, int(0.02 * rate))
    n_fr = len(y) // hop
    if n_fr < 5:
        return y, 0.0
    frames = y[:n_fr * hop].reshape(n_fr, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    peak = float(rms.max())
    if peak < 0.002:
        return y, 0.0
    # Use a low threshold so soft breaths and consonant tails are preserved intact
    sil_thresh = max(0.0008, min(0.0035, peak * 0.018))
    active = np.flatnonzero(rms > sil_thresh)
    if not active.size:
        return y, 0.0

    trimmed_s = 0.0
    pad_edge = min(int(0.12 * rate), int(max_edge_s * 0.5 * rate))
    edge_limit = int(max_edge_s * rate)
    s_start = max(0, int(active[0]) * hop - pad_edge) if int(active[0]) * hop > edge_limit else 0
    s_end_idx = int(active[-1] + 1) * hop
    s_end = min(len(y), s_end_idx + pad_edge) if (len(y) - s_end_idx) > edge_limit else len(y)
    if s_start > 0 or s_end < len(y):
        trimmed_s += (s_start + (len(y) - s_end)) / rate
        y = y[s_start:s_end].copy()

    # Recompute frames after edge trim to compress abnormal internal dead air
    n_fr = len(y) // hop
    if n_fr < 10:
        return y, round(trimmed_s, 2)
    frames = y[:n_fr * hop].reshape(n_fr, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    is_sil = rms <= sil_thresh
    max_fr = max(4, int(round(max_internal_s / 0.02)))
    tgt_fr = max(2, int(round(target_internal_s / 0.02)))
    half_fr = tgt_fr // 2

    keep_chunks: list[np.ndarray] = []
    ov = min(int(0.025 * rate), half_fr * hop // 2)
    cursor = 0
    i = 0
    while i < n_fr:
        if not is_sil[i]:
            i += 1
            continue
        j = i
        while j < n_fr and is_sil[j]:
            j += 1
        # Only compress strictly interior dead-silence runs longer than max_fr
        if i > 0 and j < n_fr and (j - i) > max_fr:
            keep_left = (i + half_fr) * hop
            keep_right = (j - (tgt_fr - half_fr)) * hop
            if keep_right > keep_left and keep_left > cursor:
                seg_l = y[cursor:keep_left]
                if keep_chunks and ov > 0 and len(keep_chunks[-1]) > ov and len(seg_l) > ov:
                    keep_chunks.append(seg_l)
                else:
                    keep_chunks.append(seg_l)
                cursor = keep_right
                trimmed_s += (keep_right - keep_left) / rate
        i = j
    keep_chunks.append(y[cursor:])
    if len(keep_chunks) == 1:
        return y, round(trimmed_s, 2)

    out = keep_chunks[0].copy()
    for nxt in keep_chunks[1:]:
        if ov > 0 and len(out) > ov and len(nxt) > ov:
            fade = np.linspace(0.0, 1.0, ov, dtype=np.float32)
            comb = out[-ov:] * (1.0 - fade) + nxt[:ov] * fade
            out = np.concatenate([out[:-ov], comb, nxt[ov:]])
        else:
            out = np.concatenate([out, nxt])
    return out, round(trimmed_s, 2)


def _apply_binaural_proximity(y48: np.ndarray, sr_out: int, ear_pan: str) -> np.ndarray:
    """Transparent binaural ear positioning: crisp airy highs + gentle ITD/ILD without muffling."""
    from scipy import signal

    pan = (ear_pan or "off").strip().lower()
    if pan in ("off", "none", ""):
        return np.stack([y48, y48], axis=1)

    # ASMR_0290 already has natural close-mic low end; only add a touch of >6.5kHz air so it never sounds muddy
    high_air = signal.sosfiltfilt(signal.butter(2, 6500, btype="highpass", fs=sr_out, output="sos"), y48)
    crisp_y = (y48 + 0.18 * high_air).astype(np.float32)

    # Contralateral ear: gentle -4.5dB (0.60x), 0.29ms (14 samples) ITD, and light >7.5kHz softening (never muffled)
    far_ear = signal.sosfiltfilt(signal.butter(1, 7500, btype="lowpass", fs=sr_out, output="sos"), crisp_y).astype(np.float32) * 0.60
    far_delayed = np.concatenate([np.zeros(14, dtype=np.float32), far_ear[:-14]])

    if pan == "left":
        return np.stack([crisp_y, far_delayed], axis=1)
    if pan == "right":
        return np.stack([far_delayed, crisp_y], axis=1)
    if pan == "alternate":
        hop = int(0.02 * sr_out)
        n_fr = len(crisp_y) // hop
        fr_rms = np.array([np.sqrt(np.mean(crisp_y[i * hop:(i + 1) * hop] ** 2)) for i in range(n_fr)])
        splits = [0]
        i = 1
        while i < n_fr - 1:
            if fr_rms[i] < 0.002:
                j = i
                while j < n_fr and fr_rms[j] < 0.002:
                    j += 1
                if (j - i) * 0.02 >= 0.38:
                    splits.append(((i + j) // 2) * hop)
                i = j
            else:
                i += 1
        splits.append(len(crisp_y))
        left_ear = np.stack([crisp_y, far_delayed], axis=1)
        right_ear = np.stack([far_delayed, crisp_y], axis=1)
        out = np.zeros_like(left_ear)
        for idx, (s0, s1) in enumerate(zip(splits[:-1], splits[1:])):
            out[s0:s1] = left_ear[s0:s1] if idx % 2 == 0 else right_ear[s0:s1]
        return out
    # "center" close-mic true phase-coherent center (no fixed sample delay, avoiding 4kHz/12kHz comb notches)
    return np.stack([crisp_y, crisp_y], axis=1)


def mix_soundscape(
    audio: np.ndarray,
    rate: int,
    ambient_bed: str = "none",
    foley_mode: str = "none",
    ambient_level: float = 0.18,
    ear_pan: str = "off",
    sfx_dir=None,
) -> tuple[np.ndarray, int, dict]:
    """Mix 32kHz mono speech into 48kHz binaural stereo with proximity ear-pan, foley, and beds."""
    from pathlib import Path
    import soundfile as sf
    from scipy import signal

    bed_key = (ambient_bed or "none").strip().lower()
    foley_key = (foley_mode or "none").strip().lower()
    pan_key = (ear_pan or "off").strip().lower()
    if bed_key in ("none", "") and foley_key in ("none", "") and pan_key in ("off", "none", ""):
        return audio, rate, {"bed": None, "foley": None, "foley_count": 0, "ear_pan": None}

    sfx_root = Path(sfx_dir) if sfx_dir else (Path(__file__).resolve().parent / "asmr_sfx")
    priv_root = sfx_root.parent / "asmr_sfx_private"

    sr_out = 48000
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if rate != sr_out:
        from math import gcd
        g = gcd(int(rate), sr_out)
        y48 = signal.resample_poly(y, sr_out // g, int(rate) // g).astype(np.float32)
    else:
        y48 = y.copy()

    active_speech = y48[np.abs(y48) > 0.008]
    speech_rms = float(np.sqrt(np.mean(active_speech ** 2))) if active_speech.size else 0.08
    speech_rms = max(0.035, min(0.18, speech_rms))
    level = float(max(0.03, min(0.55, ambient_level)))

    stereo_speech = _apply_binaural_proximity(y48, sr_out, pan_key)
    foley_count = 0

    # 1. Inter-clause Foley insertion (strictly in silence gaps between sentences, never overlapping speech)
    if foley_key == "rustle" and sfx_root.is_dir():
        pool = sorted((sfx_root / "foley").glob("cloth_rustle_*.wav"))

        def _load_foley(clip_path: Path, gain_mul: float = 1.0) -> np.ndarray:
            clip, c_sr = sf.read(clip_path, dtype="float32")
            if clip.ndim == 1:
                clip = np.stack([clip, clip], axis=1)
            if c_sr != sr_out:
                from math import gcd
                g = gcd(int(c_sr), sr_out)
                clip = signal.resample_poly(clip, sr_out // g, int(c_sr) // g, axis=0).astype(np.float32)
            c_rms = float(np.sqrt(np.mean(clip ** 2))) + 1e-8
            target_f_rms = speech_rms * (0.20 + 0.65 * level) * gain_mul
            return clip * min(2.5, target_f_rms / c_rms)

        if pool:
            # Find major inter-clause silence gaps (>= 0.45s, spaced >= 8.5s apart so movement is occasional, not every sentence)
            hop = int(0.02 * sr_out)
            n_fr = len(y48) // hop
            fr_rms = np.array([np.sqrt(np.mean(y48[i * hop:(i + 1) * hop] ** 2)) for i in range(n_fr)])
            voiced_idx = np.flatnonzero(fr_rms > 0.005)
            gaps = []
            if voiced_idx.size >= 2:
                first_v, last_v = int(voiced_idx[0]), int(voiced_idx[-1])
                i = first_v + 1
                last_gap_sample = -int(9.0 * sr_out)
                while i < last_v:
                    if fr_rms[i] < 0.002:
                        j = i
                        while j < last_v and fr_rms[j] < 0.002:
                            j += 1
                        mid_sample = ((i + j) // 2) * hop
                        if (j - i) * 0.02 >= 0.45 and (mid_sample - last_gap_sample) >= int(8.5 * sr_out):
                            gaps.append((i * hop, j * hop, mid_sample))
                            last_gap_sample = mid_sample
                        i = j
                    else:
                        i += 1
            if gaps:
                pieces = []
                prev = 0
                pad = int(0.08 * sr_out)
                max_f_len = int(0.95 * sr_out)
                for g_idx, (g_start, g_end, _) in enumerate(gaps):
                    pieces.append(stereo_speech[prev:g_start])
                    pieces.append(np.zeros((pad, 2), dtype=np.float32))
                    f_clip = _load_foley(pool[g_idx % len(pool)])[:max_f_len]
                    pieces.append(f_clip)
                    pieces.append(np.zeros((pad, 2), dtype=np.float32))
                    prev = g_end
                    foley_count += 1
                pieces.append(stereo_speech[prev:])
                stereo_speech = np.concatenate(pieces, axis=0)
            elif foley_key == "rustle" and pool:
                # Single sentence with no internal gap: add a subtle lead-in rustle
                f_clip = _load_foley(pool[0], gain_mul=0.85)
                pad = np.zeros((int(0.10 * sr_out), 2), dtype=np.float32)
                stereo_speech = np.concatenate([f_clip[:int(0.85 * sr_out)], pad, stereo_speech], axis=0)
                foley_count = 1

    # 2. Continuous Ambient Bed (strictly environmental rain / bedroom only, never oral/R18 sounds)
    bed_map = {
        "rain": sfx_root / "beds/rain_gentle.wav",
        "bedding": sfx_root / "beds/bedding_room.wav",
    }
    bed_used = None
    if bed_key in bed_map and bed_map[bed_key].is_file():
        bed_used = bed_key
        bed_wav, b_sr = sf.read(bed_map[bed_key], dtype="float32")
        if bed_wav.ndim == 1:
            bed_wav = np.stack([bed_wav, bed_wav], axis=1)
        pre_s = int(0.32 * sr_out)
        post_s = int(0.48 * sr_out)
        stereo_speech = np.concatenate(
            [np.zeros((pre_s, 2), dtype=np.float32), stereo_speech, np.zeros((post_s, 2), dtype=np.float32)],
            axis=0,
        )
        total_len = len(stereo_speech)
        reps = (total_len // len(bed_wav)) + 2
        bed_tiled = np.tile(bed_wav, (reps, 1))[:total_len].astype(np.float32)
        b_rms = float(np.sqrt(np.mean(bed_tiled ** 2))) + 1e-8
        target_b_rms = speech_rms * level * (0.85 if bed_key == "rain" else 0.72)
        bed_tiled *= min(3.0, target_b_rms / b_rms)

        # Gentle sidechain ducking (-2.2 dB when speech is active)
        env_win = int(0.06 * sr_out)
        sp_mono_sq = np.mean(stereo_speech ** 2, axis=1)
        sp_env = np.sqrt(np.convolve(sp_mono_sq, np.ones(env_win) / env_win, mode="same") + 1e-12)
        duck = np.where(sp_env > 0.010, 0.78, 1.0).astype(np.float32)
        smooth_w = int(0.08 * sr_out)
        duck = np.convolve(duck, np.ones(smooth_w) / smooth_w, mode="same")[:, None]
        bed_tiled *= duck

        # Smooth fade-in and fade-out on bed
        f_in = min(total_len // 4, int(0.25 * sr_out))
        f_out = min(total_len // 4, int(0.40 * sr_out))
        if f_in > 0:
            bed_tiled[:f_in] *= np.linspace(0.0, 1.0, f_in, dtype=np.float32)[:, None]
        if f_out > 0:
            bed_tiled[-f_out:] *= np.linspace(1.0, 0.0, f_out, dtype=np.float32)[:, None]
        stereo_speech = stereo_speech + bed_tiled

    stereo_speech = np.clip(stereo_speech, -0.98, 0.98).astype(np.float32)
    return stereo_speech, sr_out, {
        "bed": bed_used,
        "foley": foley_key if foley_count else None,
        "foley_count": foley_count,
        "ear_pan": pan_key if pan_key not in ("off", "none", "") else None,
    }


_BREATH_RR: dict[str, int] = {}


@lru_cache(maxsize=4)
def _load_breath_manifest(breath_dir_str: str) -> dict[str, list[dict]]:
    from pathlib import Path
    bdir = Path(breath_dir_str)
    mpath = bdir / "manifest.json"
    if not mpath.is_file():
        return {}
    by_cat: dict[str, list[dict]] = {}
    for item in json.loads(mpath.read_text(encoding="utf-8")):
        cat = item.get("category")
        if cat and (bdir / f"{item['id']}.wav").is_file():
            by_cat.setdefault(cat, []).append(item)
    return by_cat


def apply_breath_intro(
    wav: np.ndarray,
    rate: int,
    text: str,
    plan: list[dict],
    *,
    breath_dir=None,
    whisper_mode: bool = False,
) -> tuple[np.ndarray, str | None]:
    """Prepend an ear-verified physical Sora breath/sob intro for dramatic openings without re-training."""
    import re
    from pathlib import Path
    import soundfile as sf

    if not plan or len(wav) < int(0.45 * rate):
        return wav, None
    if re.search(r"<(?:oral|breath|sob|gasp|pant)>", text or ""):
        return wav, None
    # If the utterance begins directly with a choked nasal/labial struggle sound (e.g. んぶっ), let it breathe natively
    if re.match(r"^[…‥\s　]*[んン][ぶぷむば]", text or ""):
        return wav, None

    first = plan[0]
    emo = first.get("primary") or "neutral"
    deliv = first.get("delivery") or "plain"
    inten = float(first.get("intensity", 0.5) or 0.5)
    has_lead_ellipsis = bool(re.match(r"^[…‥\s　]+", text or ""))

    cat = None
    if deliv == "tearful" or first.get("event") == "sob" or (emo == "hurt" and inten >= 0.68):
        cat = "sob_catch"
    elif deliv == "startled" or (emo in ("fear", "surprise") and inten >= 0.70):
        cat = "inhale_tense"
    elif (has_lead_ellipsis and emo in ("shy", "tender", "hurt", "fear", "resolve") and inten >= 0.45) or (whisper_mode and inten >= 0.40):
        cat = "sigh_gentle" if (emo in ("tender", "neutral") and deliv in ("soft", "restrained") and not whisper_mode) else "inhale_soft"

    if not cat:
        return wav, None

    bdir = Path(breath_dir) if breath_dir else (ROOT / "sr_breath_tokens")
    by_cat = _load_breath_manifest(str(bdir))
    pool = by_cat.get(cat) or by_cat.get("inhale_soft")
    if not pool:
        return wav, None

    # Locate speech and lower-level breath separately on 20 ms RMS frames.
    frame = max(1, rate // 50)
    frames = wav[:len(wav) // frame * frame].reshape(-1, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    speech_level = float(np.percentile(rms, 90))
    speech_frames = np.flatnonzero(rms >= max(0.006, speech_level * 0.24))
    if not len(speech_frames):
        return wav, None
    first_speech = int(speech_frames[0])
    onset = first_speech * frame
    lead = rms[:max(0, first_speech - 2)]
    if np.count_nonzero(lead >= max(0.0015, speech_level * 0.035)) >= 4:
        return wav, None

    rr_idx = _BREATH_RR.get(cat, 0) % len(pool)
    _BREATH_RR[cat] = rr_idx + 1
    chosen = pool[rr_idx]
    b_wav, b_sr = sf.read(str(bdir / f"{chosen['id']}.wav"), dtype="float32")
    if b_wav.ndim > 1:
        b_wav = b_wav.mean(axis=1)
    if int(b_sr) != int(rate):
        g = gcd(int(b_sr), int(rate))
        b_wav = resample_poly(b_wav, int(rate) // g, int(b_sr) // g).astype(np.float32)

    active_samples = wav[np.abs(wav) > 0.012]
    speech_rms = float(np.sqrt(np.mean(active_samples ** 2))) if len(active_samples) > 100 else 0.08
    b_rms = float(np.sqrt(np.mean(b_wav ** 2))) + 1e-8
    target_b_rms = float(np.clip(speech_rms * (0.28 if whisper_mode else 0.34), 0.006, 0.028))
    b_wav = (b_wav * (target_b_rms / b_rms)).astype(np.float32)
    b_peak = float(np.max(np.abs(b_wav)))
    if b_peak > 0.14:
        b_wav = b_wav * (0.14 / b_peak)

    fade = min(int(0.025 * rate), len(b_wav) // 4)
    if fade > 0:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade, dtype=np.float32)))
        b_wav[:fade] *= ramp
        b_wav[-fade:] *= ramp[::-1]

    lead_trim = max(0, onset - int(0.055 * rate))
    speech_tail = wav[lead_trim:].astype(np.float32)
    gap = np.zeros(int(0.045 * rate), dtype=np.float32)
    out = np.concatenate([b_wav, gap, speech_tail]).astype(wav.dtype, copy=False)
    return out, chosen["id"]


def score_take_expressiveness(wav_path, plan: list[dict], *, whisper_mode=False) -> dict:
    """Measure F0 span and vocal energy dynamics relative to plan intensity so expressive takes win ties."""
    import soundfile as sf
    w, sr = sf.read(str(wav_path), dtype="float32")
    return score_dry_expressiveness(w, sr, plan, whisper_mode=whisper_mode)


def score_dry_expressiveness(w, sr, plan, *, whisper_mode=False):
    from align_v2_phoneme_pitch import estimate_pitch

    if len(w) < sr // 100:
        return {"expr_score": 0.0, "f0_med": 0.0, "f0_range": 0.0, "rms_dyn_db": 0.0}
    if w.ndim > 1:
        w = w.mean(axis=1)
    if int(sr) != 32000:
        g = gcd(int(sr), 32000)
        w32 = resample_poly(w, 32000 // g, int(sr) // g).astype(np.float32)
    else:
        w32 = w.astype(np.float32)

    pinfo = estimate_pitch(w32)
    voiced = pinfo["f0_hz"][pinfo["voiced"]]
    f0_med = float(np.median(voiced)) if len(voiced) else 300.0
    f0_range = float(np.percentile(voiced, 90) - np.percentile(voiced, 10)) if len(voiced) >= 5 else 120.0

    hop = 320
    nf = max(1, len(w32) // hop)
    frms = np.sqrt(np.mean(w32[: nf * hop].reshape(nf, hop) ** 2, axis=1) + 1e-12)
    active = frms[frms > max(0.008, float(np.percentile(frms, 90)) * 0.12)]
    if len(active) >= 4:
        db = 20.0 * np.log10(active + 1e-6)
        rms_dyn_db = float(np.percentile(db, 95) - np.percentile(db, 25))
    else:
        rms_dyn_db = 8.0

    best_sp = max(plan, key=lambda s: float(s.get("intensity", 0.5)), default={}) if plan else {}
    peak_int = float(best_sp.get("intensity", 0.5))
    main_emo = best_sp.get("primary") or "neutral"

    if whisper_mode:
        # F0 estimates are unreliable for near-unvoiced delivery.
        dyn_fit = max(0.0, 1.0 - abs(rms_dyn_db - 8.0) / 16.0)
        expr = 0.55 + 0.45 * dyn_fit
    elif main_emo in ("anger", "hurt", "fear", "surprise", "impatient") or peak_int >= 0.65:
        range_fit = min(1.0, f0_range / 360.0)
        dyn_fit = min(1.0, rms_dyn_db / 22.0)
        reg_fit = min(1.0, max(0.0, (f0_med - 220.0) / 240.0))
        expr = 0.55 * range_fit + 0.25 * dyn_fit + 0.20 * reg_fit
    else:
        range_fit = min(1.0, f0_range / 230.0)
        dyn_fit = min(1.0, rms_dyn_db / 18.0)
        soft_fit = 1.0 if f0_med <= 390.0 else max(0.3, 1.0 - (f0_med - 390.0) / 250.0)
        expr = 0.50 * range_fit + 0.25 * dyn_fit + 0.25 * soft_fit

    return {
        "expr_score": round(float(np.clip(expr, 0.0, 1.0)), 4),
        "f0_med": round(f0_med, 1),
        "f0_range": round(f0_range, 1),
        "rms_dyn_db": round(rms_dyn_db, 1),
    }
