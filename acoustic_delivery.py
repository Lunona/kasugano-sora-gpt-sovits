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
    sample_index = np.flatnonzero(np.abs(semitones) >= 0.02)
    if not sample_index.size:
        return audio, False
    changed = False
    for index in range(1, count + 1):
        t = float(call(original_tier, "Get time from index", index))
        hz = float(call(original_tier, "Get value at index", index))
        amount = float(semitones[min(len(audio)-1, max(0, round(rate*t)))])
        changed |= abs(amount) >= 0.02
        call(tier, "Add point", t, hz * 2 ** (amount / 12))
    if not changed:
        return audio, False
    call([tier, manipulation], "Replace pitch tier")
    shifted = call(manipulation, "Get resynthesis (overlap-add)").values[0].astype(np.float32)
    if shifted.shape != audio.shape or not np.isfinite(shifted).all():
        raise ValueError("Pitch resynthesis did not preserve the sample count")
    weight = np.clip(np.abs(semitones) / 0.15, 0, 1) if wet_weight is None else wet_weight
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
def check_take(path, text, plan, *, device, whisper_mode=False):
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32")
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

    active = np.flatnonzero(rms > max(0.0005, peak * 0.02))
    allowed_pause = max(1.2, max(span.get("pause_s", 0) for span in plan) + 0.7)
    longest_gap = float(np.diff(active).max(initial=0) * hop / rate)
    issues = []
    if longest_gap > allowed_pause:
        issues.append(f"句中静音 {longest_gap:.1f} 秒")
    edge = max(active[0], len(frames) - 1 - active[-1]) * hop / rate
    if edge > 1.2:
        issues.append(f"首尾静音 {edge:.1f} 秒")
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
        return [], None
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
        wet = np.maximum(np.clip(np.abs(curves[0]) / 0.15, 0, 1), wobble_gate)
        pitched, pitch_applied = _pitch_resynthesis(audio, rate, combined, wet)
    tremble_applied = pitch_applied and bool(wobble.any())
    result, adapted = _apply_loudness(pitched, curves[1])
    result *= np.power(10.0, wobble_db * tremble_applied / 20.0)
    peak = float(np.abs(result).max())
    headroom = min(1.0, 0.98 / max(peak, 1e-6))
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
    return np.clip(y, -0.99, 0.99).astype(np.float32)
