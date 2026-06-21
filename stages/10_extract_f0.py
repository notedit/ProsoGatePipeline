"""Stage 10: F0 extraction with two-pass strategy.

Pass 1: bootstrap (60-600 Hz) for every utt -> median per speaker.
Pass 2: speaker-adaptive f0_min/f0_max -> final per-utt F0 metrics + npy dump.

pyworld preferred; autocorrelation fallback if pyworld import fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from prosogate.audio_io import read_wav
from prosogate.logging_utils import get_logger
from prosogate.manifest import read_jsonl, write_jsonl

log = get_logger(__name__)

try:
    import pyworld as pw  # type: ignore

    _HAVE_PYWORLD = True
except Exception as e:  # pragma: no cover
    log.warning(f"pyworld unavailable ({e}); using autocorrelation fallback")
    _HAVE_PYWORLD = False


# ---------------------------------------------------------------------------
# F0 extraction backends
# ---------------------------------------------------------------------------


def _f0_pyworld(audio: np.ndarray, sr: int, f0_min: float, f0_max: float, frame_hop_ms: float
                ) -> tuple[np.ndarray, np.ndarray]:
    """Return (f0_hz_per_frame, confidence_per_frame). 0 = unvoiced.

    Note: pyworld's f0_floor/f0_ceil are advisory — Harvest can still emit
    values slightly outside the range (e.g. subharmonics in low-pitched
    speech). We hard-clamp here: any frame outside [f0_min, f0_max] is
    marked unvoiced.
    """
    audio64 = audio.astype(np.float64, copy=False)
    frame_period_ms = float(frame_hop_ms)
    f0, t = pw.harvest(
        audio64, sr,
        f0_floor=float(f0_min),
        f0_ceil=float(f0_max),
        frame_period=frame_period_ms,
    )
    f0 = pw.stonemask(audio64, f0, t, sr)
    # Hard clamp out-of-range frames -> unvoiced
    f0 = np.where((f0 >= f0_min) & (f0 <= f0_max), f0, 0.0)
    voiced = f0 > 0
    # crude confidence proxy: ratio of energy at f0 lag to local energy
    conf = np.zeros_like(f0, dtype=np.float64)
    hop = int(round(sr * frame_hop_ms / 1000.0))
    win = max(hop * 4, int(0.025 * sr))
    for i, fv in enumerate(f0):
        if fv <= 0:
            continue
        center = int(t[i] * sr) if i < len(t) else i * hop
        s = max(0, center - win // 2)
        e = min(len(audio), center + win // 2)
        seg = audio[s:e]
        if len(seg) < 8:
            continue
        lag = int(round(sr / fv))
        if lag <= 0 or lag >= len(seg):
            continue
        a = seg[:-lag]
        b = seg[lag:]
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        c = float(np.dot(a, b) / denom)
        conf[i] = max(0.0, min(1.0, (c + 1.0) / 2.0))
    return f0.astype(np.float32), conf.astype(np.float32)


def _f0_autocorr(audio: np.ndarray, sr: int, f0_min: float, f0_max: float, frame_hop_ms: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Pure-numpy autocorrelation fallback (~30-500 Hz default)."""
    f0_min = max(30.0, float(f0_min))
    f0_max = min(500.0, float(f0_max))
    hop = int(round(sr * frame_hop_ms / 1000.0))
    win = int(round(sr * 0.040))  # 40ms window
    if win < hop * 2:
        win = hop * 2
    n_frames = max(1, 1 + (len(audio) - win) // hop)
    f0 = np.zeros(n_frames, dtype=np.float32)
    conf = np.zeros(n_frames, dtype=np.float32)
    min_lag = max(2, int(sr / f0_max))
    max_lag = int(sr / f0_min)
    rms_global = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-12))
    silence_thr = max(1e-4, 0.1 * rms_global)
    for i in range(n_frames):
        s = i * hop
        seg = audio[s : s + win].astype(np.float64)
        if len(seg) < min_lag + 4:
            continue
        seg = seg - np.mean(seg)
        rms = float(np.sqrt(np.mean(seg * seg) + 1e-12))
        if rms < silence_thr:
            continue
        # autocorrelate via direct (small windows)
        ac = np.correlate(seg, seg, mode="full")
        ac = ac[len(seg) - 1 :]  # zero-lag onward
        ac0 = ac[0] + 1e-12
        if max_lag >= len(ac):
            max_lag = len(ac) - 1
        if max_lag <= min_lag:
            continue
        sub = ac[min_lag : max_lag + 1]
        peak = int(np.argmax(sub)) + min_lag
        peak_val = float(ac[peak] / ac0)
        if peak_val < 0.3:
            continue
        f0[i] = sr / peak
        conf[i] = max(0.0, min(1.0, peak_val))
    return f0, conf


def _extract_f0(audio: np.ndarray, sr: int, f0_min: float, f0_max: float, frame_hop_ms: float
                ) -> tuple[np.ndarray, np.ndarray]:
    if _HAVE_PYWORLD:
        try:
            return _f0_pyworld(audio, sr, f0_min, f0_max, frame_hop_ms)
        except Exception as e:
            log.warning(f"pyworld extraction failed ({e}); falling back to autocorrelation")
    return _f0_autocorr(audio, sr, f0_min, f0_max, frame_hop_ms)


# ---------------------------------------------------------------------------
# Octave correction & metrics
# ---------------------------------------------------------------------------


def _octave_correct(f0: np.ndarray, f0_min: float = 0.0, f0_max: float = float("inf")) -> np.ndarray:
    """Two-stage octave correction with hard range clamp at end.

    Stage A (global): build a histogram of voiced log-F0 over the whole utt,
    locate the mode. Any voiced frame more than 6 semitones from the mode is
    snapped to {0.5x, 1x, 2x} of itself — whichever is closest to the mode.
    This catches systematic octave errors where the wrong harmonic is locked
    for long stretches (pyworld DIO failure mode in low-pitched male voices).

    Stage B (local): also flatten relative-frame jumps > 6 semitones the
    same way; mark unvoiced if still > 4 after correction.

    Final: any frame outside [f0_min, f0_max] is forced unvoiced.
    """
    out = f0.copy()
    voiced_mask = out > 0
    if voiced_mask.sum() >= 4:
        log_f0 = np.log2(out[voiced_mask])
        # Mode via histogram (10 cents = 0.0083 octave per bin, ~120 bins for 4-300Hz)
        hist, edges = np.histogram(log_f0, bins=120)
        peak = int(np.argmax(hist))
        mode_log = 0.5 * (edges[peak] + edges[peak + 1])
        # Snap outliers to nearest octave-corrected value of the mode
        idx = np.where(voiced_mask)[0]
        for k, i in enumerate(idx):
            v = out[i]
            diff_st = abs(12.0 * (np.log2(v) - mode_log))
            if diff_st > 6.0:
                cands = [v, v * 2.0, v * 0.5, v * 4.0, v * 0.25]
                cands_log = [np.log2(c) for c in cands]
                best = int(np.argmin([abs(cl - mode_log) for cl in cands_log]))
                v_corr = cands[best]
                if abs(12.0 * (np.log2(v_corr) - mode_log)) <= 6.0:
                    out[i] = v_corr
                else:
                    out[i] = 0.0  # cannot recover, mark unvoiced

    # Stage B: relative-frame
    prev = 0.0
    for i in range(len(out)):
        v = out[i]
        if v <= 0:
            continue
        if prev <= 0:
            prev = v
            continue
        diff = abs(12.0 * np.log2(v / prev))
        if diff > 6.0:
            cands = [v, v * 2.0, v * 0.5]
            diffs = [abs(12.0 * np.log2(c / prev)) for c in cands]
            best_idx = int(np.argmin(diffs))
            v_corr = cands[best_idx]
            best_diff = diffs[best_idx]
            if best_diff > 4.0:
                out[i] = 0.0
                continue
            out[i] = v_corr
            prev = v_corr
        else:
            prev = v

    # Final hard clamp: post-octave correction may produce out-of-range values.
    out = np.where((out >= f0_min) & (out <= f0_max), out, 0.0).astype(out.dtype)
    return out


def _metrics_from_f0(f0: np.ndarray, conf: np.ndarray, speaker_median_hz: float
                     ) -> dict[str, Any]:
    voiced_mask = f0 > 0
    n_total = int(len(f0))
    voiced_ratio = float(voiced_mask.sum() / max(1, n_total))
    if voiced_mask.sum() < 2 or speaker_median_hz <= 0:
        return {
            "f0_mean_hz": 0.0,
            "f0_median_hz": 0.0,
            "f0_std_st": 0.0,
            "f0_range_st": 0.0,
            "f0_delta_p95_st": 0.0,
            "voiced_ratio": voiced_ratio,
            "f0_confidence": float(conf[voiced_mask].mean()) if voiced_mask.any() else 0.0,
        }
    voiced_hz = f0[voiced_mask].astype(np.float64)
    voiced_st = 12.0 * np.log2(voiced_hz / speaker_median_hz)
    p5, p95 = np.percentile(voiced_st, [5, 95])
    # Delta over consecutive voiced indices
    voiced_indices = np.where(voiced_mask)[0]
    deltas: list[float] = []
    for k in range(1, len(voiced_indices)):
        a, b = voiced_indices[k - 1], voiced_indices[k]
        if b - a == 1:
            d = abs(voiced_st[k] - voiced_st[k - 1])
            deltas.append(float(d))
    f0_delta_p95_st = float(np.percentile(deltas, 95)) if deltas else 0.0
    return {
        "f0_mean_hz": float(voiced_hz.mean()),
        "f0_median_hz": float(np.median(voiced_hz)),
        "f0_std_st": float(np.std(voiced_st)),
        "f0_range_st": float(p95 - p5),
        "f0_delta_p95_st": f0_delta_p95_st,
        "voiced_ratio": voiced_ratio,
        "f0_confidence": float(conf[voiced_mask].mean()),
    }


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------


def run(cfg) -> int:
    upstream = cfg.paths.manifests.spk_consistency
    out_path = cfg.paths.manifests.f0
    output_root = Path(cfg.paths.get_path("output_root", "tts_dataset"))
    features_dir = output_root / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    f0_cfg = cfg.f0
    frame_hop_ms = float(f0_cfg.get_path("frame_hop_ms", 10))
    boot_min = float(f0_cfg.get_path("bootstrap.f0_min", 60))
    boot_max = float(f0_cfg.get_path("bootstrap.f0_max", 600))
    factor_low = float(f0_cfg.get_path("speaker_adaptive_range.factor_low", 0.5))
    factor_high = float(f0_cfg.get_path("speaker_adaptive_range.factor_high", 2.0))

    records = list(read_jsonl(upstream))
    log.info(f"stage10: read {len(records)} records from {upstream}")

    # Pass 1: bootstrap F0 for active records, collect per-speaker medians
    pass1_cache: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}  # utt_id -> (f0, conf, sr)
    speaker_voiced_hz: dict[str, list[float]] = {}

    for r in records:
        if r.get("status") == "rejected":
            continue
        utt_id = r["utt_id"]
        wav_path = r["wav"]
        try:
            audio, sr = read_wav(wav_path)
        except Exception as e:
            log.warning(f"stage10: cannot read {wav_path} for {utt_id}: {e}")
            continue
        f0, conf = _extract_f0(audio, sr, boot_min, boot_max, frame_hop_ms)
        f0 = _octave_correct(f0, boot_min, boot_max)
        pass1_cache[utt_id] = (f0, conf, sr)
        voiced = f0[f0 > 0]
        if voiced.size > 0:
            speaker_voiced_hz.setdefault(r["speaker_id"], []).extend(voiced.tolist())

    speaker_median: dict[str, float] = {}
    for spk, hz_list in speaker_voiced_hz.items():
        if hz_list:
            speaker_median[spk] = float(np.median(hz_list))
    log.info(f"stage10: speaker medians = {speaker_median}")

    # Pass 2: speaker-adaptive extraction; reuse pass1 result if range identical
    out_records: list[dict[str, Any]] = []
    for r in records:
        if r.get("status") == "rejected":
            out_records.append(r)
            continue

        utt_id = r["utt_id"]
        wav_path = r["wav"]
        spk = r["speaker_id"]
        spk_med = speaker_median.get(spk, 0.0)

        if spk_med > 0:
            f0_min = max(boot_min, spk_med * factor_low)
            f0_max = min(boot_max, spk_med * factor_high)
        else:
            f0_min, f0_max = boot_min, boot_max

        # If adaptive bounds equal bootstrap (within 1Hz), reuse pass 1
        cached = pass1_cache.get(utt_id)
        same_range = (abs(f0_min - boot_min) < 1.0 and abs(f0_max - boot_max) < 1.0)
        if cached is not None and same_range:
            f0_arr, conf_arr, sr = cached
        else:
            try:
                audio, sr = read_wav(wav_path)
            except Exception as e:
                log.warning(f"stage10: pass2 read failed for {utt_id}: {e}")
                rec = dict(r)
                rec.setdefault("reject_reasons", []).append("f0_io_error")
                rec["status"] = "rejected"
                out_records.append(rec)
                continue
            f0_arr, conf_arr = _extract_f0(audio, sr, f0_min, f0_max, frame_hop_ms)
            f0_arr = _octave_correct(f0_arr, f0_min, f0_max)

        ref_med = spk_med if spk_med > 0 else (
            float(np.median(f0_arr[f0_arr > 0])) if (f0_arr > 0).any() else 0.0
        )
        metrics = _metrics_from_f0(f0_arr, conf_arr, ref_med)

        npy_path = features_dir / f"{utt_id}_f0.npy"
        try:
            np.save(npy_path, f0_arr.astype(np.float32))
        except Exception as e:
            log.warning(f"stage10: failed to save {npy_path}: {e}")

        rec = dict(r)
        rec.update(metrics)
        rec["f0_npy_path"] = str(npy_path)
        rec["speaker_median_f0_hz"] = float(ref_med)
        out_records.append(rec)

    n = write_jsonl(out_path, out_records)
    log.info(f"stage10: wrote {n} records -> {out_path}")
    return 0
