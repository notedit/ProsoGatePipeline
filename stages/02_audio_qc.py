"""Stage 02: audio QC — duration, SNR, LUFS, peak, clipping, effective BW.

Mock behavior: pyloudnorm is optional; missing it sets lufs=None and emits a
warning. Smoke-test mode (`audio_qc.smoke_test=true`) skips all hard thresholds
so synthetic clips still flow downstream while metrics are still recorded.
Reads a small head of the wav for SNR/clipping/BW; full decode only on short
files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from prosogate.audio_io import read_wav
from prosogate.config import Config
from prosogate.logging_utils import get_logger
from prosogate.manifest import read_jsonl, write_jsonl

log = get_logger(__name__)

_QC_KEYS = ("studio", "podcast", "interview", "field")


def _safe_lufs(audio: np.ndarray, sr: int) -> float | None:
    try:
        import pyloudnorm as pyln  # type: ignore

        meter = pyln.Meter(sr)
        return float(meter.integrated_loudness(audio))
    except Exception as e:  # noqa: BLE001
        log.warning("LUFS computation failed (returning None): %s", e)
        return None


def _snr_db(audio: np.ndarray) -> float:
    if audio.size == 0 or float(np.max(np.abs(audio))) < 1e-9:
        return 0.0
    # Frame-level RMS, signal=above P10, noise=below P10.
    frame = max(1, int(len(audio) / 1000))
    n_frames = max(1, len(audio) // frame)
    energies = np.array(
        [float(np.mean(audio[i * frame : (i + 1) * frame] ** 2)) for i in range(n_frames)],
        dtype=np.float64,
    )
    if energies.size == 0:
        return 0.0
    p10 = float(np.percentile(energies, 10))
    sig = energies[energies >= p10]
    noi = energies[energies < p10]
    if sig.size == 0 or noi.size == 0:
        return 0.0
    sig_e = float(np.mean(sig))
    noi_e = float(np.mean(noi)) + 1e-12
    return float(10.0 * np.log10(max(sig_e / noi_e, 1e-12)))


def _peak_db(audio: np.ndarray) -> float:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0:
        return -120.0
    return float(20.0 * np.log10(peak))


def _clipping_ratio(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.mean(np.abs(audio) >= 0.99))


def _effective_bw_hz(audio: np.ndarray, sr: int) -> float:
    if audio.size == 0:
        return 0.0
    n = min(len(audio), 1 << 16)
    seg = audio[:n]
    spec = np.abs(np.fft.rfft(seg)) ** 2
    if float(np.sum(spec)) <= 0:
        return 0.0
    cum = np.cumsum(spec) / float(np.sum(spec))
    idx = int(np.searchsorted(cum, 0.95))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    idx = max(0, min(idx, len(freqs) - 1))
    return float(freqs[idx])


def _qc_for(rec_type: str, cfg: Config) -> dict[str, Any]:
    if rec_type in _QC_KEYS and rec_type in cfg.audio_qc:
        return dict(cfg.audio_qc[rec_type])
    # Default to podcast-level looseness if recording_type is unknown.
    return dict(cfg.audio_qc.get("podcast", {"min_snr_db": 8, "max_clipping_ratio": 0.01, "min_effective_bw_hz": 6000}))


def run(cfg: Config) -> int:
    base = Path.cwd()
    in_path = base / cfg.paths.manifests.ingest
    out_path = base / cfg.paths.manifests.audio_qc

    if not in_path.exists():
        log.error("upstream manifest missing: %s", in_path)
        return 2

    smoke = bool(cfg.audio_qc.get("smoke_test", False)) if isinstance(cfg.audio_qc, dict) else False
    min_duration = float(cfg.audio_qc.get("min_duration_sec", 30))
    lufs_range = cfg.audio_qc.get("loudness_lufs_range", [-35, -12])

    out: list[dict[str, Any]] = []
    n_pass = 0
    for rec in read_jsonl(in_path):
        rec.setdefault("reject_reasons", [])
        # Pass-through prior rejects.
        if rec.get("status") == "rejected":
            out.append(rec)
            continue

        audio_path = rec.get("audio_path", "")
        thresholds = _qc_for(rec.get("recording_type", ""), cfg)

        try:
            info = sf.info(audio_path)
            duration_sec = float(info.duration)
            sr_native = int(info.samplerate)
        except Exception as e:  # noqa: BLE001
            rec["status"] = "rejected"
            rec["reject_reasons"].append(f"sf_info_failed:{e}")
            out.append(rec)
            continue

        try:
            audio, sr = read_wav(audio_path)
        except Exception as e:  # noqa: BLE001
            rec["status"] = "rejected"
            rec["reject_reasons"].append(f"read_failed:{e}")
            out.append(rec)
            continue

        snr = _snr_db(audio)
        lufs = _safe_lufs(audio, sr)
        peak = _peak_db(audio)
        clip = _clipping_ratio(audio)
        bw = _effective_bw_hz(audio, sr)

        rec.update(
            {
                "duration_sec": duration_sec,
                "native_sr": sr_native,
                "snr_db": snr,
                "lufs": lufs,
                "peak_db": peak,
                "clipping_ratio": clip,
                "effective_bw_hz": bw,
            }
        )

        reasons: list[str] = []
        if not smoke:
            if duration_sec < min_duration:
                reasons.append(f"duration<{min_duration}")
            min_snr = float(thresholds.get("min_snr_db", 0))
            if snr < min_snr:
                reasons.append(f"snr<{min_snr}")
            max_clip = float(thresholds.get("max_clipping_ratio", 1.0))
            if clip > max_clip:
                reasons.append(f"clipping>{max_clip}")
            min_bw = float(thresholds.get("min_effective_bw_hz", 0))
            if bw < min_bw:
                reasons.append(f"effective_bw<{min_bw}")
            if lufs is not None and isinstance(lufs_range, (list, tuple)) and len(lufs_range) == 2:
                lo, hi = float(lufs_range[0]), float(lufs_range[1])
                if not (lo <= lufs <= hi):
                    reasons.append(f"lufs_out_of_range[{lo},{hi}]")

        if reasons:
            rec["status"] = "rejected"
            rec["reject_reasons"].extend(reasons)
            rec["qc_status"] = "rejected"
        else:
            rec["status"] = "passed"
            rec["qc_status"] = "passed"
            n_pass += 1

        out.append(rec)

    write_jsonl(out_path, out)
    log.info(
        "audio_qc: %d total, %d passed (smoke_test=%s) -> %s",
        len(out),
        n_pass,
        smoke,
        out_path,
    )
    return 0
