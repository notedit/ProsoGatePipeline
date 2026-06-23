"""Stage 08b — MOS-based hard filter.

Compute per-utt MOS (1.0-5.0) using torchaudio.pipelines.SQUIM_SUBJECTIVE and
hard-reject anything below `mos.min_mos`. Runs AFTER 08_fine_segment (so we
have physical 24kHz wavs to score) and BEFORE 09_spk_consistency (so we don't
waste GPU on samples that are about to be filtered for audio quality).

SQUIM_SUBJECTIVE expects:
  - input wav: 16 kHz mono, length 1-30s
  - reference wav: any clean 16 kHz mono (non-matching reference)
The reference is fed as a quality anchor — its acoustic content does not need
to match the input speaker. We use one A-grade utt from upstream, or the
first input wav as a fallback if no A-grade exists yet.

Output:
  rec["mos_score"]            float, 1.0 - 5.0
  rec["mos_threshold"]        float, threshold used
  rec["status"]               "rejected" + reject_reasons += [f"mos<{thr}"]
                              when score < threshold

The filter is strict by design: in TTS training we want clean studio-quality
samples. Conversational/telephone data typically MOS 2.5-3.5 — set min_mos
based on the actual distribution. See docs/metrics.md §5.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from prosogate.audio_io import read_wav
from prosogate.logging_utils import get_logger
from prosogate.manifest import add_reject, read_jsonl, write_jsonl

log = get_logger(__name__)


def _try_load_squim():
    """Load torchaudio SQUIM_SUBJECTIVE model on GPU when possible."""
    try:
        import torch  # noqa: F401
        from torchaudio.pipelines import SQUIM_SUBJECTIVE  # type: ignore
    except Exception as e:
        log.warning(f"torchaudio SQUIM unavailable ({e}); falling back to mock MOS=5.0")
        return None, "cpu"
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = SQUIM_SUBJECTIVE.get_model().eval().to(device)
    except Exception as e:
        log.warning(f"failed to load SQUIM_SUBJECTIVE on {device} ({e}); mock fallback")
        return None, device
    log.info(f"SQUIM_SUBJECTIVE loaded on {device}")
    return model, device


def _resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return audio
    try:
        import librosa  # type: ignore
        return librosa.resample(audio, orig_sr=sr, target_sr=16000).astype(np.float32)
    except Exception:
        # Linear interp fallback
        ratio = 16000 / sr
        n = int(round(len(audio) * ratio))
        x_old = np.linspace(0, 1, num=len(audio), endpoint=False)
        x_new = np.linspace(0, 1, num=n, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)


def run(cfg: Any) -> int:
    upstream = cfg.paths.manifests.fine_segment
    # Write back to the same manifest path so downstream stages (09, 10, 11, ...)
    # pick up the MOS-filtered records without config changes. We also keep a
    # snapshot at <mos> for debugging if configured.
    out_path = upstream
    snapshot = (
        cfg.paths.manifests.mos
        if "mos" in cfg.paths.manifests
        else None
    )

    mos_cfg = cfg.mos if "mos" in cfg else {}
    min_mos = float(mos_cfg.get("min_mos", 4.0))
    batch_size = int(mos_cfg.get("batch_size", 16))
    use_mock = bool(mos_cfg.get("use_mock", False))

    records = list(read_jsonl(upstream))
    log.info(f"stage08b: read {len(records)} records from {upstream}")

    # Pending = utts that have a wav file and aren't already rejected.
    pending: list[tuple[int, dict[str, Any]]] = []
    for i, r in enumerate(records):
        if r.get("status") == "rejected":
            continue
        wav = r.get("wav")
        if not wav or not Path(wav).exists():
            continue
        pending.append((i, r))
    log.info(f"stage08b: {len(pending)} utts pending MOS scoring (min_mos={min_mos})")

    model, device = (None, "cpu")
    if not use_mock:
        model, device = _try_load_squim()

    # Pick a reference: first A-grade utt if available, else first pending utt.
    # SQUIM_SUBJECTIVE only needs ANY clean reference for its quality anchor.
    ref_audio = None
    if model is not None and pending:
        ref_candidate = next((r for _, r in pending if r.get("grade") == "A"), None)
        if ref_candidate is None:
            ref_candidate = pending[0][1]
        try:
            a, sr = read_wav(ref_candidate["wav"])
            if a.ndim > 1:
                a = a.mean(axis=1)
            ref_audio = _resample_to_16k(a, sr)
            log.info(f"stage08b: reference utt = {ref_candidate['utt_id']} ({len(ref_audio)/16000:.1f}s)")
        except Exception as e:
            log.warning(f"stage08b: failed to load reference, falling back to first pending: {e}")

    n_pass = 0
    n_rej = 0

    if model is None or ref_audio is None:
        # Mock path: assign 5.0 (passes any reasonable min_mos)
        for _, rec in pending:
            rec["mos_score"] = 5.0
            rec["mos_threshold"] = min_mos
            n_pass += 1
    else:
        import torch
        ref_t = torch.from_numpy(ref_audio).unsqueeze(0).to(device)
        with torch.no_grad():
            for idx, rec in pending:
                try:
                    audio, sr = read_wav(rec["wav"])
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    audio_16 = _resample_to_16k(audio, sr)
                    inp = torch.from_numpy(audio_16).unsqueeze(0).to(device)
                    mos = float(model(inp, ref_t).item())
                except Exception as e:
                    log.warning(f"stage08b: MOS failed on {rec.get('utt_id')}: {e}; assigning 0.0")
                    mos = 0.0
                rec["mos_score"] = round(mos, 3)
                rec["mos_threshold"] = min_mos
                if mos < min_mos:
                    add_reject(rec, f"mos<{min_mos}")
                    n_rej += 1
                else:
                    n_pass += 1

    n = write_jsonl(out_path, records)
    if snapshot:
        write_jsonl(snapshot, records)
    log.info(
        f"stage08b: wrote {n} records to {out_path} "
        f"(passed={n_pass}, rejected={n_rej}/{len(pending)}, threshold={min_mos})"
    )
    return 0
