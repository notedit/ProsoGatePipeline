"""Stage 03: produce 24kHz training + 16kHz alignment WAVs.

Mock behavior: librosa is the only resampler; if it is unavailable the stage
falls back to scipy-style polyphase via numpy linear interpolation so the
pipeline still runs end-to-end on synthetic data. Already-rejected upstream
records pass through untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from prosogate.audio_io import read_wav, write_wav
from prosogate.config import Config
from prosogate.logging_utils import get_logger
from prosogate.manifest import read_jsonl, write_jsonl

log = get_logger(__name__)


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    try:
        import librosa  # type: ignore

        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32, copy=False)
    except Exception as e:  # noqa: BLE001
        log.warning("librosa resample failed (%s), falling back to linear interp", e)
        ratio = target_sr / float(orig_sr)
        n_out = int(round(len(audio) * ratio))
        if n_out <= 1:
            return np.zeros(0, dtype=np.float32)
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32, copy=False)


def run(cfg: Config) -> int:
    base = Path.cwd()
    in_path = base / cfg.paths.manifests.audio_qc
    out_path = base / cfg.paths.manifests.resample
    work_root = base / str(cfg.paths.work_root)
    train_dir = work_root / "audio_train"
    align_dir = work_root / "audio_align"
    train_dir.mkdir(parents=True, exist_ok=True)
    align_dir.mkdir(parents=True, exist_ok=True)

    train_sr = int(cfg.resample.training_sr)
    align_sr = int(cfg.resample.alignment_sr)

    if not in_path.exists():
        log.error("upstream manifest missing: %s", in_path)
        return 2

    out: list[dict[str, Any]] = []
    n_pass = 0
    for rec in read_jsonl(in_path):
        rec.setdefault("reject_reasons", [])
        if rec.get("status") == "rejected":
            out.append(rec)
            continue

        audio_id = rec["audio_id"]
        try:
            audio, sr = read_wav(rec["audio_path"])
        except Exception as e:  # noqa: BLE001
            rec["status"] = "rejected"
            rec["reject_reasons"].append(f"read_failed:{e}")
            out.append(rec)
            continue

        try:
            train_audio = _resample(audio, sr, train_sr)
            align_audio = _resample(audio, sr, align_sr)
        except Exception as e:  # noqa: BLE001
            rec["status"] = "rejected"
            rec["reject_reasons"].append(f"resample_failed:{e}")
            out.append(rec)
            continue

        train_path = train_dir / f"{audio_id}.wav"
        align_path = align_dir / f"{audio_id}.wav"
        try:
            write_wav(train_path, train_audio, train_sr, subtype="PCM_16")
            write_wav(align_path, align_audio, align_sr, subtype="PCM_16")
        except Exception as e:  # noqa: BLE001
            rec["status"] = "rejected"
            rec["reject_reasons"].append(f"write_failed:{e}")
            out.append(rec)
            continue

        rec["audio_train_path"] = str(train_path)
        rec["audio_align_path"] = str(align_path)
        rec["train_sr"] = train_sr
        rec["align_sr"] = align_sr
        rec["status"] = "passed"
        n_pass += 1
        out.append(rec)

    write_jsonl(out_path, out)
    log.info("resample: %d total, %d passed -> %s", len(out), n_pass, out_path)
    return 0
