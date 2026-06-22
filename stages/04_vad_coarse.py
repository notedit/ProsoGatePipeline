"""Stage 04: coarse VAD + (optional) speaker diarization -> segment manifest.

Mock behavior:
- silero-vad: import-attempted; if missing, falls back to an energy-based VAD
  (frames whose RMS is below the P5 of all frame energies are treated as
  silence).
- pyannote diarization: import-attempted with HUGGINGFACE_TOKEN env var; if
  any of {disabled in cfg, no token, import error, runtime error} hits, the
  whole audio is treated as a single diarization speaker that re-uses the
  metadata `speaker_id` as `speaker_label`. So in offline / smoke runs every
  segment is single-speaker by construction.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from prosogate.audio_io import read_wav
from prosogate.config import Config
from prosogate.logging_utils import get_logger
from prosogate.manifest import read_jsonl, write_jsonl

log = get_logger(__name__)


# ------------------------------ VAD ------------------------------

def _silero_speech_timestamps(audio: np.ndarray, sr: int) -> list[dict[str, float]] | None:
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad  # type: ignore

        import torch  # type: ignore

        model = load_silero_vad()
        ts = get_speech_timestamps(
            torch.from_numpy(audio.astype(np.float32, copy=False)),
            model,
            sampling_rate=sr,
            return_seconds=True,
        )
        return [{"start": float(t["start"]), "end": float(t["end"])} for t in ts]
    except Exception as e:  # noqa: BLE001
        log.warning("silero-vad unavailable (%s); using energy fallback", e)
        return None


def _energy_speech_timestamps(audio: np.ndarray, sr: int) -> list[dict[str, float]]:
    """RMS-frame fallback. Frames below P5 = silence; merge speech frames."""
    if audio.size == 0:
        return []
    frame_len = int(0.030 * sr)  # 30ms
    hop = frame_len  # non-overlap
    n_frames = max(1, len(audio) // hop)
    rms = np.array(
        [
            float(np.sqrt(np.mean(audio[i * hop : i * hop + frame_len] ** 2) + 1e-12))
            for i in range(n_frames)
        ],
        dtype=np.float64,
    )
    if rms.size == 0:
        return []
    p5 = float(np.percentile(rms, 5))
    threshold = max(p5 * 1.5, 1e-4)
    is_speech = rms > threshold

    # Merge into intervals
    out: list[dict[str, float]] = []
    i = 0
    while i < n_frames:
        if is_speech[i]:
            j = i
            while j < n_frames and is_speech[j]:
                j += 1
            out.append({"start": (i * hop) / sr, "end": (j * hop) / sr})
            i = j
        else:
            i += 1
    return out


def _silences_from_speech(speech: list[dict[str, float]], total_dur: float) -> list[tuple[float, float]]:
    """Return silence intervals as [(start, end), ...] in seconds."""
    silences: list[tuple[float, float]] = []
    cursor = 0.0
    for s in speech:
        if s["start"] > cursor:
            silences.append((cursor, float(s["start"])))
        cursor = max(cursor, float(s["end"]))
    if cursor < total_dur:
        silences.append((cursor, total_dur))
    return silences


# --------------------------- Diarization ---------------------------

def _safe_cuda() -> bool:
    """Actually allocate on CUDA to confirm it works (is_available() can lie
    when a driver/runtime mismatch is present)."""
    try:
        import torch  # type: ignore
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda")
        return True
    except Exception:
        return False


def _diarize(audio_path: str, cfg: Config) -> list[tuple[float, float, str]] | None:
    """Return [(start, end, label), ...] or None when unavailable / disabled."""
    diar_cfg = cfg.vad_coarse.diarization if "diarization" in cfg.vad_coarse else {}
    if not diar_cfg or not diar_cfg.get("enabled", False):
        return None
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    model_ref = diar_cfg.get("model", "")
    # Local path bypasses HF download and works without token.
    from pathlib import Path as _Path
    is_local = model_ref and _Path(model_ref).exists()
    if not is_local and not token:
        log.info("diarization disabled: HUGGINGFACE_TOKEN not set and no local model")
        return None
    try:
        from pyannote.audio import Pipeline  # type: ignore
        import torch  # type: ignore
        import soundfile as sf  # type: ignore

        if is_local:
            # model_ref points to a local pipeline config.yaml (or its dir)
            cfg_path = _Path(model_ref)
            if cfg_path.is_dir():
                cfg_path = cfg_path / "config.yaml"
            log.info(f"diarization: loading local pipeline from {cfg_path}")
            pipeline = Pipeline.from_pretrained(str(cfg_path))
        else:
            log.info(f"diarization: loading HF pipeline {model_ref}")
            pipeline = Pipeline.from_pretrained(model_ref, use_auth_token=token)

        # Move to GPU if available — pyannote diarization is ~20x faster on GPU.
        # _safe_cuda guards against driver mismatch even when torch.cuda.is_available() lies.
        if _safe_cuda():
            pipeline.to(torch.device("cuda"))
            log.info("diarization: moved pipeline to cuda")
        else:
            log.info("diarization: CPU mode (no compatible GPU)")

        # In-memory waveform to avoid torchcodec
        audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import numpy as _np
            ratio = 16000 / sr
            n_target = int(len(audio) * ratio)
            idx = (_np.arange(n_target) / ratio).astype(_np.int64)
            idx = _np.clip(idx, 0, len(audio) - 1)
            audio = audio[idx]
            sr = 16000
        waveform = torch.from_numpy(audio).unsqueeze(0)
        diar = pipeline({"waveform": waveform, "sample_rate": sr})

        # pyannote 4.x returns DiarizeOutput with .speaker_diarization
        annotation = getattr(diar, "speaker_diarization", diar)

        out: list[tuple[float, float, str]] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            out.append((float(turn.start), float(turn.end), str(speaker)))
        out.sort(key=lambda t: t[0])
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("diarization failed (%s); falling back to single-speaker", e)
        return None


# --------------------------- Segmentation ---------------------------

def _diar_boundaries(diar: list[tuple[float, float, str]]) -> list[float]:
    bounds: list[float] = []
    prev_label: str | None = None
    for start, _, label in diar:
        if prev_label is not None and label != prev_label:
            bounds.append(float(start))
        prev_label = label
    return bounds


def _snap_to_silence(t: float, silences: list[tuple[float, float]], window_ms: float) -> float:
    """Snap t to the midpoint of the nearest silence whose midpoint is within window_ms."""
    if not silences:
        return t
    win = window_ms / 1000.0
    best = t
    best_dist = float("inf")
    for s, e in silences:
        mid = 0.5 * (s + e)
        d = abs(mid - t)
        if d <= win and d < best_dist:
            best = mid
            best_dist = d
    return best


def _label_at(diar: list[tuple[float, float, str]], t: float, default: str) -> str:
    for s, e, lab in diar:
        if s <= t < e:
            return lab
    return default


def _labels_in_range(
    diar: list[tuple[float, float, str]], start: float, end: float
) -> list[tuple[float, float, str]]:
    out = []
    for s, e, lab in diar:
        s2, e2 = max(s, start), min(e, end)
        if e2 > s2:
            out.append((s2, e2, lab))
    return out


def _force_cut_long(
    seg_start: float,
    seg_end: float,
    long_silence_mids: list[float],
    max_seg: float,
) -> list[float]:
    """Yield additional cut points to keep all sub-segments <= max_seg."""
    cuts: list[float] = []
    cur = seg_start
    while seg_end - cur > max_seg:
        target = cur + max_seg
        # nearest long-silence midpoint <= target
        candidate = None
        for m in long_silence_mids:
            if cur < m <= target:
                if candidate is None or abs(target - m) < abs(target - candidate):
                    candidate = m
        if candidate is None:
            candidate = target  # hard cut; better short than nothing
        cuts.append(candidate)
        cur = candidate
    return cuts


def run(cfg: Config) -> int:
    base = Path.cwd()
    in_path = base / cfg.paths.manifests.resample
    out_path = base / cfg.paths.manifests.vad_coarse

    if not in_path.exists():
        log.error("upstream manifest missing: %s", in_path)
        return 2

    vc = cfg.vad_coarse
    min_silence_ms = float(vc.get("min_silence_ms", 500))
    min_seg = float(vc.get("min_segment_sec", 10))
    max_seg = float(vc.get("max_segment_sec", 30))
    diar_cfg = vc.diarization if "diarization" in vc else {}
    snap_ms = float(diar_cfg.get("snap_to_silence_ms", 200)) if diar_cfg else 200.0
    min_diar_conf = float(diar_cfg.get("min_diar_confidence", 0.6)) if diar_cfg else 0.6

    out_segs: list[dict[str, Any]] = []
    n_audio = 0
    n_seg_pass = 0
    n_seg_short = 0
    for rec in read_jsonl(in_path):
        if rec.get("status") == "rejected":
            continue
        n_audio += 1
        audio_id = rec["audio_id"]
        align_path = rec.get("audio_align_path") or rec.get("audio_path")
        train_path = rec.get("audio_train_path") or rec.get("audio_path")
        meta_speaker = rec.get("speaker_id", "unknown")

        try:
            audio, sr = read_wav(align_path)
        except Exception as e:  # noqa: BLE001
            log.warning("failed to read align audio for %s: %s", audio_id, e)
            continue
        total_dur = float(len(audio) / sr) if sr else 0.0
        if total_dur <= 0:
            continue

        # 1. VAD
        speech = _silero_speech_timestamps(audio, sr)
        if speech is None:
            speech = _energy_speech_timestamps(audio, sr)
        silences = _silences_from_speech(speech, total_dur)
        long_silences = [(s, e) for (s, e) in silences if (e - s) * 1000.0 >= min_silence_ms]
        long_silence_mids = [0.5 * (s + e) for (s, e) in long_silences]

        # 2. Diarization
        diar = _diarize(align_path, cfg)
        diar_used = diar is not None and len(diar) > 0
        if not diar_used:
            diar = [(0.0, total_dur, meta_speaker)]

        # 3. Multi-speaker source: rename per label
        unique_labels = sorted({lab for _, _, lab in diar})
        if len(unique_labels) > 1:
            label_map = {
                lab: f"{meta_speaker}_{chr(ord('a') + i)}" for i, lab in enumerate(unique_labels)
            }
        else:
            label_map = {unique_labels[0]: meta_speaker}
        diar_renamed = [(s, e, label_map[lab]) for (s, e, lab) in diar]

        # 4. Build cut points: long silences + (snapped) diar changepoints
        diar_bounds = _diar_boundaries(diar_renamed)
        snapped = [
            _snap_to_silence(b, long_silences, snap_ms) for b in diar_bounds
        ]
        cut_points = sorted(set([0.0, total_dur] + long_silence_mids + snapped))
        cut_points = [c for c in cut_points if 0.0 <= c <= total_dur]

        # 5. Build initial segments
        raw_segs: list[tuple[float, float]] = []
        for i in range(len(cut_points) - 1):
            s, e = cut_points[i], cut_points[i + 1]
            if e - s > 1e-6:
                raw_segs.append((s, e))

        # 6. Force-cut long segments
        expanded: list[tuple[float, float]] = []
        for s, e in raw_segs:
            if e - s <= max_seg:
                expanded.append((s, e))
                continue
            extra = _force_cut_long(s, e, long_silence_mids, max_seg)
            chain = [s] + extra + [e]
            for i in range(len(chain) - 1):
                expanded.append((chain[i], chain[i + 1]))

        # 7. Split segs by diarization label changes inside a seg
        final_segs: list[tuple[float, float, str]] = []
        for s, e in expanded:
            chunks = _labels_in_range(diar_renamed, s, e)
            if not chunks:
                # No diarization turn covers this range — pyannote says nobody
                # is speaking here. Drop it (don't emit a `multi` placeholder
                # which would inflate the speaker count).
                continue
            # collapse consecutive same-label chunks
            collapsed: list[tuple[float, float, str]] = []
            for cs, ce, lab in chunks:
                if collapsed and collapsed[-1][2] == lab and abs(collapsed[-1][1] - cs) < 1e-3:
                    collapsed[-1] = (collapsed[-1][0], ce, lab)
                else:
                    collapsed.append((cs, ce, lab))
            for cs, ce, lab in collapsed:
                final_segs.append((cs, ce, lab))

        # 8. Emit segments
        # Propagate source-level QC metrics so stage 12 can score audio quality
        # per utt instead of using defaults (otherwise audio_quality_score is constant).
        src_qc = {
            k: rec.get(k)
            for k in ("snr_db", "lufs", "peak_db", "clipping_ratio", "effective_bw_hz", "native_sr")
            if rec.get(k) is not None
        }

        seg_idx = 0
        for s, e, label in final_segs:
            duration = float(e - s)
            seg_idx += 1
            seg_id = f"{label}_{audio_id}_seg{seg_idx:03d}"
            seg_rec: dict[str, Any] = {
                "seg_id": seg_id,
                "source_audio_id": audio_id,
                "audio_align_path": align_path,
                "audio_train_path": train_path,
                "speaker_id": meta_speaker,
                "speaker_label": label,
                "start": float(s),
                "end": float(e),
                "duration": duration,
                "diar_confidence": 1.0 if not diar_used else min_diar_conf,
                "status": "passed",
                "reject_reasons": [],
                **src_qc,
            }
            if duration < min_seg:
                seg_rec["status"] = "rejected"
                seg_rec["reject_reasons"].append("seg_too_short")
                n_seg_short += 1
            else:
                n_seg_pass += 1
            out_segs.append(seg_rec)

    write_jsonl(out_path, out_segs)
    log.info(
        "vad_coarse: %d audio -> %d segs (%d passed, %d short) -> %s",
        n_audio,
        len(out_segs),
        n_seg_pass,
        n_seg_short,
        out_path,
    )
    return 0
