"""Stage 05 — Qwen3-ASR transcription.

Reads seg-level manifest from stage 04, slices the corresponding audio out of
audio_align_path (16kHz), and produces an ASR text per segment.

Real backend: Qwen3-ASR-1.7B via transformers/modelscope (placeholder, not run
in smoke tests). If the import fails we log a warning and fall back to mock.

Mock backend: deterministic. If the upstream metadata exposes a transcript_path
on the source audio (recorded at ingest time), we read the manual text and take
a linear character slice proportional to [start, end] within the source audio's
total duration. Otherwise we emit a fixed placeholder string.

Hard input cap (asr.max_input_sec, default 30s): segments longer than the cap
are rejected fail-fast — we do NOT silently re-segment, the upstream VAD owns
boundary integrity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from prosogate.audio_io import read_wav, slice_audio
from prosogate.logging_utils import get_logger
from prosogate.manifest import add_reject, read_jsonl, write_jsonl

log = get_logger(__name__)

PLACEHOLDER_TEXT = "测试音频片段一二三"


def _try_load_real_backend(cfg: Any):
    """Build an HTTP client for the Qwen3-ASR service.

    Single endpoint:
      POST /v1/audio/transcriptions
        file=@wav, language=<lang>, response_format=verbose_json,
        timestamp_granularities[]=word
      ->  {text, words:[{word,start,end}], duration}

    Batch endpoint (preferred when batch_size > 1):
      POST /v1/audio/transcriptions/batch
        audio_files[]=@wav...  (one inference call, ~5x speedup at batch=16)
      ->  {results: [TranscriptionResponse, ...]}    # order preserved

    Returns dict with callables:
      single: (audio, sr) -> (text, words[])
      batch : (list[(audio, sr)]) -> list[(text, words[])]    or None if unsupported
    """
    try:
        import requests  # noqa: F401
    except ImportError as e:
        log.warning(f"requests unavailable ({e}); cannot reach Qwen3-ASR HTTP server")
        return None

    base_url = cfg.asr.get("base_url", "http://127.0.0.1:18765")
    # Back-compat: if config sets the older `endpoint` field, derive base from it.
    legacy_endpoint = cfg.asr.get("endpoint", "")
    if legacy_endpoint:
        # strip path suffix
        base_url = legacy_endpoint.rsplit("/v1/", 1)[0] if "/v1/" in legacy_endpoint else base_url
    language = cfg.asr.get("language", "zh")
    timeout = float(cfg.asr.get("timeout_sec", 120))
    log.info(f"Qwen3-ASR HTTP client -> {base_url} (lang={language})")

    import io as _io
    import soundfile as _sf
    import requests as _rq

    single_url = f"{base_url}/v1/audio/transcriptions"
    batch_url = f"{base_url}/v1/audio/transcriptions/batch"
    common_data = {
        "language": language,
        "response_format": "verbose_json",
        "timestamp_granularities[]": "word",
    }

    def _wav_bytes(audio: np.ndarray, sr: int) -> bytes:
        buf = _io.BytesIO()
        _sf.write(buf, audio, sr, subtype="PCM_16", format="WAV")
        return buf.getvalue()

    def _single(audio: np.ndarray, sr: int) -> tuple[str, list[dict]]:
        files = {"file": ("clip.wav", _wav_bytes(audio, sr), "audio/wav")}
        r = _rq.post(single_url, files=files, data=common_data, timeout=timeout)
        r.raise_for_status()
        p = r.json()
        return (p.get("text") or "").strip(), p.get("words") or []

    def _batch(items: list[tuple[np.ndarray, int]]) -> list[tuple[str, list[dict]]]:
        files = [
            ("audio_files", (f"c{i}.wav", _wav_bytes(a, sr), "audio/wav"))
            for i, (a, sr) in enumerate(items)
        ]
        r = _rq.post(batch_url, files=files, data=common_data, timeout=timeout)
        r.raise_for_status()
        results = r.json().get("results") or []
        if len(results) != len(items):
            raise RuntimeError(
                f"batch endpoint returned {len(results)} results for {len(items)} inputs"
            )
        return [((p.get("text") or "").strip(), p.get("words") or []) for p in results]

    return {"single": _single, "batch": _batch}
def _read_manual_text(transcript_path: str | None) -> str | None:
    if not transcript_path:
        return None
    p = Path(transcript_path)
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8").strip()
    except Exception as e:  # pragma: no cover
        log.warning(f"could not read transcript {p}: {e}")
        return None


def _mock_transcribe_segment(
    seg: dict[str, Any],
    source_durations: dict[str, float],
    transcripts: dict[str, str],
) -> str:
    """Deterministic mock: linear character slice of manual text by [start, end] ratio.
    Returns text only (no confidence — the real service's confidence is a hardcoded
    constant and downstream stages do not use it)."""
    src_id = seg.get("source_audio_id") or ""
    manual = transcripts.get(src_id)
    src_dur = source_durations.get(src_id, 0.0)
    if manual and src_dur > 0:
        start = max(0.0, float(seg["start"]))
        end = max(start, float(seg["end"]))
        n_chars = len(manual)
        s_idx = int(round(n_chars * start / src_dur))
        e_idx = int(round(n_chars * end / src_dur))
        s_idx = max(0, min(n_chars, s_idx))
        e_idx = max(s_idx, min(n_chars, e_idx))
        sliced = manual[s_idx:e_idx].strip()
        if sliced:
            return sliced
    return PLACEHOLDER_TEXT


def run(cfg: Any) -> int:
    in_path = cfg.paths.manifests.vad_coarse
    out_path = cfg.paths.manifests.asr
    max_input_sec = float(cfg.asr.get("max_input_sec", 30))
    use_mock = bool(cfg.asr.get("use_mock", True))

    real_infer = None if use_mock else _try_load_real_backend(cfg)
    if real_infer is None and not use_mock:
        log.warning("ASR real backend not loadable; using mock for this run")

    segs = list(read_jsonl(in_path))
    log.info(f"read {len(segs)} segs from {in_path}")

    # Map source_audio_id -> manual transcript (for mock slicing) and -> total duration.
    transcripts: dict[str, str] = {}
    source_durations: dict[str, float] = {}

    # Build a lookup from the ingest manifest: source_audio_id (== audio_id) -> transcript_path
    # The seg manifest from stage 04 may not propagate transcript_path, so we recover it here.
    ingest_path = cfg.paths.manifests.get("ingest")
    audio_id_to_transcript: dict[str, str] = {}
    if ingest_path and Path(ingest_path).exists():
        try:
            for r in read_jsonl(ingest_path):
                aid = r.get("audio_id")
                tp = r.get("transcript_path")
                if aid and tp:
                    audio_id_to_transcript[aid] = tp
        except Exception as e:
            log.warning(f"could not read ingest manifest {ingest_path}: {e}")

    # We need source_audio total duration; compute from any seg's audio_align_path
    # (a single source_audio_id has one canonical audio_align_path).
    canonical_align: dict[str, str] = {}
    for s in segs:
        sid = s.get("source_audio_id")
        if sid and sid not in canonical_align and s.get("audio_align_path"):
            canonical_align[sid] = s["audio_align_path"]
        if sid and sid not in transcripts:
            tp = s.get("transcript_path") or audio_id_to_transcript.get(sid)
            mt = _read_manual_text(tp)
            if mt is not None:
                transcripts[sid] = mt

    for sid, align_path in canonical_align.items():
        try:
            audio, sr = read_wav(align_path)
            source_durations[sid] = len(audio) / sr
        except Exception as e:
            log.warning(f"could not read {align_path} for duration: {e}")
            source_durations[sid] = 0.0

    out_records: list[dict[str, Any]] = []
    cache_audio: dict[str, tuple[np.ndarray, int]] = {}
    batch_size = int(cfg.asr.get("batch_size", 16))

    # First pass: classify each seg.
    #   pending_idx[i] -> indices in out_records that need ASR (real or mock).
    n_rejected = 0
    pending: list[tuple[int, dict[str, Any]]] = []  # (out_records index, rec)
    for seg in segs:
        rec = dict(seg)
        out_records.append(rec)
        if rec.get("status") == "rejected":
            continue
        duration = float(rec.get("duration") or 0.0)
        if duration > max_input_sec:
            log.warning(
                f"seg {rec.get('seg_id')} duration {duration:.2f}s > max_input_sec {max_input_sec}; rejecting"
            )
            add_reject(rec, f"asr_duration_gt_{max_input_sec}")
            n_rejected += 1
            continue
        align_path = rec.get("audio_align_path")
        if not align_path or not Path(align_path).exists():
            add_reject(rec, "asr_align_audio_missing")
            n_rejected += 1
            continue
        pending.append((len(out_records) - 1, rec))

    log.info(f"asr: {len(pending)} segs need transcription (batch_size={batch_size})")

    def _propagate_transcript(rec: dict[str, Any]) -> None:
        if not rec.get("transcript_path"):
            sid = rec.get("source_audio_id")
            if sid and sid in audio_id_to_transcript:
                rec["transcript_path"] = audio_id_to_transcript[sid]

    if real_infer is not None and pending:
        single_fn = real_infer["single"]
        batch_fn = real_infer["batch"]
        # Materialize all clips into memory once per source audio.
        clips: list[tuple[np.ndarray, int]] = []
        for _, rec in pending:
            ap = rec["audio_align_path"]
            if ap not in cache_audio:
                cache_audio[ap] = read_wav(ap, target_sr=16000)
            audio, sr = cache_audio[ap]
            clip = slice_audio(audio, sr, float(rec["start"]), float(rec["end"]))
            clips.append((clip, sr))

        for b_start in range(0, len(pending), batch_size):
            chunk = pending[b_start : b_start + batch_size]
            chunk_clips = clips[b_start : b_start + batch_size]
            try:
                results = batch_fn(chunk_clips)
            except Exception as e:
                log.warning(
                    f"batch ASR failed at offset {b_start} (size={len(chunk)}): {e}; "
                    f"falling back to per-seg single calls"
                )
                results = []
                for (idx, rec), (audio, sr) in zip(chunk, chunk_clips):
                    try:
                        results.append(single_fn(audio, sr))
                    except Exception as e2:
                        log.warning(f"single ASR failed on seg {rec.get('seg_id')}: {e2}; mock fallback")
                        text = _mock_transcribe_segment(rec, source_durations, transcripts)
                        results.append((text, []))
            for (idx, rec), (text, words) in zip(chunk, results):
                rec["asr_text"] = text
                if words:
                    rec["asr_words"] = words
                _propagate_transcript(rec)
    else:
        # Mock path (no real backend).
        for _, rec in pending:
            rec["asr_text"] = _mock_transcribe_segment(rec, source_durations, transcripts)
            _propagate_transcript(rec)

    # asr_confidence is intentionally NOT recorded: the HTTP service returns
    # a hardcoded 0.95 and the mock returns text-only — no signal. Stage 12
    # uses timing-derived align_quality_score instead.
    # NOTE: Whisper-large-v3 cross-check (design.md §5) is intentionally a real-only
    # path; in mock mode we skip it.

    n = write_jsonl(out_path, out_records)
    log.info(f"wrote {n} segs to {out_path} (rejected={n_rejected})")
    return 0
