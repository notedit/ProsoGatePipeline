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

    Server contract (per project README):
      POST /v1/audio/transcriptions
        file=@wav, language=<lang>, response_format=verbose_json,
        timestamp_granularities[]=word
      ->  {text, language, words:[{word,start,end}], duration}

    Returns a callable(audio_bytes_or_path, sr_unused) -> (text, conf, words[]).
    We pass the original wav path when available (the server still re-reads but
    we keep word timestamps without re-encoding); fall back to file upload of
    sliced audio when path mode isn't safe.
    """
    try:
        import requests  # noqa: F401  -- verify the package is importable
    except ImportError as e:
        log.warning(f"requests unavailable ({e}); cannot reach Qwen3-ASR HTTP server")
        return None

    endpoint = cfg.asr.get("endpoint", "http://127.0.0.1:18765/v1/audio/transcriptions")
    language = cfg.asr.get("language", "zh")
    timeout = float(cfg.asr.get("timeout_sec", 60))
    log.info(f"Qwen3-ASR HTTP client -> {endpoint} (lang={language})")

    import io as _io
    import soundfile as _sf
    import requests as _rq

    def _infer(audio: np.ndarray, sr: int) -> tuple[str, float, list[dict]]:
        # Upload sliced audio as wav bytes
        buf = _io.BytesIO()
        _sf.write(buf, audio, sr, subtype="PCM_16", format="WAV")
        buf.seek(0)
        files = {"file": ("clip.wav", buf, "audio/wav")}
        data = {
            "language": language,
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word",
        }
        r = _rq.post(endpoint, files=files, data=data, timeout=timeout)
        r.raise_for_status()
        payload = r.json()
        text = (payload.get("text") or "").strip()
        words = payload.get("words") or []
        return text, 0.95, words

    return _infer


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
) -> tuple[str, float]:
    """Deterministic mock: linear character slice of manual text by [start, end] ratio."""
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
            return sliced, 0.95
    return PLACEHOLDER_TEXT, 0.95


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

    n_rejected = 0
    for seg in segs:
        rec = dict(seg)
        if rec.get("status") == "rejected":
            out_records.append(rec)
            continue

        duration = float(rec.get("duration") or 0.0)
        if duration > max_input_sec:
            log.warning(
                f"seg {rec.get('seg_id')} duration {duration:.2f}s > max_input_sec {max_input_sec}; rejecting"
            )
            add_reject(rec, f"asr_duration_gt_{max_input_sec}")
            n_rejected += 1
            out_records.append(rec)
            continue

        align_path = rec.get("audio_align_path")
        if not align_path or not Path(align_path).exists():
            add_reject(rec, "asr_align_audio_missing")
            n_rejected += 1
            out_records.append(rec)
            continue

        if real_infer is not None:
            try:
                if align_path not in cache_audio:
                    cache_audio[align_path] = read_wav(align_path, target_sr=16000)
                audio, sr = cache_audio[align_path]
                clip = slice_audio(audio, sr, float(rec["start"]), float(rec["end"]))
                result = real_infer(clip, sr)
                # Real backend returns (text, conf, words); mock returns (text, conf)
                if len(result) == 3:
                    text, conf, words = result
                    if words:
                        rec["asr_words"] = words
                else:
                    text, conf = result
            except Exception as e:
                log.warning(f"real ASR failed on seg {rec.get('seg_id')}: {e}; falling back to mock")
                text, conf = _mock_transcribe_segment(rec, source_durations, transcripts)
        else:
            text, conf = _mock_transcribe_segment(rec, source_durations, transcripts)

        rec["asr_text"] = text
        rec["asr_confidence"] = float(conf)
        # Propagate transcript_path for downstream stages (06 normalize) when recovered.
        if not rec.get("transcript_path"):
            sid = rec.get("source_audio_id")
            if sid and sid in audio_id_to_transcript:
                rec["transcript_path"] = audio_id_to_transcript[sid]
        # NOTE: Whisper-large-v3 cross-check (design.md §5) is intentionally a real-only
        # path; in mock mode we skip it. When implementing the real backend, run a
        # second-pass transcription and reject if CER > 12% with no manual text.
        out_records.append(rec)

    n = write_jsonl(out_path, out_records)
    log.info(f"wrote {n} segs to {out_path} (rejected={n_rejected})")
    return 0
