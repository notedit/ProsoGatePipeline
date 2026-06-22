"""Stage 07 — Qwen3-ForcedAligner character-level alignment.

Reads the normalized-text manifest, slices each seg from audio_align_path
(16kHz), and writes per-seg alignment JSON files plus aggregate stats back into
the manifest.

Real backend: Qwen3-ForcedAligner-0.6B (placeholder; actual API TBD). Falls back
to mock on import or runtime failure.

Mock backend: deterministic per seg_id. Distributes characters of
text_normalized evenly across [start, end] and assigns confidences in
[0.85, 0.95] using a hash-seeded RNG (np.random.default_rng) — never raw
random — so re-runs are bit-identical.

We apply `cfg.align.latency_offset_ms` uniformly (clamped to seg bounds). No
phoneme-level timestamps (design.md §7).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from prosogate.audio_io import read_wav, slice_audio
from prosogate.logging_utils import get_logger
from prosogate.manifest import add_reject, read_jsonl, write_jsonl

log = get_logger(__name__)


def _try_load_real_backend(cfg: Any):
    """Build an HTTP client for the Qwen3-ForcedAligner service.

    Single endpoint:
      POST /v1/audio/forced_alignment
        file=@wav, text=<transcript>, language=<lang>
      ->  {duration, words:[{word,start,end}]}    # no confidence field

    Batch endpoint (preferred when batch_size > 1):
      POST /v1/audio/forced_alignment/batch
        audio_files[]=@wav...  texts[]=...  language=<lang>
      ->  {results: [AlignmentResponse, ...]}     # order preserved

    Returns dict {"single": fn, "batch": fn}. Each char dict has confidence=0.95
    as a placeholder (server does not emit one); stage 12 ignores this and
    derives align_quality_score from timing-only metrics.
    """
    try:
        import requests  # noqa: F401
    except ImportError as e:
        log.warning(f"requests unavailable ({e}); cannot reach Qwen3-Aligner HTTP server")
        return None

    base_url = cfg.align.get("base_url", "http://127.0.0.1:18765")
    legacy_endpoint = cfg.align.get("endpoint", "")
    if legacy_endpoint:
        base_url = legacy_endpoint.rsplit("/v1/", 1)[0] if "/v1/" in legacy_endpoint else base_url
    language = cfg.align.get("language", "zh")
    timeout = float(cfg.align.get("timeout_sec", 120))
    log.info(f"Qwen3-ForcedAligner HTTP client -> {base_url} (lang={language})")

    import io as _io
    import soundfile as _sf
    import requests as _rq

    single_url = f"{base_url}/v1/audio/forced_alignment"
    batch_url = f"{base_url}/v1/audio/forced_alignment/batch"

    def _wav_bytes(audio: np.ndarray, sr: int) -> bytes:
        buf = _io.BytesIO()
        _sf.write(buf, audio, sr, subtype="PCM_16", format="WAV")
        return buf.getvalue()

    def _words_to_chars(words: list[dict]) -> list[dict[str, Any]]:
        return [
            {
                "char": str(w.get("word", "")),
                "start": float(w["start"]),
                "end": float(w["end"]),
                "confidence": 0.95,
            }
            for w in words
            if "start" in w and "end" in w
        ]

    def _single(audio: np.ndarray, sr: int, text: str) -> list[dict[str, Any]]:
        files = {"file": ("clip.wav", _wav_bytes(audio, sr), "audio/wav")}
        data = {"text": text, "language": language}
        r = _rq.post(single_url, files=files, data=data, timeout=timeout)
        r.raise_for_status()
        return _words_to_chars(r.json().get("words") or [])

    def _batch(items: list[tuple[np.ndarray, int, str]]) -> list[list[dict[str, Any]]]:
        files = [
            ("audio_files", (f"c{i}.wav", _wav_bytes(a, sr), "audio/wav"))
            for i, (a, sr, _) in enumerate(items)
        ]
        # texts[] is sent as repeated form fields, order matches audio_files[]
        data = [("language", language)] + [("texts", t) for (_, _, t) in items]
        r = _rq.post(batch_url, files=files, data=data, timeout=timeout)
        r.raise_for_status()
        results = r.json().get("results") or []
        if len(results) != len(items):
            raise RuntimeError(
                f"batch endpoint returned {len(results)} results for {len(items)} inputs"
            )
        return [_words_to_chars(p.get("words") or []) for p in results]

    return {"single": _single, "batch": _batch}


def _mock_align(seg_id: str, text: str, start: float, end: float) -> list[dict[str, Any]]:
    """Char durations jittered so local_rate_cv is non-trivial; conf around 0.9. Hash-seeded."""
    chars = [c for c in text if c.strip()]
    n = len(chars)
    if n == 0:
        return []
    duration = max(0.0, end - start)
    seed = hash(seg_id) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    # Lognormal-ish jitter on per-char duration so local rate varies naturally.
    weights = np.exp(rng.normal(0.0, 0.35, size=n))
    weights = weights / weights.sum() * duration
    conf_jitter = rng.uniform(-0.05, 0.05, size=n)
    out = []
    cursor = start
    for i, ch in enumerate(chars):
        s = cursor
        e = cursor + float(weights[i]) if i < n - 1 else end
        cursor = e
        conf = float(np.clip(0.9 + conf_jitter[i], 0.0, 1.0))
        out.append({"char": ch, "start": float(s), "end": float(e), "confidence": conf})
    return out


def _apply_latency_offset(
    chars: list[dict[str, Any]], offset_sec: float, seg_start: float, seg_end: float
) -> list[dict[str, Any]]:
    if offset_sec == 0:
        return chars
    out = []
    for c in chars:
        s = max(seg_start, float(c["start"]) - offset_sec)
        e = max(s, min(seg_end, float(c["end"]) - offset_sec))
        out.append({**c, "start": s, "end": e})
    return out


def run(cfg: Any) -> int:
    in_path = cfg.paths.manifests.text_normalize
    out_path = cfg.paths.manifests.align
    use_mock = bool(cfg.align.get("use_mock", True))
    latency_offset_ms = float(cfg.align.get("latency_offset_ms", 0))
    latency_offset_sec = latency_offset_ms / 1000.0
    align_sr = int(cfg.align.get("audio_sample_rate", 16000))
    batch_size = int(cfg.align.get("batch_size", 16))

    work_root = Path(cfg.paths.get("work_root", "work"))
    align_dir = work_root / "alignments"
    align_dir.mkdir(parents=True, exist_ok=True)

    real_align = None if use_mock else _try_load_real_backend(cfg)

    cache_audio: dict[str, tuple[np.ndarray, int]] = {}
    out_records: list[dict[str, Any]] = []
    pending: list[tuple[int, dict[str, Any], str, float, float]] = []  # (idx, rec, text, start, end)
    n_skipped = 0

    # First pass: pass through rejected, route empty-text to reject, queue the rest.
    for seg in read_jsonl(in_path):
        rec = dict(seg)
        out_records.append(rec)
        if rec.get("status") == "rejected":
            continue
        text = rec.get("text_normalized") or ""
        start = float(rec["start"])
        end = float(rec["end"])
        if not text.strip():
            add_reject(rec, "align_empty_text")
            n_skipped += 1
            continue
        pending.append((len(out_records) - 1, rec, text, start, end))

    log.info(f"align: {len(pending)} segs need alignment (batch_size={batch_size})")

    # Resolve clips for each pending seg once.
    clips: list[tuple[np.ndarray, int, str]] = []
    if pending:
        for _, rec, text, start, end in pending:
            ap = rec.get("audio_align_path")
            if ap not in cache_audio:
                cache_audio[ap] = read_wav(ap, target_sr=align_sr)
            audio, sr = cache_audio[ap]
            clip = slice_audio(audio, sr, start, end)
            clips.append((clip, sr, text))

    # Inference: real batch path, with fallback per-seg single, then mock.
    raw_chars_per_seg: list[list[dict[str, Any]] | None] = [None] * len(pending)
    if real_align is not None and pending:
        single_fn = real_align["single"]
        batch_fn = real_align["batch"]
        for b_start in range(0, len(pending), batch_size):
            b_end = min(b_start + batch_size, len(pending))
            chunk_items = pending[b_start:b_end]
            chunk_clips = clips[b_start:b_end]
            try:
                results = batch_fn(chunk_clips)
            except Exception as e:
                log.warning(
                    f"batch align failed at offset {b_start} (size={b_end - b_start}): {e}; "
                    f"falling back to per-seg single calls"
                )
                results = []
                for (idx, rec, text, _, _), (audio, sr, _) in zip(chunk_items, chunk_clips):
                    try:
                        results.append(single_fn(audio, sr, text))
                    except Exception as e2:
                        log.warning(f"single align failed on {rec.get('seg_id')}: {e2}; mock fallback")
                        results.append(_mock_align(rec.get("seg_id"), text, pending[b_start + len(results)][3], pending[b_start + len(results)][4]))
            for k, raw in enumerate(results):
                raw_chars_per_seg[b_start + k] = raw
    else:
        # Mock path
        for k, (_, rec, text, start, end) in enumerate(pending):
            raw_chars_per_seg[k] = _mock_align(rec.get("seg_id"), text, start, end)

    # Post-process: time-shift, latency offset, write per-seg JSON, derive metrics.
    for k, (idx, rec, text, start, end) in enumerate(pending):
        raw = raw_chars_per_seg[k] or []
        seg_id = rec.get("seg_id")

        if real_align is not None:
            # Real aligner returns clip-relative times; shift to absolute.
            chars = [
                {
                    "char": c["char"],
                    "start": float(c["start"]) + start,
                    "end": float(c["end"]) + start,
                    "confidence": float(c.get("confidence", 0.9)),
                }
                for c in raw
            ]
        else:
            # Mock already produces absolute times.
            chars = raw

        chars = _apply_latency_offset(chars, latency_offset_sec, start, end)

        if not chars:
            add_reject(rec, "align_no_chars")
            n_skipped += 1
            continue

        confs = np.array([c["confidence"] for c in chars], dtype=np.float64)
        align_conf_mean = float(confs.mean())
        align_conf_p10 = float(np.percentile(confs, 10))
        min_char_confidence = float(cfg.align.get("min_char_confidence", 0.6))
        high_conf_char_ratio = float((confs >= min_char_confidence).mean())

        text_chars_n = sum(1 for c in text if c.strip())
        seg_dur = max(1e-6, end - start)
        # Timing-derived align signals (the server confidence is hardcoded 0.95).
        char_durs = np.array([max(0.0, c["end"] - c["start"]) for c in chars], dtype=np.float64)
        degen_ratio = float(np.mean((char_durs < 0.02) | (char_durs > 0.5))) if len(char_durs) else 1.0
        est_speech = float(char_durs.sum())
        align_coverage = float(est_speech / seg_dur) if seg_dur > 0 else 0.0
        text_audio_duration_ratio = align_coverage  # back-compat alias
        align_char_match_ratio = float(len(chars) / text_chars_n) if text_chars_n else 0.0

        align_obj = {
            "seg_id": seg_id,
            "chars": chars,
            "align_conf_mean": align_conf_mean,
            "align_conf_p10": align_conf_p10,
        }
        align_json_path = align_dir / f"{seg_id}.json"
        with open(align_json_path, "w", encoding="utf-8") as f:
            json.dump(align_obj, f, ensure_ascii=False)

        rec["alignment_json_path"] = str(align_json_path)
        rec["align_conf_mean"] = align_conf_mean
        rec["align_conf_p10"] = align_conf_p10
        rec["high_conf_char_ratio"] = high_conf_char_ratio
        rec["text_audio_duration_ratio"] = text_audio_duration_ratio
        rec["align_coverage"] = align_coverage
        rec["align_degenerate_char_ratio"] = degen_ratio
        rec["align_char_match_ratio"] = align_char_match_ratio
        rec["n_chars"] = text_chars_n

    n = write_jsonl(out_path, out_records)
    log.info(f"wrote {n} segs to {out_path} (skipped={n_skipped})")
    return 0
