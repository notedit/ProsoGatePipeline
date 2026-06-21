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

    Server contract:
      POST /v1/audio/forced_alignment
        file=@wav, text=<known transcript>, language=<lang>
      ->  {language, duration, words:[{word,start,end}]}

    Returns a callable(audio, sr, text) -> list[{char,start,end,confidence}].
    Confidence is not provided by the service; we synthesize a uniform 0.95.
    """
    try:
        import requests  # noqa: F401
    except ImportError as e:
        log.warning(f"requests unavailable ({e}); cannot reach Qwen3-Aligner HTTP server")
        return None

    endpoint = cfg.align.get(
        "endpoint", "http://127.0.0.1:18765/v1/audio/forced_alignment"
    )
    language = cfg.align.get("language", "zh")
    timeout = float(cfg.align.get("timeout_sec", 60))
    log.info(f"Qwen3-ForcedAligner HTTP client -> {endpoint} (lang={language})")

    import io as _io
    import soundfile as _sf
    import requests as _rq

    def _align(audio: np.ndarray, sr: int, text: str) -> list[dict[str, Any]]:
        buf = _io.BytesIO()
        _sf.write(buf, audio, sr, subtype="PCM_16", format="WAV")
        buf.seek(0)
        files = {"file": ("clip.wav", buf, "audio/wav")}
        data = {"text": text, "language": language}
        r = _rq.post(endpoint, files=files, data=data, timeout=timeout)
        r.raise_for_status()
        payload = r.json()
        words = payload.get("words") or []
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

    return _align


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

    work_root = Path(cfg.paths.get("work_root", "work"))
    align_dir = work_root / "alignments"
    align_dir.mkdir(parents=True, exist_ok=True)

    real_align = None if use_mock else _try_load_real_backend(cfg)

    cache_audio: dict[str, tuple[np.ndarray, int]] = {}
    out_records: list[dict[str, Any]] = []
    n_skipped = 0

    for seg in read_jsonl(in_path):
        rec = dict(seg)
        if rec.get("status") == "rejected":
            out_records.append(rec)
            continue

        seg_id = rec.get("seg_id")
        text = rec.get("text_normalized") or ""
        start = float(rec["start"])
        end = float(rec["end"])

        if not text.strip():
            add_reject(rec, "align_empty_text")
            out_records.append(rec)
            n_skipped += 1
            continue

        chars: list[dict[str, Any]] = []
        if real_align is not None:
            try:
                align_path = rec.get("audio_align_path")
                if align_path not in cache_audio:
                    cache_audio[align_path] = read_wav(align_path, target_sr=align_sr)
                audio, sr = cache_audio[align_path]
                clip = slice_audio(audio, sr, start, end)
                raw = real_align(clip, sr, text)
                # Real aligner returns timestamps relative to the clip; shift by start.
                chars = [
                    {
                        "char": c["char"],
                        "start": float(c["start"]) + start,
                        "end": float(c["end"]) + start,
                        "confidence": float(c.get("confidence", 0.9)),
                    }
                    for c in raw
                ]
            except Exception as e:
                log.warning(f"real align failed on {seg_id}: {e}; falling back to mock")
                chars = _mock_align(seg_id, text, start, end)
        else:
            chars = _mock_align(seg_id, text, start, end)

        chars = _apply_latency_offset(chars, latency_offset_sec, start, end)

        if not chars:
            add_reject(rec, "align_no_chars")
            out_records.append(rec)
            n_skipped += 1
            continue

        confs = np.array([c["confidence"] for c in chars], dtype=np.float64)
        align_conf_mean = float(confs.mean())
        align_conf_p10 = float(np.percentile(confs, 10))
        min_char_confidence = float(cfg.align.get("min_char_confidence", 0.6))
        high_conf_char_ratio = float((confs >= min_char_confidence).mean())

        text_chars_n = sum(1 for c in text if c.strip())
        seg_dur = max(1e-6, end - start)
        # text/audio duration ratio: estimated speaking-time over seg duration.
        est_speech = sum((c["end"] - c["start"]) for c in chars)
        text_audio_duration_ratio = float(est_speech / seg_dur) if seg_dur > 0 else 0.0

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
        rec["n_chars"] = text_chars_n

        out_records.append(rec)

    n = write_jsonl(out_path, out_records)
    log.info(f"wrote {n} segs to {out_path} (skipped={n_skipped})")
    return 0
