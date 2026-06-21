"""Stage 11: speech-rate metrics from char-level alignment JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from prosogate.logging_utils import get_logger
from prosogate.manifest import read_jsonl, write_jsonl

log = get_logger(__name__)


def _load_alignment(path: str) -> list[dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log.warning(f"stage11: cannot read alignment {path}: {e}")
        return []
    chars = data.get("chars", []) or []
    out = []
    for c in chars:
        try:
            s = float(c.get("start", 0.0))
            e = float(c.get("end", s))
            if e > s:
                out.append({"char": c.get("char", ""), "start": s, "end": e,
                            "confidence": float(c.get("confidence", 0.0))})
        except Exception:
            continue
    return out


def _compute_rate(record: dict[str, Any], cfg) -> dict[str, Any]:
    rate_cfg = cfg.rate
    window_chars = int(rate_cfg.get_path("window_chars", 6))
    step_chars = int(rate_cfg.get_path("step_chars", 3))
    min_chars_for_local = int(rate_cfg.get_path("min_chars_for_local", 12))

    align_path = record.get("alignment_json_path")
    chars = _load_alignment(align_path) if align_path else []
    n_chars = len(chars)
    duration = float(record.get("duration", 0.0))

    metrics: dict[str, Any] = {
        "n_chars": n_chars,
        "global_rate_cps": 0.0,
        "local_rate_mean": 0.0,
        "local_rate_std": 0.0,
        "local_rate_cv": 0.0,
        "local_rate_p5_p95_range": 0.0,
        "pause_ratio": 0.0,
        "long_pause_count": 0,
        "local_rate_skipped": True,
    }

    if n_chars == 0:
        return metrics

    # Global rate: chars / sum of char durations (excluding inter-char silences)
    speech_dur = sum(c["end"] - c["start"] for c in chars)
    if speech_dur > 0:
        metrics["global_rate_cps"] = float(n_chars / speech_dur)

    # Pauses: gaps between chars
    pause_total = 0.0
    long_pause_count = 0
    for i in range(1, n_chars):
        gap = chars[i]["start"] - chars[i - 1]["end"]
        if gap > 0.200:
            pause_total += gap
        if gap > 0.500:
            long_pause_count += 1
    if duration > 0:
        metrics["pause_ratio"] = float(pause_total / duration)
    metrics["long_pause_count"] = int(long_pause_count)

    # Local sliding window
    if n_chars >= min_chars_for_local and n_chars >= window_chars:
        rates: list[float] = []
        for i in range(0, n_chars - window_chars + 1, step_chars):
            span = chars[i + window_chars - 1]["end"] - chars[i]["start"]
            if span > 0:
                rates.append(window_chars / span)
        if len(rates) >= 2:
            arr = np.asarray(rates, dtype=np.float64)
            mean = float(arr.mean())
            std = float(arr.std())
            cv = float(std / mean) if mean > 0 else 0.0
            p5, p95 = np.percentile(arr, [5, 95])
            metrics.update({
                "local_rate_mean": mean,
                "local_rate_std": std,
                "local_rate_cv": cv,
                "local_rate_p5_p95_range": float(p95 - p5),
                "local_rate_skipped": False,
            })

    return metrics


def run(cfg) -> int:
    upstream = cfg.paths.manifests.f0
    out_path = cfg.paths.manifests.rate

    records = list(read_jsonl(upstream))
    log.info(f"stage11: read {len(records)} records from {upstream}")

    out_records: list[dict[str, Any]] = []
    for r in records:
        if r.get("status") == "rejected":
            out_records.append(r)
            continue
        rec = dict(r)
        rec.update(_compute_rate(r, cfg))
        out_records.append(rec)

    n = write_jsonl(out_path, out_records)
    log.info(f"stage11: wrote {n} records -> {out_path}")
    return 0
