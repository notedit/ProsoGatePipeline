"""Stage 12: hard-rule filtering + composite scoring + grading."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from prosogate.logging_utils import get_logger
from prosogate.manifest import read_jsonl, write_jsonl

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Score curves
# ---------------------------------------------------------------------------


def _bell(value: float, lo: float, hi: float, decay: float) -> float:
    """Plateau in [lo, hi], linear decay to 0 at lo-decay / hi+decay."""
    if value <= lo - decay or value >= hi + decay:
        return 0.0
    if value < lo:
        return max(0.0, (value - (lo - decay)) / decay)
    if value > hi:
        return max(0.0, ((hi + decay) - value) / decay)
    return 1.0


def _ramp_then_decay(value: float, full_lo: float, full_hi: float, zero_at: float) -> float:
    """1.0 in [full_lo, full_hi], linearly to 0 at zero_at (zero_at > full_hi)."""
    if value <= full_hi:
        return 1.0 if value >= full_lo else 0.0
    if value >= zero_at:
        return 0.0
    span = max(1e-6, zero_at - full_hi)
    return max(0.0, 1.0 - (value - full_hi) / span)


def _audio_quality_score(rec: dict[str, Any]) -> float:
    snr = float(rec.get("snr_db", 0.0) or 0.0)
    clip = float(rec.get("clipping_ratio", 0.0) or 0.0)
    lufs = float(rec.get("lufs", -25.0) or -25.0)
    snr_score = max(0.0, min(1.0, (snr - 5.0) / (30.0 - 5.0)))  # 5dB -> 0, 30dB -> 1
    clip_score = max(0.0, min(1.0, 1.0 - clip / 0.01))           # 0 -> 1, >=1% -> 0
    # LUFS bell around -23
    lufs_score = max(0.0, 1.0 - abs(lufs - (-23.0)) / 12.0)
    return float(0.5 * snr_score + 0.2 * clip_score + 0.3 * lufs_score)


def _pitch_score(rec: dict[str, Any], cfg) -> float:
    std_lo, std_hi = cfg.score.get_path("pitch_optimal_std_st_range", [2.0, 6.0])
    delta_full_lo, delta_full_hi = cfg.score.get_path(
        "pitch_delta_p95_full_score_range", [0.0, 4.0]
    )
    delta_zero_at = float(cfg.score.get_path("pitch_delta_p95_zero_at", 6.0))

    std_v = float(rec.get("f0_std_st", 0.0) or 0.0)
    delta_v = float(rec.get("f0_delta_p95_st", 0.0) or 0.0)
    std_score = _bell(std_v, float(std_lo), float(std_hi), 2.0)
    delta_score = _ramp_then_decay(delta_v, float(delta_full_lo), float(delta_full_hi), delta_zero_at)
    return float(0.5 * (std_score + delta_score))


def _rate_score(rec: dict[str, Any], cfg) -> float:
    cps_lo, cps_hi = cfg.score.get_path("rate_optimal_cps_range", [3.5, 6.0])
    cv_lo, cv_hi = cfg.score.get_path("rate_optimal_cv_range", [0.12, 0.45])
    cps = float(rec.get("global_rate_cps", 0.0) or 0.0)
    cps_score = _bell(cps, float(cps_lo), float(cps_hi), 2.0)
    if rec.get("local_rate_skipped"):
        return cps_score
    cv = float(rec.get("local_rate_cv", 0.0) or 0.0)
    cv_score = _bell(cv, float(cv_lo), float(cv_hi), 0.20)
    return float(0.5 * (cps_score + cv_score))


def _text_quality_score(rec: dict[str, Any]) -> float:
    cer = rec.get("cer_vs_manual")
    if cer is None:
        return 0.85
    try:
        return float(max(0.0, min(1.0, 1.0 - float(cer))))
    except Exception:
        return 0.85


# ---------------------------------------------------------------------------
# Hard-rule filtering
# ---------------------------------------------------------------------------


def _hard_rule_check(rec: dict[str, Any], cfg, spk_thresholds: dict[str, dict[str, float]]
                      ) -> list[str]:
    reasons: list[str] = []
    f0_filt = cfg.f0.get_path("filter", {})
    rate_filt = cfg.rate.get_path("filter", {})

    voiced_ratio = float(rec.get("voiced_ratio", 0.0) or 0.0)
    if voiced_ratio < float(f0_filt.get("min_voiced_ratio", 0.45)):
        reasons.append(f"voiced_ratio<{f0_filt['min_voiced_ratio']}")
    f0_conf = float(rec.get("f0_confidence", 0.0) or 0.0)
    if f0_conf < float(f0_filt.get("min_f0_confidence", 0.75)):
        reasons.append(f"f0_confidence<{f0_filt['min_f0_confidence']}")
    delta_p95 = float(rec.get("f0_delta_p95_st", 0.0) or 0.0)
    if delta_p95 > float(f0_filt.get("max_f0_delta_p95_st", 6.0)):
        reasons.append(f"f0_delta_p95_st>{f0_filt['max_f0_delta_p95_st']}")

    spk = rec.get("speaker_id", "")
    spk_f0 = spk_thresholds.get("f0", {}).get(spk)
    if spk_f0 is not None:
        std_lo, std_hi, range_lo, range_hi = spk_f0
    else:
        std_lo = float(f0_filt.get("min_f0_std_st", 1.0))
        std_hi = float(f0_filt.get("max_f0_std_st", 10.0))
        range_lo = float(f0_filt.get("min_f0_range_st", 3.0))
        range_hi = float(f0_filt.get("max_f0_range_st", 20.0))
    f0_std = float(rec.get("f0_std_st", 0.0) or 0.0)
    if f0_std < std_lo:
        reasons.append(f"f0_std_st<{std_lo:.2f}")
    if f0_std > std_hi:
        reasons.append(f"f0_std_st>{std_hi:.2f}")
    f0_range = float(rec.get("f0_range_st", 0.0) or 0.0)
    if f0_range < range_lo:
        reasons.append(f"f0_range_st<{range_lo:.2f}")
    if f0_range > range_hi:
        reasons.append(f"f0_range_st>{range_hi:.2f}")

    spk_rate = spk_thresholds.get("rate", {}).get(spk)
    if spk_rate is not None:
        cps_lo, cps_hi, cv_lo, cv_hi = spk_rate
    else:
        cps_lo = float(rate_filt.get("min_global_rate_cps", 2.0))
        cps_hi = float(rate_filt.get("max_global_rate_cps", 8.0))
        cv_lo = float(rate_filt.get("min_local_rate_cv", 0.06))
        cv_hi = float(rate_filt.get("max_local_rate_cv", 0.65))
    cps = float(rec.get("global_rate_cps", 0.0) or 0.0)
    if cps < cps_lo:
        reasons.append(f"global_rate_cps<{cps_lo:.2f}")
    if cps > cps_hi:
        reasons.append(f"global_rate_cps>{cps_hi:.2f}")
    if not rec.get("local_rate_skipped"):
        cv = float(rec.get("local_rate_cv", 0.0) or 0.0)
        if cv < cv_lo:
            reasons.append(f"local_rate_cv<{cv_lo:.2f}")
        if cv > cv_hi:
            reasons.append(f"local_rate_cv>{cv_hi:.2f}")
    pause_ratio = float(rec.get("pause_ratio", 0.0) or 0.0)
    if pause_ratio > float(rate_filt.get("max_pause_ratio", 0.45)):
        reasons.append(f"pause_ratio>{rate_filt['max_pause_ratio']}")

    align_conf_mean = float(rec.get("align_conf_mean", 1.0) or 0.0)
    min_align = float(cfg.align.get_path("min_char_confidence", 0.6))
    if align_conf_mean < min_align:
        reasons.append(f"align_conf_mean<{min_align}")
    high_conf_ratio = rec.get("high_conf_char_ratio")
    if high_conf_ratio is not None:
        min_hr = float(cfg.align.get_path("min_high_conf_char_ratio", 0.85))
        if float(high_conf_ratio) < min_hr:
            reasons.append(f"high_conf_char_ratio<{min_hr}")

    return reasons


# ---------------------------------------------------------------------------
# Speaker-adaptive thresholds (P10-P95)
# ---------------------------------------------------------------------------


def _speaker_adaptive_thresholds(records: list[dict[str, Any]], cfg
                                  ) -> dict[str, dict[str, tuple[float, float, float, float]]]:
    out: dict[str, dict[str, tuple[float, float, float, float]]] = {"f0": {}, "rate": {}}

    # F0 adaptive
    f0_adapt = cfg.f0.get_path("speaker_adaptive", {})
    if f0_adapt.get("enabled", False):
        min_n = int(f0_adapt.get("min_samples_per_speaker", 200))
        plo, phi = f0_adapt.get("keep_percentile", [10, 95])
        by_spk: dict[str, list[dict[str, float]]] = {}
        for r in records:
            if r.get("status") == "rejected":
                continue
            spk = r.get("speaker_id", "")
            by_spk.setdefault(spk, []).append({
                "std": float(r.get("f0_std_st", 0.0) or 0.0),
                "range": float(r.get("f0_range_st", 0.0) or 0.0),
            })
        for spk, lst in by_spk.items():
            if len(lst) < min_n:
                continue
            stds = np.asarray([x["std"] for x in lst], dtype=np.float64)
            ranges = np.asarray([x["range"] for x in lst], dtype=np.float64)
            std_lo, std_hi = np.percentile(stds, [plo, phi])
            r_lo, r_hi = np.percentile(ranges, [plo, phi])
            out["f0"][spk] = (float(std_lo), float(std_hi), float(r_lo), float(r_hi))

    # Rate adaptive
    rate_adapt = cfg.rate.get_path("speaker_adaptive", {})
    if rate_adapt.get("enabled", False):
        min_n = int(rate_adapt.get("min_samples_per_speaker", 200))
        plo, phi = rate_adapt.get("keep_percentile", [10, 95])
        by_spk2: dict[str, list[dict[str, float]]] = {}
        for r in records:
            if r.get("status") == "rejected":
                continue
            spk = r.get("speaker_id", "")
            by_spk2.setdefault(spk, []).append({
                "cps": float(r.get("global_rate_cps", 0.0) or 0.0),
                "cv": float(r.get("local_rate_cv", 0.0) or 0.0),
            })
        for spk, lst in by_spk2.items():
            if len(lst) < min_n:
                continue
            cps = np.asarray([x["cps"] for x in lst], dtype=np.float64)
            cv = np.asarray([x["cv"] for x in lst], dtype=np.float64)
            cps_lo, cps_hi = np.percentile(cps, [plo, phi])
            cv_lo, cv_hi = np.percentile(cv, [plo, phi])
            out["rate"][spk] = (float(cps_lo), float(cps_hi), float(cv_lo), float(cv_hi))

    return out


# ---------------------------------------------------------------------------
# Grade & prosody bucket
# ---------------------------------------------------------------------------


def _grade(score: float, cfg) -> str:
    th = cfg.score.get_path("grade_thresholds", {"A": 0.85, "B": 0.75, "C": 0.60})
    if score >= float(th.get("A", 0.85)):
        return "A"
    if score >= float(th.get("B", 0.75)):
        return "B"
    if score >= float(th.get("C", 0.60)):
        return "C"
    return "D"


def _prosody_bucket(rec: dict[str, Any]) -> str:
    std = float(rec.get("f0_std_st", 0.0) or 0.0)
    cv = float(rec.get("local_rate_cv", 0.0) or 0.0)
    skipped = bool(rec.get("local_rate_skipped"))
    if std < 2.0 or (not skipped and cv < 0.10):
        return "flat"
    if 2.0 <= std < 5.0 and (skipped or 0.10 <= cv < 0.35):
        return "natural"
    if (5.0 <= std <= 8.0) or (not skipped and 0.35 <= cv <= 0.55):
        return "expressive"
    return "chaotic"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run(cfg) -> int:
    upstream = cfg.paths.manifests.rate
    out_path = cfg.paths.manifests.filter_score
    output_root = Path(cfg.paths.get_path("output_root", "tts_dataset"))
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    rejected_csv = reports_dir / "rejected_samples.csv"

    records = list(read_jsonl(upstream))
    log.info(f"stage12: read {len(records)} records from {upstream}")

    spk_thresholds = _speaker_adaptive_thresholds(records, cfg)
    if spk_thresholds["f0"] or spk_thresholds["rate"]:
        log.info(
            f"stage12: speaker-adaptive thresholds active "
            f"(f0: {len(spk_thresholds['f0'])} spk, rate: {len(spk_thresholds['rate'])} spk)"
        )

    weights = cfg.score.get_path("weights", {})
    w_audio = float(weights.get("audio", 0.25))
    w_align = float(weights.get("align", 0.20))
    w_pitch = float(weights.get("pitch", 0.25))
    w_rate = float(weights.get("rate", 0.20))
    w_text = float(weights.get("text", 0.10))

    out_records: list[dict[str, Any]] = []
    reason_counter: Counter[str] = Counter()

    for r in records:
        rec = dict(r)
        if rec.get("status") == "rejected":
            for rs in rec.get("reject_reasons", []) or []:
                reason_counter[rs] += 1
            out_records.append(rec)
            continue

        # Hard rules
        reasons = _hard_rule_check(rec, cfg, spk_thresholds)

        audio_q = _audio_quality_score(rec)
        align_q = float(rec.get("align_conf_mean", 0.0) or 0.0)
        pitch_q = _pitch_score(rec, cfg)
        rate_q = _rate_score(rec, cfg)
        text_q = _text_quality_score(rec)

        score = (
            w_audio * audio_q
            + w_align * align_q
            + w_pitch * pitch_q
            + w_rate * rate_q
            + w_text * text_q
        )
        score = float(max(0.0, min(1.0, score)))

        rec.update({
            "audio_quality_score": float(audio_q),
            "align_quality_score": float(align_q),
            "pitch_score": float(pitch_q),
            "rate_score": float(rate_q),
            "text_quality_score": float(text_q),
            "quality_score": score,
            "grade": _grade(score, cfg),
            "prosody_bucket": _prosody_bucket(rec),
        })

        if reasons:
            existing = rec.get("reject_reasons") or []
            rec["reject_reasons"] = list(existing) + reasons
            rec["status"] = "rejected"
            for rs in reasons:
                reason_counter[rs] += 1
        else:
            rec.setdefault("status", "ok")
            rec.setdefault("reject_reasons", [])

        out_records.append(rec)

    n = write_jsonl(out_path, out_records)
    log.info(f"stage12: wrote {n} records -> {out_path}")

    # Rejected reasons CSV
    try:
        with open(rejected_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["reject_reason", "count"])
            for reason, cnt in reason_counter.most_common():
                w.writerow([reason, cnt])
        log.info(f"stage12: rejected_samples.csv -> {rejected_csv} ({len(reason_counter)} reasons)")
    except Exception as e:
        log.warning(f"stage12: cannot write {rejected_csv}: {e}")
    return 0
