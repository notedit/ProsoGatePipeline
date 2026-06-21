"""Stage 13: train/valid/test/rejected split with leakage protection + dedup."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from prosogate.logging_utils import get_logger
from prosogate.manifest import read_jsonl, write_jsonl

log = get_logger(__name__)


# Fields to retain in final manifests (design.md §14)
_KEEP_FIELDS = [
    "utt_id", "speaker_id", "speaker_label", "wav", "text", "text_normalized",
    "prev_text", "next_text", "duration", "sample_rate", "source_audio_id",
    "source_audio", "start", "end",
    "align_conf_mean", "high_conf_char_ratio", "alignment_json_path",
    "f0_mean_hz", "f0_median_hz", "f0_std_st", "f0_range_st", "f0_delta_p95_st",
    "voiced_ratio", "f0_confidence", "f0_npy_path",
    "global_rate_cps", "local_rate_mean", "local_rate_std", "local_rate_cv",
    "local_rate_p5_p95_range", "pause_ratio", "long_pause_count", "n_chars",
    "audio_quality_score", "align_quality_score", "pitch_score", "rate_score",
    "text_quality_score", "quality_score", "grade", "prosody_bucket",
    "status", "reject_reasons",
    "language", "domain", "recording_type",
]


def _project(rec: dict[str, Any]) -> dict[str, Any]:
    out = {k: rec[k] for k in _KEEP_FIELDS if k in rec}
    # default sample_rate if upstream missing
    if "sample_rate" not in out:
        out["sample_rate"] = 24000
    return out


def _five_grams(text: str) -> set[str]:
    text = (text or "").strip()
    if len(text) < 5:
        return {text} if text else set()
    return {text[i : i + 5] for i in range(len(text) - 4)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _dedup(records: list[dict[str, Any]], threshold: float) -> tuple[list[dict[str, Any]], int]:
    """Within-speaker 5-gram Jaccard dedup. Keep highest quality_score."""
    by_spk: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_spk.setdefault(r.get("speaker_id", ""), []).append(r)

    kept: list[dict[str, Any]] = []
    dropped = 0
    for spk, lst in by_spk.items():
        # sort by quality desc so the first to enter is the strongest representative
        lst_sorted = sorted(lst, key=lambda r: float(r.get("quality_score", 0.0) or 0.0), reverse=True)
        kept_grams: list[tuple[set[str], dict[str, Any]]] = []
        for r in lst_sorted:
            grams = _five_grams(r.get("text") or r.get("text_normalized") or "")
            duplicate = False
            for g, _ in kept_grams:
                if _jaccard(grams, g) > threshold:
                    duplicate = True
                    break
            if duplicate:
                dropped += 1
                continue
            kept_grams.append((grams, r))
            kept.append(r)
    return kept, dropped


def _assign_split(groups: list[tuple[Any, list[dict[str, Any]]]], ratios: dict[str, float],
                   rng: random.Random) -> dict[str, list[dict[str, Any]]]:
    """Assign whole groups to train/valid/test based on cumulative durations."""
    keys = list(groups)
    rng.shuffle(keys)
    total_dur = sum(sum(float(r.get("duration", 0.0) or 0.0) for r in recs) for _, recs in keys)
    target = {k: total_dur * float(v) for k, v in ratios.items()}
    accum = {k: 0.0 for k in ratios}
    out: dict[str, list[dict[str, Any]]] = {k: [] for k in ratios}

    for _, recs in keys:
        gd = sum(float(r.get("duration", 0.0) or 0.0) for r in recs)
        # pick split with largest remaining capacity (relative)
        best = max(ratios.keys(), key=lambda k: target[k] - accum[k])
        out[best].extend(recs)
        accum[best] += gd
    return out


def run(cfg) -> int:
    upstream = cfg.paths.manifests.filter_score
    output_root = Path(cfg.paths.get_path("output_root", "tts_dataset"))
    manifests_dir = output_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    split_cfg = cfg.split
    ratios = dict(split_cfg.get_path("ratios", {"train": 0.92, "valid": 0.04, "test": 0.04}))
    speaker_holdout = bool(split_cfg.get_path("speaker_holdout", False))
    dedup_thr = float(split_cfg.get_path("dedup_similarity", 0.9))
    seed = int(split_cfg.get_path("random_seed", 42))
    rng = random.Random(seed)

    records = list(read_jsonl(upstream))
    log.info(f"stage13: read {len(records)} records from {upstream}")

    rejected: list[dict[str, Any]] = []
    a_b: list[dict[str, Any]] = []
    c_only: list[dict[str, Any]] = []

    for r in records:
        if r.get("status") == "rejected":
            rejected.append(r)
            continue
        grade = r.get("grade", "D")
        if grade in ("A", "B"):
            a_b.append(r)
        elif grade == "C":
            c_only.append(r)
        else:
            # D: should already be rejected; safety net
            rec = dict(r)
            rec["status"] = "rejected"
            rec.setdefault("reject_reasons", []).append("grade_D")
            rejected.append(rec)

    # Dedup within speaker (a_b + c_only). Skip rejected.
    keep_pool = a_b + c_only
    keep_pool, dropped = _dedup(keep_pool, dedup_thr)
    log.info(f"stage13: dedup dropped {dropped} samples (threshold={dedup_thr})")
    a_b_set = {id(r) for r in a_b}
    a_b = [r for r in keep_pool if id(r) in a_b_set]
    c_only = [r for r in keep_pool if id(r) not in a_b_set]

    # Group by leakage key
    def _group_key(r: dict[str, Any]) -> Any:
        if speaker_holdout:
            return r.get("speaker_id", "")
        return r.get("source_audio_id") or r.get("source_audio") or r.get("speaker_id", "")

    groups: dict[Any, list[dict[str, Any]]] = {}
    for r in a_b:
        groups.setdefault(_group_key(r), []).append(r)

    assignment = _assign_split(list(groups.items()), ratios, rng)

    train = list(assignment.get("train", []))
    valid = list(assignment.get("valid", []))
    test = list(assignment.get("test", []))

    # C-only goes into train with split_weight 0.5; keep leakage rule by group key
    c_groups: dict[Any, list[dict[str, Any]]] = {}
    for r in c_only:
        c_groups.setdefault(_group_key(r), []).append(r)
    # If a C-group's key is already in valid/test, keep with that set group; otherwise train
    valid_keys = {_group_key(r) for r in valid}
    test_keys = {_group_key(r) for r in test}
    for k, recs in c_groups.items():
        if k in valid_keys or k in test_keys:
            # don't pollute valid/test with C; drop them to train anyway (low weight)
            for r in recs:
                rr = dict(r)
                rr["split_weight"] = 0.5
                train.append(rr)
        else:
            for r in recs:
                rr = dict(r)
                rr["split_weight"] = 0.5
                train.append(rr)

    # Project + write
    train_path = manifests_dir / "train.jsonl"
    valid_path = manifests_dir / "valid.jsonl"
    test_path = manifests_dir / "test.jsonl"
    rejected_path = manifests_dir / "rejected.jsonl"

    def _project_with_extra(rec: dict[str, Any]) -> dict[str, Any]:
        out = _project(rec)
        if "split_weight" in rec:
            out["split_weight"] = rec["split_weight"]
        return out

    n_train = write_jsonl(train_path, [_project_with_extra(r) for r in train])
    n_valid = write_jsonl(valid_path, [_project_with_extra(r) for r in valid])
    n_test = write_jsonl(test_path, [_project_with_extra(r) for r in test])
    n_rej = write_jsonl(rejected_path, [_project(r) for r in rejected])
    log.info(
        f"stage13: train={n_train} valid={n_valid} test={n_test} rejected={n_rej} "
        f"(seed={seed}, holdout_speaker={speaker_holdout})"
    )
    return 0
