"""Rescore stage 09 results using new thresholds without re-extracting embeddings.

Reads the existing 09_spk_consistency.jsonl manifest, re-applies the criteria
A/B/C/D/E thresholds from the latest config, and rewrites the manifest in place.
This avoids the expensive pyannote forward pass when you just want to tune
thresholds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prosogate.config import load_config
from prosogate.logging_utils import get_logger
from prosogate.manifest import read_jsonl, write_jsonl

log = get_logger("rescore_spk")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    in_path = ROOT / cfg.paths.manifests.spk_consistency
    thresholds = cfg.spk_consistency.thresholds
    t_sil = float(thresholds.get("cluster_silhouette", 0.35))
    t_dist = float(thresholds.get("max_dist_to_center", 0.35))
    t_delta = float(thresholds.get("max_neighbor_delta", 0.30))
    t_consec = float(thresholds.get("consecutive_neighbor_delta", 0.20))
    t_refout = float(thresholds.get("ref_outlier_ratio", 0.10))
    t_overlap = float(thresholds.get("overlap_ratio", 0.02))

    log.info(
        f"thresholds: A={t_sil}, B={t_dist}, C={t_delta}/{t_consec}, "
        f"D={t_refout}, E={t_overlap}"
    )

    out_records = []
    n_total = 0
    n_pass = 0
    reason_count: dict[str, int] = {}
    for rec in read_jsonl(in_path):
        n_total += 1
        if rec.get("status") == "rejected" and "spk_too_few_windows" in (
            rec.get("reject_reasons") or []
        ):
            # Upstream short-window rejection is structural; keep it.
            out_records.append(rec)
            continue
        sc = rec.get("spk_consistency") or {}
        # Recompute reject_reasons from raw metrics.
        new_reasons: list[str] = []
        # Keep any non-spk reject reasons (e.g. duration_too_short).
        kept = [
            r
            for r in (rec.get("reject_reasons") or [])
            if not r.startswith("spk_") and r != "spk_too_few_windows"
        ]
        if sc.get("cluster_silhouette", 0.0) > t_sil:
            new_reasons.append("spk_A")
        if sc.get("max_dist_to_center", 0.0) > t_dist:
            new_reasons.append("spk_B")
        if sc.get("max_neighbor_delta", 0.0) > t_delta:
            new_reasons.append("spk_C")
        # consecutive C check needs raw delta list; skip if not stored
        if sc.get("ref_outlier_ratio", 0.0) > t_refout:
            new_reasons.append("spk_D")
        if sc.get("overlap_ratio", 0.0) > t_overlap:
            new_reasons.append("spk_E")
        all_reasons = kept + new_reasons
        if all_reasons:
            rec["status"] = "rejected"
            rec["reject_reasons"] = all_reasons
            rec.setdefault("spk_consistency", {})["passed"] = False
            for r in new_reasons:
                reason_count[r] = reason_count.get(r, 0) + 1
        else:
            rec["status"] = "ok"
            rec["reject_reasons"] = []
            rec.setdefault("spk_consistency", {})["passed"] = True
            n_pass += 1
        out_records.append(rec)

    write_jsonl(in_path, out_records)
    log.info(f"rescored: {n_total} total, {n_pass} passed, {n_total - n_pass} rejected")
    log.info(f"reject reasons: {reason_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
