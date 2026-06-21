"""Stage 01: ingest metadata.csv -> ingest manifest.

Mock behavior: no external models needed. Missing audio files or required
columns are written as `status: rejected` records (with reasons), never raised.
Empty `transcript_path` is allowed (downstream goes the ASR-only path).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from prosogate.config import Config
from prosogate.hash_utils import file_hash
from prosogate.logging_utils import get_logger
from prosogate.manifest import write_jsonl

log = get_logger(__name__)

REQUIRED_COLS = ("audio_path", "speaker_id", "language", "domain", "recording_type")


def _resolve(path_str: str, base: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base / p).resolve()


def run(cfg: Config) -> int:
    base = Path.cwd()
    csv_path = _resolve(str(cfg.input.metadata_csv), base)
    out_path = _resolve(str(cfg.paths.manifests.ingest), base)

    if not csv_path.exists():
        log.error("metadata_csv not found: %s", csv_path)
        return 2

    records: list[dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            audio_path_raw = (row.get("audio_path") or "").strip()
            speaker_id = (row.get("speaker_id") or "").strip()
            session_stem = Path(audio_path_raw).stem if audio_path_raw else f"row{row_idx:04d}"
            audio_id = f"{speaker_id}_{session_stem}" if speaker_id else f"_{session_stem}"

            transcript_raw = (row.get("transcript_path") or "").strip()
            transcript_resolved = ""
            if transcript_raw:
                tp = _resolve(transcript_raw, base)
                if tp.exists():
                    transcript_resolved = str(tp)
                else:
                    log.warning(
                        "transcript missing for %s, will run ASR-only path: %s",
                        audio_id,
                        tp,
                    )

            audio_resolved = _resolve(audio_path_raw, base) if audio_path_raw else None

            rec: dict[str, Any] = {
                "audio_id": audio_id,
                "audio_path": str(audio_resolved) if audio_resolved else "",
                "speaker_id": speaker_id,
                "language": (row.get("language") or "").strip(),
                "domain": (row.get("domain") or "").strip(),
                "recording_type": (row.get("recording_type") or "").strip(),
                "transcript_path": transcript_resolved,
                "audio_hash": "",
                "status": "passed",
                "reject_reasons": [],
            }

            missing = [c for c in REQUIRED_COLS if not (row.get(c) or "").strip()]
            if missing:
                rec["status"] = "rejected"
                rec["reject_reasons"].append(f"missing_columns:{','.join(missing)}")
            elif audio_resolved is None or not audio_resolved.exists():
                rec["status"] = "rejected"
                rec["reject_reasons"].append("audio_missing")
            else:
                try:
                    rec["audio_hash"] = file_hash(audio_resolved)
                except Exception as e:  # noqa: BLE001
                    rec["status"] = "rejected"
                    rec["reject_reasons"].append(f"hash_failed:{e}")

            records.append(rec)

    n_total = len(records)
    n_pass = sum(1 for r in records if r["status"] == "passed")
    write_jsonl(out_path, records)
    log.info("ingest: %d total, %d passed -> %s", n_total, n_pass, out_path)
    return 0
