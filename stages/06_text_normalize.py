"""Stage 06 — Text normalization.

Reads ASR manifest, applies a single normalization pipeline to either the manual
transcript (preferred when available) or the ASR output, and writes
`text_normalized` for downstream alignment.

Normalization steps (best effort; each library is optional):
  - Traditional -> Simplified Chinese (opencc)
  - Number -> Chinese characters (cn2an)
  - Punctuation -> Chinese full-width
  - Whitespace cleanup

Also computes CER between manual and ASR text when both are available; samples
with CER > 0.08 are tagged `status: review` (NOT rejected — surfaced for
downstream scoring).
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from prosogate.logging_utils import get_logger
from prosogate.manifest import read_jsonl, write_jsonl

log = get_logger(__name__)

# Half-width -> Chinese full-width punctuation map.
_PUNCT_MAP = {
    ",": "，",
    ".": "。",
    "?": "？",
    "!": "！",
    ";": "；",
    ":": "：",
    "(": "（",
    ")": "）",
    "[": "［",
    "]": "］",
}

CER_REVIEW_THRESHOLD = 0.08


def _try_opencc():
    try:
        from opencc import OpenCC  # type: ignore

        return OpenCC("t2s")
    except Exception as e:
        log.warning(f"opencc unavailable ({e!r}); skipping zh-Hans conversion")
        return None


def _try_cn2an():
    try:
        import cn2an  # type: ignore

        return cn2an
    except Exception as e:
        log.warning(f"cn2an unavailable ({e!r}); skipping number normalization")
        return None


def _normalize_text(text: str, opencc, cn2an_mod) -> str:
    if not text:
        return ""
    s = text
    if opencc is not None:
        try:
            s = opencc.convert(s)
        except Exception:
            pass
    if cn2an_mod is not None:
        try:
            # transform="an2cn" converts arabic numerals embedded in text
            s = cn2an_mod.transform(s, "an2cn")
        except Exception:
            pass
    # Punctuation: only convert ASCII punctuation that is NOT inside digit-only
    # sequences (we already converted those). Conservative simple replace.
    s = "".join(_PUNCT_MAP.get(ch, ch) for ch in s)
    # Collapse whitespace; in Chinese we drop inter-character spaces.
    s = re.sub(r"\s+", " ", s).strip()
    # Drop spaces that sit between two non-ASCII (CJK) characters.
    s = re.sub(r"(?<=[^\x00-\x7f]) (?=[^\x00-\x7f])", "", s)
    return s


def _cer(ref: str, hyp: str) -> float:
    """Character error rate via SequenceMatcher (rough approximation, sufficient for review-flagging)."""
    if not ref:
        return 0.0 if not hyp else 1.0
    sm = SequenceMatcher(a=ref, b=hyp, autojunk=False)
    matched = sum(blk.size for blk in sm.get_matching_blocks())
    edits = max(len(ref), len(hyp)) - matched
    return edits / max(1, len(ref))


def _read_manual(transcript_path: str | None) -> str | None:
    if not transcript_path:
        return None
    p = Path(transcript_path)
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def run(cfg: Any) -> int:
    in_path = cfg.paths.manifests.asr
    out_path = cfg.paths.manifests.text_normalize

    opencc = _try_opencc()
    cn2an_mod = _try_cn2an()

    out_records: list[dict[str, Any]] = []
    n_review = 0

    # If a seg's source_audio has a manual transcript, we slice it the same way
    # stage 05 did so manual_text aligns with the seg time range.
    transcripts: dict[str, str] = {}
    source_durations: dict[str, float] = {}

    segs = list(read_jsonl(in_path))
    for s in segs:
        sid = s.get("source_audio_id")
        if sid and sid not in transcripts:
            mt = _read_manual(s.get("transcript_path"))
            if mt is not None:
                transcripts[sid] = mt

    # Estimate source duration: use max(end) per source_audio_id as a cheap proxy
    # (avoids re-reading audio). This gives the proportional slice for manual_text.
    for s in segs:
        sid = s.get("source_audio_id")
        if not sid:
            continue
        end = float(s.get("end") or 0.0)
        source_durations[sid] = max(source_durations.get(sid, 0.0), end)

    for seg in segs:
        rec = dict(seg)

        sid = rec.get("source_audio_id")
        manual_full = transcripts.get(sid) if sid else None
        manual_text: str | None = None
        if manual_full and source_durations.get(sid, 0) > 0:
            src_dur = source_durations[sid]
            n_chars = len(manual_full)
            s_idx = int(round(n_chars * float(rec["start"]) / src_dur))
            e_idx = int(round(n_chars * float(rec["end"]) / src_dur))
            s_idx = max(0, min(n_chars, s_idx))
            e_idx = max(s_idx, min(n_chars, e_idx))
            cut = manual_full[s_idx:e_idx].strip()
            manual_text = cut if cut else None

        asr_text = rec.get("asr_text") or ""
        raw_text = manual_text if manual_text else asr_text

        text_normalized = _normalize_text(raw_text, opencc, cn2an_mod)
        rec["text_normalized"] = text_normalized
        if manual_text is not None:
            rec["manual_text"] = manual_text

        if manual_text and asr_text:
            cer = _cer(manual_text, asr_text)
            rec["cer_vs_manual"] = float(cer)
            if cer > CER_REVIEW_THRESHOLD and rec.get("status") != "rejected":
                rec["status"] = "review"
                n_review += 1

        out_records.append(rec)

    n = write_jsonl(out_path, out_records)
    log.info(f"wrote {n} segs to {out_path} (review={n_review})")
    return 0
