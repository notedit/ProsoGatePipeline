"""Stage 14: aggregate quality report (HTML / PNG / CSV) with text fallback."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prosogate.logging_utils import get_logger
from prosogate.manifest import read_jsonl

log = get_logger(__name__)


def _load_split(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(read_jsonl(path))


def _speaker_stats(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_spk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_spk[r.get("speaker_id", "")].append(r)
    rows: list[dict[str, Any]] = []
    for spk, recs in sorted(by_spk.items()):
        n = len(recs)
        total_dur = sum(float(r.get("duration", 0.0) or 0.0) for r in recs)
        scores = [float(r.get("quality_score", 0.0) or 0.0) for r in recs]
        avg_q = (sum(scores) / len(scores)) if scores else 0.0
        grade_counts = Counter(r.get("grade", "?") for r in recs)
        rejected = sum(1 for r in recs if r.get("status") == "rejected")
        rows.append({
            "speaker_id": spk,
            "n_utt": n,
            "total_duration_sec": round(total_dur, 2),
            "avg_quality_score": round(avg_q, 4),
            "grade_A": grade_counts.get("A", 0),
            "grade_B": grade_counts.get("B", 0),
            "grade_C": grade_counts.get("C", 0),
            "grade_D": grade_counts.get("D", 0),
            "rejected": rejected,
        })
    return rows


def _write_speaker_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = [
        "speaker_id", "n_utt", "total_duration_sec", "avg_quality_score",
        "grade_A", "grade_B", "grade_C", "grade_D", "rejected",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _try_plot_pitch(records: list[dict[str, Any]], out_path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log.warning(f"stage14: matplotlib unavailable, skipping {out_path.name}: {e}")
        return False
    grades = ["A", "B", "C", "D"]
    colors = {"A": "#2ca02c", "B": "#1f77b4", "C": "#ff7f0e", "D": "#d62728"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    has_any = False
    for g in grades:
        vals = [
            float(r.get("f0_std_st", 0.0) or 0.0)
            for r in records
            if r.get("grade") == g and r.get("f0_std_st") is not None
        ]
        if vals:
            ax.hist(vals, bins=30, alpha=0.6, label=f"grade {g} (n={len(vals)})",
                    color=colors[g], range=(0, 12))
            has_any = True
    if not has_any:
        plt.close(fig)
        return False
    ax.set_xlabel("f0_std_st (semitones)")
    ax.set_ylabel("count")
    ax.set_title("F0 std distribution by grade")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return True


def _try_plot_rate(records: list[dict[str, Any]], out_path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log.warning(f"stage14: matplotlib unavailable, skipping {out_path.name}: {e}")
        return False
    vals = [
        float(r.get("global_rate_cps", 0.0) or 0.0)
        for r in records
        if r.get("global_rate_cps") is not None
    ]
    vals = [v for v in vals if v > 0]
    if not vals:
        return False
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(vals, bins=30, color="#1f77b4", alpha=0.8, range=(0, 10))
    ax.set_xlabel("global_rate_cps (chars/sec)")
    ax.set_ylabel("count")
    ax.set_title("Speech rate distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return True


def _summary(records: list[dict[str, Any]],
             splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    n_total = len(records)
    n_rejected = sum(1 for r in records if r.get("status") == "rejected")
    n_active = n_total - n_rejected
    grades = Counter(r.get("grade", "?") for r in records if r.get("status") != "rejected")
    bucket = Counter(r.get("prosody_bucket", "?") for r in records if r.get("status") != "rejected")
    total_dur = sum(float(r.get("duration", 0.0) or 0.0) for r in records)
    accepted_dur = sum(float(r.get("duration", 0.0) or 0.0)
                       for r in records if r.get("status") != "rejected")
    return {
        "n_total": n_total,
        "n_rejected": n_rejected,
        "n_active": n_active,
        "total_duration_sec": round(total_dur, 2),
        "accepted_duration_sec": round(accepted_dur, 2),
        "grade_counts": dict(grades),
        "prosody_bucket_counts": dict(bucket),
        "split_counts": {k: len(v) for k, v in splits.items()},
    }


def _write_html(path: Path, summary: dict[str, Any], speaker_rows: list[dict[str, Any]],
                pitch_png: Path, rate_png: Path) -> bool:
    try:
        from jinja2 import Template
    except Exception as e:
        log.warning(f"stage14: jinja2 unavailable, falling back to text report: {e}")
        return False

    tmpl = Template(
        """<!doctype html>
<html><head><meta charset="utf-8"><title>ProsoGate Quality Report</title>
<style>
body { font-family: sans-serif; margin: 24px; }
table { border-collapse: collapse; margin: 12px 0; }
th, td { border: 1px solid #ccc; padding: 4px 10px; }
th { background: #eee; text-align: left; }
img { max-width: 700px; }
.kv { margin: 4px 0; }
</style></head>
<body>
<h1>ProsoGate Quality Report</h1>

<h2>Summary</h2>
<div class="kv"><b>Total utts:</b> {{ s.n_total }}</div>
<div class="kv"><b>Active:</b> {{ s.n_active }}</div>
<div class="kv"><b>Rejected:</b> {{ s.n_rejected }}</div>
<div class="kv"><b>Total duration (s):</b> {{ s.total_duration_sec }}</div>
<div class="kv"><b>Accepted duration (s):</b> {{ s.accepted_duration_sec }}</div>

<h3>Grades</h3>
<table>
<tr><th>Grade</th><th>Count</th></tr>
{% for g, c in s.grade_counts.items() %}
<tr><td>{{ g }}</td><td>{{ c }}</td></tr>
{% endfor %}
</table>

<h3>Prosody buckets</h3>
<table>
<tr><th>Bucket</th><th>Count</th></tr>
{% for b, c in s.prosody_bucket_counts.items() %}
<tr><td>{{ b }}</td><td>{{ c }}</td></tr>
{% endfor %}
</table>

<h3>Splits</h3>
<table>
<tr><th>Split</th><th>Count</th></tr>
{% for k, v in s.split_counts.items() %}
<tr><td>{{ k }}</td><td>{{ v }}</td></tr>
{% endfor %}
</table>

<h2>Speakers</h2>
<table>
<tr>
  <th>speaker_id</th><th>n_utt</th><th>total_dur(s)</th><th>avg_score</th>
  <th>A</th><th>B</th><th>C</th><th>D</th><th>rejected</th>
</tr>
{% for r in rows %}
<tr>
  <td>{{ r.speaker_id }}</td>
  <td>{{ r.n_utt }}</td>
  <td>{{ r.total_duration_sec }}</td>
  <td>{{ r.avg_quality_score }}</td>
  <td>{{ r.grade_A }}</td>
  <td>{{ r.grade_B }}</td>
  <td>{{ r.grade_C }}</td>
  <td>{{ r.grade_D }}</td>
  <td>{{ r.rejected }}</td>
</tr>
{% endfor %}
</table>

{% if pitch_png_exists %}<h2>Pitch distribution</h2><img src="{{ pitch_png }}">{% endif %}
{% if rate_png_exists %}<h2>Rate distribution</h2><img src="{{ rate_png }}">{% endif %}
</body></html>"""
    )
    html_str = tmpl.render(
        s=summary,
        rows=speaker_rows,
        pitch_png=pitch_png.name,
        rate_png=rate_png.name,
        pitch_png_exists=pitch_png.exists(),
        rate_png_exists=rate_png.exists(),
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_str)
    return True


def _write_text_fallback(path: Path, summary: dict[str, Any],
                          speaker_rows: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("ProsoGate Quality Report")
    lines.append("=" * 40)
    for k, v in summary.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("Speakers:")
    for r in speaker_rows:
        lines.append(
            f"  {r['speaker_id']}  n={r['n_utt']}  dur={r['total_duration_sec']}s  "
            f"avg={r['avg_quality_score']}  A/B/C/D/rej="
            f"{r['grade_A']}/{r['grade_B']}/{r['grade_C']}/{r['grade_D']}/{r['rejected']}"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run(cfg) -> int:
    upstream = cfg.paths.manifests.filter_score
    output_root = Path(cfg.paths.get_path("output_root", "tts_dataset"))
    manifests_dir = output_root / "manifests"
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    records = list(read_jsonl(upstream))
    splits = {
        "train": _load_split(manifests_dir / "train.jsonl"),
        "valid": _load_split(manifests_dir / "valid.jsonl"),
        "test": _load_split(manifests_dir / "test.jsonl"),
        "rejected": _load_split(manifests_dir / "rejected.jsonl"),
    }
    log.info(f"stage14: building report from {len(records)} records")

    summary = _summary(records, splits)
    speaker_rows = _speaker_stats(records)

    csv_path = reports_dir / "speaker_stats.csv"
    _write_speaker_csv(csv_path, speaker_rows)

    pitch_png = reports_dir / "pitch_distribution.png"
    rate_png = reports_dir / "rate_distribution.png"
    _try_plot_pitch(records, pitch_png)
    _try_plot_rate(records, rate_png)

    html_path = reports_dir / "quality_report.html"
    ok = _write_html(html_path, summary, speaker_rows, pitch_png, rate_png)
    if not ok:
        txt_path = reports_dir / "quality_report.txt"
        _write_text_fallback(txt_path, summary, speaker_rows)
        log.info(f"stage14: wrote text report -> {txt_path}")
    else:
        log.info(f"stage14: wrote HTML report -> {html_path}")
    log.info(f"stage14: speaker_stats.csv -> {csv_path}")
    return 0
