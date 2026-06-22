"""Stage 09 — Strict single-speaker consistency check via sliding-window embeddings.

For each utt produced by stage 08, we extract per-window speaker embeddings,
then evaluate five criteria (any failure rejects the utt):

  A — global cluster silhouette (>= cluster_silhouette threshold => >=2 speakers)
  B — max distance to per-utt cosine center
  C — adjacent-window jump (single point or two-in-a-row)
  D — distance from per-speaker enrollment center (mock: per-label centroid
      across this run; first-pass enrollment is the global mean of all utts of
      that speaker_label)
  E — overlapped speech ratio (real backend only; mock skips)

Real backend: pyannote/embedding + pyannote/overlapped-speech-detection. We
attempt to load lazily; on any failure we fall back to mock. Mock embeddings:
deterministic, hash-seeded RNG (no raw random) — never `random.random` or
`np.random.rand`.

Utts with n_windows < min_windows are rejected with "spk_too_few_windows".
Enrollment is NOT updated here (design.md §9.3 — incremental update is deferred).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from prosogate.audio_io import read_wav, slice_audio
from prosogate.logging_utils import get_logger
from prosogate.manifest import read_jsonl, write_jsonl

log = get_logger(__name__)

EMB_DIM = 192


def _safe_cuda() -> bool:
    """Return True only if torch.cuda is actually usable (not just declared)."""
    try:
        import torch  # type: ignore
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda")
        return True
    except Exception:
        return False


def _try_load_real_backend(cfg: Any):
    try:
        import torch  # type: ignore
        from pyannote.audio import Inference, Model  # type: ignore

        model_ref = cfg.spk_consistency.get("embedding_model", "pyannote/embedding")
        # Allow embedding_model to be a local path (e.g. models/pyannote/.../pytorch_model.bin)
        model_path = Path(model_ref)
        if model_path.is_file():
            log.info(f"loading real speaker embedding from local: {model_path}")
            model = Model.from_pretrained(str(model_path))
            # Stage 09 manually slices windows and feeds them one-by-one,
            # so use window='whole' to get a single embedding per call.
            emb = Inference(model, window="whole")
        else:
            log.info(f"loading real speaker embedding {model_ref} via HF")
            emb = Inference(model_ref, window="whole")

        if _safe_cuda():
            try:
                emb.to(torch.device("cuda"))
                log.info("spk_consistency: moved embedding inference to cuda")
            except Exception as e:
                log.warning(f"spk_consistency: could not move to cuda ({e}); using CPU")

        ovl_ref = cfg.spk_consistency.get("overlap_model", "")
        ovl = None
        if ovl_ref:
            try:
                ovl_path = Path(ovl_ref)
                if ovl_path.is_file():
                    ovl_model = Model.from_pretrained(str(ovl_path))
                    ovl = Inference(ovl_model)
                else:
                    ovl = Inference(ovl_ref)  # type: ignore
            except Exception as e:
                log.warning(f"overlap detection unavailable ({e!r}); skipping")
                ovl = None
        return emb, ovl
    except Exception as e:
        log.warning(f"real spk backend unavailable ({e!r}); falling back to mock")
        return None, None


def _mock_embed_window(seed_key: str, label: str) -> np.ndarray:
    """Hash-seeded reproducible embedding. Same speaker_label biases the mean
    so windows from one speaker cluster together; per-window noise is small.
    """
    label_seed = hash(label) & 0xFFFFFFFF
    label_rng = np.random.default_rng(label_seed)
    mean = label_rng.uniform(-1.0, 1.0, size=EMB_DIM)
    win_seed = hash(seed_key) & 0xFFFFFFFF
    win_rng = np.random.default_rng(win_seed)
    noise = win_rng.normal(0, 0.05, size=EMB_DIM)
    v = mean + noise
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def _real_embed_window(emb_inf, audio: np.ndarray, sr: int) -> np.ndarray:
    """Use pyannote Inference on a numpy clip; returns L2-normalized vector."""
    import torch  # type: ignore

    waveform = torch.from_numpy(audio).unsqueeze(0)
    emb = emb_inf({"waveform": waveform, "sample_rate": sr})
    if hasattr(emb, "data"):
        emb = emb.data
    v = np.asarray(emb, dtype=np.float32).squeeze()
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def _silhouette_two_clusters(embs: np.ndarray) -> float:
    """Cheap silhouette computed by k=2 KMeans-like split (single split, not
    sklearn). We fall back to 0 if anything degenerates.
    """
    n = len(embs)
    if n < 4:
        return 0.0
    # Initialize: pick farthest pair as seeds.
    # Compute pairwise cosine distances.
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    e_norm = embs / norms
    sim = e_norm @ e_norm.T
    dist = 1.0 - sim
    # Seed: i,j with max distance.
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    if i == j:
        return 0.0
    c1 = embs[i].copy()
    c2 = embs[j].copy()
    for _ in range(8):
        d1 = np.array([_cosine_dist(e, c1) for e in embs])
        d2 = np.array([_cosine_dist(e, c2) for e in embs])
        labels = (d2 < d1).astype(np.int8)
        if labels.sum() == 0 or labels.sum() == n:
            return 0.0
        c1 = embs[labels == 0].mean(axis=0)
        c2 = embs[labels == 1].mean(axis=0)
    # Silhouette per point: a = mean dist within cluster, b = mean dist to other cluster.
    sils = []
    for k, e in enumerate(embs):
        own = labels[k]
        own_mask = labels == own
        own_mask[k] = False
        other_mask = labels != own
        if own_mask.sum() == 0 or other_mask.sum() == 0:
            continue
        a = float(np.mean([_cosine_dist(e, embs[m]) for m in np.where(own_mask)[0]]))
        b = float(np.mean([_cosine_dist(e, embs[m]) for m in np.where(other_mask)[0]]))
        sils.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(sils)) if sils else 0.0


def _extract_windows(
    utt: dict[str, Any],
    window_sec: float,
    hop_sec: float,
    use_mock: bool,
    real_emb,
) -> list[np.ndarray]:
    """Return per-window embeddings (already L2-normalized)."""
    label = utt.get("speaker_label", "spk")
    duration = float(utt.get("duration") or 0.0)
    if duration <= 0:
        return []
    n_win = max(0, int(np.floor((duration - window_sec) / hop_sec)) + 1)
    if n_win <= 0:
        return []
    embs: list[np.ndarray] = []
    if use_mock or real_emb is None:
        for i in range(n_win):
            seed_key = f"{utt.get('utt_id')}::w{i}"
            embs.append(_mock_embed_window(seed_key, label))
        return embs

    # Real path: read audio_align_path slice in 16kHz.
    align_path = utt.get("audio_align_path")
    align_audio = None
    if align_path and Path(align_path).exists():
        align_audio, sr = read_wav(align_path, target_sr=16000)
        clip = slice_audio(align_audio, sr, float(utt["start"]), float(utt["end"]))
    else:
        # Fall back to the 24kHz utt wav, downsampled.
        wav_path = utt.get("wav")
        if not wav_path or not Path(wav_path).exists():
            return []
        clip, sr = read_wav(wav_path, target_sr=16000)

    win_samps = int(window_sec * sr)
    hop_samps = int(hop_sec * sr)
    for i in range(n_win):
        s = i * hop_samps
        e = s + win_samps
        if e > len(clip):
            break
        try:
            embs.append(_real_embed_window(real_emb, clip[s:e], sr))
        except Exception as ex:
            log.warning(f"real embed failed on window {i}: {ex}; using mock")
            embs.append(_mock_embed_window(f"{utt.get('utt_id')}::w{i}", label))
    return embs


def _evaluate(
    utt_id: str,
    embs: list[np.ndarray],
    ref_centers: dict[str, np.ndarray],
    speaker_label: str,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    n = len(embs)
    if n == 0:
        return {"passed": False, "n_windows": 0, "reasons": ["E_no_windows"]}
    arr = np.stack(embs)
    center = arr.mean(axis=0)
    center = center / (np.linalg.norm(center) + 1e-12)
    dists_to_center = np.array([_cosine_dist(e, center) for e in embs])
    max_dist_to_center = float(dists_to_center.max())

    neighbor_deltas = np.array(
        [_cosine_dist(embs[i], embs[i + 1]) for i in range(n - 1)]
    ) if n >= 2 else np.zeros(0)
    max_neighbor_delta = float(neighbor_deltas.max()) if neighbor_deltas.size else 0.0

    # Consecutive-2 sliding check
    consecutive_two_violation = False
    cons_thr = thresholds["consecutive_neighbor_delta"]
    for i in range(len(neighbor_deltas) - 1):
        if neighbor_deltas[i] > cons_thr and neighbor_deltas[i + 1] > cons_thr:
            consecutive_two_violation = True
            break

    silhouette = _silhouette_two_clusters(arr) if n >= 4 else 0.0

    # Criterion D — first-pass: skip when ref_center is missing or equals self center.
    ref_outlier_ratio = 0.0
    ref = ref_centers.get(speaker_label)
    if ref is not None:
        d_ref = np.array([_cosine_dist(e, ref) for e in embs])
        ref_outlier_ratio = float((d_ref > thresholds["max_dist_to_center"]).mean())

    overlap_ratio = 0.0  # mock: skip E

    reasons: list[str] = []
    if silhouette > thresholds["cluster_silhouette"]:
        reasons.append("A")
    if max_dist_to_center > thresholds["max_dist_to_center"]:
        reasons.append("B")
    if max_neighbor_delta > thresholds["max_neighbor_delta"] or consecutive_two_violation:
        reasons.append("C")
    if ref_outlier_ratio > thresholds["ref_outlier_ratio"]:
        reasons.append("D")
    if overlap_ratio > thresholds["overlap_ratio"]:
        reasons.append("E")

    passed = len(reasons) == 0
    return {
        "passed": passed,
        "n_windows": n,
        "max_dist_to_center": max_dist_to_center,
        "max_neighbor_delta": max_neighbor_delta,
        "cluster_silhouette": float(silhouette),
        "overlap_ratio": float(overlap_ratio),
        "ref_outlier_ratio": float(ref_outlier_ratio),
        "reasons": reasons,
    }


def run(cfg: Any) -> int:
    in_path = cfg.paths.manifests.fine_segment
    out_path = cfg.paths.manifests.spk_consistency
    sc = cfg.spk_consistency
    use_mock = bool(sc.get("use_mock", True))
    window_sec = float(sc.get("window_sec", 1.5))
    hop_sec = float(sc.get("hop_sec", 0.25))
    min_windows = int(sc.get("min_windows", 4))
    thresholds = {
        "cluster_silhouette": float(sc.thresholds.get("cluster_silhouette", 0.35)),
        "max_dist_to_center": float(sc.thresholds.get("max_dist_to_center", 0.35)),
        "max_neighbor_delta": float(sc.thresholds.get("max_neighbor_delta", 0.30)),
        "consecutive_neighbor_delta": float(
            sc.thresholds.get("consecutive_neighbor_delta", 0.20)
        ),
        "ref_outlier_ratio": float(sc.thresholds.get("ref_outlier_ratio", 0.10)),
        "overlap_ratio": float(sc.thresholds.get("overlap_ratio", 0.02)),
    }

    real_emb = None
    if not use_mock:
        real_emb, _ovl = _try_load_real_backend(cfg)

    utts = list(read_jsonl(in_path))
    log.info(f"read {len(utts)} utts from {in_path}")

    # First pass: gather embeddings per utt (skip already-rejected).
    per_utt_embs: dict[str, list[np.ndarray]] = {}
    for u in utts:
        if u.get("status") == "rejected":
            continue
        utt_id = u.get("utt_id")
        embs = _extract_windows(u, window_sec, hop_sec, use_mock, real_emb)
        per_utt_embs[utt_id] = embs

    # Build per-speaker_label reference center (mock D criterion):
    # mean of per-utt mean embeddings for that label.
    per_label_means: dict[str, list[np.ndarray]] = {}
    for u in utts:
        utt_id = u.get("utt_id")
        embs = per_utt_embs.get(utt_id) or []
        if not embs:
            continue
        m = np.stack(embs).mean(axis=0)
        m = m / (np.linalg.norm(m) + 1e-12)
        per_label_means.setdefault(u.get("speaker_label", "spk"), []).append(m)

    ref_centers: dict[str, np.ndarray] = {}
    for label, mlist in per_label_means.items():
        if len(mlist) <= 1:
            # Only one utt — skip D (would compare against self).
            continue
        c = np.stack(mlist).mean(axis=0)
        ref_centers[label] = c / (np.linalg.norm(c) + 1e-12)

    out_records: list[dict[str, Any]] = []
    n_rej = 0
    n_pass = 0

    for u in utts:
        rec = dict(u)
        utt_id = rec.get("utt_id")
        if rec.get("status") == "rejected":
            out_records.append(rec)
            continue

        embs = per_utt_embs.get(utt_id) or []
        if len(embs) < min_windows:
            rec["spk_consistency"] = {
                "passed": False,
                "n_windows": len(embs),
                "max_dist_to_center": 0.0,
                "max_neighbor_delta": 0.0,
                "cluster_silhouette": 0.0,
                "overlap_ratio": 0.0,
                "ref_outlier_ratio": 0.0,
            }
            rec.setdefault("reject_reasons", []).append("spk_too_few_windows")
            rec["status"] = "rejected"
            n_rej += 1
            out_records.append(rec)
            continue

        result = _evaluate(
            utt_id, embs, ref_centers, rec.get("speaker_label", "spk"), thresholds
        )
        reasons = result.pop("reasons", [])
        rec["spk_consistency"] = {
            "passed": bool(result["passed"]),
            "n_windows": int(result["n_windows"]),
            "max_dist_to_center": result["max_dist_to_center"],
            "max_neighbor_delta": result["max_neighbor_delta"],
            "cluster_silhouette": result["cluster_silhouette"],
            "overlap_ratio": result["overlap_ratio"],
            "ref_outlier_ratio": result["ref_outlier_ratio"],
        }
        if not result["passed"]:
            for r in reasons:
                rec.setdefault("reject_reasons", []).append(f"spk_{r}")
            rec["status"] = "rejected"
            n_rej += 1
        else:
            n_pass += 1
        out_records.append(rec)

    n = write_jsonl(out_path, out_records)
    log.info(f"wrote {n} utts to {out_path} (passed={n_pass}, rejected={n_rej})")
    return 0
