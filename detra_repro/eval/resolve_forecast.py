"""RESOLVE forecast rules on top of the existing minADE / minFDE code.

Primary subset: an agent counts only when all 10 future waypoints (2 Hz, 5 s)
are valid. Those waypoints are later centers of the same track inside the same
sequence, so a sequence-level split keeps them out of the other splits.
Partial tracks stay in the cache for training masks and are reported separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from detra_repro.data.resolve_taxonomy import CLASS_NAMES, FUTURE_STEPS, HORIZON_S
from detra_repro.eval.forecasting_metrics import trajectory_metrics

# ViP3D (Gu et al., CVPR 2023): match on the current center, then a hit also
# needs minFDE <= tau. False positives are penalized, so EPA is not a recall.
EPA_MATCH_THRESH_M = 2.0
EPA_HIT_FDE_M = 2.0
EPA_FP_PENALTY = 0.5
# Same score gate as RESOLVE TransFusion ``POST_PROCESSING.SCORE_THRESH``.
EPA_SCORE_THRESH = 0.1


def score_agents(future_xy: np.ndarray, gt_future: np.ndarray, gt_mask: np.ndarray, mode_probs: np.ndarray) -> dict:
    """Score one agent. ``future_xy`` is ``[K, T, 2]`` and ``mode_probs`` is ``[K]``."""

    full = bool(gt_mask.shape[0] == FUTURE_STEPS and gt_mask.all())
    metrics = trajectory_metrics(future_xy, gt_future, gt_mask, mode_probs)
    if metrics is None:
        return {"full_horizon": full}
    metrics["full_horizon"] = full
    return metrics


def _center_matches(gt_xy: np.ndarray, pred_xy: np.ndarray, thresh: float) -> list[tuple[int, int]]:
    """Hungarian matches with current-center distance at most ``thresh`` meters."""

    if len(gt_xy) == 0 or len(pred_xy) == 0:
        return []
    from scipy.optimize import linear_sum_assignment

    dist = np.linalg.norm(gt_xy[:, None, :] - pred_xy[None, :, :], axis=-1)
    cost = np.where(dist <= thresh, dist, 1.0e6)
    rows, cols = linear_sum_assignment(cost)
    return [(int(row), int(col)) for row, col in zip(rows, cols) if dist[row, col] <= thresh]


def _future_errors(future_xy: np.ndarray, gt_future: np.ndarray, mode_probs: np.ndarray) -> tuple[float, float, float, float]:
    """Return top-1 ADE/FDE and best-mode ADE/FDE. ``future_xy`` is ``[K, T, 2]``."""

    future_xy = np.asarray(future_xy, dtype=np.float64)
    gt_future = np.asarray(gt_future, dtype=np.float64)
    if future_xy.ndim != 3 or gt_future.ndim != 2:
        raise ValueError(f"future shapes {future_xy.shape} {gt_future.shape}, expected [K,T,2] and [T,2]")
    error = np.linalg.norm(future_xy - gt_future[None, :, :], axis=-1)
    ade = error.mean(axis=1)
    fde = error[:, -1]
    top = int(np.argmax(mode_probs)) if len(mode_probs) else 0
    return float(ade[top]), float(fde[top]), float(ade.min()), float(fde.min())


def evaluate_end_to_end(
    gt_by_token: dict,
    pred_by_token: dict,
    *,
    score_thresh: float = EPA_SCORE_THRESH,
    match_thresh_m: float = EPA_MATCH_THRESH_M,
    hit_fde_m: float = EPA_HIT_FDE_M,
    fp_penalty: float = EPA_FP_PENALTY,
) -> dict:
    """ViP3D end-to-end prediction accuracy plus minADE / minFDE.

    Ground-truth agents count only when all ``FUTURE_STEPS`` waypoints are
    valid. Detections are matched to every same-class box within
    ``match_thresh_m``, including partial tracks, so a detection on a partial
    track is not a false positive. A hit is a matched full-horizon agent whose
    final-step error is within ``hit_fde_m``. Counts are summed over frames,
    then EPA is averaged over classes that have at least one full-horizon box.
    """

    totals = {
        name: {
            "n_gt": 0,
            "hits": 0,
            "hits_top1": 0,
            "fp": 0,
            "n_pred": 0,
            "n_matched": 0,
            "ade_k1": 0.0,
            "fde_k1": 0.0,
            "ade_k": 0.0,
            "fde_k": 0.0,
        }
        for name in CLASS_NAMES
    }
    for token, gt_boxes in gt_by_token.items():
        preds = [box for box in pred_by_token.get(token, []) if float(box["score"]) >= score_thresh]
        for name in CLASS_NAMES:
            gt_cls = [box for box in gt_boxes if box["name"] == name]
            pred_cls = [box for box in preds if box["name"] == name]
            bucket = totals[name]
            bucket["n_pred"] += len(pred_cls)
            if not gt_cls and not pred_cls:
                continue
            gt_xy = np.asarray([box["translation"][:2] for box in gt_cls], dtype=np.float64).reshape(-1, 2)
            pred_xy = np.asarray([box["translation"][:2] for box in pred_cls], dtype=np.float64).reshape(-1, 2)
            matched_gt = set()
            for gt_idx, pred_idx in _center_matches(gt_xy, pred_xy, match_thresh_m):
                matched_gt.add(gt_idx)
                gt = gt_cls[gt_idx]
                mask = np.asarray(gt["future_mask"], dtype=bool)
                if mask.shape[0] != FUTURE_STEPS or not bool(mask.all()):
                    continue
                pred = pred_cls[pred_idx]
                ade_k1, fde_k1, ade_k, fde_k = _future_errors(
                    np.asarray(pred["future_xy"], dtype=np.float64),
                    np.asarray(gt["future_xy"], dtype=np.float64),
                    np.asarray(pred["mode_probs"], dtype=np.float64),
                )
                bucket["n_matched"] += 1
                bucket["ade_k1"] += ade_k1
                bucket["fde_k1"] += fde_k1
                bucket["ade_k"] += ade_k
                bucket["fde_k"] += fde_k
                if fde_k <= hit_fde_m:
                    bucket["hits"] += 1
                if fde_k1 <= hit_fde_m:
                    bucket["hits_top1"] += 1
            bucket["fp"] += len(pred_cls) - len(matched_gt)
            for gt in gt_cls:
                mask = np.asarray(gt["future_mask"], dtype=bool)
                if mask.shape[0] == FUTURE_STEPS and bool(mask.all()):
                    bucket["n_gt"] += 1

    per_class = {}
    epa_values = []
    epa_top1_values = []
    for name, bucket in totals.items():
        n_gt = bucket["n_gt"]
        n_matched = bucket["n_matched"]
        row = {
            "n_gt": n_gt,
            "n_pred": bucket["n_pred"],
            "n_matched": n_matched,
            "hits": bucket["hits"],
            "hits_top1": bucket["hits_top1"],
            "fp": bucket["fp"],
            "epa": None if n_gt == 0 else (bucket["hits"] - fp_penalty * bucket["fp"]) / n_gt,
            "epa_top1": None if n_gt == 0 else (bucket["hits_top1"] - fp_penalty * bucket["fp"]) / n_gt,
            "minade_k1": None if n_matched == 0 else bucket["ade_k1"] / n_matched,
            "minfde_k1": None if n_matched == 0 else bucket["fde_k1"] / n_matched,
            "minade": None if n_matched == 0 else bucket["ade_k"] / n_matched,
            "minfde": None if n_matched == 0 else bucket["fde_k"] / n_matched,
        }
        per_class[name] = row
        if n_gt > 0:
            epa_values.append(row["epa"])
            epa_top1_values.append(row["epa_top1"])
    n_gt = sum(bucket["n_gt"] for bucket in totals.values())
    hits = sum(bucket["hits"] for bucket in totals.values())
    hits_top1 = sum(bucket["hits_top1"] for bucket in totals.values())
    fp = sum(bucket["fp"] for bucket in totals.values())
    n_matched = sum(bucket["n_matched"] for bucket in totals.values())
    ade_k1 = sum(bucket["ade_k1"] for bucket in totals.values())
    fde_k1 = sum(bucket["fde_k1"] for bucket in totals.values())
    ade_k = sum(bucket["ade_k"] for bucket in totals.values())
    fde_k = sum(bucket["fde_k"] for bucket in totals.values())
    return {
        "protocol": {
            "name": "ViP3D EPA",
            "match_thresh_m": match_thresh_m,
            "hit_fde_m": hit_fde_m,
            "fp_penalty": fp_penalty,
            "score_thresh": score_thresh,
            "future_steps": FUTURE_STEPS,
            "horizon_s": HORIZON_S,
            "epa": "macro mean over classes with a full-horizon ground-truth box",
            "minade_minfde": "micro mean over full-horizon boxes matched within the center threshold",
            "k1": "most likely mode",
            "k": "best mode, ADE and FDE may choose different modes",
        },
        "epa": float(np.mean(epa_values)) if epa_values else None,
        "epa_top1": float(np.mean(epa_top1_values)) if epa_top1_values else None,
        "epa_micro": None if n_gt == 0 else (hits - fp_penalty * fp) / n_gt,
        "epa_micro_top1": None if n_gt == 0 else (hits_top1 - fp_penalty * fp) / n_gt,
        "minade_k1": None if n_matched == 0 else ade_k1 / n_matched,
        "minfde_k1": None if n_matched == 0 else fde_k1 / n_matched,
        "minade": None if n_matched == 0 else ade_k / n_matched,
        "minfde": None if n_matched == 0 else fde_k / n_matched,
        "n_gt_full_horizon": n_gt,
        "n_matched": n_matched,
        "hits": hits,
        "hits_top1": hits_top1,
        "fp": fp,
        "per_class": per_class,
    }


def audit_cache(cache_dir: str | Path, split_json: str | Path, split: str) -> dict:
    """Check cached futures stay inside the published sequence split."""

    payload = json.loads(Path(split_json).read_text(encoding="utf-8"))
    allowed = set(payload[split])
    files = sorted(Path(cache_dir).glob("sample_*.npz"))
    files.extend(sorted(Path(cache_dir).glob("*/sample_*.npz")))
    if not files:
        raise FileNotFoundError(f"no samples under {cache_dir}")
    agents = 0
    full = 0
    map_tokens = 0
    for path in files:
        batch_id = path.parent.name
        if batch_id not in allowed:
            raise AssertionError(f"{path} batch {batch_id} is not in split {split}")
        with np.load(path, allow_pickle=True) as data:
            if data["map_xy"].shape[0] != 0:
                map_tokens += int(data["map_xy"].shape[0])
            mask = data["gt_future_mask"].astype(bool)
            future = data["gt_future"]
            if not np.isfinite(future[mask]).all():
                raise AssertionError(f"{path}: non-finite future")
            if mask.shape[1] != FUTURE_STEPS:
                raise AssertionError(f"{path}: future steps {mask.shape[1]}")
            agents += int(mask.shape[0])
            full += int(mask.all(axis=1).sum()) if mask.size else 0
            if int(data["ego_pose"].shape[0]) != 4:
                raise AssertionError(f"{path}: ego pose")
    report = {
        "split": split,
        "samples": len(files),
        "agents": agents,
        "full_horizon_agents": full,
        "map_tokens": map_tokens,
        "horizon_s": HORIZON_S,
        "class_names": list(CLASS_NAMES),
        "primary_subset": "full_horizon_agents",
    }
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a RESOLVE forecast cache")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    args = parser.parse_args()
    audit_cache(args.cache, args.split_json, args.split)


if __name__ == "__main__":
    main()
