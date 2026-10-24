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


def score_agents(future_xy: np.ndarray, gt_future: np.ndarray, gt_mask: np.ndarray, mode_probs: np.ndarray) -> dict:
    """Score one agent. ``future_xy`` is ``[K, T, 2]`` and ``mode_probs`` is ``[K]``."""

    full = bool(gt_mask.shape[0] == FUTURE_STEPS and gt_mask.all())
    metrics = trajectory_metrics(future_xy, gt_future, gt_mask, mode_probs)
    if metrics is None:
        return {"full_horizon": full}
    metrics["full_horizon"] = full
    return metrics


def audit_cache(cache_dir: str | Path, split_json: str | Path, split: str) -> dict:
    """Check cached futures stay inside the published sequence split."""

    payload = json.loads(Path(split_json).read_text(encoding="utf-8"))
    allowed = set(payload[split])
    files = sorted(Path(cache_dir).glob("**/sample_*.npz"))
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
