"""RESOLVE roadside sequences -> MC-DeTra cache.

No HD map is available, so map tokens are empty. The map encoder stays in the
model and simply receives no tokens. Point clouds are the aligned 128-beam
fusion (``128_128b_sync``). Forecast targets are later centers of the same
track id, in the static lidar frame.

Official ``split_dataset_divisible`` shuffles frame tokens (default ratio 0.8,
no test split). That leaks future frames across splits, so the published split
assigns each whole ``batchId`` sequence to train, val, or test at 7:2:1.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from detra_repro.config import DataConfig
from detra_repro.data.cache_v2 import CACHE_SCHEMA_VERSION, atomic_save_sample, atomic_write_json
from detra_repro.data.resolve_taxonomy import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    FRAME_STRIDE,
    FUTURE_DT_S,
    FUTURE_STEPS,
    HISTORY_SWEEPS,
    HORIZON_S,
    HZ,
    LABEL_TO_CLASS,
    LIDAR_DIR,
    MAP_FEATURE_DIM,
    ROI_M,
    SPLIT_RATIOS,
    SPLIT_SEED,
    TIMESTAMP_TOL_S,
    VEHICLE_CLASS_NAMES,
    Z_RANGE_M,
)
from detra_repro.data.types import SceneSample


VEHICLE_CLASS_IDS = {CLASS_TO_INDEX[name] for name in VEHICLE_CLASS_NAMES}


@dataclass
class AnnFrame:
    frame_index: int
    stem: str
    timestamp_s: float
    boxes: np.ndarray
    class_ids: np.ndarray
    track_ids: np.ndarray
    velocity: np.ndarray


@dataclass
class Sequence:
    batch_id: str
    rosbag: str
    frames: list[AnnFrame]

    @property
    def num_frames(self) -> int:
        return len(self.frames)


def resolve_data_config() -> DataConfig:
    cfg = DataConfig(
        history_sweeps=HISTORY_SWEEPS,
        future_steps=FUTURE_STEPS,
        frame_stride=FRAME_STRIDE,
        roi_m=ROI_M,
        voxel_size_m=0.1,
        z_range_m=Z_RANGE_M,
        class_names=CLASS_NAMES,
        class_id_mode="contiguous",
    )
    height = int(round((cfg.roi_m[3] - cfg.roi_m[1]) / cfg.voxel_size_m))
    width = int(round((cfg.roi_m[2] - cfg.roi_m[0]) / cfg.voxel_size_m))
    if height % 16 or width % 16:
        raise ValueError(f"BEV grid {(height, width)} is not divisible by encoder stride 16")
    return cfg


def _inside_xyz(xyz: np.ndarray, cfg: DataConfig) -> np.ndarray:
    """Inclusive range, matching OpenPCDet ``mask_points_by_range`` plus z."""

    x_min, y_min, x_max, y_max = cfg.roi_m
    z_min, z_max = cfg.z_range_m
    return (
        (xyz[:, 0] >= x_min)
        & (xyz[:, 0] <= x_max)
        & (xyz[:, 1] >= y_min)
        & (xyz[:, 1] <= y_max)
        & (xyz[:, 2] >= z_min)
        & (xyz[:, 2] <= z_max)
    )


def _stem_timestamp(stem: str) -> float:
    sec, nano = stem.split("-")
    return int(sec) + int(nano) / 1e9


def load_sequences(root: str | Path) -> dict[str, Sequence]:
    """Parse ``HSQJN_3.json`` into per-sequence box tracks. Points are not read."""

    root = Path(root)
    with (root / "HSQJN_3.json").open("r", encoding="utf-8") as handle:
        raw_frames = json.load(handle)

    grouped: dict[str, list] = defaultdict(list)
    for frame in raw_frames:
        grouped[str(frame["batchId"])].append(frame)

    sequences: dict[str, Sequence] = {}
    for batch_id, frames in grouped.items():
        frames = sorted(frames, key=lambda item: int(item["frameIndex"]))
        indices = [int(item["frameIndex"]) for item in frames]
        if indices != list(range(indices[-1] + 1)):
            raise ValueError(f"{batch_id}: frameIndex is not a contiguous 0..N-1 range")
        info = frames[0]["info"]
        rosbag = next(part for part in info.split("/") if part.startswith("rosbag2_"))
        ann_frames: list[AnnFrame] = []
        for frame in frames:
            stem = Path(frame["info"]).stem
            boxes, class_ids, track_ids = [], [], []
            for label in frame.get("labels") or []:
                if label.get("drawType") != "box3d":
                    continue
                pts = label.get("points") or []
                if len(pts) < 9:
                    continue
                name = LABEL_TO_CLASS.get(str(label.get("label", "")).strip())
                if name is None:
                    continue
                boxes.append(
                    [
                        float(pts[0]),
                        float(pts[1]),
                        float(pts[2]),
                        float(pts[6]),
                        float(pts[7]),
                        float(pts[8]),
                        float(pts[5]),
                    ]
                )
                class_ids.append(CLASS_TO_INDEX[name])
                track_ids.append(f"{batch_id}_{int(label.get('id', -1))}")
            n = len(boxes)
            ann_frames.append(
                AnnFrame(
                    frame_index=int(frame["frameIndex"]),
                    stem=stem,
                    timestamp_s=_stem_timestamp(stem),
                    boxes=np.asarray(boxes, dtype=np.float32).reshape(n, 7),
                    class_ids=np.asarray(class_ids, dtype=np.int64),
                    track_ids=np.asarray(track_ids),
                    velocity=np.zeros((n, 2), dtype=np.float32),
                )
            )
        _fill_velocity(ann_frames)
        sequences[batch_id] = Sequence(batch_id=batch_id, rosbag=rosbag, frames=ann_frames)
    return sequences


def _fill_velocity(frames: list[AnnFrame]) -> None:
    """vx, vy from the previous observation. The first frame copies the second."""

    prev: dict[str, tuple[np.ndarray, float]] = {}
    seen: dict[str, int] = {}
    first: dict[str, tuple[int, int]] = {}
    for frame_pos, frame in enumerate(frames):
        for obj_idx, track_id in enumerate(frame.track_ids.tolist()):
            xy = frame.boxes[obj_idx, :2].astype(np.float32)
            count = seen.get(track_id, 0)
            if count >= 1 and track_id in prev:
                prev_xy, prev_ts = prev[track_id]
                dt = float(frame.timestamp_s - prev_ts)
                if 1e-3 < dt < 0.25:
                    vel = (xy - prev_xy) / dt
                    frame.velocity[obj_idx] = vel
                    if count == 1:
                        first_pos, first_obj = first[track_id]
                        frames[first_pos].velocity[first_obj] = vel
            if count == 0:
                first[track_id] = (frame_pos, obj_idx)
            seen[track_id] = count + 1
            prev[track_id] = (xy, float(frame.timestamp_s))


def _track_index(seq: Sequence) -> dict[tuple[str, int], int]:
    index: dict[tuple[str, int], int] = {}
    for frame in seq.frames:
        for obj_idx, track_id in enumerate(frame.track_ids.tolist()):
            index[(track_id, frame.frame_index)] = obj_idx
    return index


def official_frame_leakage(sequences: dict[str, Sequence]) -> dict[str, float | int]:
    """Replay ``split_dataset_divisible`` (seed 42, ratio 0.8) and count leaks."""

    tokens = []
    neighbors: dict[str, set[str]] = defaultdict(set)
    future: dict[str, set[str]] = defaultdict(set)
    for seq in sequences.values():
        stems = [frame.stem for frame in seq.frames]
        for pos, stem in enumerate(stems):
            tokens.append(stem)
            if pos > 0:
                neighbors[stem].add(stems[pos - 1])
            if pos + 1 < len(stems):
                neighbors[stem].add(stems[pos + 1])
            for step in range(1, FUTURE_STEPS + 1):
                fut = pos + step * FRAME_STRIDE
                if fut < len(stems):
                    future[stem].add(stems[fut])
    unique = sorted(set(tokens))
    rng = random.Random(42)
    rng.shuffle(unique)
    # Match the official helper at its default ratio, ignoring the batch-16 trim.
    train_len = int(round(len(unique) * 0.8))
    train = set(unique[:train_len])
    val = [token for token in unique[train_len:]]
    neighbor_leak = sum(any(item in train for item in neighbors[token]) for token in val)
    future_leak = sum(any(item in train for item in future[token]) for token in val)
    return {
        "script": "RESOLVE/tools/sunlakes_dataset_utils.py:split_dataset_divisible",
        "ratio": 0.8,
        "seed": 42,
        "has_test_split": False,
        "shuffles_frames": True,
        "num_val_frames": len(val),
        "val_frames_with_train_neighbor": neighbor_leak,
        "val_frames_whose_2hz_future_intersects_train": future_leak,
        "neighbor_leak_fraction": neighbor_leak / max(len(val), 1),
        "future_leak_fraction": future_leak / max(len(val), 1),
    }


def assign_sequences(
    sequences: dict[str, Sequence],
    ratios: dict[str, float] = SPLIT_RATIOS,
    seed: int = SPLIT_SEED,
) -> dict[str, str]:
    """Assign each sequence to one split. Stable sort keeps the seed for ties."""

    total = sum(seq.num_frames for seq in sequences.values())
    targets = {name: ratios[name] * total for name in ("train", "val", "test")}
    order = list(sequences)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda batch_id: sequences[batch_id].num_frames, reverse=True)
    counts = {name: 0 for name in targets}
    assignment: dict[str, str] = {}
    for batch_id in order:
        split = max(targets, key=lambda name: targets[name] - counts[name])
        assignment[batch_id] = split
        counts[split] += sequences[batch_id].num_frames
    return assignment


def _full_horizon_frames(seq: Sequence) -> int:
    return sum(1 for frame in seq.frames if frame.frame_index + FUTURE_STEPS * FRAME_STRIDE < seq.num_frames)


def write_split(path: str | Path, sequences: dict[str, Sequence], assignment: dict[str, str]) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    splits = {name: [] for name in ("train", "val", "test")}
    for batch_id, split in sorted(assignment.items()):
        splits[split].append(batch_id)
    counts = {}
    for name, batch_ids in splits.items():
        frames = sum(sequences[batch_id].num_frames for batch_id in batch_ids)
        full = sum(_full_horizon_frames(sequences[batch_id]) for batch_id in batch_ids)
        counts[name] = {
            "sequences": len(batch_ids),
            "frames": frames,
            "full_horizon_anchors": full,
        }
    payload = {
        "name": "resolve_forecast_7_2_1",
        "seed": SPLIT_SEED,
        "ratios": SPLIT_RATIOS,
        "unit": "batchId sequence (contiguous 10 Hz frames, one rosbag)",
        "lidar_dir": LIDAR_DIR,
        "hd_map": False,
        "official_random_split_audit": official_frame_leakage(sequences),
        "protocol": {
            "hz": HZ,
            "history_sweeps": HISTORY_SWEEPS,
            "history_lidar_stride_frames": 1,
            "actor_history_steps": HISTORY_SWEEPS,
            "actor_history_stride_frames": FRAME_STRIDE,
            "future_steps": FUTURE_STEPS,
            "future_stride_frames": FRAME_STRIDE,
            "future_hz": HZ / FRAME_STRIDE,
            "horizon_s": HORIZON_S,
            "waypoint_times_s": [(step + 1) * FUTURE_DT_S for step in range(FUTURE_STEPS)],
            "timestamp_tolerance_s": TIMESTAMP_TOL_S,
            "coordinate_frame": "static roadside lidar, ego pose is identity",
            "track_id": "{batchId}_{label id}, unique inside a sequence",
            "velocity": "adjacent-frame center displacement / dt; first frame copies the second",
            "min_lidar_points": 1,
            "primary_forecast_subset": "agents whose 10 future waypoints are all valid in-sequence",
            "detection": "nuScenes CVPR-2019 center distance, RESOLVE class names, range bins 0-50/50-70/70-inf at origin (-22, 0)",
            "temporal_rule": "a val/test anchor and its future waypoints never come from a train sequence",
        },
        "counts": counts,
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        "sequences": {
            batch_id: {
                "split": assignment[batch_id],
                "rosbag": sequences[batch_id].rosbag,
                "num_frames": sequences[batch_id].num_frames,
                "full_horizon_anchors": _full_horizon_frames(sequences[batch_id]),
            }
            for batch_id in sorted(sequences)
        },
    }
    _assert_isolated(sequences, assignment)
    atomic_write_json(path, payload)
    for name, batch_ids in splits.items():
        text = "".join(f"{batch_id}\n" for batch_id in batch_ids)
        (path.parent / f"resolve_{name}_sequences.txt").write_text(text, encoding="utf-8")
    return payload


def _assert_isolated(sequences: dict[str, Sequence], assignment: dict[str, str]) -> None:
    """Val/test futures must not land on a training frame of the same capture."""

    stem_split: dict[str, str] = {}
    for batch_id, seq in sequences.items():
        split = assignment[batch_id]
        for frame in seq.frames:
            prev = stem_split.get(frame.stem)
            if prev is not None and prev != split:
                raise AssertionError(f"stem {frame.stem} is in both {prev} and {split}")
            stem_split[frame.stem] = split
    for batch_id, seq in sequences.items():
        split = assignment[batch_id]
        if split == "train":
            continue
        stems = [frame.stem for frame in seq.frames]
        for pos, stem in enumerate(stems):
            for step in range(1, FUTURE_STEPS + 1):
                fut = pos + step * FRAME_STRIDE
                if fut >= len(stems):
                    break
                if stem_split[stems[fut]] == "train":
                    raise AssertionError(f"{batch_id} frame {pos} future is in train")


def _lidar_path(root: Path, seq: Sequence, frame: AnnFrame) -> Path:
    return root / seq.rosbag / LIDAR_DIR / f"{frame.stem}.bin"


def _read_points(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"{path}: float count {raw.size} is not divisible by 4")
    return raw.reshape(-1, 4)


def _count_points(points_xyz: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Points inside center-form boxes. Yaw matches the official corner test."""

    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int32)
    dx = points_xyz[:, None, 0] - boxes[None, :, 0]
    dy = points_xyz[:, None, 1] - boxes[None, :, 1]
    dz = points_xyz[:, None, 2] - boxes[None, :, 2]
    yaw = boxes[:, 6]
    cos = np.cos(yaw)[None, :]
    sin = np.sin(yaw)[None, :]
    local_x = cos * dx + sin * dy
    local_y = -sin * dx + cos * dy
    inside = (
        (np.abs(local_x) <= boxes[None, :, 3] * 0.5)
        & (np.abs(local_y) <= boxes[None, :, 4] * 0.5)
        & (np.abs(dz) <= boxes[None, :, 5] * 0.5)
    )
    return inside.sum(axis=0).astype(np.int32)


def _time_ok(actual_s: float, expected_s: float) -> bool:
    return abs(actual_s - expected_s) <= TIMESTAMP_TOL_S


def make_sample(
    root: Path,
    seq: Sequence,
    frame_pos: int,
    track_at: dict[tuple[str, int], int],
    cfg: DataConfig,
    *,
    filter_boxes_outside_range: bool = True,
) -> tuple[SceneSample, np.ndarray] | None:
    frame = seq.frames[frame_pos]
    current_path = _lidar_path(root, seq, frame)
    if not current_path.exists():
        return None
    sweeps = []
    current_xyz = None
    first = frame_pos - cfg.history_sweeps + 1
    for sweep_pos in range(max(0, first), frame_pos + 1):
        sweep = seq.frames[sweep_pos]
        path = _lidar_path(root, seq, sweep)
        if not path.exists():
            continue
        xyz = _read_points(path)
        dt = sweep.timestamp_s - frame.timestamp_s
        if sweep_pos != frame_pos and not _time_ok(dt, (sweep.frame_index - frame.frame_index) / HZ):
            continue
        if sweep_pos == frame_pos:
            current_xyz = xyz[:, :3]
        kept = xyz[_inside_xyz(xyz[:, :3], cfg)]
        dt_col = np.full((len(kept), 1), dt, dtype=np.float32)
        sweeps.append(np.concatenate([kept[:, :3], dt_col], axis=1))
    points = np.concatenate(sweeps, axis=0) if sweeps else np.zeros((0, 4), dtype=np.float32)

    boxes = frame.boxes
    class_ids = frame.class_ids
    track_ids = frame.track_ids
    velocity = frame.velocity
    if current_xyz is None:
        point_counts = np.zeros((len(boxes),), dtype=np.int32)
    else:
        point_counts = _count_points(current_xyz, boxes)
    keep = point_counts > 0
    # OpenPCDet drops out-of-range centers only while training. Val/test keep
    # them so detection GT matches the official eval infos.
    if filter_boxes_outside_range and len(boxes):
        keep = keep & _inside_xyz(boxes[:, :3], cfg)
    boxes = boxes[keep]
    class_ids = class_ids[keep]
    track_ids = track_ids[keep]
    velocity = velocity[keep]
    point_counts = point_counts[keep]

    m = len(boxes)
    future = np.zeros((m, cfg.future_steps, 2), dtype=np.float32)
    future_mask = np.zeros((m, cfg.future_steps), dtype=bool)
    history = np.zeros((m, cfg.history_sweeps, 2), dtype=np.float32)
    history_mask = np.zeros((m, cfg.history_sweeps), dtype=bool)
    by_index = {item.frame_index: item for item in seq.frames}
    for obj_idx, track_id in enumerate(track_ids.tolist()):
        for step in range(cfg.future_steps):
            fut_index = frame.frame_index + (step + 1) * cfg.frame_stride
            slot = track_at.get((track_id, fut_index))
            fut = by_index.get(fut_index)
            if slot is None or fut is None:
                continue
            if not _time_ok(fut.timestamp_s - frame.timestamp_s, (step + 1) * FUTURE_DT_S):
                continue
            future[obj_idx, step] = fut.boxes[slot, :2]
            future_mask[obj_idx, step] = True
        for hist_idx in range(cfg.history_sweeps):
            steps_back = cfg.history_sweeps - 1 - hist_idx
            hist_index = frame.frame_index - steps_back * cfg.frame_stride
            slot = track_at.get((track_id, hist_index))
            hist = by_index.get(hist_index)
            if slot is None or hist is None:
                continue
            if steps_back and not _time_ok(frame.timestamp_s - hist.timestamp_s, steps_back * FUTURE_DT_S):
                continue
            history[obj_idx, hist_idx] = hist.boxes[slot, :2]
            history_mask[obj_idx, hist_idx] = True

    zeros_i8 = np.zeros((m,), dtype=np.int8)
    return SceneSample(
        points=points.astype(np.float32),
        gt_boxes=boxes.astype(np.float32).reshape(m, 7),
        gt_track_ids=track_ids.astype(str),
        gt_types=class_ids.astype(np.int64),
        gt_future=future,
        gt_future_mask=future_mask,
        gt_history=history,
        gt_history_mask=history_mask,
        map_xy=np.zeros((0, 2), dtype=np.float32),
        map_features=np.zeros((0, MAP_FEATURE_DIM), dtype=np.float32),
        gt_num_lidar_points=point_counts,
        gt_detection_difficulty=zeros_i8,
        gt_tracking_difficulty=zeros_i8.copy(),
        paper_vehicle_eligible=np.isin(class_ids, list(VEHICLE_CLASS_IDS)) & (point_counts >= 1),
        context_name=seq.rosbag,
        frame_index=frame.frame_index,
        frame_timestamp_micros=int(round(frame.timestamp_s * 1e6)),
        ego_pose=np.eye(4, dtype=np.float64),
        nlz_xy=np.zeros((0, 2), dtype=np.float32),
        nlz_offsets=np.zeros((1,), dtype=np.int32),
    ), velocity


def _cache_sequence_job(job: tuple) -> dict:
    """Write one sequence. Returns counts and the best preview frame, not points."""

    root, seq, split_dir, limit, filter_outside = job
    cfg = resolve_data_config()
    track_at = _track_index(seq)
    n_samples = 0
    best: tuple | None = None
    seg_dir = Path(split_dir) / seq.batch_id
    for frame_pos in range(limit):
        built = make_sample(
            Path(root),
            seq,
            frame_pos,
            track_at,
            cfg,
            filter_boxes_outside_range=filter_outside,
        )
        if built is None:
            continue
        sample, velocity = built
        dest = seg_dir / f"sample_{sample.frame_index:06d}.npz"
        atomic_save_sample(
            dest,
            sample,
            str(_lidar_path(Path(root), seq, seq.frames[frame_pos])),
            extras={
                "gt_velocity": velocity.astype(np.float32),
                "gt_class_names": np.asarray(CLASS_NAMES),
            },
        )
        n_samples += 1
        score = _preview_score(sample)
        if best is None or score > best[0]:
            best = (score, frame_pos)
    return {"batch_id": seq.batch_id, "n_samples": n_samples, "best": best}


def cache_sequences(
    root: str | Path,
    split_json: str | Path,
    out_root: str | Path,
    splits: list[str],
    max_sequences: int = 0,
    max_frames: int = 0,
    sample_out: str | Path | None = None,
    workers: int = 8,
) -> dict[str, int]:
    root = Path(root)
    out_root = Path(out_root)
    payload = json.loads(Path(split_json).read_text(encoding="utf-8"))
    sequences = load_sequences(root)
    cfg = resolve_data_config()
    written = {}
    preview = None
    for split in splits:
        batch_ids = list(payload[split])
        if max_sequences > 0:
            batch_ids = sorted(batch_ids, key=lambda item: sequences[item].num_frames)[:max_sequences]
        split_dir = out_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        filter_outside = split == "train"
        jobs = []
        for batch_id in batch_ids:
            seq = sequences[batch_id]
            limit = seq.num_frames if max_frames <= 0 else min(max_frames, seq.num_frames)
            jobs.append((str(root), seq, str(split_dir), limit, filter_outside))
        n_samples = 0
        done = 0
        best_job: tuple | None = None
        pool_workers = max(1, min(workers, len(jobs)))
        with ProcessPoolExecutor(max_workers=pool_workers) as pool:
            futures = [pool.submit(_cache_sequence_job, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                done += 1
                n_samples += int(result["n_samples"])
                best = result["best"]
                if best is not None and (best_job is None or best[0] > best_job[0]):
                    best_job = (best[0], result["batch_id"], best[1])
                if done % 10 == 0 or done == len(jobs):
                    print(
                        f"{split}: {done}/{len(jobs)} sequences, {n_samples} frames",
                        flush=True,
                    )
        if best_job is not None:
            _, batch_id, frame_pos = best_job
            seq = sequences[batch_id]
            built = make_sample(
                root,
                seq,
                frame_pos,
                _track_index(seq),
                cfg,
                filter_boxes_outside_range=filter_outside,
            )
            if built is not None:
                sample, velocity = built
                dest = split_dir / batch_id / f"sample_{sample.frame_index:06d}.npz"
                candidate = (_preview_score(sample), dest, sample, velocity, seq, seq.frames[frame_pos])
                if preview is None or candidate[0] > preview[0]:
                    preview = candidate
        manifest = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "dataset": "RESOLVE",
            "split": split,
            "class_id_mode": "contiguous",
            "class_names": list(CLASS_NAMES),
            "lidar_dir": LIDAR_DIR,
            "hd_map": False,
            "split_json": str(Path(split_json).resolve()),
            "num_samples": n_samples,
            "max_sequences": max_sequences,
            "max_frames": max_frames,
            "config": {
                "point_cloud_range": [cfg.roi_m[0], cfg.roi_m[1], cfg.z_range_m[0], cfg.roi_m[2], cfg.roi_m[3], cfg.z_range_m[1]],
                "roi_m": list(cfg.roi_m),
                "z_range_m": list(cfg.z_range_m),
                "voxel_size_m": cfg.voxel_size_m,
                "filter_boxes_outside_range": split == "train",
                "history_sweeps": cfg.history_sweeps,
                "future_steps": cfg.future_steps,
                "frame_stride": cfg.frame_stride,
            },
        }
        atomic_write_json(split_dir / "manifest.json", manifest)
        written[split] = n_samples
        print(f"{split}: wrote {n_samples} samples under {split_dir}", flush=True)
    if sample_out is not None and preview is not None:
        export_preview(sample_out, *preview[1:])
    return written


def _preview_score(sample: SceneSample) -> tuple[int, int, float]:
    """Prefer a multi-sweep frame with a fully observed, moving track."""

    if len(sample.points):
        n_dt = len({round(float(v), 3) for v in sample.points[:, 3]})
    else:
        n_dt = 0
    full = int(sample.gt_future_mask.all(axis=1).any()) if len(sample.gt_future_mask) else 0
    disp = 0.0
    for idx in range(len(sample.gt_boxes)):
        valid = sample.gt_future_mask[idx]
        if not valid.any():
            continue
        last = int(np.flatnonzero(valid)[-1])
        disp = max(disp, float(np.linalg.norm(sample.gt_future[idx, last] - sample.gt_boxes[idx, :2])))
    return (full, n_dt, disp)


def export_preview(sample_out, npz_path, sample: SceneSample, velocity, seq: Sequence, frame: AnnFrame) -> None:
    """Write one human-readable sample next to a copy of its npz."""

    sample_out = Path(sample_out)
    sample_out.mkdir(parents=True, exist_ok=True)
    dest_npz = sample_out / "sample.npz"
    dest_npz.write_bytes(Path(npz_path).read_bytes())
    objects = []
    for idx in range(len(sample.gt_boxes)):
        objects.append(
            {
                "track_id": str(sample.gt_track_ids[idx]),
                "class_name": CLASS_NAMES[int(sample.gt_types[idx])],
                "box_xyzlwh_yaw": sample.gt_boxes[idx].tolist(),
                "velocity_xy": velocity[idx].tolist(),
                "num_lidar_points": int(sample.gt_num_lidar_points[idx]),
                "future_xy": sample.gt_future[idx].tolist(),
                "future_mask": sample.gt_future_mask[idx].astype(bool).tolist(),
                "history_xy": sample.gt_history[idx].tolist(),
                "history_mask": sample.gt_history_mask[idx].astype(bool).tolist(),
            }
        )
    pts = sample.points
    payload = {
        "npz": str(dest_npz),
        "source_cache": str(npz_path),
        "rosbag": seq.rosbag,
        "batch_id": seq.batch_id,
        "frame_index": frame.frame_index,
        "stem": frame.stem,
        "timestamp_s": frame.timestamp_s,
        "lidar_dir": LIDAR_DIR,
        "point_cloud_range_xyz": [ROI_M[0], ROI_M[1], Z_RANGE_M[0], ROI_M[2], ROI_M[3], Z_RANGE_M[1]],
        "hd_map_tokens": 0,
        "ego_pose": "identity",
        "num_points": int(len(pts)),
        "point_dt_unique": sorted({round(float(v), 4) for v in pts[:, 3]}) if len(pts) else [],
        "point_xyz_min": pts[:, :3].min(axis=0).tolist() if len(pts) else None,
        "point_xyz_max": pts[:, :3].max(axis=0).tolist() if len(pts) else None,
        "num_objects": len(objects),
        "objects": objects,
        "protocol": {
            "future_waypoint_times_s": [(step + 1) * FUTURE_DT_S for step in range(FUTURE_STEPS)],
            "horizon_s": HORIZON_S,
            "note": "future_xy is the same track's center in this static lidar frame",
        },
    }
    (sample_out / "sample.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"sample preview: {sample_out / 'sample.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="RESOLVE split and MC-DeTra cache")
    sub = parser.add_subparsers(dest="cmd", required=True)

    split_parser = sub.add_parser("split")
    split_parser.add_argument("--root", required=True)
    split_parser.add_argument("--out", required=True)

    cache_parser = sub.add_parser("cache")
    cache_parser.add_argument("--root", required=True)
    cache_parser.add_argument("--split-json", required=True)
    cache_parser.add_argument("--out", required=True)
    cache_parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    cache_parser.add_argument("--max-sequences", type=int, default=0)
    cache_parser.add_argument("--max-frames", type=int, default=0)
    cache_parser.add_argument("--sample-out", default=None)
    cache_parser.add_argument("--workers", type=int, default=8)

    args = parser.parse_args()
    if args.cmd == "split":
        sequences = load_sequences(args.root)
        assignment = assign_sequences(sequences)
        payload = write_split(args.out, sequences, assignment)
        print(json.dumps(payload["counts"], indent=2))
        audit = payload["official_random_split_audit"]
        print(
            "official neighbor leak "
            f"{audit['val_frames_with_train_neighbor']}/{audit['num_val_frames']} "
            f"future leak {audit['val_frames_whose_2hz_future_intersects_train']}/{audit['num_val_frames']}"
        )
        return
    cache_sequences(
        args.root,
        args.split_json,
        args.out,
        args.splits,
        max_sequences=args.max_sequences,
        max_frames=args.max_frames,
        sample_out=args.sample_out,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
