#!/usr/bin/env python3
"""Project RESOLVE lidar points and 3D boxes onto the four roadside cameras.

The points are the raw synchronized ``128_128b_sync`` frame named by the
camera cache, not the ROI-cropped multi-sweep tensor. Boxes are the cache
ground truth in that same lidar frame. Depth coloring and box colors follow
``SSLFusion/tools/visualize_labels.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detra_repro.data.resolve_camera import (  # noqa: E402
    BOX_EDGES,
    CAMERAS,
    CLASS_BGR,
    CLASS_NAMES,
    boxes_to_corners_3d,
    project_points,
    read_lidar_xyz,
)


def _depth_colors(depth: np.ndarray, depth_max: float) -> np.ndarray:
    """BGR colors. Near is orange, mid is yellow-green, far is blue."""

    depth_norm = np.clip(depth, 0.0, depth_max) / depth_max
    near = np.array([0, 191, 255], dtype=np.float32)
    mid = np.array([0, 255, 210], dtype=np.float32)
    far = np.array([255, 144, 30], dtype=np.float32)
    colors = np.zeros((len(depth_norm), 3), dtype=np.float32)
    mid_mask = depth_norm <= 0.5
    far_mask = ~mid_mask
    if mid_mask.any():
        ratio = depth_norm[mid_mask] / 0.5
        colors[mid_mask] = near + (mid - near) * ratio[:, None]
    if far_mask.any():
        ratio = (depth_norm[far_mask] - 0.5) / 0.5
        colors[far_mask] = mid + (far - mid) * ratio[:, None]
    return np.clip(colors, 0, 255).astype(np.uint8)


def _class_bgr(name: str) -> tuple[int, int, int]:
    return CLASS_BGR.get(name, (244, 196, 42))


def draw_projection(
    image: np.ndarray,
    points_xyz: np.ndarray,
    boxes: np.ndarray,
    class_names: list[str],
    lidar2cam: np.ndarray,
    k_mat: np.ndarray,
    distortion: np.ndarray,
    depth_max: float,
) -> tuple[np.ndarray, dict]:
    canvas = image.copy()
    height, width = canvas.shape[:2]
    uv, depth = project_points(points_xyz, lidar2cam, k_mat, distortion)
    mask = (
        (depth > 0.0)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < height)
    )
    pixels = uv[mask]
    visible_depth = depth[mask]
    order = np.argsort(-visible_depth)
    pixels = pixels[order]
    visible_depth = visible_depth[order]
    colors = _depth_colors(visible_depth, depth_max)
    for (u_pix, v_pix), color in zip(pixels, colors):
        cv2.circle(canvas, (int(u_pix), int(v_pix)), 2, color.tolist(), -1, lineType=cv2.LINE_AA)

    boxes_drawn = 0
    if len(boxes):
        corners = boxes_to_corners_3d(boxes)
        for index, box_corners in enumerate(corners):
            corner_uv, corner_z = project_points(box_corners, lidar2cam, k_mat, distortion)
            inside = (
                (corner_z > 0.0)
                & (corner_uv[:, 0] >= 0.0)
                & (corner_uv[:, 0] < width)
                & (corner_uv[:, 1] >= 0.0)
                & (corner_uv[:, 1] < height)
            )
            if not np.any(inside):
                continue
            boxes_drawn += 1
            name = class_names[index] if index < len(class_names) else ""
            color = _class_bgr(name)
            for start, end in BOX_EDGES:
                if corner_z[start] <= 0.0 or corner_z[end] <= 0.0:
                    continue
                p1 = (int(round(corner_uv[start, 0])), int(round(corner_uv[start, 1])))
                p2 = (int(round(corner_uv[end, 0])), int(round(corner_uv[end, 1])))
                ok, q1, q2 = cv2.clipLine((0, 0, width, height), p1, p2)
                if ok:
                    cv2.line(canvas, q1, q2, color, 2, lineType=cv2.LINE_AA)
    stats = {
        "points_projected": int(mask.sum()),
        "boxes_drawn": boxes_drawn,
        "image_hw": [int(height), int(width)],
    }
    return canvas, stats


def _label_panel(image: np.ndarray, text: str) -> np.ndarray:
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(panel, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def _mosaic(panels: list[np.ndarray]) -> np.ndarray:
    resized = [cv2.resize(panel, (960, 540), interpolation=cv2.INTER_AREA) for panel in panels]
    top = np.concatenate(resized[:2], axis=1)
    bottom = np.concatenate(resized[2:], axis=1)
    return np.concatenate([top, bottom], axis=0)


def _load_boxes(point_npz: Path) -> tuple[np.ndarray, list[str]]:
    with np.load(point_npz, allow_pickle=True) as data:
        boxes = data["gt_boxes"].astype(np.float64)
        types = data["gt_types"].astype(np.int64)
        if "gt_class_names" in data.files:
            names = [str(item) for item in data["gt_class_names"].tolist()]
        else:
            names = list(CLASS_NAMES)
    class_names = [names[int(type_id)] if 0 <= int(type_id) < len(names) else "other" for type_id in types]
    return boxes, class_names


def _box_count(cache_root: Path, item: dict) -> int:
    with np.load(cache_root / item["point_npz"], allow_pickle=True) as data:
        return int(len(data["gt_boxes"]))


def _pick_frame(cache_root: Path, frames: list[dict], *, prefer_mismatch: bool) -> dict:
    """Among a few frames in one bag, keep the one with the most cached boxes."""

    ordered = sorted(frames, key=lambda item: int(item["frame_index"]))
    if prefer_mismatch:
        mismatched = [item for item in ordered if int(item["json_matches"]) < 4]
        if mismatched:
            ordered = mismatched
    if len(ordered) > 7:
        positions = np.linspace(0, len(ordered) - 1, 7).astype(int)
        ordered = [ordered[int(pos)] for pos in positions]
    best = ordered[0]
    best_count = -1
    for item in ordered:
        count = _box_count(cache_root, item)
        if count > best_count:
            best = item
            best_count = count
    return best


def _pick_samples(cache_root: Path, manifest: dict, num_samples: int) -> list[dict]:
    """Spread frames across bags and splits, including one json-stem mismatch."""

    grouped: dict[tuple[str, str], list[dict]] = {}
    for item in manifest["samples"]:
        if int(item["cameras_present"]) != 4:
            continue
        grouped.setdefault((item["split"], item["rosbag"]), []).append(item)
    if not grouped:
        return []
    split_order = {"train": 0, "val": 1, "test": 2}
    keys = sorted(grouped, key=lambda key: (split_order.get(key[0], 9), key[1]))
    stride = max(1, len(keys) // num_samples)
    chosen_keys: list[tuple[str, str]] = []
    for key in keys[::stride]:
        if key not in chosen_keys:
            chosen_keys.append(key)
        if len(chosen_keys) >= num_samples:
            break
    mismatch_key = next(
        (key for key in keys if any(int(item["json_matches"]) < 4 for item in grouped[key])),
        None,
    )
    if mismatch_key is not None and mismatch_key not in chosen_keys:
        chosen_keys[-1] = mismatch_key
    return [
        _pick_frame(cache_root, grouped[key], prefer_mismatch=(key == mismatch_key))
        for key in chosen_keys[:num_samples]
    ]


def export_and_render(
    cache_root: Path,
    output_dir: Path,
    num_samples: int,
    depth_max: float,
) -> list[dict]:
    manifest = json.loads((cache_root / "cameras_manifest.json").read_text(encoding="utf-8"))
    picked = _pick_samples(cache_root, manifest, num_samples)
    if len(picked) < 5:
        raise RuntimeError(f"only found {len(picked)} four-camera frames")
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = []
    for rank, item in enumerate(picked):
        slug = f"{rank:02d}_{item['split']}_{item['rosbag']}_f{int(item['frame_index']):06d}"
        sample_dir = output_dir / slug
        image_dir = sample_dir / "images"
        vis_dir = sample_dir / "vis"
        image_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)
        camera_src = cache_root / item["camera_npz"]
        point_src = cache_root / item["point_npz"]
        _publish_file(point_src, sample_dir / "sample.npz")
        _publish_file(camera_src, sample_dir / "camera.npz")
        boxes, class_names = _load_boxes(point_src)
        with np.load(camera_src, allow_pickle=False) as camera:
            points = read_lidar_xyz(str(camera["source_lidar"]))
            panels = []
            per_camera = []
            for index, camera_name in enumerate(CAMERAS):
                camera_name = str(camera["camera_names"][index])
                image_path = cache_root / item["split"] / item["batch_id"] / "images" / camera_name / f"sample_{int(item['frame_index']):06d}.jpg"
                _publish_file(image_path, image_dir / f"{camera_name}.jpg")
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(image_path)
                canvas, stats = draw_projection(
                    image,
                    points,
                    boxes,
                    class_names,
                    camera["lidar2cam"][index],
                    camera["K"][index],
                    camera["distortion"][index],
                    depth_max,
                )
                text = (
                    f"{camera_name}  points {stats['points_projected']}  "
                    f"boxes {stats['boxes_drawn']}/{len(boxes)}"
                )
                labeled = _label_panel(canvas, text)
                cv2.imwrite(str(vis_dir / f"{camera_name}.jpg"), labeled, [cv2.IMWRITE_JPEG_QUALITY, 90])
                panels.append(labeled)
                per_camera.append(
                    {
                        "camera": camera_name,
                        "image_stem": str(camera["image_stem"][index]),
                        "json_image_stem": str(camera["json_image_stem"][index]),
                        "json_matches_sync": bool(camera["json_matches_sync"][index]),
                        "image_timestamp_s": float(camera["image_timestamp_s"][index]),
                        "K": camera["K"][index].tolist(),
                        "distortion": camera["distortion"][index].tolist(),
                        "lidar2cam": camera["lidar2cam"][index].tolist(),
                        **stats,
                    }
                )
            mosaic = _mosaic(panels)
            mosaic_path = vis_dir / "mosaic.jpg"
            cv2.imwrite(str(mosaic_path), mosaic, [cv2.IMWRITE_JPEG_QUALITY, 90])
            record = {
                "id": slug,
                "split": item["split"],
                "rosbag": item["rosbag"],
                "batch_id": item["batch_id"],
                "frame_index": int(item["frame_index"]),
                "lidar_stem": str(camera["lidar_stem"]),
                "lidar_timestamp_s": float(camera["lidar_timestamp_s"]),
                "source_lidar": str(camera["source_lidar"]),
                "point_npz": "sample.npz",
                "camera_npz": "camera.npz",
                "num_points_raw": int(len(points)),
                "num_objects": int(len(boxes)),
                "objects": [
                    {"class_name": name, "box_xyzlwh_yaw": boxes[obj_index].tolist()}
                    for obj_index, name in enumerate(class_names)
                ],
                "cameras": per_camera,
                "mosaic": str(mosaic_path.relative_to(output_dir)),
                "note": (
                    "Images, calibration, and boxes are the frame paired by "
                    "{image_stem}__{lidar_stem}_calibration.txt. Points are the "
                    "full unrotated 128_128b_sync cloud."
                ),
            }
        (sample_dir / "sample.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        catalog.append(
            {
                "id": slug,
                "split": item["split"],
                "rosbag": item["rosbag"],
                "frame_index": int(item["frame_index"]),
                "num_objects": int(len(boxes)),
                "points_projected": [entry["points_projected"] for entry in per_camera],
                "boxes_drawn": [entry["boxes_drawn"] for entry in per_camera],
                "json_matches_sync": [entry["json_matches_sync"] for entry in per_camera],
                "mosaic": record["mosaic"],
            }
        )
        print(
            f"{slug}: objects {len(boxes)} projected {catalog[-1]['points_projected']} "
            f"boxes {catalog[-1]['boxes_drawn']}",
            flush=True,
        )
    (output_dir / "index.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    return catalog


def _publish_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        try:
            if src.stat().st_ino == dst.stat().st_ino and src.stat().st_dev == dst.stat().st_dev:
                return
        except OSError:
            pass
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        os.symlink(src, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "data" / "resolve_cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "samples" / "resolve_multimodal")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--depth-max", type=float, default=80.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_and_render(args.cache_root, args.output_dir, args.num_samples, args.depth_max)


if __name__ == "__main__":
    main()
