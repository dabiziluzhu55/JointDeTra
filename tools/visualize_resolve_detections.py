#!/usr/bin/env python3
"""Project MC-DeTra detections onto the four RESOLVE roadside cameras.

Same lidar-to-image projection as ``visualize_resolve_cameras.py``. The boxes
are the checkpoint's predictions on validation frames, not the ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detra_repro.data.dataset import CachedWaymoDataset, collate_cached_waymo  # noqa: E402
from detra_repro.data.resolve_camera import CAMERAS, project_points, read_lidar_xyz  # noqa: E402
from detra_repro.data.resolve_taxonomy import CLASS_NAMES  # noqa: E402
from detra_repro.eval.resolve_detection import DET_MAX_BOXES, load_resolve_modules  # noqa: E402
from detra_repro.losses import current_boxes_from_pred  # noqa: E402
from tools.visualize_resolve_cameras import (  # noqa: E402
    _box_count,
    _label_panel,
    _mosaic,
    _publish_file,
    draw_projection,
)


def _pick_frame(cache_root: Path, frames: list[dict]) -> dict:
    """Keep the frame with the most annotated objects so the scene is not empty."""

    ordered = sorted(frames, key=lambda item: int(item["frame_index"]))
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


def pick_val_samples(cache_root: Path, manifest: dict, num_samples: int) -> list[dict]:
    """Spread four-camera validation frames across different rosbags."""

    grouped: dict[str, list[dict]] = {}
    for item in manifest["samples"]:
        if item["split"] != "val" or int(item["cameras_present"]) != 4:
            continue
        grouped.setdefault(item["rosbag"], []).append(item)
    if not grouped:
        raise RuntimeError(f"no four-camera validation frames under {cache_root}")
    keys = sorted(grouped)
    stride = max(1, len(keys) // num_samples)
    chosen = []
    for key in keys[::stride]:
        if key not in chosen:
            chosen.append(key)
        if len(chosen) >= num_samples:
            break
    return [_pick_frame(cache_root, grouped[key]) for key in chosen[:num_samples]]


def _predict_frame(sample_path: Path, modules: dict, device: str, score_thresh: float) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Run one cached frame. Returns boxes (N, 7), class names, and scores."""

    import torch

    dataset = CachedWaymoDataset(sample_path.parent.parent)
    dataset.files = [sample_path]
    batch = collate_cached_waymo(
        [dataset[0]],
        max_points=400000,
        max_objects=max(modules["model_cfg"].num_queries, 4096),
        max_map_tokens=16,
    )
    lidar = modules["lidar"]
    proposal = modules["proposal"]
    map_encoder = modules["map_encoder"]
    model = modules["model"]
    data_cfg = modules["data_cfg"]
    model_cfg = modules["model_cfg"]
    with torch.no_grad():
        points = batch["points"].to(device)
        point_mask = batch["point_mask"].to(device)
        encoded = lidar(points, point_mask=point_mask)
        prop_out = proposal(encoded["feature_map"])
        proposals = proposal.decode(
            prop_out,
            data_cfg,
            num_proposals=model_cfg.num_queries,
            output_stride=model_cfg.proposal_output_stride,
            nms_radius_m=0.1,
            center_mode="regression",
            peak_filter=True,
            nms_mode="rotated",
        )
        map_tokens = map_xy = None
        if map_encoder is not None and batch["map_mask"].any():
            map_xy = batch["map_xy"].to(device)
            map_tokens = map_encoder(batch["map_features"].to(device), map_xy, batch["map_mask"].to(device))
        pred = model(
            encoded["multi_scale_features"],
            proposals["poses"],
            initial_boxes=proposals["boxes"],
            map_tokens=map_tokens,
            map_xy=map_xy,
        )
        boxes = current_boxes_from_pred(pred)[0]
        objectness = pred["object_logits"][0].sigmoid()
        class_prob = pred["class_logits"][0].softmax(dim=-1)
        class_id = class_prob.argmax(dim=-1)
        score = objectness * class_prob.gather(1, class_id[:, None]).squeeze(1)
        keep = torch.topk(score, k=min(DET_MAX_BOXES, score.shape[0])).indices
        keep = keep[score[keep] >= score_thresh]
        chosen = boxes[keep].detach().cpu().numpy()
        names = [CLASS_NAMES[int(class_id[idx])] for idx in keep.tolist()]
        scores = score[keep].detach().cpu().numpy()
    return chosen.astype(np.float64), names, scores.astype(np.float64)


def _draw_score_labels(
    canvas: np.ndarray,
    boxes: np.ndarray,
    names: list[str],
    scores: np.ndarray,
    lidar2cam: np.ndarray,
    k_mat: np.ndarray,
    distortion: np.ndarray,
) -> None:
    if len(boxes) == 0:
        return
    height, width = canvas.shape[:2]
    uv, depth = project_points(boxes[:, :3], lidar2cam, k_mat, distortion)
    order = np.argsort(-depth)
    for index in order:
        if depth[index] <= 0.0:
            continue
        u_pix, v_pix = int(round(uv[index, 0])), int(round(uv[index, 1]))
        if not (0 <= u_pix < width and 0 <= v_pix < height):
            continue
        text = f"{names[index]} {scores[index]:.2f}"
        origin = (u_pix, max(16, v_pix - 4))
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def export_detections(
    cache_root: Path,
    output_dir: Path,
    checkpoint: Path,
    device: str,
    num_samples: int,
    score_thresh: float,
    depth_max: float,
) -> list[dict]:
    manifest = json.loads((cache_root / "cameras_manifest.json").read_text(encoding="utf-8"))
    picked = pick_val_samples(cache_root, manifest, num_samples)
    if len(picked) < num_samples:
        raise RuntimeError(f"only found {len(picked)} validation frames with four cameras")
    ckpt, data_cfg, model_cfg, lidar, proposal, map_encoder, model = load_resolve_modules(checkpoint, device)
    modules = {
        "data_cfg": data_cfg,
        "model_cfg": model_cfg,
        "lidar": lidar,
        "proposal": proposal,
        "map_encoder": map_encoder,
        "model": model,
    }
    print(f"checkpoint_step={ckpt.get('step')} samples={len(picked)} score_thresh={score_thresh}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = []
    for rank, item in enumerate(picked):
        slug = f"{rank:02d}_val_{item['rosbag']}_f{int(item['frame_index']):06d}"
        sample_dir = output_dir / slug
        image_dir = sample_dir / "images"
        vis_dir = sample_dir / "vis"
        image_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)
        point_src = cache_root / item["point_npz"]
        camera_src = cache_root / item["camera_npz"]
        _publish_file(point_src, sample_dir / "sample.npz")
        _publish_file(camera_src, sample_dir / "camera.npz")
        boxes, names, scores = _predict_frame(point_src, modules, device, score_thresh)
        with np.load(camera_src, allow_pickle=False) as camera:
            points = read_lidar_xyz(str(camera["source_lidar"]))
            panels = []
            per_camera = []
            for index in range(len(CAMERAS)):
                camera_name = str(camera["camera_names"][index])
                image_path = (
                    cache_root
                    / item["split"]
                    / item["batch_id"]
                    / "images"
                    / camera_name
                    / f"sample_{int(item['frame_index']):06d}.jpg"
                )
                _publish_file(image_path, image_dir / f"{camera_name}.jpg")
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(image_path)
                canvas, stats = draw_projection(
                    image,
                    points,
                    boxes,
                    names,
                    camera["lidar2cam"][index],
                    camera["K"][index],
                    camera["distortion"][index],
                    depth_max,
                )
                _draw_score_labels(
                    canvas,
                    boxes,
                    names,
                    scores,
                    camera["lidar2cam"][index],
                    camera["K"][index],
                    camera["distortion"][index],
                )
                text = (
                    f"{camera_name}  points {stats['points_projected']}  "
                    f"det {stats['boxes_drawn']}/{len(boxes)}  score>={score_thresh:.2f}"
                )
                labeled = _label_panel(canvas, text)
                cv2.imwrite(str(vis_dir / f"{camera_name}.jpg"), labeled, [cv2.IMWRITE_JPEG_QUALITY, 90])
                panels.append(labeled)
                per_camera.append({"camera": camera_name, **stats})
            mosaic_path = vis_dir / "mosaic.jpg"
            cv2.imwrite(str(mosaic_path), _mosaic(panels), [cv2.IMWRITE_JPEG_QUALITY, 90])
        record = {
            "id": slug,
            "split": "val",
            "rosbag": item["rosbag"],
            "batch_id": item["batch_id"],
            "frame_index": int(item["frame_index"]),
            "checkpoint": str(checkpoint),
            "checkpoint_step": int(ckpt.get("step") or 0),
            "score_thresh": score_thresh,
            "num_detections": int(len(boxes)),
            "detections": [
                {
                    "class_name": names[obj_index],
                    "score": float(scores[obj_index]),
                    "box_xyzlwh_yaw": boxes[obj_index].tolist(),
                }
                for obj_index in range(len(boxes))
            ],
            "cameras": per_camera,
            "mosaic": str(mosaic_path.relative_to(output_dir)),
            "note": "Boxes are MC-DeTra detections. Ground truth is not drawn.",
        }
        (sample_dir / "sample.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        catalog.append(
            {
                "id": slug,
                "rosbag": item["rosbag"],
                "frame_index": int(item["frame_index"]),
                "num_detections": int(len(boxes)),
                "boxes_drawn": [entry["boxes_drawn"] for entry in per_camera],
                "mosaic": record["mosaic"],
            }
        )
        print(f"{slug}: detections {len(boxes)} drawn {catalog[-1]['boxes_drawn']}", flush=True)
    (output_dir / "index.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    return catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "data" / "resolve_cache")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "resolve_mc_detra_step59730.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "samples" / "resolve_mcdetra")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--score-thresh", type=float, default=0.1)
    parser.add_argument("--depth-max", type=float, default=80.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_detections(
        args.cache_root,
        args.output_dir,
        args.checkpoint,
        args.device,
        args.num_samples,
        args.score_thresh,
        args.depth_max,
    )


if __name__ == "__main__":
    main()
