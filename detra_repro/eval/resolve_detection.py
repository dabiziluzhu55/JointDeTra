"""RESOLVE 3D detection metrics, matching the official nuScenes-style eval.

The numbers follow ``RESOLVE/pcdet/datasets/carla/sunlakes_utils.py``:
center-distance matching, thresholds 0.5/1/2/4 m, TP at 2 m, and range bins
0-50 / 50-70 / 70-inf around (-22, 0). ``class_range`` is not applied, because
that script does not apply it either. GT attributes are blank, so attribute
error stays 1 and does not raise NDS (same as the published summaries).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from detra_repro.data.resolve_taxonomy import (
    CLASS_NAMES,
    DET_DIST_TH_TP,
    DET_DIST_THS,
    DET_MAX_BOXES,
    DET_MEAN_AP_WEIGHT,
    DET_MIN_PRECISION,
    DET_MIN_RECALL,
    DET_RANGE_BINS,
    DET_RANGE_ORIGIN_XY,
    FUTURE_DT_S,
)


def _angle_diff(x: float, y: float, period: float) -> float:
    diff = (x - y + period / 2) % period - period / 2
    if diff > np.pi:
        diff = diff - (2 * np.pi)
    return diff


def _yaw_diff(yaw_gt: float, yaw_est: float, period: float = 2 * np.pi) -> float:
    return abs(_angle_diff(yaw_gt, yaw_est, period))


def _scale_iou(size_gt: np.ndarray, size_est: np.ndarray) -> float:
    inter = float(np.prod(np.minimum(size_gt, size_est)))
    union = float(np.prod(size_gt) + np.prod(size_est) - inter)
    return inter / max(union, 1e-8)


def _cummean(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    if np.isnan(values).all():
        return np.ones(len(values))
    summed = np.nancumsum(values.astype(np.float64))
    count = np.cumsum(~np.isnan(values))
    out = np.zeros_like(summed)
    np.divide(summed, count, out=out, where=count != 0)
    return out


def _attribute(name: str, velocity: np.ndarray) -> str:
    """OpenPCDet attribute rule. Unused for NDS while GT attributes are blank."""

    speed = float(np.linalg.norm(velocity))
    if speed > 0.2:
        if name in {"car", "construction_vehicle", "bus", "truck", "trailer"}:
            return "vehicle.moving"
        if name in {"bicycle", "motorcycle"}:
            return "cycle.with_rider"
    elif name == "pedestrian":
        return "pedestrian.standing"
    elif name == "bus":
        return "vehicle.stopped"
    fallback = {
        "car": "vehicle.parked",
        "truck": "vehicle.parked",
        "bus": "vehicle.moving",
        "trailer": "vehicle.parked",
        "construction_vehicle": "vehicle.parked",
        "pedestrian": "pedestrian.moving",
        "motorcycle": "cycle.without_rider",
        "bicycle": "cycle.without_rider",
        "traffic_cone": "cycle.with_rider",
        "barrier": "cycle.with_rider",
    }
    return fallback[name]


def _range_m(box: dict) -> float:
    ox, oy = DET_RANGE_ORIGIN_XY
    return float(np.hypot(box["translation"][0] - ox, box["translation"][1] - oy))


def _accumulate(gt_by_token: dict, pred_by_token: dict, class_name: str, dist_th: float) -> dict:
    gt_list = [box for boxes in gt_by_token.values() for box in boxes if box["name"] == class_name]
    npos = len(gt_list)
    if npos == 0:
        return {"ap": 0.0, "tp": {key: 1.0 for key in ("trans_err", "scale_err", "orient_err", "vel_err", "attr_err")}}
    preds = [box for boxes in pred_by_token.values() for box in boxes if box["name"] == class_name]
    preds.sort(key=lambda box: box["score"], reverse=True)
    taken: set[tuple[str, int]] = set()
    tp, fp, conf = [], [], []
    match = {key: [] for key in ("trans_err", "vel_err", "scale_err", "orient_err", "attr_err", "conf")}
    gt_index = {token: boxes for token, boxes in gt_by_token.items()}
    for pred in preds:
        best, best_idx = np.inf, None
        for gt_idx, gt in enumerate(gt_index.get(pred["token"], [])):
            if gt["name"] != class_name or (pred["token"], gt_idx) in taken:
                continue
            dist = float(np.linalg.norm(np.asarray(pred["translation"][:2]) - np.asarray(gt["translation"][:2])))
            if dist < best:
                best, best_idx = dist, gt_idx
        if best < dist_th and best_idx is not None:
            taken.add((pred["token"], best_idx))
            gt = gt_index[pred["token"]][best_idx]
            tp.append(1.0)
            fp.append(0.0)
            conf.append(pred["score"])
            match["trans_err"].append(best)
            match["vel_err"].append(float(np.linalg.norm(np.asarray(pred["velocity"]) - np.asarray(gt["velocity"]))))
            match["scale_err"].append(1.0 - _scale_iou(np.asarray(gt["size"]), np.asarray(pred["size"])))
            period = np.pi if class_name == "barrier" else 2 * np.pi
            match["orient_err"].append(_yaw_diff(gt["yaw"], pred["yaw"], period))
            # Blank GT attributes reproduce the official mAAE of 1.
            acc = np.nan if gt["attr"] == "" else float(gt["attr"] == pred["attr"])
            match["attr_err"].append(1.0 - acc)
            match["conf"].append(pred["score"])
        else:
            tp.append(0.0)
            fp.append(1.0)
            conf.append(pred["score"])
    if not match["trans_err"]:
        return {"ap": 0.0, "tp": {key: 1.0 for key in ("trans_err", "scale_err", "orient_err", "vel_err", "attr_err")}}
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    conf_a = np.asarray(conf, dtype=np.float64)
    prec = tp_c / np.maximum(tp_c + fp_c, 1e-8)
    rec = tp_c / float(npos)
    rec_i = np.linspace(0, 1, 101)
    prec_i = np.interp(rec_i, rec, prec, right=0)
    conf_i = np.interp(rec_i, rec, conf_a, right=0)
    curves = {}
    match_conf = np.asarray(match["conf"], dtype=np.float64)
    match_order = np.argsort(match_conf)
    for key in ("trans_err", "vel_err", "scale_err", "orient_err", "attr_err"):
        averaged = _cummean(np.asarray(match[key], dtype=np.float64))
        if float(np.ptp(match_conf)) < 1e-8:
            curves[key] = np.full(conf_i.shape, float(averaged[-1]))
        else:
            curves[key] = np.interp(conf_i[::-1], match_conf[match_order], averaged[match_order])[::-1]
    prec_clip = prec_i[round(100 * DET_MIN_RECALL) + 1 :] - DET_MIN_PRECISION
    prec_clip[prec_clip < 0] = 0
    ap = float(np.mean(prec_clip)) / (1.0 - DET_MIN_PRECISION)
    nonzero = np.nonzero(conf_i)[0]
    max_recall = 0 if len(nonzero) == 0 else int(nonzero[-1])
    first = round(100 * DET_MIN_RECALL) + 1
    tp_errors = {}
    for key, curve in curves.items():
        if max_recall < first:
            tp_errors[key] = 1.0
        else:
            tp_errors[key] = float(np.mean(curve[first : max_recall + 1]))
    return {"ap": ap, "tp": tp_errors}


def evaluate_boxes(gt_by_token: dict, pred_by_token: dict) -> dict:
    label_aps = {}
    label_tp = {}
    for name in CLASS_NAMES:
        label_aps[name] = {}
        tp_ref = None
        for dist_th in DET_DIST_THS:
            result = _accumulate(gt_by_token, pred_by_token, name, dist_th)
            label_aps[name][f"{dist_th:.1f}"] = result["ap"]
            if abs(dist_th - DET_DIST_TH_TP) < 1e-6:
                tp_ref = result["tp"]
        label_tp[name] = tp_ref
    mean_dist = {name: float(np.mean(list(aps.values()))) for name, aps in label_aps.items()}
    mean_ap = float(np.mean(list(mean_dist.values())))
    tp_errors = {}
    for key in ("trans_err", "scale_err", "orient_err", "vel_err", "attr_err"):
        tp_errors[key] = float(np.nanmean([label_tp[name][key] for name in CLASS_NAMES]))
    tp_scores = {key: max(0.0, 1.0 - err) for key, err in tp_errors.items()}
    nds = (DET_MEAN_AP_WEIGHT * mean_ap + sum(tp_scores.values())) / (DET_MEAN_AP_WEIGHT + len(tp_scores))
    summary = {
        "label_aps": label_aps,
        "mean_dist_aps": mean_dist,
        "mean_ap": mean_ap,
        "label_tp_errors": label_tp,
        "tp_errors": tp_errors,
        "tp_scores": tp_scores,
        "nd_score": float(nds),
        "class_names": list(CLASS_NAMES),
        "dist_ths": list(DET_DIST_THS),
        "dist_th_tp": DET_DIST_TH_TP,
    }
    bins = {}
    for lo, hi in DET_RANGE_BINS:
        name = f"{int(lo)}-{('inf' if np.isinf(hi) else int(hi))}m"
        gt_bin, pred_bin = {}, {}
        for token, boxes in gt_by_token.items():
            kept = [box for box in boxes if lo <= _range_m(box) < hi]
            if kept:
                gt_bin[token] = kept
        for token, boxes in pred_by_token.items():
            kept = [box for box in boxes if lo <= _range_m(box) < hi]
            if kept:
                pred_bin[token] = kept
        bins[name] = evaluate_boxes_flat(gt_bin, pred_bin) if gt_bin else {"mean_ap": 0.0, "note": "no gt"}
    summary["distance_bins"] = {name: {"mean_ap": item["mean_ap"], "nd_score": item.get("nd_score")} for name, item in bins.items()}
    return summary


def evaluate_boxes_flat(gt_by_token: dict, pred_by_token: dict) -> dict:
    """Same metric as ``evaluate_boxes`` without the distance-bin recursion."""

    label_aps = {}
    label_tp = {}
    for name in CLASS_NAMES:
        label_aps[name] = {}
        tp_ref = None
        for dist_th in DET_DIST_THS:
            result = _accumulate(gt_by_token, pred_by_token, name, dist_th)
            label_aps[name][f"{dist_th:.1f}"] = result["ap"]
            if abs(dist_th - DET_DIST_TH_TP) < 1e-6:
                tp_ref = result["tp"]
        label_tp[name] = tp_ref
    mean_dist = {name: float(np.mean(list(aps.values()))) for name, aps in label_aps.items()}
    mean_ap = float(np.mean(list(mean_dist.values())))
    tp_errors = {key: float(np.nanmean([label_tp[name][key] for name in CLASS_NAMES])) for key in label_tp[CLASS_NAMES[0]]}
    tp_scores = {key: max(0.0, 1.0 - err) for key, err in tp_errors.items()}
    nds = (DET_MEAN_AP_WEIGHT * mean_ap + sum(tp_scores.values())) / (DET_MEAN_AP_WEIGHT + len(tp_scores))
    return {
        "label_aps": label_aps,
        "mean_dist_aps": mean_dist,
        "mean_ap": mean_ap,
        "label_tp_errors": label_tp,
        "tp_errors": tp_errors,
        "nd_score": float(nds),
    }


def boxes_from_cache(cache_dir: str | Path, *, as_predictions: bool = False) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    paths = sorted(Path(cache_dir).glob("sample_*.npz"))
    paths.extend(sorted(Path(cache_dir).glob("*/sample_*.npz")))
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            token = f"{data['context_name']}_{int(data['frame_index'])}"
            boxes = data["gt_boxes"].astype(np.float32).reshape(-1, 7)
            types = data["gt_types"].astype(np.int64)
            vel = data["gt_velocity"].astype(np.float32) if "gt_velocity" in data.files else np.zeros((len(boxes), 2), np.float32)
            future = data["gt_future"].astype(np.float32) if "gt_future" in data.files else np.zeros((len(boxes), 0, 2), np.float32)
            future_mask = (
                data["gt_future_mask"].astype(bool)
                if "gt_future_mask" in data.files
                else np.zeros((len(boxes), future.shape[1]), dtype=bool)
            )
            items = []
            for idx in range(len(boxes)):
                name = CLASS_NAMES[int(types[idx])]
                items.append(
                    {
                        "token": token,
                        "translation": boxes[idx, :3].tolist(),
                        "size": boxes[idx, 3:6].tolist(),
                        "yaw": float(boxes[idx, 6]),
                        "velocity": vel[idx].tolist(),
                        "name": name,
                        "score": 1.0,
                        "attr": "" if not as_predictions else _attribute(name, vel[idx]),
                        "future_xy": future[idx],
                        "future_mask": future_mask[idx],
                        "mode_probs": np.ones((1,), dtype=np.float32),
                    }
                )
            grouped[token] = items
    return grouped


def _norm_type_from_state(state: dict) -> str:
    """BatchNorm keeps running stats. GroupNorm does not."""

    if any(str(key).endswith("running_mean") for key in state):
        return "batch"
    return "group"


def _history_channels_from_state(state: dict) -> int:
    """Width of the motion branch that is added back through ``motion_context_proj``."""

    weight = state.get("history_occupancy_net.0.weight")
    has_motion = any(str(key).startswith(("motion_aux_head.", "motion_context_proj.")) for key in state)
    if weight is None:
        if has_motion:
            raise RuntimeError(
                "checkpoint has motion_aux_head/motion_context_proj but no history_occupancy_net"
            )
        return 0
    channels = int(weight.shape[1])
    proj = state.get("motion_context_proj.weight")
    head = state.get("motion_aux_head.3.weight")
    if proj is None or head is None:
        raise RuntimeError("history occupancy weights are present but the motion residual is not")
    if int(proj.shape[1]) != channels * 3 or int(head.shape[0]) != channels * 3:
        raise RuntimeError(
            f"motion_context_proj width {tuple(proj.shape)} does not match {channels} occupancy channels"
        )
    return channels


def _load_exact(module, state: dict, name: str) -> None:
    """Load every checkpoint tensor. Mismatched architecture is an error."""

    try:
        module.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"{name} does not match the checkpoint architecture") from exc


def load_resolve_modules(checkpoint, device: str):
    """Rebuild the trained stack from the checkpoint, then strict-load every weight.

    The saved ``DataConfig`` / ``ModelConfig`` fix the detection head, past-prediction
    head, and map encoder. Normalization and the motion residual are read from the
    lidar state dict, because those change tensor names without changing ``ModelConfig``.
    """

    import torch

    from detra_repro.checkpoints import load_checkpoint
    from detra_repro.config import DataConfig, ModelConfig
    from detra_repro.models.bev import PaperLikeLidarEncoder
    from detra_repro.models.proposals import PaperProposalHead, ProposalHead
    from detra_repro.models.refinement import DeTraMini, build_map_encoder

    ckpt = checkpoint if isinstance(checkpoint, dict) else load_checkpoint(checkpoint, device)
    data_cfg = ckpt.get("data_cfg")
    model_cfg = ckpt.get("model_cfg")
    if not isinstance(data_cfg, DataConfig) or not isinstance(model_cfg, ModelConfig):
        raise RuntimeError(
            "checkpoint is missing data_cfg/model_cfg; refusing to rebuild a different network"
        )
    for name in ("lidar_encoder", "proposal_head", "map_encoder", "model"):
        if name not in ckpt:
            raise RuntimeError(f"checkpoint is missing {name}")

    lidar_state = ckpt["lidar_encoder"]
    proposal_state = ckpt["proposal_head"]
    norm_type = _norm_type_from_state(lidar_state)
    saved_norm = ckpt.get("norm_type")
    if saved_norm is not None and saved_norm != norm_type:
        raise RuntimeError(f"checkpoint norm_type={saved_norm} but lidar weights look like {norm_type}")
    history_channels = _history_channels_from_state(lidar_state)
    saved_channels = ckpt.get("encoder_history_channels")
    if saved_channels is not None and int(saved_channels) != history_channels:
        raise RuntimeError(
            f"checkpoint encoder_history_channels={saved_channels} but weights have {history_channels}"
        )
    proposal_norm = _norm_type_from_state(proposal_state)
    head_max_idx = max(
        (
            int(key.split(".")[1])
            for key in proposal_state
            if key.startswith("heatmap.") and key.split(".")[1].isdigit()
        ),
        default=0,
    )

    lidar = PaperLikeLidarEncoder(
        data_cfg,
        point_feature_dim=model_cfg.point_feature_dim,
        hidden_dim=model_cfg.hidden_dim,
        dynamic_conv_experts=model_cfg.dynamic_conv_experts,
        norm_type=norm_type,
        history_occupancy_channels=history_channels,
    ).to(device)
    if head_max_idx > 2:
        proposal = PaperProposalHead(
            model_cfg.hidden_dim,
            num_convs=4,
            norm_type=proposal_norm if proposal_norm in {"batch", "group"} else "group",
        ).to(device)
    else:
        proposal = ProposalHead(model_cfg.hidden_dim).to(device)
    map_encoder = build_map_encoder(model_cfg).to(device)
    model = DeTraMini(model_cfg, data_cfg).to(device)
    _load_exact(lidar, lidar_state, "lidar_encoder")
    _load_exact(proposal, proposal_state, "proposal_head")
    _load_exact(map_encoder, ckpt["map_encoder"], "map_encoder")
    _load_exact(model, ckpt["model"], "model")
    lidar.eval()
    proposal.eval()
    map_encoder.eval()
    model.eval()
    return ckpt, data_cfg, model_cfg, lidar, proposal, map_encoder, model


def predictions_from_model(cache_dir: str | Path, checkpoint: str | Path, device: str) -> dict[str, list[dict]]:
    """Run the checkpoint's full stack, including the motion residual on the BEV map."""

    import torch
    from torch.utils.data import DataLoader

    from detra_repro.data.dataset import CachedWaymoDataset, collate_cached_waymo
    from detra_repro.losses import current_boxes_from_pred

    ckpt, data_cfg, model_cfg, lidar, proposal, map_encoder, model = load_resolve_modules(checkpoint, device)
    dataset = CachedWaymoDataset(cache_dir)
    print(f"checkpoint_step={ckpt.get('step')} device={device} frames={len(dataset)}", flush=True)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda samples: collate_cached_waymo(
            samples,
            max_points=400000,
            max_objects=max(model_cfg.num_queries, 4096),
            max_map_tokens=16,
        ),
    )
    grouped: dict[str, list[dict]] = {}
    seen = 0
    with torch.no_grad():
        for batch in loader:
            seen += 1
            if seen == 1 or seen % 50 == 0:
                print(f"infer {seen}/{len(dataset)}", flush=True)
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
            obj = pred["object_logits"][0].sigmoid()
            prob = pred["class_logits"][0].softmax(dim=-1)
            cls = prob.argmax(dim=-1)
            score = obj * prob.gather(1, cls[:, None]).squeeze(1)
            mode_probs = pred["mode_logits"][0].softmax(dim=-1)
            mode = mode_probs.argmax(dim=-1)
            future_all = pred["future_xy"][0]
            future_top = future_all[torch.arange(mode.shape[0], device=device), mode, 0, :]
            velocity = (future_top - boxes[:, :2]) / FUTURE_DT_S
            keep = torch.topk(score, k=min(DET_MAX_BOXES, score.shape[0])).indices
            token = f"{batch['context_names'][0]}_{int(batch['frame_indices'][0])}"
            future_keep = future_all[keep].detach().cpu().numpy()
            prob_keep = mode_probs[keep].detach().cpu().numpy()
            items = []
            for local_idx, idx in enumerate(keep.tolist()):
                name = CLASS_NAMES[int(cls[idx])]
                vel = velocity[idx].detach().cpu().numpy()
                box = boxes[idx].detach().cpu().numpy()
                items.append(
                    {
                        "token": token,
                        "translation": box[:3].tolist(),
                        "size": box[3:6].tolist(),
                        "yaw": float(box[6]),
                        "velocity": vel.tolist(),
                        "name": name,
                        "score": float(score[idx]),
                        "attr": _attribute(name, vel),
                        "future_xy": future_keep[local_idx],
                        "mode_probs": prob_keep[local_idx],
                    }
                )
            grouped[token] = items
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="RESOLVE nuScenes-style detection eval")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--gt-as-pred", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    gt = boxes_from_cache(args.cache, as_predictions=False)
    if args.gt_as_pred:
        pred = boxes_from_cache(args.cache, as_predictions=True)
        for boxes in pred.values():
            for box in boxes:
                box["attr"] = ""
    elif args.checkpoint:
        pred = predictions_from_model(args.cache, args.checkpoint, args.device)
    else:
        raise SystemExit("pass --gt-as-pred or --checkpoint")
    summary = evaluate_boxes(gt, pred)
    from detra_repro.eval.resolve_forecast import evaluate_end_to_end

    if pred and all("future_xy" in box for boxes in pred.values() for box in boxes[:1]):
        summary["forecast"] = evaluate_end_to_end(gt, pred)
    text = json.dumps(
        {k: summary[k] for k in ("mean_ap", "nd_score", "tp_errors", "mean_dist_aps", "distance_bins", "forecast") if k in summary},
        indent=2,
    )
    print(text)
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
