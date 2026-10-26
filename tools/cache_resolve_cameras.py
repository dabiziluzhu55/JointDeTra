#!/usr/bin/env python3
"""Pair RESOLVE images and calibration with the existing lidar cache.

Writes next to each ``sample_XXXXXX.npz`` without rewriting it:

    {split}/{batch}/images/axis{1-4}/sample_XXXXXX.jpg
    {split}/{batch}/calib/axis{1-4}/sample_XXXXXX.txt
    {split}/{batch}/cameras/sample_XXXXXX.npz

Images and calibration texts are hardlinks (same filesystem) to the
synchronized source files. The npz stores K, distortion, and lidar2cam.
A frame with no local image still gets a camera npz so every lidar sample
has an explicit camera record.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detra_repro.data.resolve_camera import (  # noqa: E402
    CALIB_LIDAR,
    CAMERAS,
    LIDAR_DIR,
    parse_calibration,
    project_points,
    read_lidar_xyz,
    stem_timestamp_s,
)


def _jpeg_hw(path: Path) -> tuple[int, int]:
    """Return (height, width) from a JPEG SOF marker, or (0, 0)."""

    try:
        with path.open("rb") as handle:
            if handle.read(2) != b"\xff\xd8":
                return (0, 0)
            while True:
                marker_byte = handle.read(1)
                if not marker_byte:
                    return (0, 0)
                if marker_byte != b"\xff":
                    continue
                marker = handle.read(1)
                while marker == b"\xff":
                    marker = handle.read(1)
                if marker in {b"\xc0", b"\xc1", b"\xc2"}:
                    handle.read(3)
                    height, width = np.frombuffer(handle.read(4), dtype=">u2")
                    return int(height), int(width)
                length_raw = handle.read(2)
                if len(length_raw) < 2:
                    return (0, 0)
                length = int.from_bytes(length_raw, "big")
                handle.seek(length - 2, os.SEEK_CUR)
    except OSError:
        return (0, 0)


def _publish_link(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        try:
            src_stat = src.stat()
            dst_stat = dst.stat()
            if src_stat.st_ino == dst_stat.st_ino and src_stat.st_dev == dst_stat.st_dev:
                return "hardlink"
        except OSError:
            pass
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return "symlink"
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        os.symlink(src, dst)
        return "symlink"


def _atomic_savez(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def index_annotations(dataset_root: Path) -> dict[str, dict[int, dict]]:
    with (dataset_root / "HSQJN_3.json").open("r", encoding="utf-8") as handle:
        frames = json.load(handle)
    grouped: dict[str, dict[int, dict]] = {}
    for frame in frames:
        info = str(frame["info"])
        rosbag = next(part for part in info.split("/") if part.startswith("rosbag2_"))
        images = {}
        for url in frame.get("imgInfo") or []:
            camera = url.split("/")[-2].replace("_label", "")
            if camera in CAMERAS:
                images[camera] = Path(url).stem
        grouped.setdefault(str(frame["batchId"]), {})[int(frame["frameIndex"])] = {
            "rosbag": rosbag,
            "lidar_stem": Path(info).stem,
            "images": images,
        }
    return grouped


def index_calibrations(dataset_root: Path) -> dict[tuple[str, str, str], dict]:
    """Map (rosbag, camera, lidar_stem) to image stem and parsed matrices.

    Roadside calibration is static inside a bag. The first and last file are
    compared; per-frame files are parsed only when those two differ.
    """

    index: dict[tuple[str, str, str], dict] = {}
    varied = 0
    for bag_dir in sorted(path for path in dataset_root.iterdir() if path.name.startswith("rosbag2_")):
        for camera in CAMERAS:
            calib_dir = bag_dir / "calibration_txt" / f"{camera}__{CALIB_LIDAR}"
            if not calib_dir.is_dir():
                continue
            files = sorted(path for path in calib_dir.iterdir() if path.name.endswith("_calibration.txt"))
            if not files:
                continue
            first_k, first_d, first_t = parse_calibration(files[0])
            if len(files) > 1:
                last_k, last_d, last_t = parse_calibration(files[-1])
                varies = not (
                    np.allclose(first_k, last_k)
                    and np.allclose(first_d, last_d)
                    and np.allclose(first_t, last_t)
                )
            else:
                varies = False
            if varies:
                varied += 1
            for path in files:
                stem = path.name[: -len("_calibration.txt")]
                if "__" not in stem:
                    continue
                image_stem, lidar_stem = stem.split("__", 1)
                if varies:
                    k_mat, distortion, lidar2cam = parse_calibration(path)
                else:
                    k_mat, distortion, lidar2cam = first_k, first_d, first_t
                index[(bag_dir.name, camera, lidar_stem)] = {
                    "image_stem": image_stem,
                    "path": path,
                    "K": k_mat,
                    "distortion": distortion,
                    "lidar2cam": lidar2cam,
                }
    print(f"calibration pairs: {len(index)}; bags/cameras whose matrices vary: {varied}", flush=True)
    return index


def _empty_pose() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.eye(3, dtype=np.float64), np.zeros((4,), dtype=np.float64), np.eye(4, dtype=np.float64)


def build_camera_cache(dataset_root: Path, cache_root: Path) -> dict:
    annotations = index_annotations(dataset_root)
    calibrations = index_calibrations(dataset_root)
    hw_cache: dict[tuple[str, str], tuple[int, int]] = {}
    samples = []
    counts = {
        "lidar_samples": 0,
        "with_4_images": 0,
        "with_any_image": 0,
        "images_linked": 0,
        "calibs_linked": 0,
        "json_stem_match": 0,
        "json_stem_mismatch": 0,
        "missing_annotation": 0,
    }
    link_kinds = {"hardlink": 0, "symlink": 0}

    for split in ("train", "val", "test"):
        split_dir = cache_root / split
        if not split_dir.is_dir():
            continue
        batch_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
        for done, batch_dir in enumerate(batch_dirs, start=1):
            batch_id = batch_dir.name
            frames = annotations.get(batch_id)
            if frames is None:
                counts["missing_annotation"] += sum(1 for _ in batch_dir.glob("sample_*.npz"))
                continue
            for sample_path in sorted(batch_dir.glob("sample_*.npz")):
                frame_index = int(sample_path.stem.split("_")[-1])
                frame = frames.get(frame_index)
                if frame is None:
                    counts["missing_annotation"] += 1
                    continue
                counts["lidar_samples"] += 1
                rosbag = frame["rosbag"]
                lidar_stem = frame["lidar_stem"]
                lidar_path = dataset_root / rosbag / LIDAR_DIR / f"{lidar_stem}.bin"
                names = []
                present = []
                image_stems = []
                image_times = []
                json_stems = []
                json_match = []
                source_images = []
                source_calibs = []
                k_stack = []
                d_stack = []
                t_stack = []
                hw_stack = []
                for camera in CAMERAS:
                    names.append(camera)
                    record = calibrations.get((rosbag, camera, lidar_stem))
                    json_stem = frame["images"].get(camera, "")
                    json_stems.append(json_stem)
                    if record is None:
                        k_mat, distortion, lidar2cam = _empty_pose()
                        present.append(False)
                        image_stems.append("")
                        image_times.append(0.0)
                        json_match.append(False)
                        source_images.append("")
                        source_calibs.append("")
                        hw_stack.append((0, 0))
                    else:
                        image_stem = record["image_stem"]
                        image_path = dataset_root / rosbag / f"{camera}_sync" / f"{image_stem}.jpg"
                        image_ok = image_path.is_file()
                        present.append(image_ok)
                        image_stems.append(image_stem)
                        image_times.append(stem_timestamp_s(image_stem))
                        matched = bool(json_stem) and json_stem == image_stem
                        json_match.append(matched)
                        if matched:
                            counts["json_stem_match"] += 1
                        elif json_stem:
                            counts["json_stem_mismatch"] += 1
                        k_mat = record["K"]
                        distortion = record["distortion"]
                        lidar2cam = record["lidar2cam"]
                        source_calibs.append(str(record["path"]))
                        calib_dst = batch_dir / "calib" / camera / f"{sample_path.stem}.txt"
                        kind = _publish_link(record["path"], calib_dst)
                        link_kinds[kind] += 1
                        counts["calibs_linked"] += 1
                        if image_ok:
                            image_dst = batch_dir / "images" / camera / f"{sample_path.stem}.jpg"
                            kind = _publish_link(image_path, image_dst)
                            link_kinds[kind] += 1
                            counts["images_linked"] += 1
                            source_images.append(str(image_path))
                            hw_key = (rosbag, camera)
                            if hw_key not in hw_cache:
                                hw_cache[hw_key] = _jpeg_hw(image_path)
                            hw_stack.append(hw_cache[hw_key])
                        else:
                            source_images.append("")
                            hw_stack.append((0, 0))
                    k_stack.append(k_mat)
                    d_stack.append(distortion)
                    t_stack.append(lidar2cam)
                n_present = int(sum(present))
                if n_present == 4:
                    counts["with_4_images"] += 1
                if n_present > 0:
                    counts["with_any_image"] += 1
                camera_path = batch_dir / "cameras" / f"{sample_path.stem}.npz"
                _atomic_savez(
                    camera_path,
                    {
                        "camera_names": np.asarray(names),
                        "image_present": np.asarray(present, dtype=bool),
                        "image_stem": np.asarray(image_stems),
                        "image_timestamp_s": np.asarray(image_times, dtype=np.float64),
                        "lidar_stem": np.asarray(lidar_stem),
                        "lidar_timestamp_s": np.asarray(stem_timestamp_s(lidar_stem), dtype=np.float64),
                        "K": np.stack(k_stack).astype(np.float64),
                        "distortion": np.stack(d_stack).astype(np.float64),
                        "lidar2cam": np.stack(t_stack).astype(np.float64),
                        "image_hw": np.asarray(hw_stack, dtype=np.int32),
                        "json_image_stem": np.asarray(json_stems),
                        "json_matches_sync": np.asarray(json_match, dtype=bool),
                        "frame_index": np.asarray(frame_index, dtype=np.int32),
                        "batch_id": np.asarray(batch_id),
                        "rosbag": np.asarray(rosbag),
                        "split": np.asarray(split),
                        "source_image": np.asarray(source_images),
                        "source_calibration": np.asarray(source_calibs),
                        "source_lidar": np.asarray(str(lidar_path)),
                        "extrinsic_frame": np.asarray(
                            "128_128b_sync unrotated; lidar2cam is X_cam = R @ X_lidar + t"
                        ),
                    },
                )
                samples.append(
                    {
                        "split": split,
                        "batch_id": batch_id,
                        "frame_index": frame_index,
                        "rosbag": rosbag,
                        "lidar_stem": lidar_stem,
                        "cameras_present": n_present,
                        "json_matches": int(sum(json_match)),
                        "camera_npz": str(camera_path.relative_to(cache_root)),
                        "point_npz": str(sample_path.relative_to(cache_root)),
                    }
                )
            if done % 20 == 0 or done == len(batch_dirs):
                print(
                    f"{split}: {done}/{len(batch_dirs)} sequences, "
                    f"{counts['with_4_images']} four-camera frames so far",
                    flush=True,
                )

    payload = {
        "version": 1,
        "dataset": "RESOLVE",
        "lidar_dir": LIDAR_DIR,
        "cameras": list(CAMERAS),
        "pairing": (
            "calibration filename {image_stem}__{lidar_stem}_calibration.txt; "
            "lidar_stem matches 128_128b_sync and the annotation info path"
        ),
        "extrinsic": (
            "lidar2cam maps the unrotated 128_128b_sync frame into the camera. "
            "Do not apply the official -142.5 deg training rotation before projecting."
        ),
        "json_imgInfo": (
            "HSQJN_3.json imgInfo is recorded for audit. The synced image is the "
            "calibration filename, which is the pair that projects onto the vehicles. "
            "imgInfo can differ by about one frame."
        ),
        "counts": counts,
        "link_kinds": link_kinds,
        "samples": samples,
    }
    manifest_path = cache_root / "cameras_manifest.json"
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, manifest_path)
    print(json.dumps({"counts": counts, "link_kinds": link_kinds}, indent=2), flush=True)
    _sanity_check(dataset_root, cache_root, samples)
    return payload


def _sanity_check(dataset_root: Path, cache_root: Path, samples: list[dict]) -> None:
    """Project one full four-camera frame and require points to land on the image."""

    chosen = next((item for item in samples if item["cameras_present"] == 4), None)
    if chosen is None:
        print("sanity: no four-camera frame to check", flush=True)
        return
    with np.load(cache_root / chosen["camera_npz"], allow_pickle=False) as camera:
        points = read_lidar_xyz(str(camera["source_lidar"]))
        if len(points) > 40000:
            points = points[:: max(1, len(points) // 40000)]
        kept = []
        for index, camera_name in enumerate(camera["camera_names"]):
            if not bool(camera["image_present"][index]):
                continue
            height, width = [int(v) for v in camera["image_hw"][index]]
            uv, depth = project_points(
                points,
                camera["lidar2cam"][index],
                camera["K"][index],
                camera["distortion"][index],
            )
            inside = (
                (depth > 0)
                & (uv[:, 0] >= 0)
                & (uv[:, 0] < width)
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < height)
            )
            kept.append((str(camera_name), int(inside.sum()), height, width))
    print(
        f"sanity {chosen['rosbag']} frame {chosen['frame_index']}: {kept}",
        flush=True,
    )
    if not kept or max(count for _, count, _, _ in kept) < 500:
        raise RuntimeError(f"projection sanity check failed: {kept}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/kemove/devdata1/zrl/dataset/RESOLVE"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "data" / "resolve_cache",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        os.nice(10)
    except OSError:
        pass
    build_camera_cache(args.dataset_root, args.cache_root)


if __name__ == "__main__":
    main()
