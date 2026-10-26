"""RESOLVE roadside cameras paired with the fused lidar cache.

The point-cloud cache stays in the unrotated ``128_128b_sync`` frame. Each
camera sample is the frame whose calibration filename is
``{image_stem}__{lidar_stem}_calibration.txt``. That filename is the sync
key: the image stem is a file in ``axis{1-4}_sync``, and the lidar stem is
the same stem as ``128_128b_sync`` and the annotation ``info`` path.

``lidar2cam`` maps that lidar frame to the camera:

    X_cam = R @ X_lidar + t

The -142.5 degree yaw used by the official detection export is a training
coordinate change. It is not applied here, and it must not be applied before
this projection. Distortion on the released calibration files is zero; the
projection still honors k1, k2, p1, p2 when they are present.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


CAMERAS = ("axis1", "axis2", "axis3", "axis4")
LIDAR_DIR = "128_128b_sync"
CALIB_LIDAR = "ouster128b"

CLASS_NAMES = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)

# BGR, matching SSLFusion ``visualize_labels.py`` after its RGB-to-BGR swap.
CLASS_BGR = {
    "car": (255, 0, 0),
    "truck": (65, 249, 249),
    "bus": (249, 135, 157),
    "trailer": (65, 249, 249),
    "construction_vehicle": (244, 196, 42),
    "barrier": (180, 180, 180),
    "motorcycle": (84, 130, 53),
    "bicycle": (102, 255, 51),
    "pedestrian": (255, 255, 0),
    "traffic_cone": (0, 165, 255),
}

BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

_INTRINSIC_PAT = re.compile(r"Intrinsic:\s*\n([^\n]+)\n([^\n]+)\n([^\n]+)")
_DISTORT_PAT = re.compile(r"Distortion:\s*\n\[(.*?)\]", re.S)
_R_PAT = re.compile(r"R:\s*(.*?)\s*t:", re.S)
_T_PAT = re.compile(r"t:\s*(.*?)\n")


def stem_timestamp_s(stem: str) -> float:
    sec, nano = stem.split("-")
    return int(sec) + int(nano) / 1e9


def parse_calibration(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``K`` (3, 3), distortion (4,), and lidar-to-camera (4, 4)."""

    content = Path(path).read_text(encoding="utf-8")
    rotation = _R_PAT.search(content)
    translation = _T_PAT.search(content)
    intrinsic = _INTRINSIC_PAT.search(content)
    if rotation is None or translation is None or intrinsic is None:
        raise ValueError(f"calibration is missing R, t, or K: {path}")
    rows = [list(map(float, line.split())) for line in rotation.group(1).strip().splitlines()]
    lidar2cam = np.eye(4, dtype=np.float64)
    lidar2cam[:3, :3] = np.asarray(rows, dtype=np.float64)
    lidar2cam[:3, 3] = np.asarray(list(map(float, translation.group(1).split())), dtype=np.float64)
    k_mat = np.vstack([np.fromstring(row, sep=" ") for row in intrinsic.groups()]).astype(np.float64)
    distort = _DISTORT_PAT.search(content)
    if distort is None:
        distortion = np.zeros((4,), dtype=np.float64)
    else:
        values = np.fromstring(distort.group(1), sep=",", dtype=np.float64)
        distortion = np.zeros((4,), dtype=np.float64)
        distortion[: min(4, values.size)] = values[:4]
    return k_mat, distortion, lidar2cam


def project_points(
    points_xyz: np.ndarray,
    lidar2cam: np.ndarray,
    k_mat: np.ndarray,
    distortion: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project lidar points. Returns pixel ``uv`` (N, 2) and camera-frame ``z`` (N,)."""

    points = np.asarray(points_xyz, dtype=np.float64)[:, :3]
    rotation = np.asarray(lidar2cam, dtype=np.float64)[:3, :3]
    translation = np.asarray(lidar2cam, dtype=np.float64)[:3, 3]
    camera = points @ rotation.T + translation
    depth = camera[:, 2]
    safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    x_n = camera[:, 0] / safe
    y_n = camera[:, 1] / safe
    if distortion is not None and np.any(np.abs(distortion) > 1e-12):
        distorted = _distort(np.stack([x_n, y_n], axis=1), np.asarray(distortion, dtype=np.float64))
        x_n = distorted[:, 0]
        y_n = distorted[:, 1]
    k_mat = np.asarray(k_mat, dtype=np.float64)
    u = k_mat[0, 0] * x_n + k_mat[0, 2]
    v = k_mat[1, 1] * y_n + k_mat[1, 2]
    return np.stack([u, v], axis=1), depth


def _distort(xy: np.ndarray, distortion: np.ndarray) -> np.ndarray:
    """OpenCV brown-conrady order k1, k2, p1, p2. k3 is unused."""

    k1 = float(distortion[0]) if distortion.size > 0 else 0.0
    k2 = float(distortion[1]) if distortion.size > 1 else 0.0
    p1 = float(distortion[2]) if distortion.size > 2 else 0.0
    p2 = float(distortion[3]) if distortion.size > 3 else 0.0
    x = xy[:, 0]
    y = xy[:, 1]
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    x2 = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y2 = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return np.stack([x2, y2], axis=1)


def boxes_to_corners_3d(boxes: np.ndarray) -> np.ndarray:
    """OpenPCDet corner order. ``boxes`` is (N, 7) xyz, lwh, yaw. Returns (N, 8, 3)."""

    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    template = np.array(
        [
            [1, 1, -1],
            [1, -1, -1],
            [-1, -1, -1],
            [-1, 1, -1],
            [1, 1, 1],
            [1, -1, 1],
            [-1, -1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float64,
    ) * 0.5
    corners = boxes[:, None, 3:6] * template[None, :, :]
    yaw = boxes[:, 6]
    cos = np.cos(yaw)
    sin = np.sin(yaw)
    rotated = np.empty_like(corners)
    rotated[:, :, 0] = cos[:, None] * corners[:, :, 0] - sin[:, None] * corners[:, :, 1]
    rotated[:, :, 1] = sin[:, None] * corners[:, :, 0] + cos[:, None] * corners[:, :, 1]
    rotated[:, :, 2] = corners[:, :, 2]
    rotated += boxes[:, None, 0:3]
    return rotated


def read_lidar_xyz(path: str | Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"{path}: float count {raw.size} is not divisible by 4")
    return raw.reshape(-1, 4)[:, :3]


def calibration_name(image_stem: str, lidar_stem: str) -> str:
    return f"{image_stem}__{lidar_stem}_calibration.txt"
