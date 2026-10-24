"""RESOLVE label taxonomy and forecast-protocol constants.

The lidar choice and the empty map are data decisions. They do not change
the DeTra / MC-DeTra modules.
"""

# Same 10 names as RESOLVE OpenPCDet configs (centerpoint / transfusion).
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

# Official converter renames. Kept so detection classes match that benchmark.
LABEL_TO_CLASS = {
    "car": "car",
    "truck": "truck",
    "construction vehicle": "construction_vehicle",
    "construction_vehicle": "construction_vehicle",
    "bus": "bus",
    "trailer": "trailer",
    "barrier": "barrier",
    "van": "barrier",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "pedestrian": "pedestrian",
    "traffic_cone": "traffic_cone",
    "traffic cone": "traffic_cone",
    "golf cart": "traffic_cone",
}

CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Moving road vehicles. Vans are labeled ``barrier`` by the official map.
VEHICLE_CLASS_NAMES = ("car", "truck", "construction_vehicle", "bus", "trailer")

# Fused pair of 128-beam lidars. Annotations were drawn on ``128_128b_label``
# and the files share those timestamps. ~72k points/frame, so five sweeps stay
# under the training point cap. 64-beam is about half as dense; 16-beam is
# vertically too sparse for far pedestrians.
LIDAR_DIR = "128_128b_sync"

# Waymo cache map width (7 type channels + 5 geometry channels). RESOLVE has
# no HD map, so every sample stores this many columns and zero tokens.
MAP_FEATURE_DIM = 12

# Paper horizon: 10 Hz raw frames, futures at 2 Hz for 5 s, 5 lidar sweeps.
HZ = 10.0
HISTORY_SWEEPS = 5
FUTURE_STEPS = 10
FRAME_STRIDE = 5
FUTURE_DT_S = FRAME_STRIDE / HZ
HORIZON_S = FUTURE_STEPS * FUTURE_DT_S
TIMESTAMP_TOL_S = 0.05

# Square BEV crop used by the paper encoder. About 1% of boxes lie outside.
ROI_M = (-75.0, -75.0, 75.0, 75.0)
# Roadside returns sit below the sensor. Official range is [-8, 0]; +2 keeps roofs.
Z_RANGE_M = (-8.0, 2.0)
VOXEL_SIZE_M = 0.1

# nuScenes CVPR 2019 thresholds used by RESOLVE's detection eval.
# RESOLVE's accumulate() does not apply class_range, so neither do we.
DET_DIST_THS = (0.5, 1.0, 2.0, 4.0)
DET_DIST_TH_TP = 2.0
DET_MIN_RECALL = 0.1
DET_MIN_PRECISION = 0.1
DET_MAX_BOXES = 500
DET_MEAN_AP_WEIGHT = 5
# Distance bins in sunlakes_utils, unrotated frame.
DET_RANGE_ORIGIN_XY = (-22.0, 0.0)
DET_RANGE_BINS = ((0.0, 50.0), (50.0, 70.0), (70.0, float("inf")))

SPLIT_SEED = 20261024
SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}
