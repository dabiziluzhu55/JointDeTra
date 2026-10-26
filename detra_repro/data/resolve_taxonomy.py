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

# ``sunlakes_dataset.yaml`` lists [-60, -60, -8, 90, 80, 0]. Every sunlakes
# detector except ``second.yaml`` overrides that with the range below
# (centerpoint, transfusion, bevfusion, pointpillar, dsvt, ...).
# (88.8 - (-60)) / 0.1 = 1488, which the BEV encoder can downsample by 16.
POINT_CLOUD_RANGE = (-60.0, -60.0, -8.0, 88.8, 88.8, 0.0)
ROI_M = (POINT_CLOUD_RANGE[0], POINT_CLOUD_RANGE[1], POINT_CLOUD_RANGE[3], POINT_CLOUD_RANGE[4])
Z_RANGE_M = (POINT_CLOUD_RANGE[2], POINT_CLOUD_RANGE[5])
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
