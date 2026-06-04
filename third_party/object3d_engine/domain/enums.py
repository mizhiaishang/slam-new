from enum import Enum


class SpatialSimilarityType(str, Enum):
    OVERLAP = "overlap"
    BBOX_IOU = "bbox_iou"
    CENTER_DISTANCE = "center_distance"


class MatchStrategy(str, Enum):
    GEOMETRY_ONLY = "geometry_only"
    GEOMETRY_APPEARANCE = "geometry_appearance"


class EngineMode(str, Enum):
    SINGLE_FRAME = "single_frame"
    TRACKING = "tracking"
