from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from object3d_engine.domain.value_objects import (
    BoundingBox2D,
    BoundingBox3D,
    CameraIntrinsics,
    EstimationMetadata,
    FeatureVector,
    Pose3D,
)


@dataclass(slots=True)
class FrameData:
    image_rgb: np.ndarray = field(repr=False)
    depth: np.ndarray = field(repr=False)
    intrinsics: CameraIntrinsics
    pose: Pose3D
    frame_id: str
    timestamp: str | None = None


@dataclass(slots=True)
class Detection2D:
    class_name: str
    confidence: float
    bbox: BoundingBox2D
    detection_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MaskDetection(Detection2D):
    mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool), repr=False)
    image_feature: FeatureVector | None = None
    text_feature: FeatureVector | None = None

    @property
    def mask_area(self) -> int:
        return int(np.asarray(self.mask, dtype=bool).sum())


@dataclass(slots=True)
class ObjectObservation3D:
    observation_id: str
    frame_id: str
    class_name: str
    confidence: float
    bbox2d: BoundingBox2D
    mask_area: int
    local_points: np.ndarray = field(repr=False)
    global_points: np.ndarray = field(repr=False)
    bbox3d: BoundingBox3D
    centroid: np.ndarray = field(repr=False)
    support_point_count: int
    metadata: EstimationMetadata = field(default_factory=EstimationMetadata)
    image_feature: FeatureVector | None = None
    text_feature: FeatureVector | None = None
    is_valid: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def global_position(self) -> np.ndarray:
        return self.centroid


@dataclass(slots=True)
class MapObject3D:
    object_id: str
    class_votes: dict[str, int]
    confidence_sum: float
    observations_count: int
    total_support_points: int
    global_points: np.ndarray = field(repr=False)
    bbox3d: BoundingBox3D
    centroid: np.ndarray = field(repr=False)
    frame_ids: list[str] = field(default_factory=list)
    image_feature: FeatureVector | None = None
    text_feature: FeatureVector | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def dominant_class_name(self) -> str:
        if not self.class_votes:
            return "unknown"
        return max(self.class_votes.items(), key=lambda item: item[1])[0]

    @property
    def mean_confidence(self) -> float:
        if self.observations_count <= 0:
            return 0.0
        return self.confidence_sum / self.observations_count

    @classmethod
    def from_observation(cls, object_id: str, observation: ObjectObservation3D) -> "MapObject3D":
        return cls(
            object_id=object_id,
            class_votes={observation.class_name: 1},
            confidence_sum=float(observation.confidence),
            observations_count=1,
            total_support_points=int(observation.support_point_count),
            global_points=np.asarray(observation.global_points, dtype=np.float64),
            bbox3d=observation.bbox3d,
            centroid=np.asarray(observation.centroid, dtype=np.float64),
            frame_ids=[observation.frame_id],
            image_feature=observation.image_feature,
            text_feature=observation.text_feature,
        )


@dataclass(slots=True)
class MatchAssignment:
    observation_index: int
    object_index: int | None
    score: float


@dataclass(slots=True)
class ObjectMapState:
    objects: list[MapObject3D] = field(default_factory=list)
    processed_frames: int = 0
    next_object_index: int = 1

    def allocate_object_id(self) -> str:
        object_id = f"object:{self.next_object_index}"
        self.next_object_index += 1
        return object_id


@dataclass(slots=True)
class EngineResult:
    frame_id: str
    observations: list[ObjectObservation3D]
    map_state: ObjectMapState
    assignments: list[MatchAssignment]
    created_object_ids: list[str]
    serialized_payload: dict[str, Any] | None = None
