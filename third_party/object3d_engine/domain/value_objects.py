from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _as_float_array(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"expected shape {shape}, got {array.shape}")
    return array


@dataclass(slots=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "CameraIntrinsics":
        matrix = _as_float_array(matrix, (3, 3))
        return cls(
            fx=float(matrix[0, 0]),
            fy=float(matrix[1, 1]),
            cx=float(matrix[0, 2]),
            cy=float(matrix[1, 2]),
        )


@dataclass(slots=True)
class Pose3D:
    matrix: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        self.matrix = _as_float_array(self.matrix, (4, 4))

    @property
    def translation(self) -> np.ndarray:
        return self.matrix[:3, 3].copy()

    @property
    def rotation(self) -> np.ndarray:
        return self.matrix[:3, :3].copy()

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        points = _as_float_array(points)
        if points.size == 0:
            return points.reshape(-1, 3)
        rotated = points @ self.rotation.T
        return rotated + self.translation

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        return self.transform_points(np.asarray(point, dtype=np.float64).reshape(1, 3))[0]

    @classmethod
    def identity(cls) -> "Pose3D":
        return cls(matrix=np.eye(4, dtype=np.float64))


@dataclass(slots=True)
class BoundingBox2D:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(slots=True)
class BoundingBox3D:
    corners: np.ndarray = field(repr=False)
    center: np.ndarray = field(repr=False)
    extent: np.ndarray = field(repr=False)
    min_bound: np.ndarray = field(repr=False)
    max_bound: np.ndarray = field(repr=False)
    volume: float
    oriented: bool = True

    def __post_init__(self) -> None:
        self.corners = _as_float_array(self.corners)
        self.center = _as_float_array(self.center, (3,))
        self.extent = _as_float_array(self.extent, (3,))
        self.min_bound = _as_float_array(self.min_bound, (3,))
        self.max_bound = _as_float_array(self.max_bound, (3,))

    @classmethod
    def from_corners(
        cls,
        corners: np.ndarray,
        center: np.ndarray | None = None,
        extent: np.ndarray | None = None,
        oriented: bool = True,
    ) -> "BoundingBox3D":
        corners = _as_float_array(corners)
        min_bound = corners.min(axis=0)
        max_bound = corners.max(axis=0)
        if center is None:
            center = corners.mean(axis=0)
        if extent is None:
            extent = max_bound - min_bound
        volume = float(np.prod(np.maximum(extent, 0.0)))
        return cls(
            corners=corners,
            center=np.asarray(center, dtype=np.float64),
            extent=np.asarray(extent, dtype=np.float64),
            min_bound=min_bound,
            max_bound=max_bound,
            volume=volume,
            oriented=oriented,
        )


@dataclass(slots=True)
class PointCloudStats:
    point_count: int
    centroid: np.ndarray = field(repr=False)
    min_bound: np.ndarray = field(repr=False)
    max_bound: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        self.centroid = _as_float_array(self.centroid, (3,))
        self.min_bound = _as_float_array(self.min_bound, (3,))
        self.max_bound = _as_float_array(self.max_bound, (3,))


@dataclass(slots=True)
class FeatureVector:
    vector: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector, dtype=np.float64).reshape(-1)
        norm = np.linalg.norm(vector)
        self.vector = vector if norm <= 0 else vector / norm

    def cosine_similarity(self, other: "FeatureVector | None") -> float:
        if other is None:
            return 0.0
        if self.vector.size == 0 or other.vector.size == 0:
            return 0.0
        if self.vector.shape != other.vector.shape:
            raise ValueError(
                f"feature shape mismatch: {self.vector.shape} != {other.vector.shape}"
            )
        return float(np.clip(np.dot(self.vector, other.vector), -1.0, 1.0))


@dataclass(slots=True)
class EstimationMetadata:
    method: str = "mask_pointcloud_centroid"
    mask_area: int = 0
    support_point_count: int = 0
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "method": self.method,
            "mask_area": self.mask_area,
            "support_point_count": self.support_point_count,
        }
        payload.update(self.notes)
        return payload
