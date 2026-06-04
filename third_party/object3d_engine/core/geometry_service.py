from __future__ import annotations

import numpy as np

from object3d_engine.domain.value_objects import BoundingBox3D, Pose3D


class GeometryService:
    @staticmethod
    def transform_points(points: np.ndarray, pose: Pose3D) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.size == 0:
            return points.reshape(-1, 3)
        return pose.transform_points(points)

    @staticmethod
    def centroid(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.size == 0:
            return np.zeros(3, dtype=np.float64)
        return points.mean(axis=0)

    @staticmethod
    def aabb_iou(box1: BoundingBox3D, box2: BoundingBox3D) -> float:
        inter_min = np.maximum(box1.min_bound, box2.min_bound)
        inter_max = np.minimum(box1.max_bound, box2.max_bound)
        inter_extent = np.maximum(0.0, inter_max - inter_min)
        inter_volume = float(np.prod(inter_extent))
        if inter_volume <= 0:
            return 0.0
        union = box1.volume + box2.volume - inter_volume
        if union <= 0:
            return 0.0
        return inter_volume / union

    @staticmethod
    def center_distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))

    @staticmethod
    def cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
        if a is None or b is None:
            return 0.0
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        if a.size == 0 or b.size == 0 or a.shape != b.shape:
            return 0.0
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm <= 0 or b_norm <= 0:
            return 0.0
        return float(np.clip(np.dot(a / a_norm, b / b_norm), -1.0, 1.0))
