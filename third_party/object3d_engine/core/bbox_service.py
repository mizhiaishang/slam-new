from __future__ import annotations

import numpy as np

try:
    import open3d as o3d
except ImportError:  # pragma: no cover
    o3d = None

from object3d_engine.config.settings import EngineSettings
from object3d_engine.domain.value_objects import BoundingBox3D


class BoundingBoxEstimator:
    def __init__(self, settings: EngineSettings) -> None:
        self.settings = settings

    def estimate_bbox(self, points: np.ndarray) -> BoundingBox3D:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(points) == 0:
            zero = np.zeros((8, 3), dtype=np.float64)
            return BoundingBox3D.from_corners(zero, center=np.zeros(3), extent=np.zeros(3), oriented=False)

        centered = points - points.mean(axis=0, keepdims=True)
        if o3d is None or len(points) < 4 or np.linalg.matrix_rank(centered) < 3:
            return self._estimate_aabb(points)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if self.settings.use_oriented_bbox:
            try:
                obb = pcd.get_oriented_bounding_box(robust=True)
                return BoundingBox3D.from_corners(
                    corners=np.asarray(obb.get_box_points()),
                    center=np.asarray(obb.center),
                    extent=np.asarray(obb.extent),
                    oriented=True,
                )
            except RuntimeError:
                pass

        return self._estimate_aabb(points)

    def _estimate_aabb(self, points: np.ndarray) -> BoundingBox3D:
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        corners = self._aabb_corners(min_bound, max_bound)
        return BoundingBox3D.from_corners(
            corners=corners,
            center=(min_bound + max_bound) / 2.0,
            extent=max_bound - min_bound,
            oriented=False,
        )

    @staticmethod
    def _aabb_corners(min_bound: np.ndarray, max_bound: np.ndarray) -> np.ndarray:
        x1, y1, z1 = min_bound
        x2, y2, z2 = max_bound
        return np.array(
            [
                [x1, y1, z1],
                [x2, y1, z1],
                [x1, y2, z1],
                [x1, y1, z2],
                [x2, y2, z2],
                [x1, y2, z2],
                [x2, y1, z2],
                [x2, y2, z1],
            ],
            dtype=np.float64,
        )
