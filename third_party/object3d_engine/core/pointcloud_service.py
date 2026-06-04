from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import open3d as o3d
except ImportError:  # pragma: no cover
    o3d = None

from object3d_engine.config.settings import EngineSettings
from object3d_engine.domain.entities import FrameData, MaskDetection
from object3d_engine.domain.value_objects import PointCloudStats


class PointCloudService:
    def __init__(self, settings: EngineSettings) -> None:
        self.settings = settings

    def project_mask_to_local_points(
        self,
        frame: FrameData,
        detection: MaskDetection,
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = np.asarray(detection.mask, dtype=bool)
        depth = np.asarray(frame.depth, dtype=np.float64)
        image = np.asarray(frame.image_rgb, dtype=np.uint8)

        if mask.shape != depth.shape:
            if cv2 is None:
                raise ValueError(
                    f"mask shape {mask.shape} != depth shape {depth.shape} and cv2 is unavailable"
                )
            mask = cv2.resize(
                mask.astype(np.uint8),
                (depth.shape[1], depth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        if image.shape[:2] != depth.shape:
            raise ValueError(
                f"image shape {image.shape[:2]} must match depth shape {depth.shape}"
            )

        valid_mask = np.logical_and(mask, np.isfinite(depth))
        valid_mask = np.logical_and(valid_mask, depth > 0)
        if valid_mask.sum() == 0:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

        intr = frame.intrinsics
        v_coords, u_coords = np.where(valid_mask)
        z = depth[valid_mask]
        x = (u_coords - intr.cx) * z / intr.fx
        y = (v_coords - intr.cy) * z / intr.fy
        points = np.stack((x, y, z), axis=-1)
        colors = image[valid_mask] / 255.0
        return points.astype(np.float64), colors.astype(np.float64)

    def transform_to_global(self, local_points: np.ndarray, frame: FrameData) -> np.ndarray:
        return frame.pose.transform_points(local_points)

    def clean(
        self,
        points: np.ndarray,
        colors: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if colors is None or len(colors) != len(points):
            colors = np.zeros((len(points), 3), dtype=np.float64)
        else:
            colors = np.asarray(colors, dtype=np.float64).reshape(-1, 3)

        if len(points) == 0:
            return points, colors
        if o3d is None:
            return points, colors

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        if self.settings.voxel_size > 0:
            pcd = pcd.voxel_down_sample(self.settings.voxel_size)

        if self.settings.dbscan_remove_noise and len(pcd.points) > 0:
            pcd = self._largest_dbscan_cluster(
                pcd,
                eps=self.settings.dbscan_eps,
                min_points=self.settings.dbscan_min_points,
            )

        points_out = np.asarray(pcd.points, dtype=np.float64)
        colors_out = np.asarray(pcd.colors, dtype=np.float64)
        return points_out.reshape(-1, 3), colors_out.reshape(-1, 3)

    def compute_stats(self, points: np.ndarray) -> PointCloudStats:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(points) == 0:
            zero = np.zeros(3, dtype=np.float64)
            return PointCloudStats(point_count=0, centroid=zero, min_bound=zero, max_bound=zero)
        return PointCloudStats(
            point_count=len(points),
            centroid=points.mean(axis=0),
            min_bound=points.min(axis=0),
            max_bound=points.max(axis=0),
        )

    @staticmethod
    def _largest_dbscan_cluster(
        pcd: "o3d.geometry.PointCloud",
        eps: float,
        min_points: int,
    ) -> "o3d.geometry.PointCloud":
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
        if labels.size == 0:
            return pcd
        kept_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        if kept_labels.size == 0:
            return pcd
        dominant = kept_labels[np.argmax(counts)]
        mask = labels == dominant
        if int(mask.sum()) < 5:
            return pcd
        result = o3d.geometry.PointCloud()
        result.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[mask])
        result.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
        return result
