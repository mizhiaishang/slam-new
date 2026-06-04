from __future__ import annotations

from dataclasses import replace

import numpy as np

from object3d_engine.config.settings import EngineSettings
from object3d_engine.core.bbox_service import BoundingBoxEstimator
from object3d_engine.core.pointcloud_service import PointCloudService
from object3d_engine.domain.entities import FrameData, MaskDetection, ObjectObservation3D
from object3d_engine.domain.value_objects import EstimationMetadata


class ObservationEstimator:
    def __init__(
        self,
        settings: EngineSettings,
        pointcloud_service: PointCloudService,
        bbox_estimator: BoundingBoxEstimator,
    ) -> None:
        self.settings = settings
        self.pointcloud_service = pointcloud_service
        self.bbox_estimator = bbox_estimator

    def estimate_observation(
        self,
        frame: FrameData,
        detection: MaskDetection,
        observation_id: str,
    ) -> ObjectObservation3D | None:
        if detection.mask_area < self.settings.mask_area_threshold:
            return None
        if detection.confidence < self.settings.mask_conf_threshold:
            return None

        local_points, colors = self.pointcloud_service.project_mask_to_local_points(frame, detection)
        if len(local_points) < max(self.settings.min_points_threshold, 1):
            return None

        local_points, colors = self.pointcloud_service.clean(local_points, colors)
        if len(local_points) < max(self.settings.min_points_threshold, 1):
            return None

        local_points, colors, foreground_notes = self._filter_foreground_points(local_points, colors)
        if len(local_points) < max(self.settings.min_points_threshold, 1):
            return None

        if self.settings.observation_max_depth_m is not None:
            median_depth_m = float(np.median(local_points[:, 2]))
            if not np.isfinite(median_depth_m) or median_depth_m > float(self.settings.observation_max_depth_m):
                return None

        global_points = self.pointcloud_service.transform_to_global(local_points, frame)
        bbox3d = self.bbox_estimator.estimate_bbox(global_points)

        stats = self.pointcloud_service.compute_stats(global_points)
        metadata = EstimationMetadata(
            method="mask_pointcloud_centroid",
            mask_area=detection.mask_area,
            support_point_count=stats.point_count,
            notes={"bbox_oriented": bbox3d.oriented, **foreground_notes},
        )

        return ObjectObservation3D(
            observation_id=observation_id,
            frame_id=frame.frame_id,
            class_name=detection.class_name,
            confidence=float(detection.confidence),
            bbox2d=detection.bbox,
            mask_area=detection.mask_area,
            local_points=local_points,
            global_points=global_points,
            bbox3d=bbox3d,
            centroid=np.asarray(stats.centroid, dtype=np.float64),
            support_point_count=stats.point_count,
            metadata=metadata,
            image_feature=detection.image_feature,
            text_feature=detection.text_feature,
            extra=dict(detection.extra),
        )

    def _filter_foreground_points(
        self,
        points: np.ndarray,
        colors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        colors = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
        min_points = max(
            int(getattr(self.settings, "foreground_min_points_threshold", 50) or 0),
            int(getattr(self.settings, "min_points_threshold", 1) or 1),
            1,
        )
        notes: dict[str, object] = {
            "foreground_filter_enabled": bool(
                getattr(self.settings, "foreground_depth_filter_enabled", True)
                or getattr(self.settings, "foreground_center_filter_enabled", True)
            ),
            "foreground_input_point_count": int(len(points)),
        }
        if len(points) < min_points:
            notes["foreground_filter_applied"] = False
            notes["foreground_filter_reason"] = "insufficient_input_points"
            return points, colors, notes

        current_points = points
        current_colors = colors
        applied_steps: list[str] = []

        if bool(getattr(self.settings, "foreground_depth_filter_enabled", True)):
            depth_window = float(getattr(self.settings, "foreground_depth_window_m", 0.30) or 0.0)
            if depth_window > 0.0:
                median_depth = float(np.median(current_points[:, 2]))
                depth_mask = np.abs(current_points[:, 2] - median_depth) <= depth_window
                candidate_points = current_points[depth_mask]
                candidate_colors = current_colors[depth_mask]
                notes["foreground_depth_median_m"] = median_depth
                notes["foreground_depth_window_m"] = depth_window
                notes["foreground_depth_filtered_point_count"] = int(len(candidate_points))
                if len(candidate_points) >= min_points:
                    current_points = candidate_points
                    current_colors = candidate_colors
                    applied_steps.append("median_depth_window")
                else:
                    notes["foreground_depth_filter_skipped_reason"] = "too_few_points_after_depth_filter"

        if bool(getattr(self.settings, "foreground_center_filter_enabled", True)) and len(current_points) >= min_points:
            percentile = float(getattr(self.settings, "foreground_center_distance_percentile", 90.0) or 0.0)
            percentile = min(100.0, max(0.0, percentile))
            if 0.0 < percentile < 100.0:
                center = np.median(current_points, axis=0)
                distances = np.linalg.norm(current_points - center.reshape(1, 3), axis=1)
                threshold = float(np.percentile(distances, percentile))
                center_mask = distances <= threshold
                candidate_points = current_points[center_mask]
                candidate_colors = current_colors[center_mask]
                notes["foreground_center"] = center.tolist()
                notes["foreground_center_distance_percentile"] = percentile
                notes["foreground_center_distance_threshold_m"] = threshold
                notes["foreground_center_filtered_point_count"] = int(len(candidate_points))
                if len(candidate_points) >= min_points:
                    current_points = candidate_points
                    current_colors = candidate_colors
                    applied_steps.append("center_distance_percentile")
                else:
                    notes["foreground_center_filter_skipped_reason"] = "too_few_points_after_center_filter"

        if bool(getattr(self.settings, "foreground_rerun_clean", True)) and applied_steps:
            candidate_points, candidate_colors = self.pointcloud_service.clean(current_points, current_colors)
            notes["foreground_rerun_clean_point_count"] = int(len(candidate_points))
            if len(candidate_points) >= min_points:
                current_points = candidate_points
                current_colors = candidate_colors
                applied_steps.append("rerun_clean")
            else:
                notes["foreground_rerun_clean_skipped_reason"] = "too_few_points_after_rerun_clean"

        notes["foreground_filter_applied"] = bool(applied_steps)
        notes["foreground_filter_steps"] = applied_steps
        notes["foreground_output_point_count"] = int(len(current_points))
        return current_points, current_colors, notes


class FrameObjectEstimator:
    def __init__(self, observation_estimator: ObservationEstimator) -> None:
        self.observation_estimator = observation_estimator

    def estimate_frame(
        self,
        frame: FrameData,
        detections: list[MaskDetection],
    ) -> list[ObjectObservation3D]:
        observations: list[ObjectObservation3D] = []
        for index, detection in enumerate(detections):
            detection_id = detection.detection_id or f"det:{frame.frame_id}:{index}"
            prepared = replace(detection, detection_id=detection_id)
            observation = self.observation_estimator.estimate_observation(
                frame=frame,
                detection=prepared,
                observation_id=f"obs:{frame.frame_id}:{index}",
            )
            if observation is not None:
                observations.append(observation)
        return observations
