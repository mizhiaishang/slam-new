from __future__ import annotations

import numpy as np

from object3d_engine.config.settings import EngineSettings
from object3d_engine.core.bbox_service import BoundingBoxEstimator
from object3d_engine.core.pointcloud_service import PointCloudService
from object3d_engine.domain.entities import MapObject3D, ObjectObservation3D
from object3d_engine.domain.value_objects import FeatureVector
from object3d_engine.interfaces.merger import IObjectMerger


class FusionService(IObjectMerger):
    def __init__(
        self,
        settings: EngineSettings,
        pointcloud_service: PointCloudService,
        bbox_estimator: BoundingBoxEstimator,
    ) -> None:
        self.settings = settings
        self.pointcloud_service = pointcloud_service
        self.bbox_estimator = bbox_estimator

    def merge(self, target: MapObject3D, observation: ObjectObservation3D) -> MapObject3D:
        merged_points = np.concatenate([target.global_points, observation.global_points], axis=0)
        merged_points, _ = self.pointcloud_service.clean(merged_points)
        if len(merged_points) == 0:
            merged_points = target.global_points

        bbox3d = self.bbox_estimator.estimate_bbox(merged_points)
        centroid = merged_points.mean(axis=0) if len(merged_points) > 0 else target.centroid

        target.class_votes[observation.class_name] = target.class_votes.get(observation.class_name, 0) + 1
        target.confidence_sum += float(observation.confidence)
        target.observations_count += 1
        target.total_support_points += int(observation.support_point_count)
        target.global_points = merged_points
        target.bbox3d = bbox3d
        target.centroid = centroid
        if observation.frame_id not in target.frame_ids:
            target.frame_ids.append(observation.frame_id)
        target.image_feature = self._merge_feature(
            current=target.image_feature,
            incoming=observation.image_feature,
            old_count=target.observations_count - 1,
        )
        target.text_feature = self._merge_feature(
            current=target.text_feature,
            incoming=observation.text_feature,
            old_count=target.observations_count - 1,
        )
        return target

    def merge_objects(self, target: MapObject3D, incoming: MapObject3D) -> MapObject3D:
        merged_points = np.concatenate([target.global_points, incoming.global_points], axis=0)
        merged_points, _ = self.pointcloud_service.clean(merged_points)
        if len(merged_points) == 0:
            merged_points = target.global_points

        bbox3d = self.bbox_estimator.estimate_bbox(merged_points)
        centroid = merged_points.mean(axis=0) if len(merged_points) > 0 else target.centroid

        for class_name, vote in incoming.class_votes.items():
            target.class_votes[class_name] = target.class_votes.get(class_name, 0) + vote
        target.confidence_sum += incoming.confidence_sum
        target.observations_count += incoming.observations_count
        target.total_support_points += incoming.total_support_points
        target.global_points = merged_points
        target.bbox3d = bbox3d
        target.centroid = centroid
        target.frame_ids = sorted(set(target.frame_ids + incoming.frame_ids))
        target.image_feature = self._merge_feature(
            current=target.image_feature,
            incoming=incoming.image_feature,
            old_count=max(target.observations_count - incoming.observations_count, 1),
            incoming_count=max(incoming.observations_count, 1),
        )
        target.text_feature = self._merge_feature(
            current=target.text_feature,
            incoming=incoming.text_feature,
            old_count=max(target.observations_count - incoming.observations_count, 1),
            incoming_count=max(incoming.observations_count, 1),
        )
        return target

    @staticmethod
    def _merge_feature(
        current: FeatureVector | None,
        incoming: FeatureVector | None,
        old_count: int,
        incoming_count: int = 1,
    ) -> FeatureVector | None:
        if current is None:
            return incoming
        if incoming is None:
            return current
        merged = (
            current.vector * max(old_count, 1) + incoming.vector * max(incoming_count, 1)
        ) / (max(old_count, 1) + max(incoming_count, 1))
        return FeatureVector(merged)
