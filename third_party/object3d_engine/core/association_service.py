from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from object3d_engine.config.settings import EngineSettings
from object3d_engine.core.geometry_service import GeometryService
from object3d_engine.domain.entities import MapObject3D, MatchAssignment, ObjectObservation3D
from object3d_engine.domain.enums import SpatialSimilarityType
from object3d_engine.interfaces.matcher import IObjectMatcher


@dataclass(slots=True)
class AssociationScores:
    spatial: float
    appearance: float
    text: float
    aggregate: float


class AssociationService(IObjectMatcher):
    def __init__(self, settings: EngineSettings) -> None:
        self.settings = settings

    def match(
        self,
        observations: list[ObjectObservation3D],
        objects: list[MapObject3D],
    ) -> list[MatchAssignment]:
        assignments: list[MatchAssignment] = []
        if not objects:
            return [MatchAssignment(i, None, float("-inf")) for i in range(len(observations))]

        for obs_index, observation in enumerate(observations):
            best_index: int | None = None
            best_score = float("-inf")
            for obj_index, obj in enumerate(objects):
                scores = self.compute_scores(observation, obj)
                if scores.aggregate > best_score:
                    best_score = scores.aggregate
                    best_index = obj_index
            if best_score < self.settings.match_threshold:
                assignments.append(MatchAssignment(obs_index, None, best_score))
            else:
                assignments.append(MatchAssignment(obs_index, best_index, best_score))
        return assignments

    def compute_scores(
        self,
        observation: ObjectObservation3D,
        obj: MapObject3D,
    ) -> AssociationScores:
        spatial = self._spatial_similarity(observation, obj)
        appearance = 0.0
        text = 0.0

        if self.settings.use_appearance and observation.image_feature and obj.image_feature:
            appearance = observation.image_feature.cosine_similarity(obj.image_feature)
        if self.settings.use_text and observation.text_feature and obj.text_feature:
            text = observation.text_feature.cosine_similarity(obj.text_feature)

        aggregate = self._aggregate(spatial=spatial, appearance=appearance, text=text)
        return AssociationScores(
            spatial=spatial,
            appearance=appearance,
            text=text,
            aggregate=aggregate,
        )

    def _spatial_similarity(
        self,
        observation: ObjectObservation3D,
        obj: MapObject3D,
    ) -> float:
        center_distance = GeometryService.center_distance(observation.centroid, obj.centroid)
        if center_distance > self.settings.max_assignment_distance:
            return 0.0

        center_similarity = max(
            0.0,
            1.0 - (center_distance / max(self.settings.max_assignment_distance, 1e-6)),
        )

        if self.settings.spatial_similarity_type == SpatialSimilarityType.CENTER_DISTANCE:
            return center_similarity

        bbox_iou = GeometryService.aabb_iou(observation.bbox3d, obj.bbox3d)
        if self.settings.spatial_similarity_type == SpatialSimilarityType.BBOX_IOU:
            return bbox_iou

        overlap = self._point_overlap_ratio(observation.global_points, obj.global_points)
        return max(center_similarity * 0.5 + bbox_iou * 0.25 + overlap * 0.25, 0.0)

    def _point_overlap_ratio(self, source: np.ndarray, target: np.ndarray) -> float:
        if len(source) == 0 or len(target) == 0:
            return 0.0
        tree = cKDTree(np.asarray(target, dtype=np.float64))
        distances, _ = tree.query(np.asarray(source, dtype=np.float64), k=1)
        threshold = self.settings.effective_overlap_distance()
        return float((distances <= threshold).sum() / len(source))

    def _aggregate(self, spatial: float, appearance: float, text: float) -> float:
        terms: list[tuple[float, float]] = [(spatial, self.settings.spatial_weight)]
        if self.settings.use_appearance:
            terms.append((appearance, self.settings.appearance_weight))
        if self.settings.use_text:
            terms.append((text, self.settings.text_weight))

        weight_sum = sum(weight for _value, weight in terms if weight > 0)
        if weight_sum <= 0:
            return 0.0
        return float(sum(value * weight for value, weight in terms if weight > 0) / weight_sum)


class GeometryOnlyMatcher(AssociationService):
    def __init__(self, settings: EngineSettings) -> None:
        settings.use_appearance = False
        settings.use_text = False
        super().__init__(settings=settings)


class GeometryAppearanceMatcher(AssociationService):
    def __init__(self, settings: EngineSettings) -> None:
        settings.use_appearance = True
        super().__init__(settings=settings)
