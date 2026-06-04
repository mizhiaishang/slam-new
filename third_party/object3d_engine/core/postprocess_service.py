from __future__ import annotations

import numpy as np

from object3d_engine.config.settings import EngineSettings
from object3d_engine.core.geometry_service import GeometryService
from object3d_engine.domain.entities import MapObject3D
from object3d_engine.interfaces.merger import IObjectMerger


class PostProcessService:
    def __init__(self, settings: EngineSettings, merger: IObjectMerger) -> None:
        self.settings = settings
        self.merger = merger

    def denoise_objects(self, objects: list[MapObject3D]) -> list[MapObject3D]:
        return [obj for obj in objects if len(obj.global_points) >= self.settings.obj_min_points]

    def filter_objects(self, objects: list[MapObject3D]) -> list[MapObject3D]:
        kept: list[MapObject3D] = []
        for obj in objects:
            if len(obj.global_points) < self.settings.obj_min_points:
                continue
            if obj.observations_count < self.settings.obj_min_detections:
                continue
            kept.append(obj)
        return kept

    def merge_overlapping_objects(self, objects: list[MapObject3D]) -> list[MapObject3D]:
        if len(objects) < 2:
            return objects

        kept = [True] * len(objects)
        for i in range(len(objects)):
            if not kept[i]:
                continue
            for j in range(i + 1, len(objects)):
                if not kept[j]:
                    continue
                if self._should_merge(objects[i], objects[j]):
                    objects[i] = self.merger.merge_objects(objects[i], objects[j])
                    kept[j] = False
        return [obj for obj, keep in zip(objects, kept) if keep]

    def postprocess(self, objects: list[MapObject3D]) -> list[MapObject3D]:
        if not self.settings.enable_postprocess:
            return objects
        objects = self.denoise_objects(objects)
        objects = self.filter_objects(objects)
        objects = self.merge_overlapping_objects(objects)
        return objects

    def _should_merge(self, left: MapObject3D, right: MapObject3D) -> bool:
        overlap = GeometryService.aabb_iou(left.bbox3d, right.bbox3d)
        if overlap < self.settings.merge_overlap_threshold:
            return False

        if self.settings.use_appearance and left.image_feature and right.image_feature:
            if left.image_feature.cosine_similarity(right.image_feature) < self.settings.merge_visual_similarity_threshold:
                return False

        if self.settings.use_text and left.text_feature and right.text_feature:
            if left.text_feature.cosine_similarity(right.text_feature) < self.settings.merge_text_similarity_threshold:
                return False

        return True
