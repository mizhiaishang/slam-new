from __future__ import annotations

from copy import deepcopy

from object3d_engine.domain.entities import (
    Detection2D,
    EngineResult,
    FrameData,
    MaskDetection,
)
from object3d_engine.interfaces.feature_provider import IFeatureProvider
from object3d_engine.interfaces.mask_provider import IMaskProvider
from object3d_engine.interfaces.serializer import ISerializer
from object3d_engine.core.object_map_service import ObjectMapService
from object3d_engine.core.observation_estimator import FrameObjectEstimator


class Object3DEngine:
    def __init__(
        self,
        frame_estimator: FrameObjectEstimator,
        object_map_service: ObjectMapService,
        mask_provider: IMaskProvider | None = None,
        feature_provider: IFeatureProvider | None = None,
        serializer: ISerializer | None = None,
    ) -> None:
        self.frame_estimator = frame_estimator
        self.object_map_service = object_map_service
        self.mask_provider = mask_provider
        self.feature_provider = feature_provider
        self.serializer = serializer

    def process_frame(
        self,
        frame: FrameData,
        detections: list[Detection2D | MaskDetection],
    ) -> EngineResult:
        prepared = self._prepare_detections(frame, detections)
        observations = self.frame_estimator.estimate_frame(frame, prepared)
        assignments, created_object_ids = self.object_map_service.update(observations)

        result = EngineResult(
            frame_id=frame.frame_id,
            observations=observations,
            map_state=self.object_map_service.export_state(),
            assignments=assignments,
            created_object_ids=created_object_ids,
        )
        if self.serializer is not None:
            result.serialized_payload = self.serializer.serialize(result)
        return result

    def export_state(self):
        return self.object_map_service.export_state()

    def reset(self) -> None:
        self.object_map_service.reset()

    def _prepare_detections(
        self,
        frame: FrameData,
        detections: list[Detection2D | MaskDetection],
    ) -> list[MaskDetection]:
        if not detections:
            return []

        if all(isinstance(det, MaskDetection) for det in detections):
            prepared = [deepcopy(det) for det in detections]  # type: ignore[arg-type]
        else:
            if self.mask_provider is None:
                raise ValueError("mask provider is required for Detection2D inputs")
            base_detections = [deepcopy(det) for det in detections]  # type: ignore[arg-type]
            prepared = self.mask_provider.generate_masks(frame, base_detections)

        if self.feature_provider is not None:
            prepared = self.feature_provider.enrich(frame, prepared)
        return prepared
