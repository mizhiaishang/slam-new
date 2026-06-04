from __future__ import annotations

from object3d_engine.domain.entities import FrameData, MaskDetection
from object3d_engine.interfaces.feature_provider import IFeatureProvider


class NullFeatureProvider(IFeatureProvider):
    def enrich(
        self,
        frame: FrameData,
        detections: list[MaskDetection],
    ) -> list[MaskDetection]:
        return detections
