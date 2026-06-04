from __future__ import annotations

from typing import Protocol

from object3d_engine.domain.entities import FrameData, MaskDetection


class IFeatureProvider(Protocol):
    def enrich(
        self,
        frame: FrameData,
        detections: list[MaskDetection],
    ) -> list[MaskDetection]:
        ...
