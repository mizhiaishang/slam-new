from __future__ import annotations

from typing import Iterable

from object3d_engine.domain.entities import Detection2D, EngineResult, FrameData, MaskDetection
from object3d_engine.core.engine import Object3DEngine


class Object3DPipeline:
    def __init__(self, engine: Object3DEngine) -> None:
        self.engine = engine

    def process_sequence(
        self,
        frames: Iterable[FrameData],
        detections_iterable: Iterable[list[Detection2D | MaskDetection]],
    ) -> list[EngineResult]:
        results: list[EngineResult] = []
        for frame, detections in zip(frames, detections_iterable):
            results.append(self.engine.process_frame(frame, detections))
        return results
