from __future__ import annotations

from typing import Protocol

from object3d_engine.domain.entities import Detection2D, FrameData, MaskDetection


class IMaskProvider(Protocol):
    def generate_masks(
        self,
        frame: FrameData,
        detections: list[Detection2D],
    ) -> list[MaskDetection]:
        ...
