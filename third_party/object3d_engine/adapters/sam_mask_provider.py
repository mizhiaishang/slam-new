from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from object3d_engine.domain.entities import Detection2D, FrameData, MaskDetection
from object3d_engine.interfaces.mask_provider import IMaskProvider


class SamMaskProvider(IMaskProvider):
    def __init__(
        self,
        predictor,
        *,
        min_component_area: int = 12,
    ) -> None:
        self.predictor = predictor
        self.min_component_area = int(min_component_area)

    def generate_masks(
        self,
        frame: FrameData,
        detections: list[Detection2D],
    ) -> list[MaskDetection]:
        image = np.asarray(frame.image_rgb, dtype=np.uint8)
        self.predictor.set_image(image)

        results: list[MaskDetection] = []
        for det in detections:
            box = np.array([det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2], dtype=np.float32)
            masks, scores, _ = self.predictor.predict(box=box, multimask_output=False)
            if len(masks) == 0:
                continue
            mask = np.asarray(masks[0], dtype=bool)
            if int(mask.sum()) < self.min_component_area:
                continue
            results.append(
                MaskDetection(
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    detection_id=det.detection_id,
                    mask=mask,
                    extra=dict(det.extra),
                )
            )
        return results


class BoundingBoxMaskProvider(IMaskProvider):
    def generate_masks(
        self,
        frame: FrameData,
        detections: list[Detection2D],
    ) -> list[MaskDetection]:
        height, width = frame.depth.shape
        results: list[MaskDetection] = []
        for det in detections:
            mask = np.zeros((height, width), dtype=bool)
            x1 = max(0, int(round(det.bbox.x1)))
            y1 = max(0, int(round(det.bbox.y1)))
            x2 = min(width, int(round(det.bbox.x2)))
            y2 = min(height, int(round(det.bbox.y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            mask[y1:y2, x1:x2] = True
            results.append(
                MaskDetection(
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    detection_id=det.detection_id,
                    mask=mask,
                    extra=dict(det.extra),
                )
            )
        return results
