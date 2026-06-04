from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from object3d_engine.domain.entities import Detection2D, MaskDetection
from object3d_engine.domain.value_objects import BoundingBox2D


class UltralyticsResultAdapter:
    def __init__(self, class_names: Mapping[int, str] | Sequence[str], mask_threshold: float = 0.5) -> None:
        if isinstance(class_names, Mapping):
            self.class_names = {int(key): str(value) for key, value in class_names.items()}
        else:
            self.class_names = {index: str(value) for index, value in enumerate(class_names)}
        self.mask_threshold = float(mask_threshold)

    def build_detections(
        self,
        result,
        *,
        prefer_masks: bool = True,
        min_confidence: float = 0.25,
        min_bbox_area: float = 36.0,
        excluded_class_names: set[str] | None = None,
    ) -> list[Detection2D | MaskDetection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        masks = self._extract_masks(result) if prefer_masks else None
        detections: list[Detection2D | MaskDetection] = []

        for index in range(len(boxes)):
            confidence = float(boxes.conf[index].item())
            if confidence < min_confidence:
                continue

            class_index = int(boxes.cls[index].item())
            class_name = self.class_names.get(class_index, str(class_index))
            if excluded_class_names and class_name in excluded_class_names:
                continue

            x1, y1, x2, y2 = (float(value) for value in boxes.xyxy[index].tolist())
            bbox = BoundingBox2D(x1=x1, y1=y1, x2=x2, y2=y2)
            if bbox.area < min_bbox_area:
                continue

            extra = {
                "source": "ultralytics",
                "source_class_index": class_index,
            }
            if masks is not None and index < len(masks):
                detections.append(
                    MaskDetection(
                        class_name=class_name,
                        confidence=confidence,
                        bbox=bbox,
                        detection_id=f"ultra:{index}",
                        mask=np.asarray(masks[index] > self.mask_threshold, dtype=bool),
                        extra=extra,
                    )
                )
            else:
                detections.append(
                    Detection2D(
                        class_name=class_name,
                        confidence=confidence,
                        bbox=bbox,
                        detection_id=f"ultra:{index}",
                        extra=extra,
                    )
                )
        return detections

    @staticmethod
    def _extract_masks(result) -> np.ndarray | None:
        masks = getattr(result, "masks", None)
        if masks is None or getattr(masks, "data", None) is None:
            return None
        data = masks.data
        if hasattr(data, "cpu"):
            data = data.cpu().numpy()
        return np.asarray(data)
