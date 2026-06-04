from __future__ import annotations

import numpy as np

from object3d_engine.domain.entities import MaskDetection
from object3d_engine.domain.value_objects import BoundingBox2D, FeatureVector


class ConceptGraphsGobsAdapter:
    def to_mask_detections(self, gobs: dict) -> list[MaskDetection]:
        classes = list(gobs.get("classes", []))
        image_feats = gobs.get("image_feats")
        text_feats = gobs.get("text_feats")
        detections: list[MaskDetection] = []

        for index, xyxy in enumerate(gobs.get("xyxy", [])):
            class_id = int(gobs["class_id"][index])
            class_name = classes[class_id] if 0 <= class_id < len(classes) else str(class_id)
            image_feature = None
            text_feature = None
            if image_feats is not None and len(image_feats) > index:
                image_feature = FeatureVector(np.asarray(image_feats[index], dtype=np.float64))
            if text_feats is not None and len(text_feats) > index:
                text_feature = FeatureVector(np.asarray(text_feats[index], dtype=np.float64))

            detections.append(
                MaskDetection(
                    class_name=class_name,
                    confidence=float(gobs["confidence"][index]) if gobs.get("confidence") is not None else 1.0,
                    bbox=BoundingBox2D(
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                    ),
                    detection_id=f"gobs:{index}",
                    mask=np.asarray(gobs["mask"][index], dtype=bool),
                    image_feature=image_feature,
                    text_feature=text_feature,
                    extra={"class_id": class_id},
                )
            )
        return detections
