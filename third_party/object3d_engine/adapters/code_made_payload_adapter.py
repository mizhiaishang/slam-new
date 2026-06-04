from __future__ import annotations

import numpy as np

from object3d_engine.domain.entities import Detection2D, FrameData, MaskDetection
from object3d_engine.domain.value_objects import BoundingBox2D, CameraIntrinsics, Pose3D


class CodeMadePayloadAdapter:
    def build_frame(
        self,
        image_rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics_matrix: np.ndarray,
        pose_matrix: np.ndarray,
        frame_id: str,
        timestamp: str | None = None,
    ) -> FrameData:
        return FrameData(
            image_rgb=np.asarray(image_rgb, dtype=np.uint8),
            depth=np.asarray(depth, dtype=np.float64),
            intrinsics=CameraIntrinsics.from_matrix(np.asarray(intrinsics_matrix, dtype=np.float64)),
            pose=Pose3D(np.asarray(pose_matrix, dtype=np.float64)),
            frame_id=frame_id,
            timestamp=timestamp,
        )

    def detections_from_payload(self, payload: list[dict]) -> list[Detection2D]:
        detections: list[Detection2D] = []
        for index, item in enumerate(payload):
            bbox_payload = item["bbox"]
            detections.append(
                Detection2D(
                    class_name=str(item["class_name"]),
                    confidence=float(item.get("confidence", 1.0)),
                    bbox=BoundingBox2D(
                        x1=float(bbox_payload["x1"]),
                        y1=float(bbox_payload["y1"]),
                        x2=float(bbox_payload["x2"]),
                        y2=float(bbox_payload["y2"]),
                    ),
                    detection_id=item.get("detection_id") or f"payload:{index}",
                    extra={k: v for k, v in item.items() if k not in {"class_name", "confidence", "bbox"}},
                )
            )
        return detections

    def mask_detections_from_payload(self, payload: list[dict]) -> list[MaskDetection]:
        detections: list[MaskDetection] = []
        for det in self.detections_from_payload(payload):
            mask = det.extra.get("mask")
            if mask is None:
                raise ValueError("mask_detections_from_payload requires a 'mask' field")
            detections.append(
                MaskDetection(
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    detection_id=det.detection_id,
                    mask=np.asarray(mask, dtype=bool),
                    extra=det.extra,
                )
            )
        return detections
