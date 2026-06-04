from __future__ import annotations

from object3d_engine.domain.entities import EngineResult
from object3d_engine.interfaces.serializer import ISerializer


class InfoJsonSerializer(ISerializer):
    def serialize(self, result: EngineResult) -> dict:
        state = result.map_state
        observations = []
        for obs in result.observations:
            observations.append(
                {
                    "class_name": obs.class_name,
                    "confidence": obs.confidence,
                    "bbox": {
                        "x1": obs.bbox2d.x1,
                        "y1": obs.bbox2d.y1,
                        "x2": obs.bbox2d.x2,
                        "y2": obs.bbox2d.y2,
                    },
                    "global_position": obs.global_position.tolist(),
                    "global_position_method": obs.metadata.method,
                    "support_point_count": obs.support_point_count,
                    "mask_area": obs.mask_area,
                    "bbox_3d_center": obs.bbox3d.center.tolist(),
                    "bbox_3d_extent": obs.bbox3d.extent.tolist(),
                    "bbox_3d_corners": obs.bbox3d.corners.tolist(),
                    "position_confidence": obs.confidence,
                    "is_3d_estimated": obs.is_valid,
                    "metadata": obs.metadata.to_dict(),
                }
            )

        return {
            "frame_id": result.frame_id,
            "object_map_stats": {
                "object_count": len(state.objects),
                "processed_frames": state.processed_frames,
            },
            "detections": observations,
        }
