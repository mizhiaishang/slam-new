from __future__ import annotations

from typing import Protocol

from object3d_engine.domain.entities import MapObject3D, ObjectObservation3D


class IObjectMerger(Protocol):
    def merge(self, target: MapObject3D, observation: ObjectObservation3D) -> MapObject3D:
        ...

    def merge_objects(self, target: MapObject3D, incoming: MapObject3D) -> MapObject3D:
        ...
