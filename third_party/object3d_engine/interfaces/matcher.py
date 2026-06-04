from __future__ import annotations

from typing import Protocol

from object3d_engine.domain.entities import MapObject3D, MatchAssignment, ObjectObservation3D


class IObjectMatcher(Protocol):
    def match(
        self,
        observations: list[ObjectObservation3D],
        objects: list[MapObject3D],
    ) -> list[MatchAssignment]:
        ...
