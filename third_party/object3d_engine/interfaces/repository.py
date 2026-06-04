from __future__ import annotations

from typing import Protocol

from object3d_engine.domain.entities import ObjectMapState


class IObjectMapRepository(Protocol):
    def save(self, state: ObjectMapState) -> None:
        ...

    def load(self) -> ObjectMapState:
        ...
