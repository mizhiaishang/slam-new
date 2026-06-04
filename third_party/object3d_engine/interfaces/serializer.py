from __future__ import annotations

from typing import Protocol

from object3d_engine.domain.entities import EngineResult


class ISerializer(Protocol):
    def serialize(self, result: EngineResult) -> dict:
        ...
