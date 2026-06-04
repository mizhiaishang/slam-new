from __future__ import annotations

from copy import deepcopy

from object3d_engine.config.settings import EngineSettings
from object3d_engine.core.postprocess_service import PostProcessService
from object3d_engine.domain.entities import (
    MapObject3D,
    MatchAssignment,
    ObjectMapState,
    ObjectObservation3D,
)
from object3d_engine.interfaces.matcher import IObjectMatcher
from object3d_engine.interfaces.merger import IObjectMerger


class ObjectMapService:
    def __init__(
        self,
        settings: EngineSettings,
        matcher: IObjectMatcher,
        merger: IObjectMerger,
        postprocess_service: PostProcessService,
        state: ObjectMapState | None = None,
    ) -> None:
        self.settings = settings
        self.matcher = matcher
        self.merger = merger
        self.postprocess_service = postprocess_service
        self.state = state or ObjectMapState()

    def update(
        self,
        observations: list[ObjectObservation3D],
    ) -> tuple[list[MatchAssignment], list[str]]:
        assignments = self.matcher.match(observations, self.state.objects)
        created_object_ids: list[str] = []

        for assignment in assignments:
            observation = observations[assignment.observation_index]
            if assignment.object_index is None:
                object_id = self.state.allocate_object_id()
                self.state.objects.append(MapObject3D.from_observation(object_id, observation))
                created_object_ids.append(object_id)
                continue

            target = self.state.objects[assignment.object_index]
            self.state.objects[assignment.object_index] = self.merger.merge(target, observation)

        self.state.processed_frames += 1
        if self._should_postprocess():
            self.state.objects = self.postprocess_service.postprocess(self.state.objects)

        return assignments, created_object_ids

    def reset(self) -> None:
        self.state = ObjectMapState()

    def export_state(self) -> ObjectMapState:
        return deepcopy(self.state)

    def _should_postprocess(self) -> bool:
        interval = self.settings.postprocess_interval
        if interval <= 0:
            return False
        return self.state.processed_frames % interval == 0
