from __future__ import annotations

from object3d_engine.adapters.null_feature_provider import NullFeatureProvider
from object3d_engine.config.settings import EngineSettings
from object3d_engine.core.association_service import GeometryAppearanceMatcher, GeometryOnlyMatcher
from object3d_engine.core.bbox_service import BoundingBoxEstimator
from object3d_engine.core.engine import Object3DEngine
from object3d_engine.core.fusion_service import FusionService
from object3d_engine.core.object_map_service import ObjectMapService
from object3d_engine.core.observation_estimator import FrameObjectEstimator, ObservationEstimator
from object3d_engine.core.pointcloud_service import PointCloudService
from object3d_engine.core.postprocess_service import PostProcessService


class Object3DEngineFactory:
    @staticmethod
    def create_single_frame_engine(
        settings: EngineSettings | None = None,
        mask_provider=None,
        feature_provider=None,
        serializer=None,
    ) -> Object3DEngine:
        settings = settings or EngineSettings()
        object_map_service = Object3DEngineFactory._build_object_map_service(settings)
        object_map_service.settings.postprocess_interval = -1
        return Object3DEngineFactory._build_engine(
            settings=settings,
            object_map_service=object_map_service,
            mask_provider=mask_provider,
            feature_provider=feature_provider,
            serializer=serializer,
        )

    @staticmethod
    def create_tracking_engine(
        settings: EngineSettings | None = None,
        mask_provider=None,
        feature_provider=None,
        serializer=None,
    ) -> Object3DEngine:
        settings = settings or EngineSettings()
        object_map_service = Object3DEngineFactory._build_object_map_service(settings)
        return Object3DEngineFactory._build_engine(
            settings=settings,
            object_map_service=object_map_service,
            mask_provider=mask_provider,
            feature_provider=feature_provider,
            serializer=serializer,
        )

    @staticmethod
    def _build_engine(
        settings: EngineSettings,
        object_map_service: ObjectMapService,
        mask_provider,
        feature_provider,
        serializer,
    ) -> Object3DEngine:
        pointcloud_service = PointCloudService(settings)
        bbox_estimator = BoundingBoxEstimator(settings)
        observation_estimator = ObservationEstimator(
            settings=settings,
            pointcloud_service=pointcloud_service,
            bbox_estimator=bbox_estimator,
        )
        frame_estimator = FrameObjectEstimator(observation_estimator)
        return Object3DEngine(
            frame_estimator=frame_estimator,
            object_map_service=object_map_service,
            mask_provider=mask_provider,
            feature_provider=feature_provider or NullFeatureProvider(),
            serializer=serializer,
        )

    @staticmethod
    def _build_object_map_service(settings: EngineSettings) -> ObjectMapService:
        pointcloud_service = PointCloudService(settings)
        bbox_estimator = BoundingBoxEstimator(settings)
        matcher = (
            GeometryAppearanceMatcher(settings)
            if settings.use_appearance
            else GeometryOnlyMatcher(settings)
        )
        merger = FusionService(
            settings=settings,
            pointcloud_service=pointcloud_service,
            bbox_estimator=bbox_estimator,
        )
        postprocess_service = PostProcessService(settings=settings, merger=merger)
        return ObjectMapService(
            settings=settings,
            matcher=matcher,
            merger=merger,
            postprocess_service=postprocess_service,
        )
