from dataclasses import dataclass

from object3d_engine.domain.enums import EngineMode, MatchStrategy, SpatialSimilarityType


@dataclass(slots=True)
class EngineSettings:
    voxel_size: float = 0.025
    dbscan_remove_noise: bool = True
    dbscan_eps: float = 0.05
    dbscan_min_points: int = 10
    min_points_threshold: int = 16
    mask_area_threshold: int = 25
    mask_conf_threshold: float = 0.2
    observation_max_depth_m: float | None = None
    foreground_depth_filter_enabled: bool = True
    foreground_depth_window_m: float = 0.30
    foreground_center_filter_enabled: bool = True
    foreground_center_distance_percentile: float = 90.0
    foreground_min_points_threshold: int = 50
    foreground_rerun_clean: bool = True
    use_oriented_bbox: bool = True

    spatial_similarity_type: SpatialSimilarityType = SpatialSimilarityType.OVERLAP
    match_strategy: MatchStrategy = MatchStrategy.GEOMETRY_ONLY
    engine_mode: EngineMode = EngineMode.TRACKING

    spatial_weight: float = 1.0
    appearance_weight: float = 0.5
    text_weight: float = 0.25
    match_threshold: float = 0.25
    max_assignment_distance: float = 2.0
    use_appearance: bool = False
    use_text: bool = False

    obj_min_points: int = 20
    obj_min_detections: int = 1
    merge_overlap_threshold: float = 0.7
    merge_visual_similarity_threshold: float = 0.7
    merge_text_similarity_threshold: float = 0.7
    enable_postprocess: bool = True
    postprocess_interval: int = -1

    bbox_padding_eps: float = 0.0
    point_overlap_distance: float | None = None

    def effective_overlap_distance(self) -> float:
        return self.point_overlap_distance or self.voxel_size
