import argparse
import copy
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import quaternion
import torch
from cv2 import imread
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics import YOLO

from current_node import CurrentNode
from topomap import Topomap
from utils_local import (
    calculate_focal_length_from_hfov_vfov,
)


def append_import_roots(root_path: str | os.PathLike | None, *, package_name: str | None = None) -> None:
    if not root_path:
        return

    root = Path(root_path).resolve()
    candidates = [root]
    src_dir = root / "src"
    if src_dir.exists():
        candidates.append(src_dir)

    if package_name:
        package_dir = root / package_name
        if package_dir.exists():
            candidates.append(package_dir.parent)
        src_package_dir = src_dir / package_name
        if src_package_dir.exists():
            candidates.append(src_package_dir.parent)

    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.append(candidate_str)


def build_default_config():
    return {
        "mode": "build",
        "database_track_dir": None,
        "data_base_path": None,
        "model_config_path": None,
        "weights_path": None,
        "depth_anything_root": None,
        "opr_root": None,
        "yolo_weights": "/home/docker_opr_ros2/yolov8n.pt",
        "yolo_device": None,
        "depth_model_path": None,
        "output_dir": "./semantic_graphs",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "max_frames": -1,
        "feature_distance_threshold": 1.8,
        "coord_distance_threshold": 2.5,
        "new_node_distance_threshold": 2.7,
        "hfov": 91.0,
        "vfov": 65.0,
        "save_step_artifacts": False,
        "prefer_recorded_depth": True,
        "recorded_depth_dir_name": "depth",
        "object3d_engine_root": "/home/zyf/Desktop/3d",
        "sam_checkpoint": str(Path(__file__).resolve().parent / "core_content" / "sam_vit_b_01ec64.pth"),
        "sam_model_type": "vit_b",
        "sam_device": None,
        "nav_graph_root": str(Path(__file__).resolve().parent / "nav_tools"),
        "nav_config_path": None,
        "nav_graph_output": None,
        "nav_contents_output": None,
        "nav_stats_output": None,
        "nav_visualization_svg": None,
        "nav_visualization_html": None,
        "nav_visualization_3d_html": None,
        "object3d_global_map_html": None,
        "object3d_tracking_summary_json": None,
        "object3d_min_consecutive_frames": 2,
        "object3d_overlap_filter_enabled": True,
        "object3d_overlap_iou_threshold": 0.05,
        "object3d_overlap_min_ratio_threshold": 0.35,
        "object3d_motion_filter_enabled": True,
        "object3d_motion_filter_classes": "person",
        "object3d_motion_unknown_filter_enabled": True,
        "object3d_motion_min_consecutive_observations": 2,
        "object3d_motion_static_max_center_span_m": 1.0,
        "object3d_motion_static_max_median_step_m": 0.35,
        "object3d_motion_static_max_single_step_m": 1.2,
        "object3d_disappearance_filter_enabled": True,
        "object3d_disappearance_max_observation_distance_m": 3.0,
        "object3d_disappearance_position_tolerance_m": 1.0,
        "object3d_disappearance_match_distance_m": 1.0,
        "object3d_disappearance_min_visible_misses": 2,
        "object3d_disappearance_fov_margin_deg": 8.0,
        "use_filtered_object3d_for_nav_graph": True,
        "waypoint_sampling_enabled": True,
        "waypoint_min_distance_m": 0.8,
        "waypoint_min_yaw_deg": 25.0,
        "waypoint_keep_first_last": True,
        "waypoint_keep_topology_change": True,
        "current_position": None,
        "object_query": "refrigerator",
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Semantic + topological mapping pipeline")
    parser.add_argument("--config", default=None, help="Path to yaml config file")
    parser.add_argument(
        "--mode",
        choices=["build", "navigate", "visualize"],
        default='build',
        help="Run build, navigation, or graph visualization mode",
    )
    parser.add_argument("--database-track-dir", default=None, help="Dataset root used for descriptor DB")
    parser.add_argument("--data-base-path", default=None, help="Dataset root for frame iteration")
    parser.add_argument("--model-config-path", default=None, help="OPR model yaml config")
    parser.add_argument("--weights-path", default=None, help="OPR model weights path")
    parser.add_argument("--depth-anything-root", default=None, help="Depth-Anything-V2 repo root")
    parser.add_argument("--opr-root", default=None, help="OpenPlaceRecognition repo root")
    parser.add_argument("--yolo-weights", default=None, help="YOLO model file")
    parser.add_argument("--yolo-device", default=None, help="YOLO inference device, e.g. cpu, cuda, cuda:0")
    parser.add_argument("--depth-model-path", default=None, help="DepthAnything metric model file")
    parser.add_argument("--output-dir", default=None, help="Output folder")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames for debug; -1 uses all")
    parser.add_argument("--feature-distance-threshold", type=float, default=None)
    parser.add_argument("--coord-distance-threshold", type=float, default=None)
    parser.add_argument("--new-node-distance-threshold", type=float, default=None)
    parser.add_argument("--hfov", type=float, default=None)
    parser.add_argument("--vfov", type=float, default=None)
    parser.add_argument(
        "--save-step-artifacts",
        action="store_true",
        help="Save step*/info.json and debug images for each frame",
    )
    parser.add_argument(
        "--prefer-recorded-depth",
        dest="prefer_recorded_depth",
        action="store_true",
        default=None,
        help="Prefer dataset depth/<front_cam_ts>.png over DepthAnything+LiDAR when available",
    )
    parser.add_argument(
        "--disable-recorded-depth",
        dest="prefer_recorded_depth",
        action="store_false",
        help="Always use DepthAnything+LiDAR depth reconstruction",
    )
    parser.add_argument(
        "--recorded-depth-dir-name",
        default=None,
        help="Depth directory name under data_base_path; default: depth",
    )
    parser.add_argument(
        "--object3d-engine-root",
        default=None,
        help="Root folder containing the standalone object3d_engine package",
    )
    parser.add_argument("--sam-checkpoint", default=None, help="SAM checkpoint used for YOLO-box-prompted masks")
    parser.add_argument("--sam-model-type", choices=["vit_b", "vit_l", "vit_h"], default=None, help="SAM model type matching the checkpoint")
    parser.add_argument("--sam-device", default=None, help="SAM inference device, e.g. cpu, cuda, cuda:0")
    parser.add_argument(
        "--nav-graph-root",
        default=None,
        help="Folder containing build_nav_graph_nx.py and nav_graph_export_utils.py",
    )
    parser.add_argument(
        "--nav-config-path",
        default=None,
        help="Path to nav_graph_config.json; defaults to <nav_graph_root>/nav_graph_config.json",
    )
    parser.add_argument("--nav-graph-output", default=None, help="Output .pkl graph path")
    parser.add_argument("--nav-contents-output", default=None, help="Output graph contents .json path")
    parser.add_argument("--nav-stats-output", default=None, help="Output graph stats .json path")
    parser.add_argument("--nav-visualization-svg", default=None, help="Output 2D graph svg path")
    parser.add_argument("--nav-visualization-html", default=None, help="Output 2D graph html path")
    parser.add_argument("--nav-visualization-3d-html", default=None, help="Output 3D graph html path")
    parser.add_argument("--object3d-global-map-html", default=None, help="Output object3d_engine object-level 3D map html path")
    parser.add_argument("--object3d-tracking-summary-json", default=None, help="Output object3d_engine tracking summary json path")
    parser.add_argument("--object3d-min-consecutive-frames", type=int, default=None, help="Minimum consecutive frames required to keep a tracked object")
    parser.add_argument("--disable-object3d-overlap-filter", dest="object3d_overlap_filter_enabled", action="store_false", default=None, help="Disable same-class 3D overlap duplicate removal")
    parser.add_argument("--object3d-overlap-iou-threshold", type=float, default=None, help="3D IoU threshold for same-class duplicate removal")
    parser.add_argument("--object3d-overlap-min-ratio-threshold", type=float, default=None, help="Intersection/min-volume threshold for same-class duplicate removal")
    parser.add_argument("--disable-object3d-motion-filter", dest="object3d_motion_filter_enabled", action="store_false", default=None, help="Disable object-level motion-state filtering")
    parser.add_argument("--object3d-motion-filter-classes", default=None, help="Comma-separated classes that should be checked by the 3m-range motion filter")
    parser.add_argument("--disable-object3d-motion-unknown-filter", dest="object3d_motion_unknown_filter_enabled", action="store_false", default=None, help="Keep motion-unknown movable objects instead of filtering them from the final map")
    parser.add_argument("--object3d-motion-min-consecutive-observations", type=int, default=None, help="Minimum consecutive 3m-range observations required before a movable object can be treated as static")
    parser.add_argument("--object3d-motion-static-max-center-span-m", type=float, default=None, help="Max same-track 3D center span in one consecutive run before marking object as moving")
    parser.add_argument("--object3d-motion-static-max-median-step-m", type=float, default=None, help="Max median consecutive-frame 3D center step before marking object as moving")
    parser.add_argument("--object3d-motion-static-max-single-step-m", type=float, default=None, help="Max single consecutive-frame 3D center step before marking object as moving")
    parser.add_argument("--disable-object3d-disappearance-filter", dest="object3d_disappearance_filter_enabled", action="store_false", default=None, help="Disable later-revisit disappearance removal for stable object3d tracks")
    parser.add_argument("--object3d-disappearance-max-observation-distance-m", type=float, default=None, help="Max camera-to-object distance/depth for counting a later frame as a disappearance visibility check")
    parser.add_argument("--object3d-disappearance-position-tolerance-m", type=float, default=None, help="Max distance from a previous observation pose for counting a later frame as a same-place revisit")
    parser.add_argument("--object3d-disappearance-match-distance-m", type=float, default=None, help="Max 3D center distance for treating a later same-class detection as the same physical object")
    parser.add_argument("--object3d-disappearance-min-visible-misses", type=int, default=None, help="Minimum valid 3m-range revisit frames with no matching detection before removing the object from the final map")
    parser.add_argument("--object3d-disappearance-fov-margin-deg", type=float, default=None, help="Safety margin subtracted from the camera FOV when evaluating whether a disappeared object should still have been visible")
    parser.add_argument(
        "--disable-filtered-object3d-nav-graph",
        dest="use_filtered_object3d_for_nav_graph",
        action="store_false",
        default=None,
        help="Keep per-frame object detections in nav graph instead of stable postprocessed object3d tracks",
    )
    parser.add_argument("--enable-waypoint-sampling", dest="waypoint_sampling_enabled", action="store_true", default=None, help="Keep only key waypoint frames in the nav graph")
    parser.add_argument("--disable-waypoint-sampling", dest="waypoint_sampling_enabled", action="store_false", help="Disable waypoint sampling")
    parser.add_argument("--waypoint-min-distance-m", type=float, default=None, help="Minimum travel distance before keeping another waypoint")
    parser.add_argument("--waypoint-min-yaw-deg", type=float, default=None, help="Minimum yaw change before keeping another waypoint")
    parser.add_argument("--disable-waypoint-keep-first-last", dest="waypoint_keep_first_last", action="store_false", default=None, help="Do not force keeping first and last frames as waypoints")
    parser.add_argument("--disable-waypoint-keep-topology-change", dest="waypoint_keep_topology_change", action="store_false", default=None, help="Do not force keeping frames when topology label changes")
    parser.add_argument("--current-position", default=None, help="Current position x,y,z for navigation mode")
    parser.add_argument("--object-query", default=None, help="Target object query for navigation mode")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config if args.config else os.path.join(script_dir, "config.yaml")

    default_cfg = OmegaConf.create(build_default_config())
    file_cfg = OmegaConf.load(config_path) if os.path.exists(config_path) else OmegaConf.create({})

    cli_overrides = {k: v for k, v in vars(args).items() if k not in {"config"} and v is not None}
    cli_cfg = OmegaConf.create(cli_overrides)
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_cfg)

    if cfg.mode == "build":
        required_fields = [
            "database_track_dir",
            "data_base_path",
            "model_config_path",
            "weights_path",
            "depth_anything_root",
            "opr_root",
            "depth_model_path",
        ]
        missing = [field for field in required_fields if not cfg.get(field)]
        if missing:
            parser.error(
                f"Missing required config fields: {', '.join(missing)}. "
                f"Set them in {config_path} or via CLI args."
            )
    elif cfg.mode == "navigate":
        if not cfg.get("current_position"):
            parser.error("Navigation mode requires --current-position, for example: --current-position 1.0,2.0,0.0")

    return cfg


def position_get(track_csv, i):
    x = track_csv["tx"].iloc[i]
    y = track_csv["ty"].iloc[i]
    z = track_csv["tz"].iloc[i]
    qw = track_csv["qw"].iloc[i]
    qx = track_csv["qx"].iloc[i]
    qy = track_csv["qy"].iloc[i]
    qz = track_csv["qz"].iloc[i]
    position = np.array([x, y, z], dtype=np.float32)
    rotation = quaternion.quaternion(qw, qx, qy, qz)
    return position, rotation


def load_depth_model(depth_model_path, device, depth_anything_root):
    append_import_roots(depth_anything_root)

    from metric_depth.depth_anything_v2.dpt import DepthAnythingV2 as DepthAnythingV2Metric

    model_configs = {
        "small": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "base": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "large": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    params = model_configs["small"]
    model = DepthAnythingV2Metric(**params, max_depth=20.0)
    model.load_state_dict(torch.load(depth_model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def set_tf_matrix():
    fx = 683.6
    fy = fx
    cx = 615.1
    cy = 345.3
    camera_matrix = {"f": fx, "cx": cx, "cy": cy}
    proj_matrix = np.array(
        [
            [fx, 0.0, cx, 0],
            [0.0, fy, cy, 0],
            [0.0, 0.0, 1.0, 0],
        ]
    )
    rotation = [-0.498, 0.498, -0.495, 0.510]
    r_mat = Rotation.from_quat(rotation).as_matrix()
    translation = np.array([[0.061], [0.049], [-0.131]])
    tf_matrix = np.concatenate([r_mat, translation], axis=1)
    tf_matrix = np.concatenate([tf_matrix, np.array([[0, 0, 0, 1]])], axis=0)
    return tf_matrix, camera_matrix, proj_matrix


class FrameSemanticPayloadBuilder:
    def __init__(
        self,
        output_root,
        hfov,
        vfov,
        save_step_artifacts=False,
        object3d_engine_root=None,
        sam_checkpoint=None,
        sam_model_type="vit_b",
        sam_device=None,
    ):
        self.output_root = output_root
        self.hfov = hfov
        self.vfov = vfov
        self.save_step_artifacts = save_step_artifacts
        self.object3d_engine_root = object3d_engine_root
        self.sam_checkpoint = sam_checkpoint
        self.sam_model_type = str(sam_model_type or "vit_b")
        self.sam_device = str(sam_device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.object3d_engine = None
        self.object3d_adapter = None
        self.object3d_mask_provider = None
        self.ultralytics_adapter_cls = None
        self.object3d_track_history = {}
        self._setup_object3d_engine()

    def build(
        self,
        result,
        rgb,
        depth_map,
        frame_idx,
        position,
        rotation,
        test_img_file,
        node_label,
        depth_source=None,
        frame_timestamp=None,
    ):
        detections_raw = []
        img_width, img_height = rgb.shape[1], rgb.shape[0]
        focal_length_x, focal_length_y = calculate_focal_length_from_hfov_vfov(
            self.hfov,
            self.vfov,
            img_width,
            img_height,
        )

        plot_img = result.plot()

        prepared_detections, observation_by_index, object_ids_by_index = self._estimate_object3d_observations(
            result=result,
            rgb=rgb,
            depth_map=depth_map,
            frame_idx=frame_idx,
            position=position,
            rotation=rotation,
            img_width=img_width,
            img_height=img_height,
            focal_length_x=focal_length_x,
            focal_length_y=focal_length_y,
            frame_timestamp=frame_timestamp,
        )

        for det_index, detection in enumerate(prepared_detections):
            bbox = detection.bbox
            observation = observation_by_index.get(det_index)

            if observation is None:
                continue

            detections_raw.append(
                {
                    "class_name": detection.class_name,
                    "bbox": {
                        "x1": float(bbox.x1),
                        "y1": float(bbox.y1),
                        "x2": float(bbox.x2),
                        "y2": float(bbox.y2),
                    },
                    "confidence": float(detection.confidence),
                    "global_position": observation.global_position.tolist(),
                    "depth": self._resolve_detection_depth(observation),
                    "test_img_file": test_img_file,
                    "global_position_method": self._observation_method(observation),
                    "support_point_count": int(observation.support_point_count),
                    "mask_area": int(detection.mask_area),
                    "bbox_3d_center": observation.bbox3d.center.tolist(),
                    "bbox_3d_extent": observation.bbox3d.extent.tolist(),
                    "bbox_3d_corners": observation.bbox3d.corners.tolist(),
                    "global_points_sample": self._sample_global_points(observation.global_points),
                    "object3d_track_id": object_ids_by_index.get(det_index),
                    "estimation_metadata": observation.metadata.to_dict(),
                }
            )

        detections = self._filter_detections_near_border(
            detections_raw,
            img_width,
            img_height,
            margin_ratio=0.05,
        )
        payload = self._build_payload(
            detections,
            position,
            rotation,
            test_img_file,
            node_label,
            depth_source,
            frame_timestamp=frame_timestamp,
        )

        if self.save_step_artifacts:
            self._save_step_artifacts(frame_idx, rgb, plot_img, payload, detections)

        return payload

    def _build_payload(self, detections, position, rotation, test_img_file, node_label, depth_source=None, frame_timestamp=None):
        q = rotation
        q_list = [float(q.w), float(q.x), float(q.y), float(q.z)]
        return {
            "timestamp": self._normalize_timestamp_value(frame_timestamp) or datetime.now().strftime("%Y%m%d%H%M%S"),
            "position": position.tolist(),
            "rotation": q_list,
            "depth_source": depth_source,
            "detections": detections,
            "class_node": node_label,
        }

    def _save_step_artifacts(self, frame_idx, rgb, plot_img, payload, detections):
        drawn = self._draw_filtered_detections(rgb, detections)
        step_dir = os.path.join(self.output_root, f"step{frame_idx}")
        os.makedirs(step_dir, exist_ok=True)
        cv2.imwrite(os.path.join(step_dir, "result.jpg"), cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(step_dir, "result_orin.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(step_dir, "result_orin_plot.jpg"), cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR))
        with open(os.path.join(step_dir, "info.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _setup_object3d_engine(self):
        if not self.object3d_engine_root:
            raise ValueError("object3d_engine_root is required for build mode.")
        if not self.sam_checkpoint:
            raise ValueError("sam_checkpoint is required for the YOLO -> SAM -> 3D chain.")

        engine_root = str(Path(self.object3d_engine_root).resolve())
        if engine_root not in sys.path:
            sys.path.append(engine_root)

        try:
            from segment_anything import SamPredictor, sam_model_registry

            from object3d_engine.adapters.code_made_payload_adapter import CodeMadePayloadAdapter
            from object3d_engine.adapters.sam_mask_provider import SamMaskProvider
            from object3d_engine.adapters.ultralytics_adapter import UltralyticsResultAdapter
            from object3d_engine.config.settings import EngineSettings
            from object3d_engine.domain.enums import SpatialSimilarityType
            from object3d_engine.runtime.factory import Object3DEngineFactory

            checkpoint = Path(str(self.sam_checkpoint)).expanduser().resolve()
            if not checkpoint.exists():
                raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
            if self.sam_model_type not in sam_model_registry:
                raise ValueError(f"unsupported SAM model type: {self.sam_model_type}")

            sam = sam_model_registry[self.sam_model_type](checkpoint=str(checkpoint))
            sam.to(device=self.sam_device)
            sam.eval()

            settings = EngineSettings(
                voxel_size=0.02,
                dbscan_remove_noise=True,
                dbscan_eps=0.10,
                dbscan_min_points=20,
                min_points_threshold=50,
                mask_area_threshold=18,
                observation_max_depth_m=3.0,
                foreground_depth_filter_enabled=True,
                foreground_depth_window_m=0.30,
                foreground_center_filter_enabled=True,
                foreground_center_distance_percentile=90.0,
                foreground_min_points_threshold=50,
                foreground_rerun_clean=True,
                use_oriented_bbox=False,
            )
            settings.spatial_similarity_type = SpatialSimilarityType.CENTER_DISTANCE
            settings.max_assignment_distance = 1.3
            settings.match_threshold = 0.35
            settings.postprocess_interval = -1
            self.object3d_engine = Object3DEngineFactory.create_tracking_engine(
                settings=settings,
            )
            self.object3d_adapter = CodeMadePayloadAdapter()
            self.object3d_mask_provider = SamMaskProvider(SamPredictor(sam))
            self.ultralytics_adapter_cls = UltralyticsResultAdapter
            print(
                "[object3d_engine] YOLO -> SAM -> depth point cloud enabled "
                f"from {engine_root}; sam={checkpoint} ({self.sam_model_type}) on {self.sam_device}"
            )
        except Exception as exc:
            raise RuntimeError(f"object3d_engine init failed: {exc}") from exc

    def _estimate_object3d_observations(
        self,
        result,
        rgb,
        depth_map,
        frame_idx,
        position,
        rotation,
        img_width,
        img_height,
        focal_length_x,
        focal_length_y,
        frame_timestamp=None,
    ):
        if (
            self.object3d_engine is None
            or self.object3d_adapter is None
            or self.object3d_mask_provider is None
            or self.ultralytics_adapter_cls is None
        ):
            return [], {}, {}

        intrinsics_matrix = np.array(
            [
                [focal_length_x, 0.0, img_width / 2.0],
                [0.0, focal_length_y, img_height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        pose_matrix = self._build_object3d_pose_matrix(position, rotation)

        try:
            frame = self.object3d_adapter.build_frame(
                image_rgb=rgb,
                depth=depth_map,
                intrinsics_matrix=intrinsics_matrix,
                pose_matrix=pose_matrix,
                frame_id=f"frame:{frame_idx}",
                timestamp=self._normalize_timestamp_value(frame_timestamp) or datetime.now().strftime("%Y%m%d%H%M%S"),
            )
            adapter = self.ultralytics_adapter_cls(result.names)
            raw_detections = adapter.build_detections(
                result,
                prefer_masks=False,
                min_confidence=0.25,
                min_bbox_area=48.0,
            )
            if not raw_detections:
                return [], {}, {}

            prepared_detections = self.object3d_mask_provider.generate_masks(frame, raw_detections)
            if not prepared_detections:
                return [], {}, {}

            engine_result = self.object3d_engine.process_frame(frame, prepared_detections)
        except Exception as exc:
            raise RuntimeError(f"object3d_engine frame {frame_idx} estimation failed: {exc}") from exc

        observations = {}
        for observation in engine_result.observations:
            try:
                det_index = int(str(observation.observation_id).rsplit(":", 1)[-1])
            except (TypeError, ValueError):
                continue
            observations[det_index] = observation
        object_ids_by_index = self._resolve_assignment_object_ids(engine_result)
        self._record_object3d_track_history(frame_idx, object_ids_by_index, observations, frame_timestamp=frame_timestamp)
        return prepared_detections, observations, object_ids_by_index

    @staticmethod
    def _resolve_assignment_object_ids(engine_result):
        resolved = {}
        created_index = 0
        for obs_position, observation in enumerate(engine_result.observations):
            try:
                detection_index = int(str(observation.observation_id).rsplit(":", 1)[-1])
            except (TypeError, ValueError):
                continue
            assignment = engine_result.assignments[obs_position]
            if assignment.object_index is None:
                object_id = (
                    engine_result.created_object_ids[created_index]
                    if created_index < len(engine_result.created_object_ids)
                    else None
                )
                created_index += 1
            else:
                object_id = engine_result.map_state.objects[assignment.object_index].object_id
            if object_id:
                resolved[detection_index] = object_id
        return resolved

    def _record_object3d_track_history(self, frame_idx, object_ids_by_index, observations, frame_timestamp=None):
        timestamp_text = self._normalize_timestamp_value(frame_timestamp)
        timestamp_s = self._timestamp_to_seconds(frame_timestamp)
        for det_index, object_id in object_ids_by_index.items():
            if not object_id:
                continue
            observation = observations.get(det_index)
            if observation is None:
                continue

            history = self.object3d_track_history.setdefault(str(object_id), [])
            history.append(
                {
                    "frame_index": int(frame_idx),
                    "frame_id": str(observation.frame_id),
                    "timestamp": timestamp_text,
                    "timestamp_s": timestamp_s,
                    "class_name": str(observation.class_name),
                    "confidence": float(observation.confidence),
                    "bbox_3d_center": np.asarray(observation.bbox3d.center, dtype=np.float64).tolist(),
                    "bbox_3d_extent": np.asarray(observation.bbox3d.extent, dtype=np.float64).tolist(),
                    "centroid": np.asarray(observation.centroid, dtype=np.float64).tolist(),
                    "support_point_count": int(observation.support_point_count),
                    "mask_area": int(observation.mask_area),
                }
            )

    def _build_object3d_pose_matrix(self, position, rotation):
        pose = np.eye(4, dtype=np.float64)
        world_rotation = np.asarray(quaternion.as_rotation_matrix(rotation), dtype=np.float64)
        optical_to_world_axes = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )
        pose[:3, :3] = world_rotation @ optical_to_world_axes
        pose[:3, 3] = np.asarray(position, dtype=np.float64)
        return pose

    @staticmethod
    def _resolve_detection_depth(observation):
        local_points = np.asarray(observation.local_points, dtype=np.float64).reshape(-1, 3)
        if len(local_points) == 0:
            return None
        return float(np.median(local_points[:, 2]))

    @staticmethod
    def _observation_method(observation):
        return "yolo_sam_depth_pointcloud"

    @staticmethod
    def _sample_global_points(points, max_points=350):
        global_points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(global_points) == 0:
            return []
        if len(global_points) > max_points:
            indexes = np.linspace(0, len(global_points) - 1, max_points, dtype=np.int32)
            global_points = global_points[indexes]
        return global_points.tolist()

    @staticmethod
    def _normalize_timestamp_value(value):
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None
        if re.fullmatch(r"\d+\.0+", text):
            text = text.split(".", 1)[0]
        return text

    @classmethod
    def _timestamp_to_seconds(cls, value):
        text = cls._normalize_timestamp_value(value)
        if text is None:
            return None
        try:
            numeric = float(text)
        except ValueError:
            return None
        if not np.isfinite(numeric):
            return None
        if abs(numeric) > 1_000_000.0:
            return float(numeric / 1_000_000_000.0)
        return float(numeric)

    def export_tracking_summary(
        self,
        camera_positions,
        min_consecutive_frames=2,
        overlap_filter_enabled=True,
        overlap_iou_threshold=0.05,
        overlap_min_ratio_threshold=0.35,
        motion_filter_enabled=True,
        motion_filter_classes=None,
        motion_unknown_filter_enabled=True,
        motion_min_consecutive_observations=2,
        motion_static_max_center_span_m=1.0,
        motion_static_max_median_step_m=0.35,
        motion_static_max_single_step_m=1.2,
    ):
        if self.object3d_engine is None:
            return {
                "object_source": "object3d_engine_tracking",
                "processed_frames": 0,
                "camera_positions": camera_positions,
                "raw_object_count": 0,
                "object_count": 0,
                "postprocess": {
                    "min_consecutive_frames": int(min_consecutive_frames),
                    "after_consecutive_object_count": 0,
                    "overlap_filter_enabled": bool(overlap_filter_enabled),
                    "overlap_iou_threshold": float(overlap_iou_threshold),
                    "overlap_min_ratio_threshold": float(overlap_min_ratio_threshold),
                    "overlap_removed_object_count": 0,
                    "motion_filter_enabled": bool(motion_filter_enabled),
                    "motion_filter_classes": self._parse_motion_filter_classes(motion_filter_classes),
                    "motion_unknown_filter_enabled": bool(motion_unknown_filter_enabled),
                    "motion_min_consecutive_observations": int(motion_min_consecutive_observations),
                    "motion_static_max_center_span_m": float(motion_static_max_center_span_m),
                    "motion_static_max_median_step_m": float(motion_static_max_median_step_m),
                    "motion_static_max_single_step_m": float(motion_static_max_single_step_m),
                    "motion_removed_object_count": 0,
                    "removed_object_count": 0,
                },
                "objects": [],
                "removed_objects": [],
            }

        state = self.object3d_engine.export_state()
        raw_objects = sorted(
            state.objects,
            key=lambda obj: (-obj.observations_count, -obj.total_support_points, obj.object_id),
        )
        kept_objects, removed_objects = self._filter_tracking_objects_by_consecutive_frames(
            raw_objects,
            min_consecutive_frames=max(1, int(min_consecutive_frames)),
        )
        after_consecutive_count = len(kept_objects)
        overlap_removed_objects = []
        if overlap_filter_enabled:
            kept_objects, overlap_removed_objects = self._filter_tracking_objects_by_class_volume_overlap(
                kept_objects,
                iou_threshold=float(overlap_iou_threshold),
                min_ratio_threshold=float(overlap_min_ratio_threshold),
            )
            removed_objects.extend(overlap_removed_objects)
        motion_metrics_by_id = self._analyze_tracking_object_motion(
            kept_objects,
            motion_filter_classes=self._parse_motion_filter_classes(motion_filter_classes),
            motion_min_consecutive_observations=max(1, int(motion_min_consecutive_observations)),
            max_center_span_m=float(motion_static_max_center_span_m),
            max_median_step_m=float(motion_static_max_median_step_m),
            max_single_step_m=float(motion_static_max_single_step_m),
        )
        motion_removed_objects = []
        if motion_filter_enabled:
            kept_objects, motion_removed_objects = self._filter_tracking_objects_by_motion_state(
                kept_objects,
                motion_metrics_by_id,
                remove_unknown=bool(motion_unknown_filter_enabled),
            )
            removed_objects.extend(motion_removed_objects)
        return {
            "object_source": "object3d_engine_tracking",
            "processed_frames": int(state.processed_frames),
            "camera_positions": camera_positions,
            "raw_object_count": len(raw_objects),
            "object_count": len(kept_objects),
            "postprocess": {
                "min_consecutive_frames": max(1, int(min_consecutive_frames)),
                "after_consecutive_object_count": after_consecutive_count,
                "overlap_filter_enabled": bool(overlap_filter_enabled),
                "overlap_iou_threshold": float(overlap_iou_threshold),
                "overlap_min_ratio_threshold": float(overlap_min_ratio_threshold),
                "overlap_removed_object_count": len(overlap_removed_objects),
                "motion_filter_enabled": bool(motion_filter_enabled),
                "motion_filter_classes": self._parse_motion_filter_classes(motion_filter_classes),
                "motion_unknown_filter_enabled": bool(motion_unknown_filter_enabled),
                "motion_min_consecutive_observations": max(1, int(motion_min_consecutive_observations)),
                "motion_static_max_center_span_m": float(motion_static_max_center_span_m),
                "motion_static_max_median_step_m": float(motion_static_max_median_step_m),
                "motion_static_max_single_step_m": float(motion_static_max_single_step_m),
                "motion_removed_object_count": len(motion_removed_objects),
                "removed_object_count": len(removed_objects),
            },
            "objects": [
                self._serialize_tracking_object(obj, motion_metrics_by_id.get(str(obj.object_id)))
                for obj in kept_objects
            ],
            "removed_objects": removed_objects,
        }

    @classmethod
    def _filter_tracking_objects_by_consecutive_frames(cls, objects, min_consecutive_frames):
        if min_consecutive_frames <= 1:
            kept = sorted(
                objects,
                key=lambda obj: (-obj.observations_count, -obj.total_support_points, obj.object_id),
            )
            return kept, []

        kept = []
        removed = []
        for obj in objects:
            longest_run = cls._longest_consecutive_frame_run(list(obj.frame_ids))
            removal_payload = {
                "object_id": obj.object_id,
                "dominant_class_name": obj.dominant_class_name,
                "observations_count": int(obj.observations_count),
                "frame_ids": list(obj.frame_ids),
                "longest_consecutive_frame_run": int(longest_run),
            }
            if longest_run >= min_consecutive_frames:
                kept.append(obj)
            else:
                removed.append(removal_payload)

        kept = sorted(
            kept,
            key=lambda obj: (-obj.observations_count, -obj.total_support_points, obj.object_id),
        )
        removed = sorted(
            removed,
            key=lambda item: (
                -int(item["longest_consecutive_frame_run"]),
                -int(item["observations_count"]),
                str(item["object_id"]),
            ),
        )
        return kept, removed

    @classmethod
    def _filter_tracking_objects_by_class_volume_overlap(
        cls,
        objects,
        iou_threshold=0.05,
        min_ratio_threshold=0.35,
    ):
        ranked_objects = sorted(objects, key=cls._tracking_object_rank, reverse=True)
        kept = []
        removed = []

        for obj in ranked_objects:
            conflict_payload = None
            for kept_obj in kept:
                if cls._normalize_object_class(obj) != cls._normalize_object_class(kept_obj):
                    continue
                overlap = cls._tracking_object_volume_overlap(obj, kept_obj)
                if overlap is None:
                    continue
                iou, min_ratio, _max_ratio = overlap
                if iou >= iou_threshold or min_ratio >= min_ratio_threshold:
                    conflict_payload = {
                        "object_id": obj.object_id,
                        "dominant_class_name": obj.dominant_class_name,
                        "observations_count": int(obj.observations_count),
                        "frame_ids": list(obj.frame_ids),
                        "longest_consecutive_frame_run": int(cls._longest_consecutive_frame_run(list(obj.frame_ids))),
                        "remove_reason": "same_class_3d_volume_overlap",
                        "kept_object_id": kept_obj.object_id,
                        "kept_observations_count": int(kept_obj.observations_count),
                        "iou_3d": float(iou),
                        "intersection_over_min_volume": float(min_ratio),
                    }
                    break
            if conflict_payload is None:
                kept.append(obj)
            else:
                removed.append(conflict_payload)

        kept = sorted(
            kept,
            key=lambda obj: (-obj.observations_count, -obj.total_support_points, obj.object_id),
        )
        removed = sorted(
            removed,
            key=lambda item: (
                str(item.get("dominant_class_name", "")),
                str(item.get("kept_object_id", "")),
                -float(item.get("iou_3d", 0.0)),
                str(item.get("object_id", "")),
            ),
        )
        return kept, removed

    def _analyze_tracking_object_motion(
        self,
        objects,
        motion_filter_classes=None,
        motion_min_consecutive_observations=2,
        max_center_span_m=1.0,
        max_median_step_m=0.35,
        max_single_step_m=1.2,
    ):
        movable_classes = self._parse_motion_filter_classes(motion_filter_classes)
        return {
            str(obj.object_id): self._build_object_motion_metrics(
                obj,
                self.object3d_track_history.get(str(obj.object_id), []),
                movable_classes=movable_classes,
                min_consecutive_observations=max(1, int(motion_min_consecutive_observations)),
                max_center_span_m=max_center_span_m,
                max_median_step_m=max_median_step_m,
                max_single_step_m=max_single_step_m,
            )
            for obj in objects
        }

    @classmethod
    def _filter_tracking_objects_by_motion_state(cls, objects, motion_metrics_by_id, remove_unknown=True):
        kept = []
        removed = []
        for obj in objects:
            metrics = motion_metrics_by_id.get(str(obj.object_id), {})
            motion_state = str(metrics.get("motion_state") or "")
            should_remove = motion_state == "moving" or (bool(remove_unknown) and motion_state == "unknown")
            if should_remove:
                removed.append(
                    {
                        "object_id": obj.object_id,
                        "dominant_class_name": obj.dominant_class_name,
                        "observations_count": int(obj.observations_count),
                        "frame_ids": list(obj.frame_ids),
                        "longest_consecutive_frame_run": int(cls._longest_consecutive_frame_run(list(obj.frame_ids))),
                        "remove_reason": "object3d_track_motion_not_static"
                        if motion_state == "moving"
                        else "object3d_track_motion_unknown_not_static",
                        "motion_state": motion_state,
                        "motion_reason": metrics.get("motion_reason"),
                        "motion_metrics": metrics,
                    }
                )
            else:
                kept.append(obj)

        kept = sorted(
            kept,
            key=lambda obj: (-obj.observations_count, -obj.total_support_points, obj.object_id),
        )
        removed = sorted(
            removed,
            key=lambda item: (
                str(item.get("motion_state", "")),
                -float(item.get("motion_metrics", {}).get("max_center_span_m", 0.0)),
                str(item.get("object_id", "")),
            ),
        )
        return kept, removed

    @classmethod
    def _build_object_motion_metrics(
        cls,
        obj,
        history,
        movable_classes=None,
        min_consecutive_observations=2,
        max_center_span_m=1.0,
        max_median_step_m=0.35,
        max_single_step_m=1.2,
    ):
        class_name = str(getattr(obj, "dominant_class_name", "") or "").strip().lower()
        movable_classes = cls._parse_motion_filter_classes(movable_classes)
        if movable_classes and class_name not in movable_classes:
            return {
                "motion_state": "static",
                "motion_reason": "class_not_in_motion_filter_classes_static_assumed",
                "motion_filter_basis": "per_track_consecutive_bbox3d_center_history",
                "motion_filter_classes": movable_classes,
                "dominant_class_name": class_name,
            }

        valid_entries = cls._normalize_motion_history(history)
        if len(valid_entries) < 2:
            return {
                "motion_state": "unknown",
                "motion_reason": "insufficient_3m_motion_history",
                "motion_filter_basis": "per_track_consecutive_bbox3d_center_history",
                "history_observation_count": len(valid_entries),
                "min_consecutive_observations": max(1, int(min_consecutive_observations)),
                "motion_filter_classes": movable_classes,
                "thresholds": {
                    "max_center_span_m": float(max_center_span_m),
                    "max_median_step_m": float(max_median_step_m),
                    "max_single_step_m": float(max_single_step_m),
                },
            }

        runs = cls._split_motion_history_into_consecutive_runs(valid_entries)
        run_metrics = [cls._motion_run_metrics(run) for run in runs if len(run) >= 2]
        if not run_metrics:
            return {
                "motion_state": "unknown",
                "motion_reason": "no_consecutive_3m_motion_history",
                "motion_filter_basis": "per_track_consecutive_bbox3d_center_history",
                "history_observation_count": len(valid_entries),
                "consecutive_run_count": len(runs),
                "min_consecutive_observations": max(1, int(min_consecutive_observations)),
                "motion_filter_classes": movable_classes,
                "thresholds": {
                    "max_center_span_m": float(max_center_span_m),
                    "max_median_step_m": float(max_median_step_m),
                    "max_single_step_m": float(max_single_step_m),
                },
            }

        max_center_span = max(float(item["center_span_m"]) for item in run_metrics)
        max_net_displacement = max(float(item["net_displacement_m"]) for item in run_metrics)
        max_single_step = max(float(item["max_step_m"]) for item in run_metrics)
        median_steps = [float(item["median_step_m"]) for item in run_metrics if item["step_count"] > 0]
        median_step = float(np.median(median_steps)) if median_steps else 0.0
        longest_run = max(int(item["frame_count"]) for item in run_metrics)
        strongest_run = max(run_metrics, key=lambda item: float(item["center_span_m"]))
        if longest_run < max(1, int(min_consecutive_observations)):
            return {
                "motion_state": "unknown",
                "motion_reason": "insufficient_consecutive_3m_observations",
                "motion_filter_basis": "per_track_consecutive_bbox3d_center_history",
                "history_observation_count": len(valid_entries),
                "consecutive_run_count": len(runs),
                "longest_motion_run": int(longest_run),
                "min_consecutive_observations": max(1, int(min_consecutive_observations)),
                "strongest_run": strongest_run,
                "motion_filter_classes": movable_classes,
                "thresholds": {
                    "max_center_span_m": float(max_center_span_m),
                    "max_median_step_m": float(max_median_step_m),
                    "max_single_step_m": float(max_single_step_m),
                },
                "dominant_class_name": class_name,
            }

        moving_by_span = (
            max_center_span > float(max_center_span_m)
            and max_net_displacement > float(max_center_span_m) * 0.6
        )
        moving_by_median_step = (
            max_center_span > float(max_center_span_m)
            and median_step > float(max_median_step_m)
        )
        moving_by_single_step = (
            max_center_span > float(max_center_span_m)
            and max_single_step > float(max_single_step_m)
        )
        is_moving = bool(moving_by_span or moving_by_median_step or moving_by_single_step)
        if moving_by_median_step:
            reason = "center_span_and_median_step_exceed_threshold"
        elif moving_by_single_step:
            reason = "center_span_and_single_step_exceed_threshold"
        elif moving_by_span:
            reason = "center_span_and_net_displacement_exceed_threshold"
        else:
            reason = "motion_within_static_threshold"

        return {
            "motion_state": "moving" if is_moving else "static",
            "motion_reason": reason,
            "motion_filter_basis": "per_track_consecutive_bbox3d_center_history",
            "history_observation_count": len(valid_entries),
            "consecutive_run_count": len(runs),
            "longest_motion_run": int(longest_run),
            "min_consecutive_observations": max(1, int(min_consecutive_observations)),
            "motion_filter_classes": movable_classes,
            "max_center_span_m": float(max_center_span),
            "max_net_displacement_m": float(max_net_displacement),
            "median_step_m": float(median_step),
            "max_single_step_m": float(max_single_step),
            "strongest_run": strongest_run,
            "thresholds": {
                "max_center_span_m": float(max_center_span_m),
                "max_median_step_m": float(max_median_step_m),
                "max_single_step_m": float(max_single_step_m),
            },
            "dominant_class_name": class_name,
        }

    @staticmethod
    def _parse_motion_filter_classes(value):
        if value is None:
            return []
        if isinstance(value, str):
            raw_items = value.split(",")
        else:
            try:
                raw_items = list(value)
            except TypeError:
                raw_items = [value]
        classes = []
        for item in raw_items:
            name = str(item or "").strip().lower()
            if name and name not in classes:
                classes.append(name)
        return classes

    @staticmethod
    def _normalize_motion_history(history):
        by_frame = {}
        for item in (history if isinstance(history, list) else []):
            if not isinstance(item, dict):
                continue
            try:
                frame_index = int(item.get("frame_index"))
                center = np.asarray(item.get("bbox_3d_center"), dtype=np.float64).reshape(3)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(center).all():
                continue
            by_frame[frame_index] = {
                **item,
                "frame_index": frame_index,
                "bbox_3d_center": center.tolist(),
            }
        return [by_frame[key] for key in sorted(by_frame)]

    @staticmethod
    def _split_motion_history_into_consecutive_runs(entries):
        runs = []
        current = []
        previous_index = None
        for item in entries:
            frame_index = int(item["frame_index"])
            if previous_index is None or frame_index == previous_index + 1:
                current.append(item)
            else:
                if current:
                    runs.append(current)
                current = [item]
            previous_index = frame_index
        if current:
            runs.append(current)
        return runs

    @staticmethod
    def _motion_run_metrics(run):
        centers = np.asarray([item["bbox_3d_center"] for item in run], dtype=np.float64).reshape(-1, 3)
        frame_indexes = [int(item["frame_index"]) for item in run]
        if len(centers) < 2:
            center_span = 0.0
            steps = np.zeros((0,), dtype=np.float64)
        else:
            pairwise = centers[:, None, :] - centers[None, :, :]
            center_span = float(np.linalg.norm(pairwise, axis=2).max())
            steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)

        return {
            "start_frame_index": int(frame_indexes[0]),
            "end_frame_index": int(frame_indexes[-1]),
            "frame_count": int(len(frame_indexes)),
            "step_count": int(len(steps)),
            "center_span_m": float(center_span),
            "net_displacement_m": float(np.linalg.norm(centers[-1] - centers[0])) if len(centers) >= 2 else 0.0,
            "median_step_m": float(np.median(steps)) if len(steps) else 0.0,
            "max_step_m": float(np.max(steps)) if len(steps) else 0.0,
            "start_center": centers[0].tolist(),
            "end_center": centers[-1].tolist(),
        }

    @classmethod
    def _tracking_object_rank(cls, obj):
        return (
            int(cls._longest_consecutive_frame_run(list(obj.frame_ids))),
            int(obj.observations_count),
            int(obj.total_support_points),
            float(obj.mean_confidence),
            -float(cls._tracking_object_volume(obj) or 0.0),
        )

    @staticmethod
    def _normalize_object_class(obj):
        return str(getattr(obj, "dominant_class_name", "") or "").strip().lower()

    @staticmethod
    def _tracking_object_aabb(obj):
        corners = np.asarray(obj.bbox3d.corners, dtype=np.float64).reshape(-1, 3)
        if len(corners) == 0:
            return None
        mins = corners.min(axis=0)
        maxs = corners.max(axis=0)
        extent = np.maximum(maxs - mins, 0.0)
        if float(np.prod(extent)) <= 0.0:
            return None
        return mins, maxs

    @classmethod
    def _tracking_object_volume(cls, obj):
        aabb = cls._tracking_object_aabb(obj)
        if aabb is None:
            return None
        mins, maxs = aabb
        return float(np.prod(np.maximum(maxs - mins, 0.0)))

    @classmethod
    def _tracking_object_volume_overlap(cls, first, second):
        first_aabb = cls._tracking_object_aabb(first)
        second_aabb = cls._tracking_object_aabb(second)
        if first_aabb is None or second_aabb is None:
            return None
        first_min, first_max = first_aabb
        second_min, second_max = second_aabb
        inter_min = np.maximum(first_min, second_min)
        inter_max = np.minimum(first_max, second_max)
        inter_extent = np.maximum(inter_max - inter_min, 0.0)
        intersection = float(np.prod(inter_extent))
        if intersection <= 0.0:
            return 0.0, 0.0, 0.0
        first_volume = float(np.prod(np.maximum(first_max - first_min, 0.0)))
        second_volume = float(np.prod(np.maximum(second_max - second_min, 0.0)))
        union = first_volume + second_volume - intersection
        if first_volume <= 0.0 or second_volume <= 0.0 or union <= 0.0:
            return None
        iou = intersection / union
        min_ratio = intersection / min(first_volume, second_volume)
        max_ratio = intersection / max(first_volume, second_volume)
        return iou, min_ratio, max_ratio

    @staticmethod
    def _longest_consecutive_frame_run(frame_ids):
        indexes = []
        for frame_id in frame_ids:
            match = re.search(r"(\d+)$", str(frame_id))
            if match is not None:
                indexes.append(int(match.group(1)))
        indexes = sorted(set(indexes))
        if not indexes:
            return 0

        longest = 1
        current = 1
        for previous, current_idx in zip(indexes, indexes[1:]):
            if current_idx == previous + 1:
                current += 1
            else:
                longest = max(longest, current)
                current = 1
        return max(longest, current)

    @classmethod
    def _build_object_time_metadata(cls, obj, history):
        entries = cls._normalize_motion_history(history)
        if entries:
            frame_indexes = [int(item["frame_index"]) for item in entries]
            first_entry = entries[0]
            last_entry = entries[-1]
            first_timestamp = cls._normalize_timestamp_value(first_entry.get("timestamp"))
            last_timestamp = cls._normalize_timestamp_value(last_entry.get("timestamp"))
            first_time_s = first_entry.get("timestamp_s")
            last_time_s = last_entry.get("timestamp_s")
        else:
            frame_indexes = []
            for frame_id in list(getattr(obj, "frame_ids", []) or []):
                match = re.search(r"(\d+)$", str(frame_id))
                if match is not None:
                    frame_indexes.append(int(match.group(1)))
            first_timestamp = None
            last_timestamp = None
            first_time_s = None
            last_time_s = None

        if not frame_indexes:
            return {
                "object3d_observed_frame_count": int(getattr(obj, "observations_count", 0) or 0),
                "object3d_lifecycle_state": "observed",
            }

        first_frame_index = int(min(frame_indexes))
        last_frame_index = int(max(frame_indexes))
        frame_count = len(set(frame_indexes))
        frame_span = int(last_frame_index - first_frame_index + 1)
        missing_frame_count = max(0, frame_span - frame_count)

        duration_s = None
        try:
            if first_time_s is not None and last_time_s is not None:
                duration_s = max(0.0, float(last_time_s) - float(first_time_s))
        except (TypeError, ValueError):
            duration_s = None

        return {
            "object3d_first_seen_frame_index": first_frame_index,
            "object3d_last_seen_frame_index": last_frame_index,
            "object3d_observed_frame_count": int(frame_count),
            "object3d_observed_frame_span": int(frame_span),
            "object3d_missing_frame_count": int(missing_frame_count),
            "object3d_observation_rate": float(frame_count / frame_span) if frame_span > 0 else None,
            "object3d_first_seen_timestamp": first_timestamp,
            "object3d_last_seen_timestamp": last_timestamp,
            "object3d_first_seen_time_s": float(first_time_s) if first_time_s is not None else None,
            "object3d_last_seen_time_s": float(last_time_s) if last_time_s is not None else None,
            "object3d_observed_duration_s": duration_s,
            "object3d_lifecycle_state": "observed",
        }

    def _serialize_tracking_object(self, obj, motion_metrics=None):
        payload = {
            "object_id": obj.object_id,
            "dominant_class_name": obj.dominant_class_name,
            "class_name": obj.dominant_class_name,
            "class_votes": dict(obj.class_votes),
            "mean_confidence": float(obj.mean_confidence),
            "observations_count": int(obj.observations_count),
            "observation_count": int(obj.observations_count),
            "total_support_points": int(obj.total_support_points),
            "longest_consecutive_frame_run": int(self._longest_consecutive_frame_run(list(obj.frame_ids))),
            "centroid": np.asarray(obj.centroid, dtype=np.float64).tolist(),
            "bbox_3d_center": np.asarray(obj.bbox3d.center, dtype=np.float64).tolist(),
            "bbox_3d_extent": np.asarray(obj.bbox3d.extent, dtype=np.float64).tolist(),
            "bbox_3d_corners": np.asarray(obj.bbox3d.corners, dtype=np.float64).tolist(),
            "global_points_sample": self._sample_global_points(obj.global_points, max_points=700),
            "frame_ids": list(obj.frame_ids),
        }
        payload.update(
            self._build_object_time_metadata(
                obj,
                self.object3d_track_history.get(str(obj.object_id), []),
            )
        )
        if isinstance(motion_metrics, dict):
            payload["motion_state"] = motion_metrics.get("motion_state")
            payload["motion_reason"] = motion_metrics.get("motion_reason")
            payload["motion_metrics"] = motion_metrics
        return payload

    @staticmethod
    def _filter_detections_near_border(detections, img_width, img_height, margin_ratio=0.05):
        margin_x = img_width * margin_ratio
        margin_y = img_height * margin_ratio
        filtered = []
        for det in detections:
            bbox = det.get("bbox") or {}
            x1 = float(bbox.get("x1", 0.0))
            y1 = float(bbox.get("y1", 0.0))
            x2 = float(bbox.get("x2", 0.0))
            y2 = float(bbox.get("y2", 0.0))
            if x1 >= margin_x and y1 >= margin_y and x2 <= img_width - margin_x and y2 <= img_height - margin_y:
                filtered.append(det)
        return filtered

    @staticmethod
    def _draw_filtered_detections(img, detections):
        drawn = img.copy()
        for det in detections:
            bbox = det.get("bbox") or {}
            x1 = int(round(float(bbox.get("x1", 0.0))))
            y1 = int(round(float(bbox.get("y1", 0.0))))
            x2 = int(round(float(bbox.get("x2", 0.0))))
            y2 = int(round(float(bbox.get("y2", 0.0))))
            class_name = str(det.get("class_name", "unknown"))
            confidence = float(det.get("confidence", 0.0))
            depth = det.get("depth")
            method = str(det.get("global_position_method", "unknown"))
            depth_text = "NA" if depth is None else f"{float(depth):.2f}"
            label = f"{class_name} {confidence:.2f} D={depth_text} {method}"
            cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                drawn,
                label,
                (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )
        return drawn


class InMemoryNavGraphBuilder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.output_dir = Path(str(cfg.output_dir)).resolve()
        self.nav_root = Path(str(cfg.nav_graph_root)).resolve()
        if str(self.nav_root) not in sys.path:
            sys.path.append(str(self.nav_root))

        import build_nav_graph_nx as nav
        from nav_graph_export_utils import export_graph_contents

        self.nav = nav
        self.export_graph_contents = export_graph_contents
        self.nav.ensure_networkx()

        self.nav_config_path = Path(str(cfg.nav_config_path)).resolve() if cfg.nav_config_path else self.nav_root / self.nav.DEFAULT_CONFIG_PATH
        self.graph_output = Path(str(cfg.nav_graph_output)).resolve() if cfg.nav_graph_output else self.output_dir / "nav_graph.pkl"
        self.contents_output = Path(str(cfg.nav_contents_output)).resolve() if cfg.nav_contents_output else self.output_dir / "nav_graph_contents.json"
        self.stats_output = Path(str(cfg.nav_stats_output)).resolve() if cfg.nav_stats_output else self.output_dir / "nav_graph_stats.json"

        self.graph_output.parent.mkdir(parents=True, exist_ok=True)
        self.contents_output.parent.mkdir(parents=True, exist_ok=True)
        self.stats_output.parent.mkdir(parents=True, exist_ok=True)

        self.graph = self.nav.new_nav_graph(self._build_nav_config())
        self.topology_index, self.waypoint_index, self.object_index = self.nav.rebuild_spatial_indexes(self.graph)
        self.previous_waypoint = None
        self.imported_waypoints = 0
        self.imported_observations = 0

    def _build_nav_config(self):
        config_payload = self.nav.load_config_file(str(self.nav_config_path))
        section_defaults = dict(self.nav.DEFAULT_CONFIG["build"])
        section_from_file = config_payload.get("build", {})
        if section_from_file is None:
            section_from_file = {}
        if not isinstance(section_from_file, dict):
            raise ValueError("config section 'build' must be a JSON object")

        merged = dict(section_defaults)
        merged.update(section_from_file)
        args = SimpleNamespace(command="build")
        for key in section_defaults:
            setattr(args, key, merged.get(key))

        dynamic_classes = self.nav.build_dynamic_classes(args.extra_dynamic_classes)
        return self.nav.build_config(args, dynamic_classes)

    def append_frame(self, payload, frame_idx):
        source_path = str(self.output_dir / f"in_memory_step_{frame_idx:08d}.json")
        sort_key = f"{frame_idx:08d}"
        self.previous_waypoint = self.nav.append_payload(
            graph=self.graph,
            payload=payload,
            source_path=source_path,
            previous_waypoint=self.previous_waypoint,
            topology_index=self.topology_index,
            waypoint_index=self.waypoint_index,
            object_index=self.object_index,
            sort_key_override=sort_key,
        )
        self.imported_waypoints += 1
        detections = payload.get("detections")
        if isinstance(detections, list):
            self.imported_observations += sum(1 for item in detections if isinstance(item, dict))

    def finalize(self):
        self.nav.attach_semantic_hierarchy(self.graph, levels=3)
        self.nav.write_graph(self.graph, self.graph_output)
        content_payload = self.export_graph_contents(self.graph, self.contents_output, self.graph_output)

        inferred_topology_links = 0
        edges = content_payload.get("edges")
        if isinstance(edges, dict):
            inferred_edges = edges.get("topology_topology_inferred", [])
            if isinstance(inferred_edges, list):
                inferred_topology_links = len(inferred_edges)

        stats_payload = {
            "graph_path": str(self.graph_output),
            "content_path": str(self.contents_output),
            "imported_waypoints": self.imported_waypoints,
            "imported_observations": self.imported_observations,
            "stats": self.nav.graph_stats(self.graph),
            "inferred_topology_links": inferred_topology_links,
        }
        self.stats_output.write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return stats_payload


class GraphRuntimeManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.output_dir = Path(str(cfg.output_dir)).resolve()
        self.nav_root = Path(str(cfg.nav_graph_root)).resolve()
        if str(self.nav_root) not in sys.path:
            sys.path.append(str(self.nav_root))

        import build_nav_graph_nx as nav
        import object_navigation_runtime as object_nav
        from nav_graph_export_utils import (
            export_graph_contents,
            export_graph_visualization,
            export_graph_visualization_3d,
            export_object3d_global_visualization,
            export_object3d_tracking_visualization,
        )

        self.nav = nav
        self.object_nav = object_nav
        self.export_graph_contents = export_graph_contents
        self.export_graph_visualization = export_graph_visualization
        self.export_graph_visualization_3d = export_graph_visualization_3d
        self.export_object3d_global_visualization = export_object3d_global_visualization
        self.export_object3d_tracking_visualization = export_object3d_tracking_visualization

        self.graph_output = Path(str(cfg.nav_graph_output)).resolve() if cfg.nav_graph_output else self.output_dir / "nav_graph.pkl"
        self.contents_output = Path(str(cfg.nav_contents_output)).resolve() if cfg.nav_contents_output else self.output_dir / "nav_graph_contents.json"
        self.object3d_tracking_summary_output = (
            Path(str(cfg.object3d_tracking_summary_json)).resolve()
            if cfg.object3d_tracking_summary_json
            else self.output_dir / "object3d_tracking_summary.json"
        )
        self.svg_output = (
            Path(str(cfg.nav_visualization_svg)).resolve()
            if cfg.nav_visualization_svg
            else self.output_dir / "nav_graph_visualization.svg"
        )
        self.html_output = (
            Path(str(cfg.nav_visualization_html)).resolve()
            if cfg.nav_visualization_html
            else self.output_dir / "nav_graph_visualization.html"
        )
        self.html_3d_output = (
            Path(str(cfg.nav_visualization_3d_html)).resolve()
            if cfg.nav_visualization_3d_html
            else self.output_dir / "nav_graph_visualization_3d.html"
        )
        self.object3d_global_map_output = (
            Path(str(cfg.object3d_global_map_html)).resolve()
            if cfg.object3d_global_map_html
            else self.output_dir / "object3d_global_map.html"
        )

    def navigate(self):
        payload = self.object_nav.navigate_from_current_position(
            current_position=self.cfg.current_position,
            object_query=self.cfg.object_query,
            graph_path=self.graph_output,
        )
        print(self.object_nav.build_console_report(payload))
        return payload

    def visualize(self):
        self.nav.ensure_networkx()
        if not self.graph_output.exists():
            raise FileNotFoundError(f"graph not found: {self.graph_output}")

        graph = self.nav.read_graph(self.graph_output)
        content_payload = self.export_graph_contents(graph, self.contents_output, self.graph_output)
        self.export_graph_visualization(content_payload, self.svg_output, self.html_output)
        self.export_graph_visualization_3d(content_payload, self.html_3d_output)
        if self.object3d_tracking_summary_output.exists():
            tracking_payload = json.loads(self.object3d_tracking_summary_output.read_text(encoding="utf-8"))
            tracking_payload["summary_path"] = str(self.object3d_tracking_summary_output)
            self.export_object3d_tracking_visualization(tracking_payload, self.object3d_global_map_output)
        else:
            self.export_object3d_global_visualization(content_payload, self.object3d_global_map_output)

        result = {
            "graph_path": str(self.graph_output),
            "content_path": str(self.contents_output),
            "svg_path": str(self.svg_output),
            "html_path": str(self.html_output),
            "html_3d_path": str(self.html_3d_output),
            "object3d_tracking_summary_json": str(self.object3d_tracking_summary_output),
            "object3d_global_map_html": str(self.object3d_global_map_output),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result


class SemanticTopomapFusionPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.output_dir = Path(str(cfg.output_dir)).resolve()
        self.device = cfg.device
        self.yolo_device = str(cfg.get("yolo_device") or self.device)
        self.model = None
        self.db_dataset = None
        self.descriptors = None
        self.depth_pipeline = None
        self.recorded_depth_available = False
        self.topomap = Topomap()
        self.current_node = CurrentNode(leaving_threshold=cfg.new_node_distance_threshold)
        self.creating_new_node = True
        self.track_csv = None
        self.camera_positions = []
        self.object3d_tracking_summary_output = (
            Path(str(cfg.object3d_tracking_summary_json)).resolve()
            if cfg.object3d_tracking_summary_json
            else self.output_dir / "object3d_tracking_summary.json"
        )
        self.object3d_global_map_output = (
            Path(str(cfg.object3d_global_map_html)).resolve()
            if cfg.object3d_global_map_html
            else self.output_dir / "object3d_global_map.html"
        )
        self.payload_builder = FrameSemanticPayloadBuilder(
            output_root=str(cfg.output_dir),
            hfov=float(cfg.hfov),
            vfov=float(cfg.vfov),
            save_step_artifacts=bool(cfg.save_step_artifacts),
            object3d_engine_root=cfg.object3d_engine_root,
            sam_checkpoint=cfg.sam_checkpoint,
            sam_model_type=cfg.sam_model_type,
            sam_device=cfg.get("sam_device") or cfg.get("yolo_device") or cfg.device,
        )
        self.nav_builder = InMemoryNavGraphBuilder(cfg)

    def setup(self):
        append_import_roots(self.cfg.opr_root, package_name="opr")
        append_import_roots(self.cfg.depth_anything_root)

        from opr.datasets.itlp import ITLPCampus
        from opr.pipelines.place_recognition import PlaceRecognitionPipeline

        self.model = YOLO(self.cfg.yolo_weights)
        if self.yolo_device.startswith("cuda"):
            self.model.to(self.yolo_device)

        model_config = OmegaConf.load(self.cfg.model_config_path)
        model_1 = instantiate(model_config)
        pipe = PlaceRecognitionPipeline(
            database_dir=self.cfg.database_track_dir,
            model=model_1,
            model_weights_path=self.cfg.weights_path,
            device=self.device,
        )
        model_1 = pipe.model
        model_1.eval()

        self.db_dataset = ITLPCampus(
            dataset_root=self.cfg.database_track_dir,
            sensors=["front_cam", "back_cam", "lidar"],
            mink_quantization_size=0.5,
            load_semantics=False,
            exclude_dynamic_classes=False,
            indoor=True,
        )
        db_dataloader = DataLoader(
            self.db_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            collate_fn=self.db_dataset.collate_fn,
        )

        descriptors = []
        with torch.no_grad():
            for batch in tqdm(db_dataloader, desc="Build descriptors"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                final_descriptor = model_1(batch)["final_descriptor"]
                descriptors.append(final_descriptor.detach().cpu().numpy())
        self.descriptors = np.concatenate(descriptors, axis=0)

        self.track_csv = pd.read_csv(os.path.join(self.cfg.data_base_path, "track.csv"))
        self.track_csv["front_cam_ts"] = self.track_csv["front_cam_ts"].astype(str)
        self.track_csv["lidar_ts"] = self.track_csv["lidar_ts"].astype(str)
        if "depth_ts" in self.track_csv.columns:
            self.track_csv["depth_ts"] = self.track_csv["depth_ts"].fillna("").astype(str)

        self.recorded_depth_available = self._has_recorded_depth()
        if self.recorded_depth_available:
            print(f"[depth] using recorded depth PNGs from {self._recorded_depth_dir()}")
        else:
            self._ensure_depth_pipeline()
            print("[depth] recorded depth not found; using DepthAnything+LiDAR fallback")

    def run(self):
        self.setup()
        total_frames = self.track_csv.shape[0]
        if self.cfg.max_frames > 0:
            total_frames = min(total_frames, self.cfg.max_frames)

        frame_payloads = []
        for frame_idx in tqdm(range(total_frames), desc="Frame loop"):
            payload = self.process_frame(frame_idx)
            frame_payloads.append(payload)
            position = payload.get("position")
            if isinstance(position, list) and len(position) >= 3:
                self.camera_positions.append(position[:3])

        if len(self.current_node.current_feature) > 0:
            self.current_node.current_to_graph(self.topomap)

        tracking_summary = self.payload_builder.export_tracking_summary(
            camera_positions=self.camera_positions,
            min_consecutive_frames=int(self.cfg.object3d_min_consecutive_frames),
            overlap_filter_enabled=bool(self.cfg.object3d_overlap_filter_enabled),
            overlap_iou_threshold=float(self.cfg.object3d_overlap_iou_threshold),
            overlap_min_ratio_threshold=float(self.cfg.object3d_overlap_min_ratio_threshold),
            motion_filter_enabled=bool(self.cfg.object3d_motion_filter_enabled),
            motion_filter_classes=self.cfg.get("object3d_motion_filter_classes", ""),
            motion_unknown_filter_enabled=bool(self.cfg.get("object3d_motion_unknown_filter_enabled", True)),
            motion_min_consecutive_observations=int(
                self.cfg.get("object3d_motion_min_consecutive_observations", 2)
            ),
            motion_static_max_center_span_m=float(self.cfg.object3d_motion_static_max_center_span_m),
            motion_static_max_median_step_m=float(self.cfg.object3d_motion_static_max_median_step_m),
            motion_static_max_single_step_m=float(self.cfg.object3d_motion_static_max_single_step_m),
        )
        tracking_summary = self._apply_object3d_disappearance_filter(
            frame_payloads,
            tracking_summary,
        )
        filtered_payloads, nav_filter_stats = self._filter_payloads_for_nav_graph(
            frame_payloads,
            tracking_summary,
        )
        nav_payloads, waypoint_sampling_stats = self._sample_waypoint_payloads(filtered_payloads)
        for frame_idx, payload in nav_payloads:
            self.nav_builder.append_frame(payload, frame_idx)

        pruned_nav_graph_object_ids = self._prune_removed_disappearance_objects_from_nav_graph(tracking_summary)
        stats = self.nav_builder.finalize()
        tracking_summary["summary_path"] = str(self.object3d_tracking_summary_output)
        self.object3d_tracking_summary_output.parent.mkdir(parents=True, exist_ok=True)
        self.object3d_tracking_summary_output.write_text(
            json.dumps(tracking_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        from nav_graph_export_utils import export_object3d_tracking_visualization

        export_object3d_tracking_visualization(tracking_summary, self.object3d_global_map_output)
        stats["object3d_tracking_summary_json"] = str(self.object3d_tracking_summary_output)
        stats["object3d_global_map_html"] = str(self.object3d_global_map_output)
        stats["object3d_raw_object_count"] = tracking_summary.get("raw_object_count")
        stats["object3d_stable_object_count"] = tracking_summary.get("object_count")
        stats["object3d_min_consecutive_frames"] = tracking_summary.get("postprocess", {}).get("min_consecutive_frames")
        stats["object3d_overlap_removed_object_count"] = tracking_summary.get("postprocess", {}).get("overlap_removed_object_count")
        stats["object3d_motion_removed_object_count"] = tracking_summary.get("postprocess", {}).get("motion_removed_object_count")
        stats["object3d_disappearance_removed_object_count"] = tracking_summary.get("postprocess", {}).get("disappearance_removed_object_count")
        stats["object3d_nav_graph_filter"] = nav_filter_stats
        stats["waypoint_sampling"] = waypoint_sampling_stats
        stats["object3d_nav_graph_pruned_removed_object_count"] = len(pruned_nav_graph_object_ids)
        self.nav_builder.stats_output.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        print(f"Done. Output directory: {os.path.abspath(self.cfg.output_dir)}")

    def _sample_waypoint_payloads(self, payloads):
        input_count = len(payloads)
        if not bool(self.cfg.get("waypoint_sampling_enabled", False)):
            return list(enumerate(payloads)), {
                "enabled": False,
                "input_frames": input_count,
                "kept_waypoints": input_count,
                "removed_waypoints": 0,
            }

        min_distance_m = max(0.0, float(self.cfg.get("waypoint_min_distance_m", 0.8)))
        min_yaw_deg = max(0.0, float(self.cfg.get("waypoint_min_yaw_deg", 25.0)))
        keep_first_last = bool(self.cfg.get("waypoint_keep_first_last", True))
        keep_topology_change = bool(self.cfg.get("waypoint_keep_topology_change", True))

        selected = []
        last_kept_position = None
        last_kept_yaw = None
        last_topology = None
        reasons_count = {}

        def add_reason(reason):
            reasons_count[reason] = int(reasons_count.get(reason, 0)) + 1

        for frame_idx, payload in enumerate(payloads):
            position = self._payload_position(payload)
            yaw_deg = self._payload_yaw_deg(payload)
            topology = str(payload.get("class_node") or "")
            reasons = []

            if frame_idx == 0 and keep_first_last:
                reasons.append("first")
            elif position is None:
                reasons.append("missing_position")
            elif last_kept_position is None:
                reasons.append("first_valid_position")
            else:
                distance = float(np.linalg.norm(position - last_kept_position))
                if distance >= min_distance_m:
                    reasons.append("distance")
                if yaw_deg is not None and last_kept_yaw is not None:
                    yaw_delta = self._angle_delta_deg(yaw_deg, last_kept_yaw)
                    if yaw_delta >= min_yaw_deg:
                        reasons.append("yaw")
                if keep_topology_change and topology and last_topology and topology != last_topology:
                    reasons.append("topology_change")

            if frame_idx == input_count - 1 and keep_first_last:
                reasons.append("last")

            if not reasons:
                continue

            next_payload = copy.deepcopy(payload)
            next_payload["waypoint_sampled_from_frame_index"] = int(frame_idx)
            next_payload["waypoint_sampling_reasons"] = sorted(set(reasons))
            selected.append((frame_idx, next_payload))
            for reason in set(reasons):
                add_reason(reason)

            if position is not None:
                last_kept_position = position
            if yaw_deg is not None:
                last_kept_yaw = yaw_deg
            if topology:
                last_topology = topology

        if not selected and payloads:
            selected.append((0, copy.deepcopy(payloads[0])))
            add_reason("fallback_first")

        return selected, {
            "enabled": True,
            "input_frames": input_count,
            "kept_waypoints": len(selected),
            "removed_waypoints": max(0, input_count - len(selected)),
            "min_distance_m": min_distance_m,
            "min_yaw_deg": min_yaw_deg,
            "keep_first_last": keep_first_last,
            "keep_topology_change": keep_topology_change,
            "reasons_count": reasons_count,
            "kept_frame_indices": [int(frame_idx) for frame_idx, _payload in selected],
        }

    @staticmethod
    def _payload_position(payload):
        try:
            value = payload.get("position")
            if not isinstance(value, (list, tuple)) or len(value) < 3:
                return None
            position = np.asarray(value[:3], dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if not np.all(np.isfinite(position)):
            return None
        return position

    @staticmethod
    def _payload_yaw_deg(payload):
        rotation = payload.get("rotation")
        if not isinstance(rotation, (list, tuple)) or len(rotation) < 4:
            return None
        try:
            q = quaternion.quaternion(float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3]))
            matrix = quaternion.as_rotation_matrix(q)
            yaw = np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))
        except (TypeError, ValueError, ZeroDivisionError):
            return None
        if not np.isfinite(yaw):
            return None
        return float(yaw)

    @staticmethod
    def _angle_delta_deg(a, b):
        return float(abs((float(a) - float(b) + 180.0) % 360.0 - 180.0))

    def _filter_payloads_for_nav_graph(self, payloads, tracking_summary):
        if not bool(self.cfg.get("use_filtered_object3d_for_nav_graph", True)):
            return payloads, {
                "enabled": False,
                "reason": "disabled_by_config",
                "input_detection_count": self._count_payload_detections(payloads),
                "kept_detection_count": self._count_payload_detections(payloads),
                "removed_detection_count": 0,
            }

        input_count = 0
        kept_count = 0
        removed_count = 0
        filtered_payloads = []
        for frame_index, payload in enumerate(payloads):
            stable_objects = self._collect_nav_graph_candidate_objects(tracking_summary, frame_index)
            stable_by_id = {
                str(obj.get("object_id")): obj
                for obj in stable_objects
                if isinstance(obj, dict) and obj.get("object_id")
            }
            next_payload = copy.deepcopy(payload)
            detections = next_payload.get("detections")
            if not isinstance(detections, list):
                next_payload["detections"] = []
                filtered_payloads.append(next_payload)
                continue

            kept_detections = []
            used_stable_object_ids = set()
            for detection in detections:
                if not isinstance(detection, dict):
                    continue
                input_count += 1
                stable_object, match_info = self._resolve_stable_object_for_detection(
                    detection=detection,
                    stable_by_id=stable_by_id,
                    stable_objects=stable_objects,
                    used_stable_object_ids=used_stable_object_ids,
                    match_distance_m=float(self.cfg.get("object3d_disappearance_match_distance_m", 1.0)),
                )
                if stable_object is None:
                    removed_count += 1
                    continue
                used_stable_object_ids.add(str(stable_object.get("object_id") or ""))
                kept_detections.append(
                    self._apply_stable_object3d_to_detection(
                        detection,
                        stable_object,
                        match_info=match_info,
                    )
                )
                kept_count += 1

            next_payload["detections"] = kept_detections
            filtered_payloads.append(next_payload)

        return filtered_payloads, {
            "enabled": True,
            "source": "object3d_tracking_postprocess_time_aware",
            "stable_object_count": len(
                tracking_summary.get("objects") if isinstance(tracking_summary.get("objects"), list) else []
            ),
            "input_detection_count": input_count,
            "kept_detection_count": kept_count,
            "removed_detection_count": removed_count,
        }

    @staticmethod
    def _resolve_first_seen_frame_index(stable_object):
        try:
            value = stable_object.get("object3d_first_seen_frame_index")
        except AttributeError:
            value = None
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass

        frame_ids = stable_object.get("frame_ids") if isinstance(stable_object, dict) else None
        candidates = []
        for item in frame_ids if isinstance(frame_ids, list) else []:
            try:
                candidates.append(int(item))
            except (TypeError, ValueError):
                continue
        if not candidates:
            return None
        return min(candidates)

    @staticmethod
    def _resolve_disappearance_removal_frame_index(stable_object):
        try:
            explicit_value = stable_object.get("object3d_removal_frame_index")
        except AttributeError:
            explicit_value = None
        if explicit_value is not None:
            try:
                return int(explicit_value)
            except (TypeError, ValueError):
                pass

        if str(stable_object.get("removal_stage") or "") != "disappearance_filter":
            return None

        decision = stable_object.get("disappearance_decision")
        if not isinstance(decision, dict):
            return None
        missed_frames = decision.get("missed_frames")
        if not isinstance(missed_frames, list) or not missed_frames:
            return None
        last_miss = missed_frames[-1]
        if not isinstance(last_miss, dict):
            return None
        try:
            return int(last_miss.get("frame_index"))
        except (TypeError, ValueError):
            return None

    @classmethod
    def _is_stable_object_active_for_nav_frame(cls, stable_object, frame_index):
        first_seen_frame_index = cls._resolve_first_seen_frame_index(stable_object)
        if first_seen_frame_index is not None and int(frame_index) < int(first_seen_frame_index):
            return False

        removal_frame_index = cls._resolve_disappearance_removal_frame_index(stable_object)
        if removal_frame_index is not None and int(frame_index) >= int(removal_frame_index):
            return False
        return True

    @classmethod
    def _collect_nav_graph_candidate_objects(cls, tracking_summary, frame_index):
        stable_objects = tracking_summary.get("objects")
        if not isinstance(stable_objects, list):
            stable_objects = []

        removed_objects = tracking_summary.get("removed_objects")
        if not isinstance(removed_objects, list):
            removed_objects = []

        candidates = []
        for obj in stable_objects:
            if isinstance(obj, dict):
                candidates.append(obj)
        for obj in removed_objects:
            if not isinstance(obj, dict):
                continue
            if str(obj.get("removal_stage") or "") != "disappearance_filter":
                continue
            candidates.append(obj)

        return [
            obj
            for obj in candidates
            if cls._is_stable_object_active_for_nav_frame(obj, frame_index)
        ]

    def _apply_object3d_disappearance_filter(self, payloads, tracking_summary):
        summary = copy.deepcopy(tracking_summary if isinstance(tracking_summary, dict) else {})
        stable_objects = summary.get("objects")
        if not isinstance(stable_objects, list):
            stable_objects = []
            summary["objects"] = stable_objects

        removed_objects = summary.get("removed_objects")
        if not isinstance(removed_objects, list):
            removed_objects = []
            summary["removed_objects"] = removed_objects

        postprocess = summary.setdefault("postprocess", {})
        enabled = bool(self.cfg.get("object3d_disappearance_filter_enabled", True))
        max_observation_distance_m = float(self.cfg.get("object3d_disappearance_max_observation_distance_m", 3.0))
        position_tolerance_m = float(self.cfg.get("object3d_disappearance_position_tolerance_m", 1.0))
        match_distance_m = float(self.cfg.get("object3d_disappearance_match_distance_m", 1.0))
        min_visible_misses = max(1, int(self.cfg.get("object3d_disappearance_min_visible_misses", 2)))
        fov_margin_deg = max(0.0, float(self.cfg.get("object3d_disappearance_fov_margin_deg", 8.0)))

        postprocess["disappearance_filter_enabled"] = enabled
        postprocess["disappearance_max_observation_distance_m"] = max_observation_distance_m
        postprocess["disappearance_position_tolerance_m"] = position_tolerance_m
        postprocess["disappearance_match_distance_m"] = match_distance_m
        postprocess["disappearance_min_visible_misses"] = min_visible_misses
        postprocess["disappearance_fov_margin_deg"] = fov_margin_deg

        if not enabled or not stable_objects:
            postprocess["disappearance_removed_object_count"] = 0
            postprocess["disappearance_suspected_object_count"] = 0
            postprocess["removed_object_count"] = len(removed_objects)
            summary["object_count"] = len(stable_objects)
            return summary

        track_history = self.payload_builder.object3d_track_history
        kept_objects = []
        disappearance_removed_objects = []
        suspected_object_count = 0
        for stable_object in stable_objects:
            decision = self._evaluate_object3d_disappearance(
                stable_object=stable_object,
                payloads=payloads,
                track_history=track_history,
                max_observation_distance_m=max_observation_distance_m,
                position_tolerance_m=position_tolerance_m,
                match_distance_m=match_distance_m,
                min_visible_misses=min_visible_misses,
                fov_margin_deg=fov_margin_deg,
            )
            if decision.get("removed"):
                removed_payload = copy.deepcopy(stable_object)
                missed_frames = decision.get("missed_frames") if isinstance(decision, dict) else []
                removal_frame_index = None
                removal_timestamp = None
                if isinstance(missed_frames, list) and missed_frames:
                    last_miss = missed_frames[-1]
                    if isinstance(last_miss, dict):
                        removal_frame_index = last_miss.get("frame_index")
                        removal_timestamp = last_miss.get("timestamp")
                removed_payload["removal_stage"] = "disappearance_filter"
                removed_payload["removal_reason"] = "missing_after_revisit_visibility_check"
                removed_payload["object3d_lifecycle_state"] = "removed_after_disappearance_filter"
                removed_payload["object3d_removal_frame_index"] = (
                    int(removal_frame_index) if removal_frame_index is not None else None
                )
                removed_payload["object3d_removal_timestamp"] = removal_timestamp
                removed_payload["object3d_removal_time_s"] = (
                    FrameSemanticPayloadBuilder._timestamp_to_seconds(removal_timestamp)
                    if removal_timestamp is not None
                    else None
                )
                removed_payload["object3d_removal_reason"] = "missing_after_revisit_visibility_check"
                removed_payload["disappearance_decision"] = decision
                disappearance_removed_objects.append(removed_payload)
            else:
                kept_payload = copy.deepcopy(stable_object)
                kept_payload["object3d_lifecycle_state"] = str(
                    decision.get("lifecycle_state") or kept_payload.get("object3d_lifecycle_state") or "active"
                )
                kept_payload["disappearance_decision"] = decision
                kept_payload["object3d_last_confirmed_frame_index"] = decision.get("last_confirmed_frame_index")
                kept_payload["object3d_last_confirmed_timestamp"] = decision.get("last_confirmed_timestamp")
                kept_payload["object3d_last_confirmed_time_s"] = decision.get("last_confirmed_time_s")
                kept_payload["object3d_disappearance_visibility_check_count"] = decision.get("visibility_opportunity_count")
                kept_payload["object3d_disappearance_visible_miss_count"] = decision.get("visible_miss_count")
                kept_payload["object3d_disappearance_consecutive_visible_miss_count"] = decision.get(
                    "consecutive_visible_miss_count"
                )
                kept_payload["object3d_disappearance_max_visible_miss_streak"] = decision.get(
                    "max_visible_miss_streak"
                )
                kept_payload["object3d_revisit_confirmed_count"] = decision.get("matched_revisit_count")
                if kept_payload["object3d_lifecycle_state"] == "suspected_missing":
                    suspected_object_count += 1
                kept_objects.append(kept_payload)

        removed_objects.extend(disappearance_removed_objects)
        summary["objects"] = kept_objects
        summary["removed_objects"] = removed_objects
        summary["object_count"] = len(kept_objects)
        postprocess["disappearance_removed_object_count"] = len(disappearance_removed_objects)
        postprocess["disappearance_suspected_object_count"] = int(suspected_object_count)
        postprocess["removed_object_count"] = len(removed_objects)
        return summary

    def _evaluate_object3d_disappearance(
        self,
        stable_object,
        payloads,
        track_history,
        max_observation_distance_m,
        position_tolerance_m,
        match_distance_m,
        min_visible_misses,
        fov_margin_deg,
    ):
        object_id = str(stable_object.get("object_id") or "")
        object_center = self._extract_object3d_center(stable_object)
        if object_center is None:
            return {
                "removed": False,
                "reason": "missing_object_center",
            }

        last_seen_frame_index = self._resolve_last_seen_frame_index(stable_object, track_history.get(object_id))
        if last_seen_frame_index is None:
            return {
                "removed": False,
                "lifecycle_state": "active",
                "reason": "missing_last_seen_frame_index",
            }

        observation_positions = self._collect_object_observation_positions(
            stable_object=stable_object,
            payloads=payloads,
            track_history_entries=track_history.get(object_id),
        )
        visible_misses = []
        visibility_opportunity_count = 0
        matched_revisit_count = 0
        total_visible_miss_count = 0
        consecutive_visible_miss_count = 0
        max_visible_miss_streak = 0
        last_confirmed_frame_index = int(last_seen_frame_index)
        last_confirmed_timestamp = stable_object.get("object3d_last_seen_timestamp")
        last_confirmed_time_s = stable_object.get("object3d_last_seen_time_s")
        for frame_idx in range(int(last_seen_frame_index) + 1, len(payloads)):
            payload = payloads[frame_idx]
            visibility = self._evaluate_object_visibility_opportunity(
                payload=payload,
                object_center=object_center,
                observation_positions=observation_positions,
                max_observation_distance_m=max_observation_distance_m,
                position_tolerance_m=position_tolerance_m,
                fov_margin_deg=fov_margin_deg,
            )
            if not visibility.get("eligible"):
                continue

            visibility_opportunity_count += 1
            match_info = self._find_matching_detection(
                payload=payload,
                stable_object=stable_object,
                object_center=object_center,
                match_distance_m=match_distance_m,
            )
            if match_info is not None:
                matched_revisit_count += 1
                consecutive_visible_miss_count = 0
                last_confirmed_frame_index = int(frame_idx)
                last_confirmed_timestamp = payload.get("timestamp")
                last_confirmed_time_s = FrameSemanticPayloadBuilder._timestamp_to_seconds(last_confirmed_timestamp)
                camera_position = self._payload_position_array(payload)
                if camera_position is not None:
                    observation_positions.append(camera_position)
                continue

            total_visible_miss_count += 1
            consecutive_visible_miss_count += 1
            max_visible_miss_streak = max(max_visible_miss_streak, consecutive_visible_miss_count)
            visible_misses.append(
                {
                    "frame_index": int(frame_idx),
                    "timestamp": payload.get("timestamp"),
                    "camera_distance_m": visibility.get("camera_distance_m"),
                    "horizontal_angle_deg": visibility.get("horizontal_angle_deg"),
                    "vertical_angle_deg": visibility.get("vertical_angle_deg"),
                    "revisit_distance_m": visibility.get("revisit_distance_m"),
                    "visible_miss_streak": int(consecutive_visible_miss_count),
                }
            )
            if consecutive_visible_miss_count >= min_visible_misses:
                return {
                    "removed": True,
                    "lifecycle_state": "removed_after_disappearance_filter",
                    "reason": "missing_after_revisit_visibility_check",
                    "last_seen_frame_index": int(last_seen_frame_index),
                    "last_seen_timestamp": stable_object.get("object3d_last_seen_timestamp"),
                    "last_confirmed_frame_index": last_confirmed_frame_index,
                    "last_confirmed_timestamp": last_confirmed_timestamp,
                    "last_confirmed_time_s": last_confirmed_time_s,
                    "visibility_opportunity_count": visibility_opportunity_count,
                    "visible_miss_count": total_visible_miss_count,
                    "consecutive_visible_miss_count": consecutive_visible_miss_count,
                    "max_visible_miss_streak": max_visible_miss_streak,
                    "matched_revisit_count": matched_revisit_count,
                    "missed_frames": visible_misses,
                }

        lifecycle_state = "suspected_missing" if consecutive_visible_miss_count > 0 else "active"
        if matched_revisit_count > 0 and consecutive_visible_miss_count == 0:
            reason = "reconfirmed_during_revisit_checks"
        elif consecutive_visible_miss_count > 0:
            reason = "recent_visible_revisit_misses_below_threshold"
        else:
            reason = "no_later_visibility_opportunity_or_no_miss"
        return {
            "removed": False,
            "lifecycle_state": lifecycle_state,
            "reason": reason,
            "last_seen_frame_index": int(last_seen_frame_index),
            "last_seen_timestamp": stable_object.get("object3d_last_seen_timestamp"),
            "last_confirmed_frame_index": last_confirmed_frame_index,
            "last_confirmed_timestamp": last_confirmed_timestamp,
            "last_confirmed_time_s": last_confirmed_time_s,
            "visibility_opportunity_count": visibility_opportunity_count,
            "visible_miss_count": total_visible_miss_count,
            "consecutive_visible_miss_count": consecutive_visible_miss_count,
            "max_visible_miss_streak": max_visible_miss_streak,
            "matched_revisit_count": matched_revisit_count,
            "missed_frames": visible_misses,
        }

    def _evaluate_object_visibility_opportunity(
        self,
        payload,
        object_center,
        observation_positions,
        max_observation_distance_m,
        position_tolerance_m,
        fov_margin_deg,
    ):
        camera_position = self._payload_position_array(payload)
        camera_rotation = self._payload_rotation_quaternion(payload)
        if camera_position is None or camera_rotation is None:
            return {"eligible": False, "reason": "missing_camera_pose"}

        camera_pose = self.payload_builder._build_object3d_pose_matrix(camera_position, camera_rotation)
        object_h = np.ones((4,), dtype=np.float64)
        object_h[:3] = object_center
        local_point = np.linalg.inv(camera_pose) @ object_h
        forward_z = float(local_point[2])
        if forward_z <= 0.15:
            return {"eligible": False, "reason": "behind_camera"}
        if forward_z > max_observation_distance_m:
            return {
                "eligible": False,
                "reason": "outside_object3d_depth_gate",
                "forward_depth_m": forward_z,
            }

        camera_distance_m = float(np.linalg.norm(local_point[:3]))
        if camera_distance_m > max_observation_distance_m:
            return {"eligible": False, "reason": "too_far"}

        horizontal_angle_deg = abs(float(np.degrees(np.arctan2(local_point[0], forward_z))))
        vertical_angle_deg = abs(float(np.degrees(np.arctan2(local_point[1], forward_z))))
        horizontal_limit = max(5.0, float(self.cfg.hfov) * 0.5 - fov_margin_deg)
        vertical_limit = max(5.0, float(self.cfg.vfov) * 0.5 - fov_margin_deg)
        if horizontal_angle_deg > horizontal_limit or vertical_angle_deg > vertical_limit:
            return {"eligible": False, "reason": "outside_safe_fov"}

        revisit_distance_m = None
        if observation_positions:
            revisit_distance_m = min(
                float(np.linalg.norm(camera_position - observed_position))
                for observed_position in observation_positions
            )
            if revisit_distance_m > position_tolerance_m:
                return {"eligible": False, "reason": "not_close_to_previous_observation_pose"}

        return {
            "eligible": True,
            "camera_distance_m": camera_distance_m,
            "forward_depth_m": forward_z,
            "horizontal_angle_deg": horizontal_angle_deg,
            "vertical_angle_deg": vertical_angle_deg,
            "revisit_distance_m": revisit_distance_m,
        }

    @staticmethod
    def _extract_object3d_center(stable_object):
        candidates = (
            stable_object.get("bbox_3d_center"),
            stable_object.get("centroid"),
            stable_object.get("global_position"),
        )
        for candidate in candidates:
            try:
                center = np.asarray(candidate, dtype=np.float64).reshape(3)
            except (TypeError, ValueError):
                continue
            if np.isfinite(center).all():
                return center
        return None

    @staticmethod
    def _payload_position_array(payload):
        try:
            position = np.asarray(payload.get("position"), dtype=np.float64).reshape(3)
        except (AttributeError, TypeError, ValueError):
            return None
        if not np.isfinite(position).all():
            return None
        return position

    @staticmethod
    def _payload_rotation_quaternion(payload):
        rotation = payload.get("rotation") if isinstance(payload, dict) else None
        if not isinstance(rotation, list) or len(rotation) < 4:
            return None
        try:
            return quaternion.quaternion(
                float(rotation[0]),
                float(rotation[1]),
                float(rotation[2]),
                float(rotation[3]),
            )
        except (TypeError, ValueError):
            return None

    @classmethod
    def _resolve_last_seen_frame_index(cls, stable_object, track_history_entries):
        frame_candidates = []
        try:
            if stable_object.get("object3d_last_seen_frame_index") is not None:
                frame_candidates.append(int(stable_object.get("object3d_last_seen_frame_index")))
        except (TypeError, ValueError):
            pass

        for item in track_history_entries if isinstance(track_history_entries, list) else []:
            if not isinstance(item, dict):
                continue
            try:
                frame_candidates.append(int(item.get("frame_index")))
            except (TypeError, ValueError):
                continue

        if not frame_candidates:
            return None
        return max(frame_candidates)

    @classmethod
    def _collect_object_observation_positions(cls, stable_object, payloads, track_history_entries):
        frame_indexes = set()
        for item in track_history_entries if isinstance(track_history_entries, list) else []:
            if not isinstance(item, dict):
                continue
            try:
                frame_indexes.add(int(item.get("frame_index")))
            except (TypeError, ValueError):
                continue

        if not frame_indexes:
            for key in ("object3d_first_seen_frame_index", "object3d_last_seen_frame_index"):
                try:
                    if stable_object.get(key) is not None:
                        frame_indexes.add(int(stable_object.get(key)))
                except (TypeError, ValueError):
                    continue

        positions = []
        for frame_index in sorted(frame_indexes):
            if frame_index < 0 or frame_index >= len(payloads):
                continue
            payload = payloads[frame_index]
            position = cls._payload_position_array(payload)
            if position is not None:
                positions.append(position)
        return positions

    @staticmethod
    def _normalize_class_name(value):
        return str(value or "").strip().lower()

    @classmethod
    def _find_matching_detection(cls, payload, stable_object, object_center, match_distance_m):
        detections = payload.get("detections") if isinstance(payload, dict) else None
        if not isinstance(detections, list):
            return None

        object_id = str(stable_object.get("object_id") or "")
        class_name = cls._normalize_class_name(
            stable_object.get("dominant_class_name") or stable_object.get("class_name")
        )
        best_spatial_match = None
        for detection in detections:
            if not isinstance(detection, dict):
                continue

            detection_track_id = str(detection.get("object3d_track_id") or "")
            if object_id and detection_track_id == object_id:
                return {
                    "detection": detection,
                    "match_mode": "track_id",
                    "distance_m": 0.0,
                }

            detection_class_name = cls._normalize_class_name(detection.get("class_name"))
            if class_name and detection_class_name != class_name:
                continue

            detection_center = cls._extract_detection_center(detection)
            if detection_center is None:
                continue

            distance_m = float(np.linalg.norm(detection_center - object_center))
            if distance_m <= match_distance_m and (
                best_spatial_match is None or distance_m < float(best_spatial_match["distance_m"])
            ):
                best_spatial_match = {
                    "detection": detection,
                    "match_mode": "class_spatial",
                    "distance_m": distance_m,
                }
        return best_spatial_match

    @classmethod
    def _resolve_stable_object_for_detection(
        cls,
        detection,
        stable_by_id,
        stable_objects,
        used_stable_object_ids,
        match_distance_m,
    ):
        track_id = str(detection.get("object3d_track_id") or "")
        stable_object = stable_by_id.get(track_id)
        if stable_object is not None:
            stable_object_id = str(stable_object.get("object_id") or "")
            if stable_object_id and stable_object_id not in used_stable_object_ids:
                return stable_object, {
                    "match_mode": "track_id",
                    "match_distance_m": 0.0,
                    "original_track_id": track_id or None,
                }

        detection_center = cls._extract_detection_center(detection)
        if detection_center is None:
            return None, None

        detection_class_name = cls._normalize_class_name(detection.get("class_name"))
        best_match = None
        for candidate in stable_objects if isinstance(stable_objects, list) else []:
            if not isinstance(candidate, dict):
                continue
            candidate_id = str(candidate.get("object_id") or "")
            if candidate_id in used_stable_object_ids:
                continue
            candidate_class_name = cls._normalize_class_name(
                candidate.get("dominant_class_name") or candidate.get("class_name")
            )
            if detection_class_name and candidate_class_name != detection_class_name:
                continue
            candidate_center = cls._extract_object3d_center(candidate)
            if candidate_center is None:
                continue
            distance_m = float(np.linalg.norm(detection_center - candidate_center))
            if distance_m > match_distance_m:
                continue
            current_best_distance = None
            if isinstance(best_match, dict):
                current_best_distance = (
                    best_match.get("match_info", {}).get("match_distance_m")
                    if isinstance(best_match.get("match_info"), dict)
                    else None
                )
            if current_best_distance is None or distance_m < float(current_best_distance):
                best_match = {
                    "stable_object": candidate,
                    "match_info": {
                        "match_mode": "class_spatial",
                        "match_distance_m": distance_m,
                        "original_track_id": track_id or None,
                    },
                }

        if best_match is None:
            return None, None
        return best_match["stable_object"], best_match["match_info"]

    @staticmethod
    def _extract_detection_center(detection):
        for key in ("bbox_3d_center", "global_position"):
            try:
                center = np.asarray(detection.get(key), dtype=np.float64).reshape(3)
            except (AttributeError, TypeError, ValueError):
                continue
            if np.isfinite(center).all():
                return center
        return None

    @staticmethod
    def _count_payload_detections(payloads):
        total = 0
        for payload in payloads:
            detections = payload.get("detections") if isinstance(payload, dict) else None
            if isinstance(detections, list):
                total += sum(1 for item in detections if isinstance(item, dict))
        return total

    @staticmethod
    def _apply_stable_object3d_to_detection(detection, stable_object, match_info=None):
        updated = copy.deepcopy(detection)
        centroid = stable_object.get("centroid") or stable_object.get("bbox_3d_center")
        if isinstance(centroid, list) and len(centroid) >= 3:
            updated["global_position"] = [float(centroid[0]), float(centroid[1]), float(centroid[2])]
        if stable_object.get("dominant_class_name") or stable_object.get("class_name"):
            updated["class_name"] = str(stable_object.get("dominant_class_name") or stable_object.get("class_name"))
        updated["object3d_track_id"] = stable_object.get("object_id")
        updated["global_position_method"] = "object3d_engine_filtered_3dbox"
        for key in (
            "bbox_3d_center",
            "bbox_3d_extent",
            "bbox_3d_corners",
            "global_points_sample",
            "motion_state",
            "motion_reason",
            "motion_metrics",
            "object3d_first_seen_frame_index",
            "object3d_last_seen_frame_index",
            "object3d_observed_frame_count",
            "object3d_observed_frame_span",
            "object3d_missing_frame_count",
            "object3d_observation_rate",
            "object3d_first_seen_timestamp",
            "object3d_last_seen_timestamp",
            "object3d_first_seen_time_s",
            "object3d_last_seen_time_s",
            "object3d_observed_duration_s",
            "object3d_lifecycle_state",
            "object3d_last_confirmed_frame_index",
            "object3d_last_confirmed_timestamp",
            "object3d_last_confirmed_time_s",
            "object3d_disappearance_visibility_check_count",
            "object3d_disappearance_visible_miss_count",
            "object3d_disappearance_consecutive_visible_miss_count",
            "object3d_disappearance_max_visible_miss_streak",
            "object3d_revisit_confirmed_count",
            "object3d_removal_frame_index",
            "object3d_removal_timestamp",
            "object3d_removal_time_s",
            "object3d_removal_reason",
        ):
            if stable_object.get(key) is not None:
                updated[key] = stable_object.get(key)
        updated["support_point_count"] = stable_object.get(
            "total_support_points",
            updated.get("support_point_count"),
        )
        updated["estimation_metadata"] = {
            **(updated.get("estimation_metadata") if isinstance(updated.get("estimation_metadata"), dict) else {}),
            "nav_graph_source": "filtered_object3d_tracking_object",
            "stable_object_id": stable_object.get("object_id"),
            "stable_observations_count": stable_object.get("observations_count"),
            "longest_consecutive_frame_run": stable_object.get("longest_consecutive_frame_run"),
            "motion_state": stable_object.get("motion_state"),
            "motion_reason": stable_object.get("motion_reason"),
            "motion_metrics": stable_object.get("motion_metrics"),
            "object3d_first_seen_timestamp": stable_object.get("object3d_first_seen_timestamp"),
            "object3d_last_seen_timestamp": stable_object.get("object3d_last_seen_timestamp"),
            "object3d_observed_duration_s": stable_object.get("object3d_observed_duration_s"),
            "object3d_lifecycle_state": stable_object.get("object3d_lifecycle_state"),
            "object3d_removal_frame_index": stable_object.get("object3d_removal_frame_index"),
            "object3d_removal_timestamp": stable_object.get("object3d_removal_timestamp"),
            "object3d_removal_time_s": stable_object.get("object3d_removal_time_s"),
            "object3d_removal_reason": stable_object.get("object3d_removal_reason"),
            "stable_match_mode": match_info.get("match_mode") if isinstance(match_info, dict) else None,
            "stable_match_distance_m": match_info.get("match_distance_m") if isinstance(match_info, dict) else None,
            "stable_match_original_track_id": match_info.get("original_track_id") if isinstance(match_info, dict) else None,
        }
        return updated

    def _prune_removed_disappearance_objects_from_nav_graph(self, tracking_summary):
        removed_objects = tracking_summary.get("removed_objects")
        if not isinstance(removed_objects, list):
            return []

        removed_track_ids = {
            str(obj.get("object_id") or "")
            for obj in removed_objects
            if isinstance(obj, dict) and str(obj.get("removal_stage") or "") == "disappearance_filter"
        }
        removed_track_ids.discard("")
        if not removed_track_ids:
            return []

        removed_node_ids = []
        affected_topology_ids = set()
        graph = self.nav_builder.graph
        for node_id, attrs in list(self.nav_builder.nav.object_nodes(graph)):
            track_id = str(attrs.get("object3d_track_id") or "")
            if track_id not in removed_track_ids:
                continue
            topology_id = str(attrs.get("topology_id") or "")
            if topology_id:
                affected_topology_ids.add(topology_id)
            graph.remove_node(node_id)
            removed_node_ids.append(str(node_id))

        for topology_id in sorted(affected_topology_ids):
            if graph.has_node(topology_id):
                self.nav_builder.nav.refresh_topology_object_summary(graph, topology_id)
        return removed_node_ids

    def process_frame(self, frame_idx):
        ts_cam = self.track_csv["front_cam_ts"].iloc[frame_idx]
        ts_lidar = self.track_csv["lidar_ts"].iloc[frame_idx]
        frame_timestamp = self._frame_timestamp(frame_idx, fallback=ts_cam)

        position, rotation = position_get(self.track_csv, frame_idx)
        test_img_file = os.path.join(self.cfg.data_base_path, "front_cam", f"{ts_cam}.png")
        test_cloud_file = os.path.join(self.cfg.data_base_path, "lidar", f"{ts_lidar}.bin")

        test_img = imread(test_img_file)
        if test_img is None:
            raise FileNotFoundError(f"front camera image not found or unreadable: {test_img_file}")
        test_img = test_img[:, :, :3]

        test_cloud = np.fromfile(test_cloud_file, dtype=np.float32).reshape((-1, 4))[:, :-1]
        test_cloud = test_cloud[test_cloud == test_cloud].reshape((-1, 3))

        yolo_results = self.model(test_img, device=self.yolo_device)
        result = yolo_results[0]

        if self.recorded_depth_available:
            depth, depth_path = self._load_recorded_depth(frame_idx, ts_cam)
            depth_source = f"recorded_depth:{depth_path}"
        else:
            depth, _, _, _ = self.depth_pipeline.get_depth_with_lidar(test_img, test_cloud[:, :3])
            depth_source = "depth_anything_lidar_fallback"

        sample = self.db_dataset[frame_idx]
        query_descriptor = self.descriptors[frame_idx].astype("float32")

        gt_pose = sample["pose"]
        gt_location = gt_pose[:3]
        if hasattr(gt_location, "cpu"):
            gt_location = gt_location.cpu().numpy()
        gt_location = gt_location.astype("float32")

        nearest_node_indices, distances = self.topomap.locate_node(query_descriptor, gt_location, k=1)

        if nearest_node_indices[0] == -1 and len(self.current_node.current_feature) == 0:
            self.creating_new_node = True
        elif nearest_node_indices[0] == -1:
            if self.current_node.leaving_current(query_descriptor, gt_location):
                self.current_node.current_to_graph(self.topomap)
                self.creating_new_node = True
        else:
            nearest_index = nearest_node_indices[0]
            feature_distance = distances[0]
            history_node_center = self.topomap.get_node_center(nearest_index)
            coord_distance = np.linalg.norm(gt_location - history_node_center)

            if (
                feature_distance <= self.cfg.feature_distance_threshold
                and coord_distance <= self.cfg.coord_distance_threshold
            ):
                self.current_node.current_to_graph(self.topomap)
                self.current_node.load_history(nearest_index, self.topomap)
            elif (
                feature_distance > self.cfg.feature_distance_threshold
                and coord_distance > self.cfg.coord_distance_threshold
            ):
                if self.current_node.leaving_current(query_descriptor, gt_location):
                    self.current_node.current_to_graph(self.topomap)
                    self.creating_new_node = True

        if self.creating_new_node:
            self.current_node.current_id = self.topomap.get_num_nodes()
            self.creating_new_node = False

        self.current_node.update_current(query_descriptor, gt_location)
        node = f"node_{self.current_node.current_id}"

        return self.payload_builder.build(
            result=result,
            rgb=test_img,
            depth_map=depth,
            frame_idx=frame_idx,
            position=position,
            rotation=rotation,
            test_img_file=test_img_file,
            node_label=node,
            depth_source=depth_source,
            frame_timestamp=frame_timestamp,
        )

    def _frame_timestamp(self, frame_idx, fallback=None):
        if self.track_csv is not None and "timestamp" in self.track_csv.columns:
            timestamp = self.track_csv["timestamp"].iloc[frame_idx]
            normalized = FrameSemanticPayloadBuilder._normalize_timestamp_value(timestamp)
            if normalized is not None:
                return normalized
        return FrameSemanticPayloadBuilder._normalize_timestamp_value(fallback)

    def _recorded_depth_dir(self):
        return Path(str(self.cfg.data_base_path)) / str(self.cfg.recorded_depth_dir_name)

    def _has_recorded_depth(self):
        if not bool(self.cfg.get("prefer_recorded_depth", True)):
            return False
        depth_dir = self._recorded_depth_dir()
        if not depth_dir.exists():
            return False
        if self.track_csv is None or self.track_csv.empty:
            return any(depth_dir.glob("*.png"))

        sample_count = min(int(self.track_csv.shape[0]), 10)
        for frame_idx in range(sample_count):
            ts_cam = self.track_csv["front_cam_ts"].iloc[frame_idx]
            if self._recorded_depth_path(frame_idx, ts_cam).exists():
                return True
        return False

    def _recorded_depth_path(self, frame_idx, ts_cam):
        depth_dir = self._recorded_depth_dir()
        candidates = []
        if "depth_ts" in self.track_csv.columns:
            depth_ts = str(self.track_csv["depth_ts"].iloc[frame_idx]).strip()
            if depth_ts and depth_ts.lower() != "nan":
                candidates.append(depth_ts)
        candidates.append(str(ts_cam))

        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            path = depth_dir / f"{candidate}.png"
            if path.exists():
                return path
        return depth_dir / f"{ts_cam}.png"

    def _load_recorded_depth(self, frame_idx, ts_cam):
        depth_path = self._recorded_depth_path(frame_idx, ts_cam)
        if not depth_path.exists():
            raise FileNotFoundError(f"recorded depth png not found: {depth_path}")
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(f"recorded depth png unreadable: {depth_path}")
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[:, :, 0]
        if np.issubdtype(depth_raw.dtype, np.integer):
            depth_m = depth_raw.astype(np.float32) / 1000.0
        else:
            depth_m = depth_raw.astype(np.float32)
        return depth_m, str(depth_path)

    def _ensure_depth_pipeline(self):
        if self.depth_pipeline is not None:
            return
        from opr.pipelines.depth_estimation import DepthEstimationPipeline

        depth_model = load_depth_model(self.cfg.depth_model_path, self.device, self.cfg.depth_anything_root)
        tf_matrix, camera_matrix, _ = set_tf_matrix()
        self.depth_pipeline = DepthEstimationPipeline(
            depth_model,
            model_type="DepthAnything",
            align_type="average",
            mode="indoor",
        )
        self.depth_pipeline.set_camera_matrix(camera_matrix)
        self.depth_pipeline.set_lidar_to_camera_transform(tf_matrix)


def main():
    cfg = parse_args()
    if cfg.mode == "build":
        SemanticTopomapFusionPipeline(cfg).run()
    elif cfg.mode == "visualize":
        GraphRuntimeManager(cfg).visualize()
    elif cfg.mode == "navigate":
        GraphRuntimeManager(cfg).navigate()
    else:
        raise ValueError(f"Unsupported mode: {cfg.mode}")


if __name__ == "__main__":
    main()
