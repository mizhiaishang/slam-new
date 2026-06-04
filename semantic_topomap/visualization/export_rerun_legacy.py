#!/usr/bin/env python3
"""Export an offline Rerun replay for cuVSLAM point cloud + semantic topology construction."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


RESULT_DIR = Path("/home/zyf/code_made/result/semantic_graphs_dataset_test3_foreground_filter_full")
DATASET_ROOT = Path("/home/zyf/code_made/core_content/dataset_test3_cuvslam")
CUVSLAM_DIR = Path("/home/zyf/imu/cuvslam/results/dataset_test3_zed_svo2_pointcloud_replay")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--cuvslam-dir", type=Path, default=CUVSLAM_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--cloud-points", type=int, default=12000)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--include-removed", action="store_true")
    parser.add_argument("--grow-pointcloud", action="store_true", help="Accumulate cuVSLAM landmark snapshots over time")
    parser.add_argument(
        "--show-full-cuvslam-trajectory",
        action="store_true",
        help="Also show the complete cuVSLAM trajectory as a static global reference",
    )
    parser.add_argument("--max-growing-cloud-points", type=int, default=30000)
    parser.add_argument("--cloud-color", default="255,255,255", help="RGB color for cuVSLAM points")
    parser.add_argument("--cloud-radius", type=float, default=0.035, help="Radius for cuVSLAM point visualization")
    parser.add_argument("--cloud-secondary-color", default="", help="Optional second RGB point color for contrast")
    parser.add_argument("--cloud-secondary-radius", type=float, default=0.018)
    parser.add_argument("--show-hierarchy", action="store_true", help="Show topology and semantic-region nodes above the map")
    parser.add_argument("--black-background", action="store_true", help="Use a solid black Rerun Spatial3D background")
    parser.add_argument("--object-label-mode", choices=["none", "centers", "boxes", "both"], default="none")
    parser.add_argument("--show-removed-labels", action="store_true")
    parser.add_argument("--hierarchy-label-color-mode", choices=["bright", "class"], default="bright")
    parser.add_argument("--topology-z", type=float, default=4.5)
    parser.add_argument("--semantic-level-z-step", type=float, default=2.0)
    return parser.parse_args()


def vec3(value) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        out = [float(value[0]), float(value[1]), float(value[2])]
    except Exception:
        return None
    return out if all(math.isfinite(v) for v in out) else None


def zed_to_code(point) -> list[float]:
    x, y, z = [float(v) for v in point]
    return [x, z, -y]


def sample_list(items: list, limit: int) -> list:
    if limit <= 0 or len(items) <= limit:
        return items
    step = max(1, len(items) // limit)
    return items[::step][:limit]


def read_ply(path: Path, limit: int) -> list[list[float]]:
    points = []
    if not path.exists():
        return points
    header = True
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if header:
                if line == "end_header":
                    header = False
                continue
            parts = line.split()
            if len(parts) >= 3:
                p = vec3(parts[:3])
                if p is not None:
                    points.append(zed_to_code(p))
    return sample_list(points, limit)


def read_cuvslam_trajectory(path: Path) -> list[list[float]]:
    points = []
    if not path.exists():
        return points
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                points.append(zed_to_code([row["tx"], row["ty"], row["tz"]]))
            except Exception:
                pass
    return points


def read_cuvslam_poses(path: Path) -> dict[int, tuple[np.ndarray, Rotation]]:
    poses: dict[int, tuple[np.ndarray, Rotation]] = {}
    if not path.exists():
        return poses
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                frame_id = int(row["frame_id"])
                position = np.array(zed_to_code([row["tx"], row["ty"], row["tz"]]), dtype=np.float64)
                quat = np.array([float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])], dtype=np.float64)
                basis = np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.0, -1.0, 0.0],
                    ],
                    dtype=np.float64,
                )
                rotation = Rotation.from_matrix(basis @ Rotation.from_quat(quat).as_matrix() @ basis.T)
                poses[frame_id] = (position, rotation)
            except Exception:
                continue
    return poses


def read_replay_snapshots(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("snapshots", []))


def color_for_label(label: str) -> list[int]:
    h = 0
    for ch in label:
        h = (h * 31 + ord(ch)) % 360
    # Small HSV to RGB helper with vivid but not neon colors.
    c = 0.82
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = 0.16
    if h < 60:
        rgb = (c, x, 0)
    elif h < 120:
        rgb = (x, c, 0)
    elif h < 180:
        rgb = (0, c, x)
    elif h < 240:
        rgb = (0, x, c)
    elif h < 300:
        rgb = (x, 0, c)
    else:
        rgb = (c, 0, x)
    return [int((v + m) * 255) for v in rgb]


def parse_rgb(text: str) -> list[int]:
    try:
        values = [int(float(v.strip())) for v in text.split(",")]
    except Exception:
        return [255, 255, 255]
    if len(values) != 3:
        return [255, 255, 255]
    return [max(0, min(255, v)) for v in values]


def elevated(point: list[float], z: float) -> list[float]:
    return [float(point[0]), float(point[1]), float(z)]


def draw_boxes_on_image(image: np.ndarray, observations: list[dict]) -> np.ndarray:
    out = image.copy()
    for obs in observations:
        bbox = obs.get("bbox") or {}
        try:
            x1, y1, x2, y2 = [int(round(float(bbox[k]))) for k in ["x1", "y1", "x2", "y2"]]
        except Exception:
            continue
        label = str(obs.get("class_name", "object"))
        color = color_for_label(label)
        cv2.rectangle(out, (x1, y1), (x2, y2), tuple(int(c) for c in color[::-1]), 2)
        text = f"{label} {float(obs.get('confidence') or 0.0):.2f}"
        cv2.putText(out, text, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, tuple(int(c) for c in color[::-1]), 2)
    return out


def normalize_object(item: dict, removed: bool = False) -> dict | None:
    center = vec3(item.get("bbox_3d_center")) or vec3(item.get("centroid"))
    extent = vec3(item.get("bbox_3d_extent"))
    if center is None or extent is None:
        return None
    return {
        "id": str(item.get("object_id", "")),
        "track": str(item.get("object3d_track_id") or item.get("object_id", "")),
        "class": str(item.get("class_name") or item.get("dominant_class_name") or "object"),
        "center": center,
        "extent": extent,
        "obs": int(item.get("observation_count") or item.get("observations_count") or 0),
        "wp": int(item.get("waypoint_count") or 0),
        "conf": float(item.get("mean_confidence") or 0.0),
        "first": item.get("object3d_first_seen_frame_index"),
        "last": item.get("object3d_last_seen_frame_index"),
        "life": str(item.get("object3d_lifecycle_state", "")),
        "removed": bool(removed),
    }


def load_payload(result_dir: Path) -> tuple[dict, dict, list[dict], list[dict], list[dict]]:
    contents = json.loads((result_dir / "nav_graph_contents.json").read_text(encoding="utf-8"))
    tracking = json.loads((result_dir / "object3d_tracking_summary.json").read_text(encoding="utf-8"))
    stats = json.loads((result_dir / "nav_graph_stats.json").read_text(encoding="utf-8"))
    objects = [obj for obj in (normalize_object(item) for item in contents.get("objects", [])) if obj]
    removed = [obj for obj in (normalize_object(item, True) for item in tracking.get("removed_objects", [])) if obj]
    return contents, stats, objects, removed, tracking.get("objects", [])


def build_hierarchy(contents: dict, topology_z: float, semantic_step: float) -> tuple[dict, dict, list[dict], dict, dict]:
    waypoints_by_id = {str(w.get("waypoint_id")): w for w in contents.get("waypoints", [])}
    objects_by_id = {str(o.get("object_id")): o for o in contents.get("objects", [])}
    topology_nodes = {}
    semantic_nodes = {}
    hierarchy_edges = []
    waypoint_positions: dict[str, list[float]] = {}
    object_positions: dict[str, list[float]] = {}

    for waypoint_id, waypoint in waypoints_by_id.items():
        pos = vec3(waypoint.get("position"))
        if pos is not None:
            waypoint_positions[waypoint_id] = pos

    for object_id, obj in objects_by_id.items():
        pos = vec3(obj.get("bbox_3d_center")) or vec3(obj.get("centroid"))
        if pos is not None:
            object_positions[object_id] = pos

    for topo in contents.get("topologies", []):
        center = vec3(topo.get("centroid"))
        if center is None:
            continue
        node = {
            "id": str(topo.get("topology_id", "")),
            "label": str(topo.get("label") or topo.get("topology_key") or topo.get("topology_id")),
            "position": elevated(center, topology_z),
            "base_position": center,
            "waypoint_ids": [str(v) for v in topo.get("waypoint_ids", [])],
            "object_ids": [str(v) for v in topo.get("object_ids", [])],
            "wp_count": int(topo.get("waypoint_count") or 0),
            "object_count": int(topo.get("object_count") or 0),
        }
        topology_nodes[node["id"]] = node

    for region in contents.get("semantic_regions", []):
        center = vec3(region.get("centroid"))
        if center is None:
            continue
        level = int(region.get("semantic_level") or 2)
        z = topology_z + max(1, level - 1) * semantic_step
        node = {
            "id": str(region.get("semantic_region_id", "")),
            "label": str(region.get("label") or region.get("region_name") or region.get("semantic_region_id")),
            "level": level,
            "position": elevated(center, z),
            "base_position": center,
            "child_ids": [str(v) for v in region.get("child_ids", [])],
            "topology_ids": [str(v) for v in region.get("topology_ids", [])],
            "parent": str(region.get("parent_semantic_region_id") or ""),
            "wp_count": int(region.get("waypoint_count") or 0),
            "object_count": int(region.get("object_count") or 0),
        }
        semantic_nodes[node["id"]] = node

    for topo in topology_nodes.values():
        for waypoint_id in topo["waypoint_ids"]:
            pos = waypoint_positions.get(waypoint_id)
            if pos is not None:
                hierarchy_edges.append({"a": topo["position"], "b": pos, "kind": "topology_waypoint", "id": f'{topo["id"]}->{waypoint_id}', "topology_id": topo["id"], "waypoint_id": waypoint_id})
        for object_id in topo["object_ids"]:
            pos = object_positions.get(object_id)
            if pos is not None:
                hierarchy_edges.append({"a": topo["position"], "b": pos, "kind": "topology_object", "id": f'{topo["id"]}->{object_id}', "topology_id": topo["id"], "object_id": object_id})

    for region in semantic_nodes.values():
        for child_id in region["child_ids"]:
            child = semantic_nodes.get(child_id) or topology_nodes.get(child_id)
            if child:
                hierarchy_edges.append({"a": region["position"], "b": child["position"], "kind": "semantic_child", "id": f'{region["id"]}->{child_id}', "semantic_id": region["id"], "child_id": child_id})
        for topo_id in region["topology_ids"]:
            child = topology_nodes.get(topo_id)
            if child:
                hierarchy_edges.append({"a": region["position"], "b": child["position"], "kind": "semantic_topology", "id": f'{region["id"]}->{topo_id}', "semantic_id": region["id"], "topology_id": topo_id})

    return topology_nodes, semantic_nodes, hierarchy_edges, waypoint_positions, object_positions


def build_frame_records(dataset_root: Path, contents: dict, stats: dict, max_frames: int) -> list[dict]:
    track_path = dataset_root / "track.csv"
    waypoints_by_frame = {}
    for wp in contents.get("waypoints", []):
        try:
            frame = int(str(wp.get("sort_key", "")).split(".")[0])
        except Exception:
            continue
        waypoints_by_frame[frame] = wp

    kept = set(int(v) for v in stats.get("waypoint_sampling", {}).get("kept_frame_indices", []))
    records = []
    with track_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if max_frames > 0:
        rows = rows[:max_frames]
    for i, row in enumerate(rows):
        timestamp = str(row.get("front_cam_ts") or row.get("timestamp") or "")
        pos = [float(row["tx"]), float(row["ty"]), float(row["tz"])]
        records.append(
            {
                "frame": i,
                "timestamp": timestamp,
                "image": str(dataset_root / "front_cam" / f"{timestamp}.png"),
                "position": pos,
                "slam_frame_id": int(float(row["slam_frame_id"])) if row.get("slam_frame_id") else None,
                "is_waypoint": i in kept,
                "waypoint": waypoints_by_frame.get(i),
            }
        )
    return records


def observations_for_frame(record: dict) -> list[dict]:
    waypoint = record.get("waypoint") or {}
    return list(waypoint.get("observations", []) or [])


def resize_image(path: str, width: int) -> np.ndarray | None:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if width > 0 and image.shape[1] > width:
        scale = width / image.shape[1]
        image = cv2.resize(image, (width, int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    return image


def log_text(rr, path: str, text: str) -> None:
    if hasattr(rr, "TextDocument"):
        rr.log(path, rr.TextDocument(text))


def main() -> int:
    args = parse_args()
    import rerun as rr
    import rerun.blueprint as rrb

    result_dir = args.result_dir.resolve()
    dataset_root = args.dataset_root.resolve()
    cuvslam_dir = args.cuvslam_dir.resolve()
    output = args.output.resolve() if args.output else result_dir / "online_semantic_build_replay.rrd"

    contents, stats, objects, removed_objects, stable_objects_raw = load_payload(result_dir)
    records = build_frame_records(dataset_root, contents, stats, args.max_frames)
    cloud = read_ply(cuvslam_dir / "final_landmarks.ply", args.cloud_points)
    cuv_traj = read_cuvslam_trajectory(cuvslam_dir / "trajectory.csv")
    cuv_poses = read_cuvslam_poses(cuvslam_dir / "trajectory.csv")
    cuv_snapshots = read_replay_snapshots(cuvslam_dir / "replay_snapshots.json")
    snapshot_by_frame = {int(s.get("frame_id")): s for s in cuv_snapshots if s.get("frame_id") is not None}
    cloud_color = parse_rgb(args.cloud_color)
    cloud_secondary_color = parse_rgb(args.cloud_secondary_color) if args.cloud_secondary_color else None
    topology_nodes, semantic_nodes, hierarchy_edges, waypoint_positions, object_positions = build_hierarchy(
        contents,
        topology_z=float(args.topology_z),
        semantic_step=float(args.semantic_level_z_step),
    )

    blueprint = None
    if args.black_background:
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin="world",
                    name="3D Online Build",
                    background=rrb.Background(color=[0, 0, 0, 255], kind=rrb.BackgroundKind.SolidColor),
                    line_grid=False,
                ),
                rrb.Vertical(
                    rrb.Spatial2DView(origin="camera", name="RGB Detection"),
                    rrb.TextDocumentView(origin="summary", name="Summary"),
                ),
                column_shares=[0.70, 0.30],
            ),
            collapse_panels=False,
        )

    rr.init("dataset_test3_online_semantic_build", spawn=False, default_blueprint=blueprint)
    rr.save(str(output), default_blueprint=blueprint)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    if not args.grow_pointcloud:
        rr.log("world/cuvslam_landmarks", rr.Points3D(cloud, colors=cloud_color, radii=float(args.cloud_radius)), static=True)
        if cloud_secondary_color:
            rr.log(
                "world/cuvslam_landmarks_highlight",
                rr.Points3D(cloud, colors=cloud_secondary_color, radii=float(args.cloud_secondary_radius)),
                static=True,
            )
    if args.show_full_cuvslam_trajectory and cuv_traj:
        rr.log("world/cuvslam_trajectory", rr.LineStrips3D([sample_list(cuv_traj, 2400)], colors=[90, 205, 130], radii=0.018), static=True)

    class_names = sorted({o["class"] for o in objects})
    log_text(
        rr,
        "summary",
        "\n".join(
            [
                "dataset_test3 online semantic build replay",
                f"frames: {len(records)}",
                f"waypoints: {len(contents.get('waypoints', []))}",
                f"object nodes: {len(objects)}",
                f"stable Object3D tracks: {len(stable_objects_raw)}",
                f"classes: {', '.join(class_names)}",
                "coordinates: cuVSLAM points converted with [x,y,z] -> [x,z,-y]",
                f"hierarchy nodes: topology={len(topology_nodes)}, semantic={len(semantic_nodes)}",
            ]
        ),
    )

    active_waypoints: list[list[float]] = []
    active_waypoint_ids: set[str] = set()
    active_object_ids: set[str] = set()
    growing_cloud_by_key: dict[int, list[float]] = {}
    object_by_id = {o["id"]: o for o in objects}
    object_first_frame = {}
    for obj in objects:
        first = obj.get("first")
        if isinstance(first, int):
            object_first_frame[obj["id"]] = first
        else:
            object_first_frame[obj["id"]] = 0

    for record in records:
        frame = int(record["frame"])
        rr.set_time("frame", sequence=frame)
        rr.set_time("timestamp", timestamp=int(record["timestamp"]) / 1e9 if record["timestamp"].isdigit() else frame)

        image = resize_image(record["image"], args.image_width)
        observations = observations_for_frame(record)
        if image is not None:
            scale = image.shape[1] / max(1, cv2.imread(record["image"], cv2.IMREAD_COLOR).shape[1])
            if observations:
                scaled_observations = []
                for obs in observations:
                    item = dict(obs)
                    bbox = dict(obs.get("bbox") or {})
                    for key in ["x1", "x2"]:
                        if key in bbox:
                            bbox[key] = float(bbox[key]) * scale
                    for key in ["y1", "y2"]:
                        if key in bbox:
                            bbox[key] = float(bbox[key]) * scale
                    item["bbox"] = bbox
                    scaled_observations.append(item)
                image = draw_boxes_on_image(image, scaled_observations)
            rr.log("camera/rgb", rr.Image(image).compress(jpeg_quality=75))
            if observations:
                boxes_mins = []
                boxes_sizes = []
                labels = []
                colors = []
                for obs in observations:
                    bbox = obs.get("bbox") or {}
                    try:
                        x1, y1, x2, y2 = [float(bbox[k]) * scale for k in ["x1", "y1", "x2", "y2"]]
                    except Exception:
                        continue
                    boxes_mins.append([x1, y1])
                    boxes_sizes.append([x2 - x1, y2 - y1])
                    label = str(obs.get("class_name", "object"))
                    labels.append(f"{label} {float(obs.get('confidence') or 0.0):.2f}")
                    colors.append(color_for_label(label))
                if boxes_mins:
                    rr.log("camera/detections", rr.Boxes2D(mins=boxes_mins, sizes=boxes_sizes, labels=labels, colors=colors, radii=2.0))
                else:
                    rr.log("camera/detections", rr.Clear(recursive=True))
            else:
                rr.log("camera/detections", rr.Clear(recursive=True))

        pos = record["position"]
        rr.log("world/current_camera", rr.Points3D([pos], colors=[255, 255, 255], radii=0.09))
        rr.log("world/trajectory_so_far", rr.LineStrips3D([[r["position"] for r in records[: frame + 1]]], colors=[118, 216, 159], radii=0.025))

        if args.grow_pointcloud:
            slam_frame_id = record.get("slam_frame_id")
            snapshot = None
            if slam_frame_id is not None:
                # The prepared 241-frame dataset sampled every ~15 SVO frames.
                candidates = [k for k in snapshot_by_frame.keys() if k <= slam_frame_id]
                if candidates:
                    snapshot = snapshot_by_frame[max(candidates)]
            if snapshot is not None:
                frame_id = int(snapshot.get("frame_id"))
                pose = cuv_poses.get(frame_id)
                if pose is not None:
                    pos, rot = pose
                    for lm in snapshot.get("landmarks", []) or []:
                        key = int(lm.get("id", len(growing_cloud_by_key)))
                        local = np.array(zed_to_code(lm.get("xyz", [0, 0, 0])), dtype=np.float64)
                        world = (rot.apply(local) + pos).tolist()
                        growing_cloud_by_key[key] = world
                    if len(growing_cloud_by_key) > args.max_growing_cloud_points:
                        keep_keys = list(growing_cloud_by_key.keys())[-args.max_growing_cloud_points :]
                        growing_cloud_by_key = {k: growing_cloud_by_key[k] for k in keep_keys}
            if growing_cloud_by_key:
                pts = list(growing_cloud_by_key.values())
                rr.log("world/cuvslam_landmarks_growing", rr.Points3D(pts, colors=cloud_color, radii=float(args.cloud_radius)))
                if cloud_secondary_color:
                    rr.log(
                        "world/cuvslam_landmarks_growing_highlight",
                        rr.Points3D(pts, colors=cloud_secondary_color, radii=float(args.cloud_secondary_radius)),
                    )

        wp = record.get("waypoint")
        if wp:
            waypoint_id = str(wp.get("waypoint_id") or "")
            if waypoint_id:
                active_waypoint_ids.add(waypoint_id)
            wp_pos = vec3(wp.get("position"))
            if wp_pos is not None:
                active_waypoints.append(wp_pos)
            for oid in wp.get("object_ids", []) or []:
                active_object_ids.add(str(oid))

        if active_waypoints:
            rr.log("world/waypoints", rr.Points3D(active_waypoints, colors=[86, 160, 255], radii=0.06))
            rr.log("world/waypoint_path", rr.LineStrips3D([active_waypoints], colors=[86, 160, 255], radii=0.018))
        else:
            rr.log("world/waypoints", rr.Clear(recursive=True))

        visible_objects = []
        for obj in objects:
            if obj["id"] in active_object_ids or object_first_frame.get(obj["id"], 0) <= frame:
                visible_objects.append(obj)

        if visible_objects:
            centers = [o["center"] for o in visible_objects]
            sizes = [o["extent"] for o in visible_objects]
            labels = [f'{o["class"]}:{o["id"].replace("object:", "")}' for o in visible_objects]
            colors = [color_for_label(o["class"]) for o in visible_objects]
            rr.log(
                "world/object3d_boxes",
                rr.Boxes3D(
                    centers=centers,
                    sizes=sizes,
                    labels=labels if args.object_label_mode in {"boxes", "both"} else None,
                    show_labels=args.object_label_mode in {"boxes", "both"},
                    colors=colors,
                    radii=0.018,
                ),
            )
            rr.log(
                "world/object3d_centers",
                rr.Points3D(
                    centers,
                    colors=colors,
                    radii=0.045,
                    labels=labels if args.object_label_mode in {"centers", "both"} else None,
                    show_labels=args.object_label_mode in {"centers", "both"},
                ),
            )
        else:
            rr.log("world/object3d_boxes", rr.Clear(recursive=True))
            rr.log("world/object3d_centers", rr.Clear(recursive=True))

        if args.include_removed:
            removed_now = [o for o in removed_objects if isinstance(o.get("last"), int) and o["last"] <= frame]
            if removed_now:
                rr.log(
                    "world/removed_object3d_boxes",
                    rr.Boxes3D(
                        centers=[o["center"] for o in removed_now],
                        sizes=[o["extent"] for o in removed_now],
                        labels=[f'removed {o["class"]}:{o["id"]}' for o in removed_now] if args.show_removed_labels else None,
                        show_labels=bool(args.show_removed_labels),
                        colors=[255, 100, 100],
                        radii=0.012,
                    ),
                )

        if args.show_hierarchy:
            active_topology_ids = set()
            active_semantic_ids = set()
            for r in records[: frame + 1]:
                wp_item = r.get("waypoint") or {}
                topo_id = str(wp_item.get("topology_id") or "")
                if topo_id:
                    active_topology_ids.add(topo_id)
            changed = True
            while changed:
                changed = False
                for node in semantic_nodes.values():
                    should_show = any(child in active_topology_ids or child in active_semantic_ids for child in node.get("child_ids", []))
                    if should_show and node["id"] not in active_semantic_ids:
                        active_semantic_ids.add(node["id"])
                        changed = True
                    parent = node.get("parent")
                    if node["id"] in active_semantic_ids and parent and parent not in active_semantic_ids:
                        active_semantic_ids.add(parent)
                        changed = True

            active_topologies = [topology_nodes[k] for k in sorted(active_topology_ids) if k in topology_nodes]
            active_semantics = [semantic_nodes[k] for k in sorted(active_semantic_ids) if k in semantic_nodes]
            if active_topologies:
                topology_label_color = [255, 255, 255] if args.hierarchy_label_color_mode == "bright" else [255, 220, 95]
                rr.log(
                    "world/hierarchy/topology_nodes",
                    rr.Points3D(
                        [n["position"] for n in active_topologies],
                        colors=[topology_label_color for _ in active_topologies],
                        radii=0.18,
                        labels=[f'TOPO {n["label"]}  wp={n["wp_count"]} obj={n["object_count"]}' for n in active_topologies],
                        show_labels=True,
                    ),
                )
                rr.log(
                    "world/hierarchy/topology_to_ground",
                    rr.LineStrips3D(
                        [[n["position"], n["base_position"]] for n in active_topologies],
                        colors=[255, 220, 95],
                        radii=0.01,
                    ),
                )
            else:
                rr.log("world/hierarchy/topology_nodes", rr.Clear(recursive=True))
                rr.log("world/hierarchy/topology_to_ground", rr.Clear(recursive=True))
            if active_semantics:
                semantic_colors = []
                for n in active_semantics:
                    if args.hierarchy_label_color_mode == "bright":
                        semantic_colors.append([255, 255, 255] if n["level"] == 2 else [255, 245, 80])
                    else:
                        semantic_colors.append([255, 126, 197] if n["level"] == 2 else [192, 126, 255])
                rr.log(
                    "world/hierarchy/semantic_regions",
                    rr.Points3D(
                        [n["position"] for n in active_semantics],
                        colors=semantic_colors,
                        radii=[0.24 if n["level"] == 2 else 0.32 for n in active_semantics],
                        labels=[f'L{n["level"]}:{n["label"]}' for n in active_semantics],
                        show_labels=True,
                    ),
                )
            else:
                rr.log("world/hierarchy/semantic_regions", rr.Clear(recursive=True))
            active_hierarchy_ids = active_topology_ids | active_semantic_ids
            active_edges = []
            for edge in hierarchy_edges:
                kind = edge.get("kind")
                if kind == "topology_waypoint":
                    if edge.get("topology_id") in active_topology_ids and edge.get("waypoint_id") in active_waypoint_ids:
                        active_edges.append(edge)
                elif kind == "topology_object":
                    if edge.get("topology_id") in active_topology_ids and edge.get("object_id") in active_object_ids:
                        active_edges.append(edge)
                elif kind == "semantic_topology":
                    if edge.get("semantic_id") in active_semantic_ids and edge.get("topology_id") in active_topology_ids:
                        active_edges.append(edge)
                elif kind == "semantic_child":
                    child_id = edge.get("child_id")
                    if edge.get("semantic_id") in active_semantic_ids and (child_id in active_semantic_ids or child_id in active_topology_ids):
                        active_edges.append(edge)
            if active_edges:
                rr.log(
                    "world/hierarchy/edges",
                    rr.LineStrips3D(
                        [[edge["a"], edge["b"]] for edge in active_edges],
                        colors=[180, 190, 255],
                        radii=0.008,
                    ),
                )
            else:
                rr.log("world/hierarchy/edges", rr.Clear(recursive=True))

        if frame % 20 == 0 or frame == records[-1]["frame"]:
            print(f"logged frame {frame + 1}/{len(records)} objects={len(visible_objects)} waypoints={len(active_waypoints)}")

    summary = {
        "output": str(output),
        "frames": len(records),
        "cloud_points": len(cloud),
        "grow_pointcloud": bool(args.grow_pointcloud),
        "growing_cloud_points_final": len(growing_cloud_by_key),
        "cloud_color": cloud_color,
        "cloud_radius": float(args.cloud_radius),
        "show_hierarchy": bool(args.show_hierarchy),
        "object_label_mode": args.object_label_mode,
        "show_removed_labels": bool(args.show_removed_labels),
        "hierarchy_label_color_mode": args.hierarchy_label_color_mode,
        "topology_nodes": len(topology_nodes),
        "semantic_region_nodes": len(semantic_nodes),
        "cuvslam_trajectory_points": len(cuv_traj),
        "waypoints": len(contents.get("waypoints", [])),
        "object_nodes": len(objects),
        "removed_objects": len(removed_objects),
        "rerun_pythonpath": "/tmp/rerun_sdk:/tmp/rerun_sdk/rerun_sdk",
        "open_command": f"PYTHONPATH=/tmp/rerun_sdk:/tmp/rerun_sdk/rerun_sdk /tmp/rerun_sdk/bin/rerun {output}",
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
