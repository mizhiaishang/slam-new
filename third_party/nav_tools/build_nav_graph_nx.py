#!/usr/bin/env python3
"""Build a hierarchical navigation graph with NetworkX.

This script maintains a three-layer navigation graph:

1. topology nodes
   A topology node represents a higher-level area or cluster. In the current
   implementation, waypoints are grouped into a topology by a configurable JSON
   field such as ``class_node``.

2. waypoint nodes
   Each input frame/JSON becomes one waypoint node. Waypoints preserve the
   original traversal order and therefore form the main path-search layer.

3. object nodes
   Object nodes represent semantically meaningful things observed from one or
   more nearby waypoints inside the same topology. They are deduplicated by
   ``class_name`` plus spatial proximity.

The graph contains several edge types:
- topology -> waypoint: containment / attachment
- waypoint -> waypoint: temporal traversal order between adjacent steps
- waypoint -> object: mounting / observation relationship

The script supports two ingestion styles:
- batch import from many JSON files in a directory
- direct in-memory ingestion by calling ``append_payload(...)``

All runtime parameters are read from ``nav_graph_config.json``. The command line
only selects the mode: ``build``, ``stats`` or ``nearby``.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from omegaconf import OmegaConf

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None

try:
    from sklearn.cluster import AgglomerativeClustering
except ImportError:  # pragma: no cover
    AgglomerativeClustering = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


SCHEMA_VERSION = "nx-hierarchy-1"
DEFAULT_DYNAMIC_CLASSES = {
    "person",
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "dog",
    "cat",
    "bird",
}
DEFAULT_CONFIG_PATH = "nav_graph_config.json"
DEFAULT_LLM_CONFIG_PATH = Path(__file__).resolve().parent.parent / "llm_private.yaml"
OBJECT_GLOBAL_POINTS_SAMPLE_LIMIT = 1200

DEFAULT_CONFIG = {
    "build": {
        "input_dir": ".",
        "output": "nav_graph.pkl",
        "recursive": False,
        "overwrite": False,
        "append": False,
        "topology_key_field": "class_node",
        "merge_distance": 1.0,
        "distance_dims": 2,
        "waypoint_cell_size": 2.0,
        "object_cell_size": 1.0,
        "topology_cell_size": 4.0,
        "min_confidence": 0.0,
        "extra_dynamic_classes": "",
    },
    "stats": {
        "graph": "nav_graph.pkl",
    },
    "nearby": {
        "graph": "nav_graph.pkl",
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "radius": 5.0,
        "topology_limit": 5,
        "waypoint_limit": 5,
        "object_limit": 10,
        "include_dynamic": False,
    },
}


@dataclass
class PendingFile:
    """Lightweight metadata for one source file before it is imported.

    The actual JSON payload is read later during the build loop. This keeps the
    scan step cheap and focused on discovery and stable ordering.
    """

    path: Path
    source_path: str
    sort_key: str
    timestamp: str
    topology_key: str


# ---------------------------------------------------------------------------
# Configuration and CLI
# ---------------------------------------------------------------------------

def ensure_networkx() -> None:
    """Fail fast if the runtime dependency is missing."""

    if nx is None:
        raise RuntimeError(
            "networkx is not installed. Please run: py -m pip install networkx"
        )


MODE_CHOICES = {
    "1": "build",
    "2": "stats",
    "3": "nearby",
    "build": "build",
    "stats": "stats",
    "nearby": "nearby",
}


def normalize_mode(mode: object) -> str:
    """Normalize and validate one mode token."""

    command = normalize_text(mode).lower()
    if command in MODE_CHOICES:
        return MODE_CHOICES[command]
    raise ValueError("unsupported mode, expected one of: 1, 2, 3, build, stats, nearby")


def select_mode() -> str:
    """Resolve the runtime mode from argv or from a tiny interactive menu.

    The script no longer exposes any parameter-level CLI. Users only choose one
    of the three supported modes, while every detailed setting still comes from
    ``nav_graph_config.json``.
    """

    if len(sys.argv) >= 2:
        return normalize_mode(sys.argv[1])

    print("请选择运行模式：")
    print("1. build")
    print("2. stats")
    print("3. nearby")
    return normalize_mode(input("请输入模式编号或名称: "))


def resolve_args(mode: Optional[object] = None) -> argparse.Namespace:
    """Build the runtime namespace from an explicit mode or from user selection."""

    command = normalize_mode(mode) if mode is not None else select_mode()
    return apply_config_to_args(argparse.Namespace(command=command))


def load_config_file(path_str: str = DEFAULT_CONFIG_PATH) -> Dict[str, object]:
    """Load the shared JSON config file.

    Returning an empty dict when the file does not exist makes it possible to
    fall back to ``DEFAULT_CONFIG`` during development.
    """

    path = Path(path_str)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"config file must contain a JSON object: {path}")
    return payload


def apply_config_to_args(args: argparse.Namespace) -> argparse.Namespace:
    """Merge the selected mode section from config into the parsed args.

    Example:
    - ``build`` reads ``DEFAULT_CONFIG["build"]`` and then overlays the
      ``build`` section from ``nav_graph_config.json``.
    - ``nearby`` does the same for its own section.
    """

    config_payload = load_config_file(DEFAULT_CONFIG_PATH)
    section_defaults = dict(DEFAULT_CONFIG.get(args.command, {}))
    section_from_file = config_payload.get(args.command, {})

    if section_from_file is None:
        section_from_file = {}
    if not isinstance(section_from_file, dict):
        raise ValueError(f"config section '{args.command}' must be a JSON object")

    merged_section = dict(section_defaults)
    merged_section.update(section_from_file)

    for key in section_defaults:
        setattr(args, key, merged_section.get(key))

    if args.command == "nearby":
        if args.x is None or args.y is None:
            raise ValueError("nearby requires x and y, provide them in nav_graph_config.json")

    return args


# ---------------------------------------------------------------------------
# Generic parsing helpers
# ---------------------------------------------------------------------------

def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_timestamp(value: object) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    digits = "".join(char for char in text if char.isdigit())
    if digits:
        return digits.zfill(20)
    return text


def timestamp_before(candidate: object, current: object) -> bool:
    candidate_key = normalize_timestamp(candidate)
    current_key = normalize_timestamp(current)
    return bool(candidate_key and (not current_key or candidate_key < current_key))


def timestamp_after(candidate: object, current: object) -> bool:
    candidate_key = normalize_timestamp(candidate)
    current_key = normalize_timestamp(current)
    return bool(candidate_key and (not current_key or candidate_key > current_key))


def safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_vec3(value: object) -> Optional[Tuple[float, float, float]]:
    if not isinstance(value, Sequence) or len(value) < 3:
        return None
    x = safe_float(value[0])
    y = safe_float(value[1])
    z = safe_float(value[2])
    if x is None or y is None or z is None:
        return None
    return (x, y, z)


def parse_vec3_list(value: object) -> Optional[List[float]]:
    vec = parse_vec3(value)
    if vec is None:
        return None
    return [float(vec[0]), float(vec[1]), float(vec[2])]


def parse_point_sample(value: object, limit: int = OBJECT_GLOBAL_POINTS_SAMPLE_LIMIT) -> List[List[float]]:
    if not isinstance(value, Sequence):
        return []

    points: List[List[float]] = []
    for item in value:
        vec = parse_vec3(item)
        if vec is None:
            continue
        points.append([float(vec[0]), float(vec[1]), float(vec[2])])
        if len(points) >= limit:
            break
    return points


def merge_vec3_average(
    old_value: object,
    new_value: Optional[Sequence[float]],
    old_count: int,
) -> Optional[List[float]]:
    if new_value is None:
        if isinstance(old_value, list) and len(old_value) >= 3:
            return [float(old_value[0]), float(old_value[1]), float(old_value[2])]
        return None
    if not isinstance(old_value, Sequence) or len(old_value) < 3 or old_count <= 0:
        return [float(new_value[0]), float(new_value[1]), float(new_value[2])]
    return [
        ((float(old_value[0]) * old_count) + float(new_value[0])) / (old_count + 1),
        ((float(old_value[1]) * old_count) + float(new_value[1])) / (old_count + 1),
        ((float(old_value[2]) * old_count) + float(new_value[2])) / (old_count + 1),
    ]


def extract_object3d_metadata(detection: dict) -> Dict[str, object]:
    support_point_count = safe_float(detection.get("support_point_count"))
    mask_area = safe_float(detection.get("mask_area"))
    estimation_metadata = detection.get("estimation_metadata")
    if not isinstance(estimation_metadata, dict):
        estimation_metadata = {}

    return {
        "object3d_track_id": normalize_text(detection.get("object3d_track_id")) or None,
        "global_position_method": normalize_text(detection.get("global_position_method")) or None,
        "support_point_count": support_point_count,
        "mask_area": mask_area,
        "bbox_3d_center": parse_vec3_list(detection.get("bbox_3d_center")),
        "bbox_3d_extent": parse_vec3_list(detection.get("bbox_3d_extent")),
        "global_points_sample": parse_point_sample(detection.get("global_points_sample")),
        "motion_state": normalize_text(detection.get("motion_state")) or None,
        "motion_reason": normalize_text(detection.get("motion_reason")) or None,
        "motion_metrics": detection.get("motion_metrics") if isinstance(detection.get("motion_metrics"), dict) else None,
        "object3d_first_seen_frame_index": safe_float(detection.get("object3d_first_seen_frame_index")),
        "object3d_last_seen_frame_index": safe_float(detection.get("object3d_last_seen_frame_index")),
        "object3d_observed_frame_count": safe_float(detection.get("object3d_observed_frame_count")),
        "object3d_observed_frame_span": safe_float(detection.get("object3d_observed_frame_span")),
        "object3d_missing_frame_count": safe_float(detection.get("object3d_missing_frame_count")),
        "object3d_observation_rate": safe_float(detection.get("object3d_observation_rate")),
        "object3d_first_seen_timestamp": normalize_text(detection.get("object3d_first_seen_timestamp")) or None,
        "object3d_last_seen_timestamp": normalize_text(detection.get("object3d_last_seen_timestamp")) or None,
        "object3d_first_seen_time_s": safe_float(detection.get("object3d_first_seen_time_s")),
        "object3d_last_seen_time_s": safe_float(detection.get("object3d_last_seen_time_s")),
        "object3d_observed_duration_s": safe_float(detection.get("object3d_observed_duration_s")),
        "object3d_lifecycle_state": normalize_text(detection.get("object3d_lifecycle_state")) or None,
        "object3d_removal_frame_index": safe_float(detection.get("object3d_removal_frame_index")),
        "object3d_removal_timestamp": normalize_text(detection.get("object3d_removal_timestamp")) or None,
        "object3d_removal_time_s": safe_float(detection.get("object3d_removal_time_s")),
        "object3d_removal_reason": normalize_text(detection.get("object3d_removal_reason")) or None,
        "estimation_metadata": estimation_metadata,
    }


def apply_object3d_metadata(
    attrs: Dict[str, object],
    metadata: Dict[str, object],
    old_count: int,
) -> None:
    method = metadata.get("global_position_method")
    if method:
        attrs["global_position_method"] = method

    track_id = metadata.get("object3d_track_id")
    if track_id:
        attrs["object3d_track_id"] = track_id

    support_point_count = metadata.get("support_point_count")
    if support_point_count is not None:
        support_float = float(support_point_count)
        attrs["mean_support_point_count"] = merge_optional_average(
            attrs.get("mean_support_point_count"),
            support_float,
            old_count,
        )
        attrs["latest_support_point_count"] = int(round(support_float))

    mask_area = metadata.get("mask_area")
    if mask_area is not None:
        mask_float = float(mask_area)
        attrs["mean_mask_area"] = merge_optional_average(attrs.get("mean_mask_area"), mask_float, old_count)
        attrs["latest_mask_area"] = int(round(mask_float))

    bbox_center = metadata.get("bbox_3d_center")
    attrs["bbox_3d_center"] = merge_vec3_average(attrs.get("bbox_3d_center"), bbox_center, old_count)
    bbox_extent = metadata.get("bbox_3d_extent")
    attrs["bbox_3d_extent"] = merge_vec3_average(attrs.get("bbox_3d_extent"), bbox_extent, old_count)

    motion_state = metadata.get("motion_state")
    if motion_state:
        attrs["motion_state"] = motion_state

    motion_reason = metadata.get("motion_reason")
    if motion_reason:
        attrs["motion_reason"] = motion_reason

    motion_metrics = metadata.get("motion_metrics")
    if isinstance(motion_metrics, dict) and motion_metrics:
        attrs["motion_metrics"] = motion_metrics

    for key in (
        "object3d_first_seen_frame_index",
        "object3d_last_seen_frame_index",
        "object3d_observed_frame_count",
        "object3d_observed_frame_span",
        "object3d_missing_frame_count",
        "object3d_removal_frame_index",
    ):
        value = metadata.get(key)
        if value is not None:
            attrs[key] = int(round(float(value)))

    for key in (
        "object3d_observation_rate",
        "object3d_first_seen_time_s",
        "object3d_last_seen_time_s",
        "object3d_observed_duration_s",
        "object3d_removal_time_s",
    ):
        value = metadata.get(key)
        if value is not None:
            attrs[key] = float(value)

    for key in (
        "object3d_first_seen_timestamp",
        "object3d_last_seen_timestamp",
        "object3d_lifecycle_state",
        "object3d_removal_timestamp",
        "object3d_removal_reason",
    ):
        value = metadata.get(key)
        if value:
            attrs[key] = value

    points = metadata.get("global_points_sample")
    if isinstance(points, list) and points:
        existing = attrs.get("global_points_sample")
        merged = parse_point_sample(existing) if isinstance(existing, list) else []
        merged.extend(parse_point_sample(points))
        attrs["global_points_sample"] = merged[:OBJECT_GLOBAL_POINTS_SAMPLE_LIMIT]

    estimation_metadata = metadata.get("estimation_metadata")
    if isinstance(estimation_metadata, dict) and estimation_metadata:
        attrs["estimation_metadata"] = estimation_metadata


def parse_vec4(value: object) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if not isinstance(value, Sequence):
        return (None, None, None, None)
    parts = [safe_float(item) for item in list(value[:4])]
    while len(parts) < 4:
        parts.append(None)
    return tuple(parts[:4])  # type: ignore[return-value]


def point_distance(a: Sequence[float], b: Sequence[float], dims: int) -> float:
    if dims == 2:
        return math.hypot(a[0] - b[0], a[1] - b[1])
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def grid_key(position: Sequence[float], cell_size: float) -> Tuple[int, int, int]:
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    return (
        math.floor(position[0] / cell_size),
        math.floor(position[1] / cell_size),
        math.floor(position[2] / cell_size),
    )


def iter_bucket_keys(center: Sequence[float], radius: float, cell_size: float) -> Iterable[Tuple[int, int, int]]:
    min_corner = (center[0] - radius, center[1] - radius, center[2] - radius)
    max_corner = (center[0] + radius, center[1] + radius, center[2] + radius)
    min_key = grid_key(min_corner, cell_size)
    max_key = grid_key(max_corner, cell_size)
    for bx in range(min_key[0], max_key[0] + 1):
        for by in range(min_key[1], max_key[1] + 1):
            for bz in range(min_key[2], max_key[2] + 1):
                yield (bx, by, bz)


def make_bucket_id(bucket: Tuple[int, int, int]) -> str:
    return f"{bucket[0]},{bucket[1]},{bucket[2]}"


def parse_bucket_id(bucket_text: object) -> Optional[Tuple[int, int, int]]:
    text = normalize_text(bucket_text)
    if not text:
        return None
    parts = text.split(",")
    if len(parts) != 3:
        return None
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def bucket_neighbors(center_key: Tuple[int, int, int]) -> Iterable[Tuple[int, int, int]]:
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                yield (center_key[0] + dx, center_key[1] + dy, center_key[2] + dz)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def topology_key_from_payload(payload: dict, topology_key_field: str) -> str:
    topology_key = normalize_text(payload.get(topology_key_field))
    if topology_key:
        return topology_key
    return "topology_default"


def build_sort_key(payload: dict, path: Path, topology_key_field: str) -> str:
    timestamp_key = normalize_timestamp(payload.get("timestamp"))
    topology_key = topology_key_from_payload(payload, topology_key_field)
    return f"{timestamp_key}|{topology_key}|{path.name}|{str(path.resolve()).lower()}"


def collect_pending_files(
    input_dir: Path,
    recursive: bool,
    topology_key_field: str,
    ignored_paths: Optional[Sequence[Path]] = None,
) -> Tuple[List[PendingFile], List[str]]:
    """Discover candidate JSON files and prepare a stable import order.

    Files are not imported immediately. Instead, we build ``PendingFile``
    records containing only the metadata required to sort and reference them
    later in the build phase.
    """

    pattern = "**/*.json" if recursive else "*.json"
    paths = sorted(input_dir.glob(pattern))
    pending: List[PendingFile] = []
    errors: List[str] = []
    ignored_resolved = {
        str(path.resolve()).lower()
        for path in (ignored_paths or [])
    }

    for path in paths:
        if not path.is_file():
            continue
        if str(path.resolve()).lower() in ignored_resolved:
            continue
        try:
            payload = load_json(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"skip {path}: {exc}")
            continue

        pending.append(
            PendingFile(
                path=path,
                source_path=str(path.resolve()),
                sort_key=build_sort_key(payload, path, topology_key_field),
                timestamp=normalize_text(payload.get("timestamp")),
                topology_key=topology_key_from_payload(payload, topology_key_field),
            )
        )

    pending.sort(key=lambda item: item.sort_key)
    return pending, errors


def print_warnings(messages: Sequence[str]) -> None:
    for message in messages:
        print(message, file=sys.stderr)


def build_dynamic_classes(extra_dynamic_classes: str) -> List[str]:
    """Combine built-in dynamic classes with user-provided extra classes."""

    classes = set(DEFAULT_DYNAMIC_CLASSES)
    for item in extra_dynamic_classes.split(","):
        name = item.strip()
        if name:
            classes.add(name)
    return sorted(classes)


def new_nav_graph(config: Dict[str, object]) -> "nx.Graph":
    """Create an empty graph and initialize graph-level metadata.

    The ``counters`` dictionary is used to generate stable node/observation ids
    without depending on external storage.
    """

    graph = nx.Graph()
    graph.graph.update(
        {
            "graph_kind": "hierarchical_nav_graph",
            "schema_version": SCHEMA_VERSION,
            "config": config,
            "counters": {
                "next_waypoint_seq": 1,
                "next_object_seq": 1,
                "next_observation_seq": 1,
                "next_semantic_region_seq": 1,
            },
        }
    )
    return graph


def read_graph(path: Path) -> "nx.Graph":
    """Load a previously saved NetworkX graph from disk."""

    with path.open("rb") as handle:
        graph = pickle.load(handle)

    # Backfill cached topology-level object names and remove legacy topology semantics.
    refresh_all_topology_object_summaries(graph)
    apply_all_topology_region_descriptions(graph)
    if not graph.graph.get("semantic_hierarchy"):
        attach_semantic_hierarchy(graph, levels=3)
    return graph


def write_graph(graph: "nx.Graph", path: Path) -> None:
    """Persist the current graph to disk using pickle."""

    with path.open("wb") as handle:
        pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)


def build_config(args: argparse.Namespace, dynamic_classes: Sequence[str]) -> Dict[str, object]:
    """Build the normalized runtime config stored inside the graph metadata."""

    return {
        "topology_key_field": args.topology_key_field,
        "merge_distance": args.merge_distance,
        "distance_dims": args.distance_dims,
        "waypoint_cell_size": args.waypoint_cell_size,
        "object_cell_size": args.object_cell_size,
        "topology_cell_size": args.topology_cell_size,
        "min_confidence": args.min_confidence,
        "dynamic_classes": list(dynamic_classes),
    }


def ensure_graph_metadata(
    graph: "nx.Graph",
    expected_config: Dict[str, object],
    append: bool,
) -> Dict[str, object]:
    """Validate that an existing graph is compatible with the requested build.

    Appending with mismatched configuration would silently corrupt semantics,
    for example by mixing different merge thresholds in the same graph.
    """

    if graph.graph.get("graph_kind") != "hierarchical_nav_graph":
        raise ValueError("graph_kind mismatch: not a hierarchical_nav_graph")

    if graph.graph.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version mismatch: expected {SCHEMA_VERSION}, got {graph.graph.get('schema_version')}"
        )

    current = dict(graph.graph.get("config", {}))
    if not current:
        graph.graph["config"] = expected_config
        return expected_config

    if append:
        for key, value in expected_config.items():
            if current.get(key) != value:
                raise ValueError(
                    f"Existing graph config mismatch for '{key}'. existing={current.get(key)!r}, requested={value!r}"
                )
        return current

    graph.graph["config"] = expected_config
    return expected_config


def next_sequence(graph: "nx.Graph", counter_name: str) -> int:
    """Increment and return a per-graph integer counter."""

    counters = graph.graph.setdefault("counters", {})
    value = int(counters.get(counter_name, 1))
    counters[counter_name] = value + 1
    return value


def new_waypoint_node_id(graph: "nx.Graph") -> str:
    return f"waypoint:{next_sequence(graph, 'next_waypoint_seq')}"


def new_object_node_id(graph: "nx.Graph") -> str:
    return f"object:{next_sequence(graph, 'next_object_seq')}"


def new_observation_id(graph: "nx.Graph") -> str:
    return f"obs:{next_sequence(graph, 'next_observation_seq')}"


def new_semantic_region_id(graph: "nx.Graph", level: int) -> str:
    return f"semantic_region:L{level}:{next_sequence(graph, 'next_semantic_region_seq')}"


def make_topology_node_id(topology_key: str) -> str:
    """Create the stable topology node id from a grouping key."""

    key = topology_key or "topology_default"
    return f"topology:{key}"


def topology_nodes(graph: "nx.Graph") -> Iterable[Tuple[str, dict]]:
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") == "topology":
            yield node_id, attrs


def waypoint_nodes(graph: "nx.Graph") -> Iterable[Tuple[str, dict]]:
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") == "waypoint":
            yield node_id, attrs


def object_nodes(graph: "nx.Graph") -> Iterable[Tuple[str, dict]]:
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") == "object":
            yield node_id, attrs


def semantic_region_nodes(graph: "nx.Graph") -> Iterable[Tuple[str, dict]]:
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") == "semantic_region":
            yield node_id, attrs


def rebuild_spatial_indexes(
    graph: "nx.Graph",
) -> Tuple[Dict[Tuple[int, int, int], set], Dict[Tuple[int, int, int], set], Dict[Tuple[int, int, int], set]]:
    """Recreate in-memory spatial hash maps from nodes already stored in graph.

    These indexes are intentionally not stored as standalone graph objects.
    Rebuilding them keeps the serialized graph simpler while still enabling
    efficient nearby queries and merge checks at runtime.
    """

    topology_index: Dict[Tuple[int, int, int], set] = {}
    waypoint_index: Dict[Tuple[int, int, int], set] = {}
    object_index: Dict[Tuple[int, int, int], set] = {}

    for node_id, attrs in graph.nodes(data=True):
        bucket = parse_bucket_id(attrs.get("bucket"))
        if bucket is None:
            continue
        node_type = attrs.get("node_type")
        if node_type == "topology":
            topology_index.setdefault(bucket, set()).add(node_id)
        elif node_type == "waypoint":
            waypoint_index.setdefault(bucket, set()).add(node_id)
        elif node_type == "object":
            object_index.setdefault(bucket, set()).add(node_id)

    return topology_index, waypoint_index, object_index


def get_existing_source_paths(graph: "nx.Graph") -> set:
    """Return source files that have already been imported as waypoints."""

    return {
        normalize_text(attrs.get("source_path"))
        for _, attrs in waypoint_nodes(graph)
        if normalize_text(attrs.get("source_path"))
    }


def get_last_waypoint(graph: "nx.Graph") -> Optional[Tuple[str, Tuple[float, float, float], str, str]]:
    """Find the last waypoint according to the persisted sort order.

    This is used by append builds so newly imported waypoints can continue the
    temporal chain from the previous end of the graph.
    """

    candidates: List[Tuple[str, Tuple[float, float, float], str, str]] = []
    for node_id, attrs in waypoint_nodes(graph):
        position = attrs.get("position")
        if not isinstance(position, list) or len(position) < 3:
            continue
        candidates.append(
            (
                node_id,
                (float(position[0]), float(position[1]), float(position[2])),
                normalize_text(attrs.get("sort_key")),
                normalize_text(attrs.get("topology_id")),
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[2], item[0]))
    return candidates[-1]


def move_index_bucket(
    node_id: str,
    old_bucket: Tuple[int, int, int],
    new_bucket: Tuple[int, int, int],
    index_map: Dict[Tuple[int, int, int], set],
) -> None:
    """Move one node from its old spatial bucket to a new one."""

    if old_bucket != new_bucket:
        current = index_map.get(old_bucket)
        if current is not None:
            current.discard(node_id)
            if not current:
                index_map.pop(old_bucket, None)
    index_map.setdefault(new_bucket, set()).add(node_id)


def get_or_create_topology(
    graph: "nx.Graph",
    topology_key: str,
    position: Sequence[float],
    timestamp: str,
    topology_index: Dict[Tuple[int, int, int], set],
) -> str:
    """Return an existing topology node or create a new one.

    Topology nodes are keyed by ``topology_key`` rather than by raw position, so
    multiple waypoints can accumulate under the same higher-level area.
    """

    topology_id = make_topology_node_id(topology_key)
    if graph.has_node(topology_id):
        return topology_id

    bucket = grid_key(position, float(graph.graph["config"]["topology_cell_size"]))
    graph.add_node(
        topology_id,
        node_type="topology",
        topology_key=topology_key,
        label=topology_key,
        centroid=[position[0], position[1], position[2]],
        waypoint_count=0,
        object_count=0,
        object_ids=[],
        object_names=[],
        object_name_counts={},
        nav_object_names=[],
        dynamic_object_names=[],
        first_seen=timestamp,
        last_seen=timestamp,
        bucket=make_bucket_id(bucket),
    )
    topology_index.setdefault(bucket, set()).add(topology_id)
    return topology_id


def update_topology_with_waypoint(
    graph: "nx.Graph",
    topology_id: str,
    position: Sequence[float],
    timestamp: str,
    topology_index: Dict[Tuple[int, int, int], set],
) -> None:
    """Update topology centroid and counters after attaching a new waypoint."""

    attrs = graph.nodes[topology_id]
    old_count = int(attrs.get("waypoint_count", 0))
    new_count = old_count + 1
    old_bucket = parse_bucket_id(attrs.get("bucket")) or (0, 0, 0)
    centroid = attrs.get("centroid") or [position[0], position[1], position[2]]

    new_centroid = [
        ((float(centroid[0]) * old_count) + position[0]) / new_count,
        ((float(centroid[1]) * old_count) + position[1]) / new_count,
        ((float(centroid[2]) * old_count) + position[2]) / new_count,
    ]

    attrs["centroid"] = new_centroid
    attrs["waypoint_count"] = new_count
    if timestamp:
        if timestamp_before(timestamp, attrs.get("first_seen")):
            attrs["first_seen"] = timestamp
        if timestamp_after(timestamp, attrs.get("last_seen")):
            attrs["last_seen"] = timestamp

    new_bucket = grid_key(new_centroid, float(graph.graph["config"]["topology_cell_size"]))
    attrs["bucket"] = make_bucket_id(new_bucket)
    move_index_bucket(topology_id, old_bucket, new_bucket, topology_index)


def refresh_topology_object_summary(graph: "nx.Graph", topology_id: str) -> None:
    """Refresh cached object-name fields stored on one topology node.

    The authoritative hierarchy remains ``topology -> waypoint -> object``.
    These extra topology attributes are a convenience cache so one topology node
    can directly answer which object names are present under all its waypoints.
    """

    if not graph.has_node(topology_id):
        return

    topology_attrs = graph.nodes[topology_id]
    if topology_attrs.get("node_type") != "topology":
        return

    object_ids = set()
    for waypoint_id in graph.neighbors(topology_id):
        if graph.nodes[waypoint_id].get("node_type") != "waypoint":
            continue
        topology_edge = graph.edges[topology_id, waypoint_id]
        if topology_edge.get("edge_type") != "contains_waypoint":
            continue

        for object_id in graph.neighbors(waypoint_id):
            if graph.nodes[object_id].get("node_type") != "object":
                continue
            waypoint_edge = graph.edges[waypoint_id, object_id]
            if waypoint_edge.get("edge_type") != "mounts_object":
                continue
            object_ids.add(object_id)

    ordered_object_ids = sorted(
        object_ids,
        key=lambda node_id: (
            normalize_text(graph.nodes[node_id].get("class_name")),
            node_id,
        ),
    )

    object_name_counts: Dict[str, int] = {}
    nav_object_names = set()
    dynamic_object_names = set()
    for object_id in ordered_object_ids:
        object_attrs = graph.nodes[object_id]
        class_name = normalize_text(object_attrs.get("class_name"))
        if not class_name:
            continue
        object_name_counts[class_name] = object_name_counts.get(class_name, 0) + 1
        if object_attrs.get("nav_candidate"):
            nav_object_names.add(class_name)
        if object_attrs.get("is_dynamic"):
            dynamic_object_names.add(class_name)

    ordered_object_names = sorted(object_name_counts)
    topology_attrs["object_ids"] = ordered_object_ids
    topology_attrs["object_names"] = ordered_object_names
    topology_attrs["object_name_counts"] = {
        class_name: object_name_counts[class_name]
        for class_name in ordered_object_names
    }
    topology_attrs["nav_object_names"] = sorted(nav_object_names)
    topology_attrs["dynamic_object_names"] = sorted(dynamic_object_names)
    topology_attrs["object_count"] = len(ordered_object_ids)


def refresh_all_topology_object_summaries(graph: "nx.Graph") -> None:
    """Refresh topology-level object summaries for the entire graph."""

    for topology_id, _attrs in topology_nodes(graph):
        refresh_topology_object_summary(graph, topology_id)


def sanitize_semantic_token(value: object) -> str:
    """Convert arbitrary object names into compact snake_case identifier parts."""

    text = normalize_text(value).lower()
    parts: List[str] = []
    current: List[str] = []
    for char in text:
        if char.isalnum():
            current.append(char)
        else:
            if current:
                parts.append("".join(current))
                current = []
    if current:
        parts.append("".join(current))
    if not parts:
        return "unknown"
    return "_".join(parts)


def sort_name_counts(name_counts: Dict[str, int]) -> List[Tuple[str, int]]:
    """Return object-name counts ordered by importance and name stability."""

    return sorted(
        (
            (normalize_text(name), int(count))
            for name, count in name_counts.items()
            if normalize_text(name) and int(count) > 0
        ),
        key=lambda item: (-item[1], item[0]),
    )


def merge_name_counts(items: Sequence[Dict[str, int]]) -> Dict[str, int]:
    """Merge multiple name-count maps into one aggregate summary."""

    merged: Dict[str, int] = {}
    for item in items:
        for name, count in item.items():
            class_name = normalize_text(name)
            if not class_name:
                continue
            merged[class_name] = merged.get(class_name, 0) + int(count)
    return merged


def load_llm_config(path: Path = DEFAULT_LLM_CONFIG_PATH) -> Dict[str, str]:
    """Load private LLM config from a YAML file."""

    if not path.exists():
        raise FileNotFoundError(
            f"LLM config file not found: {path}. "
            "Create it with fields: base_url, api_key, model."
        )

    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"LLM config must be a YAML object: {path}")

    llm_cfg = payload.get("llm", payload)
    if not isinstance(llm_cfg, dict):
        raise ValueError(f"LLM config section must be a YAML object: {path}")

    base_url = normalize_text(llm_cfg.get("base_url"))
    api_key = normalize_text(llm_cfg.get("api_key"))
    model = normalize_text(llm_cfg.get("model"))
    extra_body = llm_cfg.get("extra_body", {})
    if extra_body is None:
        extra_body = {}
    if not isinstance(extra_body, dict):
        raise ValueError(f"LLM extra_body must be a YAML object: {path}")

    missing = [name for name, value in [("base_url", base_url), ("api_key", api_key), ("model", model)] if not value]
    if missing:
        raise ValueError(f"Missing required LLM config fields in {path}: {', '.join(missing)}")

    return {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "extra_body": extra_body,
    }


def call_llm(prompt: str) -> str:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed")

    llm_config = load_llm_config()
    client = OpenAI(
        base_url=llm_config["base_url"],
        api_key=llm_config["api_key"],
    )
    response = client.chat.completions.create(
        model=llm_config["model"],
        messages=[
            {
                "role": "user",
                "content": f"{prompt}",
            }
        ],
        stream=True,
        extra_body=llm_config["extra_body"],
    )

    reasoning_parts: List[str] = []
    answer_parts: List[str] = []

    for chunk in response:
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue

        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue

        reasoning_chunk = getattr(delta, "reasoning_content", None)
        answer_chunk = getattr(delta, "content", None)

        if reasoning_chunk:
            reasoning_parts.append(str(reasoning_chunk))
        if answer_chunk:
            answer_parts.append(str(answer_chunk))

    final_answer = "".join(answer_parts).strip()
    if final_answer:
        return final_answer

    final_reasoning = "".join(reasoning_parts).strip()
    if final_reasoning:
        return final_reasoning

    raise RuntimeError("LLM returned no content in stream chunks")


def filter(response: str) -> Dict[str, str]:
    name_pattern = r"AREA_NAME:[ \t]*(?:<<)?([^>\n]+?)(?:>>)?[ \t]*$"
    summary_pattern = r"AREA_SUMMARY:[ \t]*(?:<<)?(.+?)(?:>>)?[ \t]*$"

    name_match = re.search(name_pattern, response, re.MULTILINE)
    summary_match = re.search(summary_pattern, response, re.MULTILINE | re.DOTALL)

    if not name_match or not summary_match:
        return {
            "name": "undefined_zone",
            "summary": "Area containing multiple objects or spaces",
        }

    name = name_match.group(1).strip().lower()
    summary = summary_match.group(1).strip()

    if not re.match(r"^[a-z0-9_]+$", name):
        name = "undefined_zone"

    return {
        "name": name,
        "summary": summary,
    }


def llm_s(dist: str) -> Dict[str, str]:
    descriptions_text = dist
    prompt = (
        f"Given these objects/areas in a 2D environment:\n"
        f"{descriptions_text}\n\n"
        "Please provide:\n"
        "1. A SHORT functional name that describes the COMBINED purpose of ALL areas/objects\n"
        "   - Use exactly 2-3 words in snake_case format (e.g., art_work_zone, meeting_dining_area)\n"
        "   - The name MUST reflect ALL major functions present\n"
        "   - Do not focus on just one function if multiple exist\n"
        "2. A single concise summary that:\n"
        "   - Describes the combined function of the space\n"
        "   - Mentions key objects/features only briefly\n"
        "   - Keep the summary within 30 Chinese characters or 30 English words\n"
        "\n"
        "Format your response EXACTLY as follows (including the <<>> markers):\n"
        "AREA_NAME: <<combined_functional_name_in_snake_case>>\n"
        "AREA_SUMMARY: <<single concise sentence covering ALL functions>>\n"
    )
    response = call_llm(prompt)
    return filter(response)


def llm(dist: Sequence[Dict[str, object]]) -> Dict[str, str]:
    descriptions_text = "\n".join(
        [
            f"Object: {obj['name']} "
            for obj in dist
        ]
    )
    prompt = (
        f"Given these objects/areas in a 2D environment:\n"
        f"{descriptions_text}\n\n"
        "Please provide:\n"
        "1. A SHORT functional name that describes the COMBINED purpose of ALL areas/objects\n"
        "   - Use exactly 2-3 words in snake_case format (e.g., art_work_zone, meeting_dining_area)\n"
        "   - The name MUST reflect ALL major functions present\n"
        "   - Do not focus on just one function if multiple exist\n"
        "2. A single concise summary that:\n"
        "   - Describes the combined function of the space\n"
        "   - Mentions key objects/features only briefly\n"
        "   - Keep the summary within 30 Chinese characters or 30 English words\n"
        "\n"
        "Format your response EXACTLY as follows (including the <<>> markers):\n"
        "AREA_NAME: <<combined_functional_name_in_snake_case>>\n"
        "AREA_SUMMARY: <<single concise sentence covering ALL functions>>\n"
    )
    response = call_llm(prompt)
    return filter(response)


def cluster_visual_entities(
    entities: Sequence[Dict[str, object]],
    n_clusters: int = 5,
) -> List[List[Dict[str, object]]]:
    members = list(entities)
    if not members:
        return []

    actual_clusters = max(1, min(int(n_clusters), len(members)))
    if actual_clusters == 1:
        return [members]

    points = []
    for item in members:
        centroid = item.get("centroid")
        if not isinstance(centroid, Sequence) or len(centroid) < 2:
            points.append([0.0, 0.0])
        else:
            points.append([float(centroid[0]), float(centroid[1])])

    if AgglomerativeClustering is None:
        return [members]

    try:
        model = AgglomerativeClustering(
            n_clusters=actual_clusters,
            metric="euclidean",
            linkage="ward",
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=actual_clusters,
            affinity="euclidean",
            linkage="ward",
        )

    labels = model.fit_predict(points)
    grouped: Dict[int, List[Dict[str, object]]] = {}
    for label, item in zip(labels, members):
        grouped.setdefault(int(label), []).append(item)
    return [grouped[key] for key in sorted(grouped)]


def build_region_name(
    name_counts: Dict[str, int],
    *,
    fallback_prefix: str,
    suffix: str,
    token_limit: int = 2,
) -> str:
    """Create a short snake_case semantic name from dominant object names."""

    tokens: List[str] = []
    for class_name, _count in sort_name_counts(name_counts):
        token = sanitize_semantic_token(class_name)
        if token not in tokens:
            tokens.append(token)
        if len(tokens) >= token_limit:
            break

    if not tokens:
        fallback = sanitize_semantic_token(fallback_prefix)
        return f"{fallback}_{suffix}"
    return "_".join(tokens + [suffix])


def build_region_summary(
    *,
    scope_label: str,
    topology_count: int,
    waypoint_count: int,
    object_count: int,
    name_counts: Dict[str, int],
) -> str:
    """Generate a deterministic Chinese area summary from object statistics."""

    dominant = sort_name_counts(name_counts)
    if dominant:
        detail = "、".join(f"{name}×{count}" for name, count in dominant[:5])
        emphasis = "、".join(name for name, _count in dominant[:3])
        return (
            f"该{scope_label}覆盖 {topology_count} 个拓扑节点、{waypoint_count} 个途径点，"
            f"汇聚了 {object_count} 个稳定物体节点，主要语义线索包括 {detail}。"
            f"从物体分布上看，该区域可概括为以 {emphasis} 为主的复合语义空间。"
        )
    return (
        f"该{scope_label}覆盖 {topology_count} 个拓扑节点、{waypoint_count} 个途径点，"
        f"当前尚未形成稳定的物体语义锚点，整体更接近过渡通行或待补充观测的区域。"
    )


def apply_topology_region_description(graph: "nx.Graph", topology_id: str) -> None:
    """Remove legacy semantic-description fields from one topology node."""

    if not graph.has_node(topology_id):
        return

    attrs = graph.nodes[topology_id]
    if attrs.get("node_type") != "topology":
        return

    attrs.pop("region_name", None)
    attrs.pop("region_summary", None)
    attrs.pop("region_keywords", None)
    attrs.pop("semantic_object_zones", None)


def apply_all_topology_region_descriptions(graph: "nx.Graph") -> None:
    """Remove legacy region-description fields from every topology node."""

    for topology_id, _attrs in topology_nodes(graph):
        apply_topology_region_description(graph, topology_id)


def clear_semantic_hierarchy(graph: "nx.Graph") -> None:
    """Remove previously attached semantic-region hierarchy nodes and links."""

    semantic_ids = [node_id for node_id, _attrs in semantic_region_nodes(graph)]
    if semantic_ids:
        graph.remove_nodes_from(semantic_ids)

    for _topology_id, attrs in topology_nodes(graph):
        attrs.pop("semantic_region_path", None)
        for level in range(1, 4):
            attrs.pop(f"semantic_region_level_{level}_id", None)


def choose_semantic_cluster_count(member_count: int, force_root: bool = False) -> int:
    """Choose a progressively coarser target cluster count for one hierarchy level."""

    if member_count <= 1:
        return 1
    if force_root:
        return 1
    return max(1, min(member_count - 1, math.ceil(member_count / 2.0)))


def cluster_entities_by_position(
    entities: Sequence[Dict[str, object]],
    n_clusters: int,
) -> List[List[Dict[str, object]]]:
    """Cluster entities in 2D using the same agglomerative idea as ``visiual.py``."""

    members = list(entities)
    if not members:
        return []

    n_clusters = max(1, min(int(n_clusters), len(members)))
    if n_clusters == 1 or len(members) == 1:
        return [members]

    points = []
    for item in members:
        centroid = item.get("centroid")
        if not isinstance(centroid, Sequence) or len(centroid) < 2:
            points.append([0.0, 0.0])
            continue
        points.append([float(centroid[0]), float(centroid[1])])

    labels: List[int]
    if AgglomerativeClustering is not None:
        try:
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="euclidean",
                linkage="ward",
            )
        except TypeError:
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity="euclidean",
                linkage="ward",
            )
        labels = [int(value) for value in model.fit_predict(points)]
    else:
        ordered_indexes = sorted(
            range(len(members)),
            key=lambda index: (points[index][0], points[index][1], normalize_text(members[index].get("node_id"))),
        )
        labels = [0] * len(members)
        for order, original_index in enumerate(ordered_indexes):
            labels[original_index] = min(n_clusters - 1, (order * n_clusters) // max(1, len(members)))

    grouped: Dict[int, List[Dict[str, object]]] = {}
    for label, item in zip(labels, members):
        grouped.setdefault(label, []).append(item)

    result = list(grouped.values())
    result.sort(
        key=lambda group: (
            float(group[0].get("centroid", [0.0, 0.0, 0.0])[0]) if group else 0.0,
            float(group[0].get("centroid", [0.0, 0.0, 0.0])[1]) if group else 0.0,
            normalize_text(group[0].get("node_id")) if group else "",
        )
    )
    return result


def weighted_centroid(entities: Sequence[Dict[str, object]]) -> List[float]:
    """Compute one centroid from child entities, weighted by topology coverage."""

    sum_x = 0.0
    sum_y = 0.0
    sum_z = 0.0
    total_weight = 0.0

    for item in entities:
        centroid = item.get("centroid")
        if not isinstance(centroid, Sequence) or len(centroid) < 3:
            continue
        weight = float(max(1, int(item.get("topology_count", len(item.get("topology_ids", [])) or 1))))
        sum_x += float(centroid[0]) * weight
        sum_y += float(centroid[1]) * weight
        sum_z += float(centroid[2]) * weight
        total_weight += weight

    if total_weight <= 0:
        return [0.0, 0.0, 0.0]
    return [sum_x / total_weight, sum_y / total_weight, sum_z / total_weight]


def attach_semantic_hierarchy(graph: "nx.Graph", levels: int = 3) -> Dict[str, object]:
    """Attach a ``visiual.py``-style hierarchy based on topology semantics first."""

    refresh_all_topology_object_summaries(graph)
    apply_all_topology_region_descriptions(graph)
    clear_semantic_hierarchy(graph)

    total_levels = max(1, levels)
    topology_entities: List[Dict[str, object]] = []
    for topology_id, attrs in topology_nodes(graph):
        centroid = attrs.get("centroid")
        if not isinstance(centroid, list) or len(centroid) < 3:
            centroid = [0.0, 0.0, 0.0]
        attrs["semantic_region_level_1_id"] = topology_id
        object_name_counts = dict(attrs.get("object_name_counts", {}))
        topology_key = normalize_text(attrs.get("topology_key")) or topology_id
        topology_entities.append(
            {
                "node_id": topology_id,
                "name": build_region_name(
                    object_name_counts,
                    fallback_prefix=topology_key,
                    suffix="zone",
                ),
                "summary": build_region_summary(
                    scope_label="拓扑节点区域",
                    topology_count=1,
                    waypoint_count=int(attrs.get("waypoint_count", 0)),
                    object_count=int(attrs.get("object_count", 0)),
                    name_counts=object_name_counts,
                ),
                "centroid": [float(centroid[0]), float(centroid[1]), float(centroid[2])],
                "topology_ids": [topology_id],
                "topology_count": 1,
                "waypoint_count": int(attrs.get("waypoint_count", 0)),
                "object_count": int(attrs.get("object_count", 0)),
                "object_name_counts": object_name_counts,
            }
        )

    topology_entities.sort(key=lambda item: normalize_text(item.get("node_id")))
    if not topology_entities:
        graph.graph["semantic_hierarchy"] = {
            "levels": 0,
            "semantic_region_nodes": 0,
            "root_region_ids": [],
        }
        return graph.graph["semantic_hierarchy"]

    current_entities = topology_entities
    created_region_ids: List[str] = []
    level_counts: Dict[str, int] = {"level_1": len(topology_entities)}

    for level in range(2, total_levels + 1):
        target_clusters = choose_semantic_cluster_count(
            len(current_entities),
            force_root=(level == total_levels),
        )
        grouped_entities = cluster_entities_by_position(current_entities, target_clusters)
        next_entities: List[Dict[str, object]] = []

        for group in grouped_entities:
            child_ids = [normalize_text(item.get("node_id")) for item in group if normalize_text(item.get("node_id"))]
            topology_ids = sorted(
                {
                    topology_id
                    for item in group
                    for topology_id in item.get("topology_ids", [])
                    if normalize_text(topology_id)
                }
            )
            waypoint_count = sum(int(item.get("waypoint_count", 0)) for item in group)
            object_name_counts = merge_name_counts(
                [dict(item.get("object_name_counts", {})) for item in group]
            )
            object_count = sum(int(item.get("object_count", 0)) for item in group)
            centroid = weighted_centroid(group)
            semantic_region_id = new_semantic_region_id(graph, level)
            region_name = build_region_name(
                object_name_counts,
                fallback_prefix=f"semantic_level_{level}",
                suffix="region",
            )
            region_summary = build_region_summary(
                scope_label=f"语义层级{level}区域",
                topology_count=len(topology_ids),
                waypoint_count=waypoint_count,
                object_count=object_count,
                name_counts=object_name_counts,
            )

            graph.add_node(
                semantic_region_id,
                node_type="semantic_region",
                semantic_level=level,
                label=region_name,
                region_name=region_name,
                region_summary=region_summary,
                centroid=centroid,
                topology_ids=topology_ids,
                topology_count=len(topology_ids),
                child_ids=child_ids,
                child_count=len(child_ids),
                object_names=[name for name, _count in sort_name_counts(object_name_counts)],
                object_name_counts=object_name_counts,
                waypoint_count=waypoint_count,
                object_count=object_count,
            )

            created_region_ids.append(semantic_region_id)

            for child_id in child_ids:
                if not graph.has_node(child_id):
                    continue
                child_type = graph.nodes[child_id].get("node_type")
                if child_type == "topology":
                    graph.add_edge(
                        semantic_region_id,
                        child_id,
                        edge_type="contains_topology_semantic",
                        relation="semantic_topology_attachment",
                        semantic_level=level,
                        traversable=False,
                    )
                    graph.nodes[child_id][f"semantic_region_level_{level}_id"] = semantic_region_id
                elif child_type == "semantic_region":
                    graph.add_edge(
                        semantic_region_id,
                        child_id,
                        edge_type="contains_semantic_region",
                        relation="semantic_region_attachment",
                        semantic_level=level,
                        traversable=False,
                    )
                    graph.nodes[child_id]["parent_semantic_region_id"] = semantic_region_id
                    for topology_id in graph.nodes[child_id].get("topology_ids", []):
                        if graph.has_node(topology_id):
                            graph.nodes[topology_id][f"semantic_region_level_{level}_id"] = semantic_region_id

            next_entities.append(
                {
                    "node_id": semantic_region_id,
                    "centroid": centroid,
                    "topology_ids": topology_ids,
                    "topology_count": len(topology_ids),
                    "waypoint_count": waypoint_count,
                    "object_count": object_count,
                    "name": region_name,
                    "summary": region_summary,
                    "object_name_counts": object_name_counts,
                }
            )

        level_counts[f"level_{level}"] = len(next_entities)
        current_entities = next_entities

    for topology_id, attrs in topology_nodes(graph):
        attrs["semantic_region_path"] = [
            attrs.get(f"semantic_region_level_{level}_id")
            for level in range(1, total_levels + 1)
        ]

    graph.graph["semantic_hierarchy"] = {
        "levels": total_levels,
        "includes_topology_level": True,
        "semantic_region_nodes": len(created_region_ids),
        "root_region_ids": [item["node_id"] for item in current_entities],
        "level_counts": level_counts,
    }
    return graph.graph["semantic_hierarchy"]


def add_waypoint_node(
    graph: "nx.Graph",
    pending: PendingFile,
    payload: dict,
    topology_id: str,
    waypoint_index: Dict[Tuple[int, int, int], set],
) -> Tuple[str, Tuple[float, float, float]]:
    """Create one waypoint node and attach it to its topology parent."""

    position = parse_vec3(payload.get("position"))
    if position is None:
        raise ValueError(f"{pending.path} missing valid 'position'")

    rotation = parse_vec4(payload.get("rotation"))
    waypoint_id = new_waypoint_node_id(graph)
    bucket = grid_key(position, float(graph.graph["config"]["waypoint_cell_size"]))
    graph.add_node(
        waypoint_id,
        node_type="waypoint",
        topology_id=topology_id,
        topology_key=pending.topology_key,
        source_path=pending.source_path,
        sort_key=pending.sort_key,
        timestamp=pending.timestamp,
        position=[position[0], position[1], position[2]],
        rotation=[rotation[0], rotation[1], rotation[2], rotation[3]],
        observations=[],
        observation_count=0,
        bucket=make_bucket_id(bucket),
    )
    waypoint_index.setdefault(bucket, set()).add(waypoint_id)
    graph.add_edge(
        topology_id,
        waypoint_id,
        edge_type="contains_waypoint",
        relation="topology_attachment",
        traversable=False,
    )
    return waypoint_id, position


def link_waypoints(
    graph: "nx.Graph",
    previous: Optional[Tuple[str, Tuple[float, float, float], str, str]],
    current: Tuple[str, Tuple[float, float, float], str, str],
) -> None:
    """Link consecutive waypoints in the global traversal sequence.

    The waypoint chain is the primary traversable structure used for step-wise
    navigation. Topology nodes are only attached to their member waypoints; we
    intentionally do not create direct topology-to-topology shortcut edges.
    """

    if previous is None:
        return
    previous_id, previous_position, previous_sort_key, _previous_topology_id = previous
    current_id, current_position, current_sort_key, _current_topology_id = current
    if previous_sort_key and current_sort_key and current_sort_key < previous_sort_key:
        return

    distance = point_distance(previous_position, current_position, dims=3)
    graph.add_edge(
        previous_id,
        current_id,
        edge_type="temporal",
        relation="adjacent_waypoint",
        distance=distance,
        weight=distance,
        traversable=True,
    )


def merge_optional_average(old_value: Optional[float], new_value: Optional[float], old_count: int) -> Optional[float]:
    """Incrementally update an average without storing all historical samples."""

    if new_value is None:
        return old_value
    if old_value is None or old_count <= 0:
        return new_value
    return ((old_value * old_count) + new_value) / (old_count + 1)


def pick_object(
    graph: "nx.Graph",
    topology_id: str,
    class_name: str,
    position: Sequence[float],
    object_index: Dict[Tuple[int, int, int], set],
) -> Optional[str]:
    """Find the best existing object candidate inside the same topology.

    Object merging is intentionally local to one topology. This avoids unrelated
    areas sharing the same object node just because they contain similar labels.
    """

    config = graph.graph["config"]
    merge_distance = float(config["merge_distance"])
    distance_dims = int(config["distance_dims"])
    object_cell_size = float(config["object_cell_size"])
    origin_key = grid_key(position, object_cell_size)

    best_id: Optional[str] = None
    best_distance = float("inf")
    for bucket_key in bucket_neighbors(origin_key):
        for node_id in object_index.get(bucket_key, ()):
            attrs = graph.nodes[node_id]
            if attrs.get("topology_id") != topology_id:
                continue
            if attrs.get("class_name") != class_name:
                continue
            centroid = attrs.get("centroid")
            if not isinstance(centroid, list) or len(centroid) < 3:
                continue
            distance = point_distance(position, centroid, distance_dims)
            if distance <= merge_distance and distance < best_distance:
                best_distance = distance
                best_id = node_id
    return best_id


def pick_object_by_object3d_track_id(graph: "nx.Graph", track_id: object) -> Optional[str]:
    """Find an object node that already represents a stable object3d track."""

    normalized_track_id = normalize_text(track_id)
    if not normalized_track_id:
        return None

    for node_id, attrs in object_nodes(graph):
        if normalize_text(attrs.get("object3d_track_id")) == normalized_track_id:
            return node_id
    return None


def create_object(
    graph: "nx.Graph",
    topology_id: str,
    class_name: str,
    position: Sequence[float],
    confidence: Optional[float],
    depth: Optional[float],
    timestamp: str,
    image_path: Optional[str],
    nav_candidate: bool,
    is_dynamic: bool,
    object_index: Dict[Tuple[int, int, int], set],
    object3d_metadata: Optional[Dict[str, object]] = None,
) -> str:
    """Create a new object node and register it in the object spatial index."""

    object_id = new_object_node_id(graph)
    bucket = grid_key(position, float(graph.graph["config"]["object_cell_size"]))
    graph.add_node(
        object_id,
        node_type="object",
        topology_id=topology_id,
        class_name=class_name,
        centroid=[position[0], position[1], position[2]],
        observation_count=1,
        waypoint_count=0,
        mean_confidence=confidence,
        min_confidence=confidence,
        max_confidence=confidence,
        mean_depth=depth,
        first_seen=timestamp,
        last_seen=timestamp,
        nav_candidate=nav_candidate,
        is_dynamic=is_dynamic,
        sample_image_path=image_path,
        bucket=make_bucket_id(bucket),
    )
    if object3d_metadata:
        apply_object3d_metadata(graph.nodes[object_id], object3d_metadata, old_count=0)
    object_index.setdefault(bucket, set()).add(object_id)
    graph.nodes[topology_id]["object_count"] = int(graph.nodes[topology_id].get("object_count", 0)) + 1
    return object_id


def update_object(
    graph: "nx.Graph",
    object_id: str,
    position: Sequence[float],
    confidence: Optional[float],
    depth: Optional[float],
    timestamp: str,
    image_path: Optional[str],
    object_index: Dict[Tuple[int, int, int], set],
    object3d_metadata: Optional[Dict[str, object]] = None,
) -> None:
    """Merge one more observation into an existing object node."""

    attrs = graph.nodes[object_id]
    centroid = attrs["centroid"]
    old_count = int(attrs.get("observation_count", 0))
    new_count = old_count + 1
    old_bucket = parse_bucket_id(attrs.get("bucket")) or (0, 0, 0)

    new_centroid = [
        ((float(centroid[0]) * old_count) + position[0]) / new_count,
        ((float(centroid[1]) * old_count) + position[1]) / new_count,
        ((float(centroid[2]) * old_count) + position[2]) / new_count,
    ]

    attrs["centroid"] = new_centroid
    attrs["observation_count"] = new_count
    attrs["mean_confidence"] = merge_optional_average(attrs.get("mean_confidence"), confidence, old_count)
    attrs["mean_depth"] = merge_optional_average(attrs.get("mean_depth"), depth, old_count)
    if object3d_metadata:
        apply_object3d_metadata(attrs, object3d_metadata, old_count)
    if confidence is not None:
        current_min = attrs.get("min_confidence")
        current_max = attrs.get("max_confidence")
        attrs["min_confidence"] = confidence if current_min is None else min(float(current_min), confidence)
        attrs["max_confidence"] = confidence if current_max is None else max(float(current_max), confidence)
    if timestamp:
        if timestamp_before(timestamp, attrs.get("first_seen")):
            attrs["first_seen"] = timestamp
        if timestamp_after(timestamp, attrs.get("last_seen")):
            attrs["last_seen"] = timestamp
    if not attrs.get("sample_image_path") and image_path:
        attrs["sample_image_path"] = image_path

    new_bucket = grid_key(new_centroid, float(graph.graph["config"]["object_cell_size"]))
    attrs["bucket"] = make_bucket_id(new_bucket)
    move_index_bucket(object_id, old_bucket, new_bucket, object_index)


def append_observation_record(
    graph: "nx.Graph",
    waypoint_id: str,
    detection: dict,
    object_id: Optional[str],
) -> None:
    """Store the raw detection under the waypoint node.

    Even if the detection cannot be merged into an object node, we still keep
    the raw observation so later debugging or reprocessing is possible.
    """

    attrs = graph.nodes[waypoint_id]
    bbox = detection.get("bbox") or {}
    global_position = parse_vec3(detection.get("global_position"))
    object3d_metadata = extract_object3d_metadata(detection)
    attrs["observations"].append(
        {
            "observation_id": new_observation_id(graph),
            "class_name": normalize_text(detection.get("class_name")),
            "confidence": safe_float(detection.get("confidence")),
            "global_position": list(global_position) if global_position is not None else None,
            "depth": safe_float(detection.get("depth")),
            "bbox": {
                "x1": safe_float(bbox.get("x1")),
                "y1": safe_float(bbox.get("y1")),
                "x2": safe_float(bbox.get("x2")),
                "y2": safe_float(bbox.get("y2")),
            },
            "image_path": normalize_text(detection.get("test_img_file")) or None,
            "object_id": object_id,
            "object3d_track_id": object3d_metadata.get("object3d_track_id"),
            "global_position_method": object3d_metadata.get("global_position_method"),
            "support_point_count": object3d_metadata.get("support_point_count"),
            "mask_area": object3d_metadata.get("mask_area"),
            "bbox_3d_center": object3d_metadata.get("bbox_3d_center"),
            "bbox_3d_extent": object3d_metadata.get("bbox_3d_extent"),
            "global_points_sample": object3d_metadata.get("global_points_sample"),
            "estimation_metadata": object3d_metadata.get("estimation_metadata"),
        }
    )
    attrs["observation_count"] = int(attrs.get("observation_count", 0)) + 1


def update_waypoint_object_edge(
    graph: "nx.Graph",
    waypoint_id: str,
    object_id: str,
    confidence: Optional[float],
    depth: Optional[float],
    timestamp: str,
) -> None:
    """Create or update the edge that says a waypoint mounts an object."""

    if graph.has_edge(waypoint_id, object_id):
        edge = graph.edges[waypoint_id, object_id]
        if edge.get("edge_type") == "mounts_object":
            old_count = int(edge.get("observation_count", 0))
            edge["observation_count"] = old_count + 1
            edge["mean_confidence"] = merge_optional_average(edge.get("mean_confidence"), confidence, old_count)
            edge["mean_depth"] = merge_optional_average(edge.get("mean_depth"), depth, old_count)
            if timestamp_after(timestamp, edge.get("last_seen")):
                edge["last_seen"] = timestamp
            return

    graph.add_edge(
        waypoint_id,
        object_id,
        edge_type="mounts_object",
        relation="waypoint_observation",
        observation_count=1,
        mean_confidence=confidence,
        mean_depth=depth,
        first_seen=timestamp,
        last_seen=timestamp,
        traversable=False,
    )
    graph.nodes[object_id]["waypoint_count"] = int(graph.nodes[object_id].get("waypoint_count", 0)) + 1


def append_payload(
    graph: "nx.Graph",
    payload: dict,
    source_path: str,
    previous_waypoint: Optional[Tuple[str, Tuple[float, float, float], str, str]],
    topology_index: Dict[Tuple[int, int, int], set],
    waypoint_index: Dict[Tuple[int, int, int], set],
    object_index: Dict[Tuple[int, int, int], set],
    sort_key_override: Optional[str] = None,
) -> Tuple[str, Tuple[float, float, float], str, str]:
    """Ingest one payload into the three-layer graph.

    Import order inside this function:
    1. determine the topology group for the payload
    2. create the waypoint node under that topology
    3. connect it to the previous waypoint if needed
    4. convert each detection into either:
       - an updated object node
       - a new object node
       - or only a raw observation if it lacks global position
    """

    config = graph.graph["config"]
    topology_key_field = str(config["topology_key_field"])
    topology_key = topology_key_from_payload(payload, topology_key_field)

    pending = PendingFile(
        path=Path(source_path),
        source_path=source_path,
        sort_key=normalize_text(sort_key_override)
        or build_sort_key(payload, Path(source_path), topology_key_field),
        timestamp=normalize_text(payload.get("timestamp")),
        topology_key=topology_key,
    )

    position = parse_vec3(payload.get("position"))
    if position is None:
        raise ValueError(f"{source_path} missing valid 'position'")

    min_confidence = float(config["min_confidence"])
    dynamic_classes = set(config["dynamic_classes"])

    # The topology layer is the parent area/container for this waypoint.
    topology_id = get_or_create_topology(
        graph=graph,
        topology_key=topology_key,
        position=position,
        timestamp=pending.timestamp,
        topology_index=topology_index,
    )
    update_topology_with_waypoint(graph, topology_id, position, pending.timestamp, topology_index)

    # The waypoint layer preserves frame-level traversal history.
    waypoint_id, waypoint_position = add_waypoint_node(
        graph=graph,
        pending=pending,
        payload=payload,
        topology_id=topology_id,
        waypoint_index=waypoint_index,
    )
    current_waypoint = (waypoint_id, waypoint_position, pending.sort_key, topology_id)
    link_waypoints(graph, previous_waypoint, current_waypoint)

    detections = payload.get("detections")
    if not isinstance(detections, list):
        detections = []
    topology_summary_dirty = False

    for detection in detections:
        if not isinstance(detection, dict):
            continue

        class_name = normalize_text(detection.get("class_name"))
        if not class_name:
            continue

        confidence = safe_float(detection.get("confidence"))
        if confidence is not None and confidence < min_confidence:
            continue

        global_position = parse_vec3(detection.get("global_position"))
        depth = safe_float(detection.get("depth"))
        image_path = normalize_text(detection.get("test_img_file")) or None
        object3d_metadata = extract_object3d_metadata(detection)
        object3d_track_id = object3d_metadata.get("object3d_track_id")
        is_dynamic = class_name in dynamic_classes
        nav_candidate = not is_dynamic
        object_id: Optional[str] = None

        if global_position is not None:
            # Stable object3d tracks should produce one object node in the
            # navigation graph; otherwise fall back to local spatial merging.
            object_id = pick_object_by_object3d_track_id(graph, object3d_track_id)
            if object_id is None:
                # Object nodes are matched only inside the same topology.
                object_id = pick_object(
                    graph=graph,
                    topology_id=topology_id,
                    class_name=class_name,
                    position=global_position,
                    object_index=object_index,
                )
            if object_id is None:
                object_id = create_object(
                    graph=graph,
                    topology_id=topology_id,
                    class_name=class_name,
                    position=global_position,
                    confidence=confidence,
                    depth=depth,
                    timestamp=pending.timestamp,
                    image_path=image_path,
                    nav_candidate=nav_candidate,
                    is_dynamic=is_dynamic,
                    object_index=object_index,
                    object3d_metadata=object3d_metadata,
                )
            else:
                update_object(
                    graph=graph,
                    object_id=object_id,
                    position=global_position,
                    confidence=confidence,
                    depth=depth,
                    timestamp=pending.timestamp,
                    image_path=image_path,
                    object_index=object_index,
                    object3d_metadata=object3d_metadata,
                )
            # This edge expresses that the object is mounted under the waypoint.
            update_waypoint_object_edge(graph, waypoint_id, object_id, confidence, depth, pending.timestamp)
            topology_summary_dirty = True

        # Raw evidence is kept regardless of whether object merging succeeded.
        append_observation_record(graph, waypoint_id, detection, object_id)

    if topology_summary_dirty:
        refresh_topology_object_summary(graph, topology_id)

    return current_waypoint


def build_graph(args: argparse.Namespace) -> int:
    """Batch-import JSON files and save the resulting graph to disk."""

    ensure_networkx()

    input_dir = Path(args.input_dir).resolve()
    output_path = Path(args.output).resolve()
    config_path = Path(DEFAULT_CONFIG_PATH).resolve()
    dynamic_classes = build_dynamic_classes(args.extra_dynamic_classes)
    config = build_config(args, dynamic_classes)

    if not input_dir.exists():
        raise FileNotFoundError(f"input directory not found: {input_dir}")

    if output_path.exists() and args.overwrite:
        output_path.unlink()
    elif output_path.exists() and not args.append:
        raise FileExistsError(
            f"output graph already exists: {output_path}. "
            f"Please update 'build.overwrite' or 'build.append' in {DEFAULT_CONFIG_PATH}."
        )

    pending_files, scan_errors = collect_pending_files(
        input_dir=input_dir,
        recursive=args.recursive,
        topology_key_field=args.topology_key_field,
        ignored_paths=[config_path],
    )
    if not pending_files:
        print_warnings(scan_errors)
        raise RuntimeError(f"no JSON files found under {input_dir}")

    if args.append and output_path.exists():
        # Append mode reuses the existing graph but demands config compatibility.
        graph = read_graph(output_path)
        ensure_graph_metadata(graph, config, append=True)
    else:
        graph = new_nav_graph(config)

    # Spatial indexes are runtime acceleration structures rebuilt from nodes.
    topology_index, waypoint_index, object_index = rebuild_spatial_indexes(graph)
    existing_sources = get_existing_source_paths(graph) if args.append else set()
    previous_waypoint = get_last_waypoint(graph) if args.append else None

    imported_waypoints = 0
    imported_observations = 0
    skipped_existing = 0
    build_errors: List[str] = []
    objects_before = sum(1 for _ in object_nodes(graph))

    for pending in pending_files:
        if pending.source_path == str(output_path):
            continue
        if pending.source_path in existing_sources:
            skipped_existing += 1
            continue

        try:
            payload = load_json(pending.path)
            previous_waypoint = append_payload(
                graph=graph,
                payload=payload,
                source_path=pending.source_path,
                previous_waypoint=previous_waypoint,
                topology_index=topology_index,
                waypoint_index=waypoint_index,
                object_index=object_index,
            )
            imported_waypoints += 1
            detections = payload.get("detections")
            if isinstance(detections, list):
                imported_observations += sum(1 for item in detections if isinstance(item, dict))
        except Exception as exc:  # noqa: BLE001
            build_errors.append(f"skip {pending.path}: {exc}")

    attach_semantic_hierarchy(graph, levels=3)
    write_graph(graph, output_path)

    objects_after = sum(1 for _ in object_nodes(graph))
    print_warnings(scan_errors)
    print_warnings(build_errors)
    print(
        json.dumps(
            {
                "graph_path": str(output_path),
                "input_dir": str(input_dir),
                "scan_files": len(pending_files),
                "imported_waypoints": imported_waypoints,
                "imported_observations": imported_observations,
                "new_objects": objects_after - objects_before,
                "skipped_existing_files": skipped_existing,
                "scan_errors": len(scan_errors),
                "build_errors": len(build_errors),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def graph_stats(graph: "nx.Graph") -> Dict[str, object]:
    """Summarize graph contents for quick inspection and debugging."""

    topology_count = 0
    waypoint_count = 0
    object_count = 0
    nav_object_count = 0
    dynamic_object_count = 0
    semantic_region_count = 0
    raw_observation_count = 0
    temporal_edge_count = 0
    waypoint_object_edge_count = 0
    semantic_region_edge_count = 0

    for _, attrs in topology_nodes(graph):
        topology_count += 1

    for _, attrs in waypoint_nodes(graph):
        waypoint_count += 1
        raw_observation_count += len(attrs.get("observations", []))

    for _, attrs in object_nodes(graph):
        object_count += 1
        if attrs.get("nav_candidate"):
            nav_object_count += 1
        if attrs.get("is_dynamic"):
            dynamic_object_count += 1

    for _, _attrs in semantic_region_nodes(graph):
        semantic_region_count += 1

    for _, _, attrs in graph.edges(data=True):
        edge_type = attrs.get("edge_type")
        if edge_type == "temporal":
            temporal_edge_count += 1
        elif edge_type == "mounts_object":
            waypoint_object_edge_count += 1
        elif edge_type in {"contains_topology_semantic", "contains_semantic_region"}:
            semantic_region_edge_count += 1

    return {
        "schema_version": graph.graph.get("schema_version"),
        "config": graph.graph.get("config", {}),
        "topology_nodes": topology_count,
        "waypoint_nodes": waypoint_count,
        "object_nodes": object_count,
        "semantic_region_nodes": semantic_region_count,
        "nav_objects": nav_object_count,
        "dynamic_objects": dynamic_object_count,
        "temporal_edges": temporal_edge_count,
        "waypoint_object_edges": waypoint_object_edge_count,
        "semantic_region_edges": semantic_region_edge_count,
        "raw_observations": raw_observation_count,
        "semantic_hierarchy": graph.graph.get("semantic_hierarchy", {}),
    }


def print_stats(args: argparse.Namespace) -> int:
    """Load a saved graph and print a compact statistics report."""

    ensure_networkx()
    graph_path = Path(args.graph).resolve()
    if not graph_path.exists():
        raise FileNotFoundError(f"graph not found: {graph_path}")
    graph = read_graph(graph_path)
    payload = {"graph_path": str(graph_path), **graph_stats(graph)}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def query_topologies(
    graph: "nx.Graph",
    center: Tuple[float, float, float],
    radius: float,
    topology_limit: int,
    topology_index: Dict[Tuple[int, int, int], set],
) -> List[dict]:
    """Query nearby topology nodes using the topology spatial index."""

    config = graph.graph["config"]
    topology_cell_size = float(config["topology_cell_size"])
    distance_dims = int(config["distance_dims"])

    candidates: List[dict] = []
    seen = set()
    for bucket in iter_bucket_keys(center, radius, topology_cell_size):
        for node_id in topology_index.get(bucket, ()):
            if node_id in seen:
                continue
            seen.add(node_id)
            attrs = graph.nodes[node_id]
            centroid = attrs.get("centroid")
            if not isinstance(centroid, list) or len(centroid) < 3:
                continue
            distance = point_distance(center, centroid, distance_dims)
            if distance <= radius:
                candidates.append(
                    {
                        "topology_id": node_id,
                        "topology_key": attrs.get("topology_key"),
                        "distance": distance,
                        "centroid": centroid,
                        "waypoint_count": attrs.get("waypoint_count"),
                        "object_count": attrs.get("object_count"),
                        "object_ids": list(attrs.get("object_ids", [])),
                        "object_names": list(attrs.get("object_names", [])),
                        "object_name_counts": dict(attrs.get("object_name_counts", {})),
                        "nav_object_names": list(attrs.get("nav_object_names", [])),
                        "dynamic_object_names": list(attrs.get("dynamic_object_names", [])),
                        "semantic_region_path": list(attrs.get("semantic_region_path", [])),
                    }
                )

    candidates.sort(key=lambda item: (item["distance"], item["topology_id"]))
    return candidates[:topology_limit]


def query_waypoints(
    graph: "nx.Graph",
    center: Tuple[float, float, float],
    radius: float,
    waypoint_limit: int,
    waypoint_index: Dict[Tuple[int, int, int], set],
) -> List[dict]:
    """Query nearby waypoint nodes using the waypoint spatial index."""

    config = graph.graph["config"]
    waypoint_cell_size = float(config["waypoint_cell_size"])
    distance_dims = int(config["distance_dims"])

    candidates: List[dict] = []
    seen = set()
    for bucket in iter_bucket_keys(center, radius, waypoint_cell_size):
        for node_id in waypoint_index.get(bucket, ()):
            if node_id in seen:
                continue
            seen.add(node_id)
            attrs = graph.nodes[node_id]
            position = attrs.get("position")
            if not isinstance(position, list) or len(position) < 3:
                continue
            distance = point_distance(center, position, distance_dims)
            if distance <= radius:
                candidates.append(
                    {
                        "waypoint_id": node_id,
                        "distance": distance,
                        "timestamp": attrs.get("timestamp"),
                        "topology_id": attrs.get("topology_id"),
                        "topology_key": attrs.get("topology_key"),
                        "source_path": attrs.get("source_path"),
                        "position": position,
                    }
                )

    candidates.sort(key=lambda item: (item["distance"], item["waypoint_id"]))
    return candidates[:waypoint_limit]


def query_objects(
    graph: "nx.Graph",
    center: Tuple[float, float, float],
    radius: float,
    object_limit: int,
    include_dynamic: bool,
    object_index: Dict[Tuple[int, int, int], set],
) -> List[dict]:
    """Query nearby object nodes using the object spatial index."""

    config = graph.graph["config"]
    object_cell_size = float(config["object_cell_size"])
    distance_dims = int(config["distance_dims"])

    candidates: List[dict] = []
    seen = set()
    for bucket in iter_bucket_keys(center, radius, object_cell_size):
        for node_id in object_index.get(bucket, ()):
            if node_id in seen:
                continue
            seen.add(node_id)
            attrs = graph.nodes[node_id]
            if not include_dynamic and not attrs.get("nav_candidate"):
                continue
            centroid = attrs.get("centroid")
            if not isinstance(centroid, list) or len(centroid) < 3:
                continue
            distance = point_distance(center, centroid, distance_dims)
            if distance <= radius:
                candidates.append(
                    {
                        "object_id": node_id,
                        "class_name": attrs.get("class_name"),
                        "topology_id": attrs.get("topology_id"),
                        "distance": distance,
                        "centroid": centroid,
                        "observation_count": attrs.get("observation_count"),
                        "waypoint_count": attrs.get("waypoint_count"),
                        "mean_confidence": attrs.get("mean_confidence"),
                        "mean_depth": attrs.get("mean_depth"),
                        "nav_candidate": bool(attrs.get("nav_candidate")),
                        "is_dynamic": bool(attrs.get("is_dynamic")),
                    }
                )

    candidates.sort(key=lambda item: (item["distance"], item["object_id"]))
    return candidates[:object_limit]


def summarize_waypoint_for_navigation(
    graph: "nx.Graph",
    waypoint_id: str,
    distance: Optional[float] = None,
) -> dict:
    """Build a compact waypoint summary used by route-planning results."""

    attrs = graph.nodes[waypoint_id]
    summary = {
        "waypoint_id": waypoint_id,
        "timestamp": attrs.get("timestamp"),
        "topology_id": attrs.get("topology_id"),
        "topology_key": attrs.get("topology_key"),
        "source_path": attrs.get("source_path"),
        "position": attrs.get("position"),
        "observation_count": attrs.get("observation_count"),
    }
    if distance is not None:
        summary["distance"] = distance
    return summary


def summarize_object_for_navigation(graph: "nx.Graph", object_id: str) -> dict:
    """Build a compact object summary used by navigation results."""

    attrs = graph.nodes[object_id]
    topology_id = normalize_text(attrs.get("topology_id"))
    topology_key = None
    if topology_id and topology_id in graph.nodes:
        topology_key = graph.nodes[topology_id].get("topology_key")

    return {
        "object_id": object_id,
        "class_name": attrs.get("class_name"),
        "topology_id": topology_id or None,
        "topology_key": topology_key,
        "centroid": attrs.get("centroid"),
        "observation_count": attrs.get("observation_count"),
        "waypoint_count": attrs.get("waypoint_count"),
        "mean_confidence": attrs.get("mean_confidence"),
        "mean_depth": attrs.get("mean_depth"),
        "nav_candidate": bool(attrs.get("nav_candidate")),
        "is_dynamic": bool(attrs.get("is_dynamic")),
    }


def find_nearest_waypoint_to_position(
    graph: "nx.Graph",
    current_position: Sequence[float],
) -> Optional[dict]:
    """Return the nearest waypoint to the current robot position."""

    distance_dims = int(graph.graph["config"]["distance_dims"])
    best_summary: Optional[dict] = None
    best_distance = float("inf")

    for waypoint_id, attrs in waypoint_nodes(graph):
        position = attrs.get("position")
        if not isinstance(position, list) or len(position) < 3:
            continue
        distance = point_distance(current_position, position, distance_dims)
        if distance < best_distance:
            best_distance = distance
            best_summary = summarize_waypoint_for_navigation(graph, waypoint_id, distance=distance)

    return best_summary


def find_object_matches(
    graph: "nx.Graph",
    object_query: object,
    include_dynamic: bool = False,
) -> List[str]:
    """Resolve one user-provided object token into candidate object node ids.

    Matching priority:
    1. exact object id
    2. exact class name
    3. substring match in object id or class name
    """

    query = normalize_text(object_query).lower()
    if not query:
        return []

    exact_id_matches: List[str] = []
    exact_class_matches: List[str] = []
    fuzzy_matches: List[str] = []

    for object_id, attrs in object_nodes(graph):
        if not include_dynamic and not attrs.get("nav_candidate"):
            continue

        object_id_text = normalize_text(object_id).lower()
        class_name_text = normalize_text(attrs.get("class_name")).lower()
        if query == object_id_text:
            exact_id_matches.append(object_id)
        elif query == class_name_text:
            exact_class_matches.append(object_id)
        elif query in object_id_text or query in class_name_text:
            fuzzy_matches.append(object_id)

    candidates = exact_id_matches or exact_class_matches or fuzzy_matches
    candidates.sort(
        key=lambda node_id: (
            normalize_text(graph.nodes[node_id].get("class_name")),
            node_id,
        )
    )
    return candidates


def linked_waypoints_for_object(graph: "nx.Graph", object_id: str) -> List[dict]:
    """List waypoint nodes directly connected to one object node."""

    if object_id not in graph.nodes or graph.nodes[object_id].get("node_type") != "object":
        return []

    object_attrs = graph.nodes[object_id]
    centroid = object_attrs.get("centroid")
    distance_dims = int(graph.graph["config"]["distance_dims"])
    linked: List[dict] = []

    for neighbor_id in graph.neighbors(object_id):
        neighbor_attrs = graph.nodes[neighbor_id]
        if neighbor_attrs.get("node_type") != "waypoint":
            continue

        edge = graph.edges[object_id, neighbor_id]
        if edge.get("edge_type") != "mounts_object":
            continue

        position = neighbor_attrs.get("position")
        distance_to_object = None
        if isinstance(centroid, list) and len(centroid) >= 3 and isinstance(position, list) and len(position) >= 3:
            distance_to_object = point_distance(centroid, position, distance_dims)

        waypoint_summary = summarize_waypoint_for_navigation(graph, neighbor_id, distance=distance_to_object)
        waypoint_summary["distance_to_object"] = distance_to_object
        waypoint_summary["edge_observation_count"] = edge.get("observation_count")
        waypoint_summary["edge_mean_confidence"] = edge.get("mean_confidence")
        waypoint_summary["edge_mean_depth"] = edge.get("mean_depth")
        waypoint_summary["first_seen"] = edge.get("first_seen")
        waypoint_summary["last_seen"] = edge.get("last_seen")
        linked.append(waypoint_summary)

    linked.sort(
        key=lambda item: (
            float("inf") if item.get("distance_to_object") is None else float(item["distance_to_object"]),
            str(item["waypoint_id"]),
        )
    )
    return linked


def build_traversable_waypoint_graph(graph: "nx.Graph") -> "nx.Graph":
    """Project the full graph into a waypoint-only traversable graph."""

    route_graph = nx.Graph()
    for waypoint_id, _ in waypoint_nodes(graph):
        route_graph.add_node(waypoint_id)

    for source_id, target_id, attrs in graph.edges(data=True):
        if attrs.get("edge_type") != "temporal":
            continue
        if not bool(attrs.get("traversable")):
            continue
        if graph.nodes[source_id].get("node_type") != "waypoint":
            continue
        if graph.nodes[target_id].get("node_type") != "waypoint":
            continue

        weight = safe_float(attrs.get("weight"))
        distance = safe_float(attrs.get("distance"))
        route_graph.add_edge(
            source_id,
            target_id,
            weight=weight if weight is not None else (distance if distance is not None else 1.0),
            distance=distance,
        )

    return route_graph


def plan_navigation_to_object(
    graph: "nx.Graph",
    current_position: Sequence[float],
    object_query: object,
    include_dynamic: bool = False,
) -> dict:
    """Plan a waypoint path from the current position to a target object.

    Workflow:
    1. resolve the user-provided object query into candidate object nodes
    2. collect all waypoints mounted under those objects
    3. choose the nearest current waypoint to the robot position
    4. search the waypoint traversal graph for a reachable path
    5. return waypoint metadata plus coordinate lists for downstream navigation
    """

    position = parse_vec3(current_position)
    if position is None:
        raise ValueError("current_position must be a 3D coordinate like [x, y, z]")

    current_waypoint = find_nearest_waypoint_to_position(graph, position)
    matched_object_ids = find_object_matches(graph, object_query, include_dynamic=include_dynamic)
    route_graph = build_traversable_waypoint_graph(graph)

    matched_objects: List[dict] = []
    target_waypoint_candidates: List[dict] = []
    best_candidate: Optional[dict] = None

    current_waypoint_id = None if current_waypoint is None else str(current_waypoint["waypoint_id"])

    for object_id in matched_object_ids:
        object_summary = summarize_object_for_navigation(graph, object_id)
        attached_waypoints = linked_waypoints_for_object(graph, object_id)
        object_summary["attached_waypoints"] = attached_waypoints
        matched_objects.append(object_summary)

        for goal_waypoint in attached_waypoints:
            goal_waypoint_id = str(goal_waypoint["waypoint_id"])
            candidate = {
                "object_id": object_summary["object_id"],
                "class_name": object_summary["class_name"],
                "topology_id": object_summary["topology_id"],
                "topology_key": object_summary["topology_key"],
                "object_centroid": object_summary["centroid"],
                "goal_waypoint": goal_waypoint,
                "reachable": False,
                "path_distance": None,
                "path_waypoint_ids": [],
            }

            if current_waypoint_id and current_waypoint_id in route_graph and goal_waypoint_id in route_graph:
                try:
                    path_waypoint_ids = nx.shortest_path(
                        route_graph,
                        source=current_waypoint_id,
                        target=goal_waypoint_id,
                        weight="weight",
                    )
                    path_distance = nx.shortest_path_length(
                        route_graph,
                        source=current_waypoint_id,
                        target=goal_waypoint_id,
                        weight="weight",
                    )
                    candidate["reachable"] = True
                    candidate["path_distance"] = path_distance
                    candidate["path_waypoint_ids"] = path_waypoint_ids
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass

            target_waypoint_candidates.append(candidate)

    target_waypoint_candidates.sort(
        key=lambda item: (
            not bool(item["reachable"]),
            float("inf") if item["path_distance"] is None else float(item["path_distance"]),
            float("inf")
            if item["goal_waypoint"].get("distance_to_object") is None
            else float(item["goal_waypoint"]["distance_to_object"]),
            str(item["class_name"]),
            str(item["object_id"]),
            str(item["goal_waypoint"]["waypoint_id"]),
        )
    )

    for candidate in target_waypoint_candidates:
        if candidate["reachable"]:
            best_candidate = candidate
            break

    path_waypoints: List[dict] = []
    path_coordinates: List[List[float]] = []
    if best_candidate is not None:
        for path_index, waypoint_id in enumerate(best_candidate["path_waypoint_ids"]):
            waypoint_summary = summarize_waypoint_for_navigation(graph, waypoint_id)
            waypoint_summary["path_index"] = path_index
            path_waypoints.append(waypoint_summary)

            position_value = waypoint_summary.get("position")
            if isinstance(position_value, list) and len(position_value) >= 3:
                path_coordinates.append(
                    [float(position_value[0]), float(position_value[1]), float(position_value[2])]
                )

    message = ""
    if not matched_object_ids:
        message = "未找到匹配的目标物体。"
    elif current_waypoint is None:
        message = "图中没有可用的 waypoint，无法规划路径。"
    elif best_candidate is None:
        message = "已找到目标物体，但当前位置对应的 waypoint 与目标关联 waypoint 之间不可达。"
    else:
        message = "已找到从当前位置到目标物体关联 waypoint 的可达路径。"

    return {
        "query": {
            "object_query": normalize_text(object_query),
            "current_position": [position[0], position[1], position[2]],
            "include_dynamic": include_dynamic,
            "distance_dims": graph.graph["config"]["distance_dims"],
        },
        "message": message,
        "current_waypoint": current_waypoint,
        "matched_objects": matched_objects,
        "target_waypoint_candidates": target_waypoint_candidates,
        "selected_target": best_candidate,
        "path_found": best_candidate is not None,
        "path_waypoints": path_waypoints,
        "path_coordinates": path_coordinates,
    }


def print_nearby(args: argparse.Namespace) -> int:
    """Load a saved graph and print the nearby query result bundle."""

    ensure_networkx()
    graph_path = Path(args.graph).resolve()
    if not graph_path.exists():
        raise FileNotFoundError(f"graph not found: {graph_path}")

    graph = read_graph(graph_path)
    topology_index, waypoint_index, object_index = rebuild_spatial_indexes(graph)
    center = (args.x, args.y, args.z)
    payload = {
        "query": {
            "x": args.x,
            "y": args.y,
            "z": args.z,
            "radius": args.radius,
            "distance_dims": graph.graph["config"]["distance_dims"],
        },
        "nearby_topologies": query_topologies(
            graph=graph,
            center=center,
            radius=args.radius,
            topology_limit=args.topology_limit,
            topology_index=topology_index,
        ),
        "nearest_waypoints": query_waypoints(
            graph=graph,
            center=center,
            radius=args.radius,
            waypoint_limit=args.waypoint_limit,
            waypoint_index=waypoint_index,
        ),
        "nearby_objects": query_objects(
            graph=graph,
            center=center,
            radius=args.radius,
            object_limit=args.object_limit,
            include_dynamic=args.include_dynamic,
            object_index=object_index,
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main(mode: Optional[object] = None) -> int:
    """Run one of the three supported modes.

    When ``mode`` is provided, the function uses it directly.
    When ``mode`` is omitted, the script falls back to argv or the interactive
    menu-based selector.
    """

    args = resolve_args(mode)
    if args.command == "build":
        return build_graph(args)
    if args.command == "stats":
        return print_stats(args)
    if args.command == "nearby":
        return print_nearby(args)
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
