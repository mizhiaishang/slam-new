#!/usr/bin/env python3
"""Export helpers for hierarchical navigation graphs."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import build_nav_graph_nx as nav


def format_position(value: object) -> str:
    """Format a coordinate list for display in compact card text."""

    if not isinstance(value, list) or len(value) < 3:
        return "-"
    return f"({float(value[0]):.2f}, {float(value[1]):.2f}, {float(value[2]):.2f})"


def extract_xy(value: object) -> Optional[tuple[float, float]]:
    """Return the x/y plane from a 3D coordinate list."""

    if not isinstance(value, list) or len(value) < 2:
        return None
    return float(value[0]), float(value[1])


def extract_xyz(value: object) -> Optional[tuple[float, float, float]]:
    """Return x/y/z from a 3D coordinate list."""

    if not isinstance(value, list) or len(value) < 3:
        return None
    return float(value[0]), float(value[1]), float(value[2])


def sorted_waypoint_ids(graph) -> List[str]:
    """Return waypoint ids in stable traversal order."""

    waypoint_ids = [node_id for node_id, _ in nav.waypoint_nodes(graph)]
    waypoint_ids.sort(
        key=lambda node_id: (
            str(graph.nodes[node_id].get("sort_key", "")),
            node_id,
        )
    )
    return waypoint_ids


def serialize_object(graph, object_id: str) -> Dict[str, object]:
    """Convert one object node into a JSON-friendly summary."""

    attrs = graph.nodes[object_id]
    waypoint_ids: List[str] = []
    for neighbor_id in graph.neighbors(object_id):
        edge = graph.edges[object_id, neighbor_id]
        if edge.get("edge_type") == "mounts_object" and graph.nodes[neighbor_id].get("node_type") == "waypoint":
            waypoint_ids.append(neighbor_id)
    waypoint_ids.sort()

    return {
        "object_id": object_id,
        "class_name": attrs.get("class_name"),
        "topology_id": attrs.get("topology_id"),
        "object3d_track_id": attrs.get("object3d_track_id"),
        "centroid": attrs.get("centroid"),
        "observation_count": attrs.get("observation_count"),
        "waypoint_count": attrs.get("waypoint_count"),
        "mean_confidence": attrs.get("mean_confidence"),
        "mean_depth": attrs.get("mean_depth"),
        "global_position_method": attrs.get("global_position_method"),
        "motion_state": attrs.get("motion_state"),
        "motion_reason": attrs.get("motion_reason"),
        "motion_metrics": attrs.get("motion_metrics", {}),
        "object3d_first_seen_frame_index": attrs.get("object3d_first_seen_frame_index"),
        "object3d_last_seen_frame_index": attrs.get("object3d_last_seen_frame_index"),
        "object3d_observed_frame_count": attrs.get("object3d_observed_frame_count"),
        "object3d_observed_frame_span": attrs.get("object3d_observed_frame_span"),
        "object3d_missing_frame_count": attrs.get("object3d_missing_frame_count"),
        "object3d_observation_rate": attrs.get("object3d_observation_rate"),
        "object3d_first_seen_timestamp": attrs.get("object3d_first_seen_timestamp"),
        "object3d_last_seen_timestamp": attrs.get("object3d_last_seen_timestamp"),
        "object3d_first_seen_time_s": attrs.get("object3d_first_seen_time_s"),
        "object3d_last_seen_time_s": attrs.get("object3d_last_seen_time_s"),
        "object3d_observed_duration_s": attrs.get("object3d_observed_duration_s"),
        "object3d_lifecycle_state": attrs.get("object3d_lifecycle_state"),
        "mean_support_point_count": attrs.get("mean_support_point_count"),
        "latest_support_point_count": attrs.get("latest_support_point_count"),
        "mean_mask_area": attrs.get("mean_mask_area"),
        "latest_mask_area": attrs.get("latest_mask_area"),
        "bbox_3d_center": attrs.get("bbox_3d_center"),
        "bbox_3d_extent": attrs.get("bbox_3d_extent"),
        "global_points_sample": attrs.get("global_points_sample", []),
        "estimation_metadata": attrs.get("estimation_metadata", {}),
        "nav_candidate": bool(attrs.get("nav_candidate")),
        "is_dynamic": bool(attrs.get("is_dynamic")),
        "sample_image_path": attrs.get("sample_image_path"),
        "first_seen": attrs.get("first_seen"),
        "last_seen": attrs.get("last_seen"),
        "waypoint_ids": waypoint_ids,
    }


def serialize_waypoint(
    graph,
    waypoint_id: str,
    sequence_index: int,
    previous_waypoint_id: Optional[str],
    next_waypoint_id: Optional[str],
) -> Dict[str, object]:
    """Convert one waypoint node plus its graph neighbors into JSON."""

    attrs = graph.nodes[waypoint_id]
    object_ids: List[str] = []
    adjacent_waypoint_ids: List[str] = []

    for neighbor_id in graph.neighbors(waypoint_id):
        edge = graph.edges[waypoint_id, neighbor_id]
        edge_type = edge.get("edge_type")
        neighbor_type = graph.nodes[neighbor_id].get("node_type")

        if edge_type == "mounts_object" and neighbor_type == "object":
            object_ids.append(neighbor_id)
        elif edge_type == "temporal" and neighbor_type == "waypoint":
            adjacent_waypoint_ids.append(neighbor_id)

    object_ids.sort(
        key=lambda node_id: (
            str(graph.nodes[node_id].get("class_name", "")),
            node_id,
        )
    )
    adjacent_waypoint_ids.sort(
        key=lambda node_id: (
            str(graph.nodes[node_id].get("sort_key", "")),
            node_id,
        )
    )

    return {
        "waypoint_id": waypoint_id,
        "sequence_index": sequence_index,
        "topology_id": attrs.get("topology_id"),
        "topology_key": attrs.get("topology_key"),
        "timestamp": attrs.get("timestamp"),
        "sort_key": attrs.get("sort_key"),
        "position": attrs.get("position"),
        "rotation": attrs.get("rotation"),
        "source_path": attrs.get("source_path"),
        "observation_count": attrs.get("observation_count"),
        "previous_waypoint_id": previous_waypoint_id,
        "next_waypoint_id": next_waypoint_id,
        "adjacent_waypoint_ids": adjacent_waypoint_ids,
        "object_ids": object_ids,
        "observations": attrs.get("observations", []),
    }


def serialize_semantic_region(graph, semantic_region_id: str) -> Dict[str, object]:
    """Convert one semantic-region node into a JSON-friendly summary."""

    attrs = graph.nodes[semantic_region_id]
    child_semantic_region_ids: List[str] = []
    child_topology_ids: List[str] = []

    for neighbor_id in graph.neighbors(semantic_region_id):
        edge = graph.edges[semantic_region_id, neighbor_id]
        edge_type = edge.get("edge_type")
        neighbor_type = graph.nodes[neighbor_id].get("node_type")
        if edge_type == "contains_semantic_region" and neighbor_type == "semantic_region":
            child_semantic_region_ids.append(neighbor_id)
        elif edge_type == "contains_topology_semantic" and neighbor_type == "topology":
            child_topology_ids.append(neighbor_id)

    child_semantic_region_ids.sort()
    child_topology_ids.sort()

    return {
        "semantic_region_id": semantic_region_id,
        "semantic_level": attrs.get("semantic_level"),
        "label": attrs.get("label"),
        "region_name": attrs.get("region_name"),
        "region_summary": attrs.get("region_summary"),
        "centroid": attrs.get("centroid"),
        "topology_ids": list(attrs.get("topology_ids", [])),
        "topology_count": attrs.get("topology_count"),
        "child_ids": list(attrs.get("child_ids", [])),
        "child_count": attrs.get("child_count"),
        "child_topology_ids": child_topology_ids,
        "child_semantic_region_ids": child_semantic_region_ids,
        "parent_semantic_region_id": attrs.get("parent_semantic_region_id"),
        "object_names": list(attrs.get("object_names", [])),
        "object_name_counts": dict(attrs.get("object_name_counts", {})),
        "waypoint_count": attrs.get("waypoint_count"),
        "object_count": attrs.get("object_count"),
    }


def infer_topology_connections(graph, waypoint_waypoint_edges: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Infer topology-to-topology adjacency from waypoint traversal edges."""

    inferred: Dict[tuple[str, str], Dict[str, object]] = {}

    for edge in waypoint_waypoint_edges:
        source_waypoint_id = str(edge.get("source_waypoint_id", ""))
        target_waypoint_id = str(edge.get("target_waypoint_id", ""))
        if source_waypoint_id not in graph.nodes or target_waypoint_id not in graph.nodes:
            continue

        source_topology_id = str(graph.nodes[source_waypoint_id].get("topology_id", ""))
        target_topology_id = str(graph.nodes[target_waypoint_id].get("topology_id", ""))
        if not source_topology_id or not target_topology_id or source_topology_id == target_topology_id:
            continue
        if source_topology_id not in graph.nodes or target_topology_id not in graph.nodes:
            continue

        topology_pair = tuple(sorted((source_topology_id, target_topology_id)))
        entry = inferred.setdefault(
            topology_pair,
            {
                "source_topology_id": topology_pair[0],
                "target_topology_id": topology_pair[1],
                "source_topology_key": graph.nodes[topology_pair[0]].get("topology_key"),
                "target_topology_key": graph.nodes[topology_pair[1]].get("topology_key"),
                "support_count": 0,
                "waypoint_pairs": [],
            },
        )
        entry["support_count"] = int(entry["support_count"]) + 1

        waypoint_pairs = entry.get("waypoint_pairs")
        if isinstance(waypoint_pairs, list):
            waypoint_pairs.append(
                {
                    "source_waypoint_id": source_waypoint_id,
                    "target_waypoint_id": target_waypoint_id,
                    "distance": edge.get("distance"),
                    "weight": edge.get("weight"),
                    "traversable": bool(edge.get("traversable")),
                }
            )

    topology_topology_edges = list(inferred.values())
    topology_topology_edges.sort(
        key=lambda item: (
            str(item.get("source_topology_key", "")),
            str(item.get("target_topology_key", "")),
            str(item.get("source_topology_id", "")),
            str(item.get("target_topology_id", "")),
        )
    )
    return topology_topology_edges


def export_graph_contents(graph, output_path: Path, graph_path: Path | None = None) -> Dict[str, object]:
    """Export an explicit graph view centered on waypoint adjacency."""

    topology_waypoint_edges: List[Dict[str, object]] = []
    waypoint_waypoint_edges: List[Dict[str, object]] = []
    waypoint_object_edges: List[Dict[str, object]] = []
    semantic_topology_edges: List[Dict[str, object]] = []
    semantic_region_edges: List[Dict[str, object]] = []

    for source_id, target_id, attrs in graph.edges(data=True):
        edge_type = attrs.get("edge_type")
        if edge_type == "contains_waypoint":
            topology_waypoint_edges.append(
                {
                    "topology_id": source_id if graph.nodes[source_id].get("node_type") == "topology" else target_id,
                    "waypoint_id": target_id if graph.nodes[target_id].get("node_type") == "waypoint" else source_id,
                    "edge_type": edge_type,
                    "relation": attrs.get("relation"),
                }
            )
        elif edge_type == "temporal":
            waypoint_waypoint_edges.append(
                {
                    "source_waypoint_id": source_id,
                    "target_waypoint_id": target_id,
                    "edge_type": edge_type,
                    "relation": attrs.get("relation"),
                    "distance": attrs.get("distance"),
                    "weight": attrs.get("weight"),
                    "traversable": bool(attrs.get("traversable")),
                }
            )
        elif edge_type == "mounts_object":
            waypoint_object_edges.append(
                {
                    "waypoint_id": source_id if graph.nodes[source_id].get("node_type") == "waypoint" else target_id,
                    "object_id": target_id if graph.nodes[target_id].get("node_type") == "object" else source_id,
                    "edge_type": edge_type,
                    "relation": attrs.get("relation"),
                    "observation_count": attrs.get("observation_count"),
                    "mean_confidence": attrs.get("mean_confidence"),
                    "mean_depth": attrs.get("mean_depth"),
                    "first_seen": attrs.get("first_seen"),
                    "last_seen": attrs.get("last_seen"),
                }
            )
        elif edge_type == "contains_topology_semantic":
            semantic_topology_edges.append(
                {
                    "semantic_region_id": source_id if graph.nodes[source_id].get("node_type") == "semantic_region" else target_id,
                    "topology_id": target_id if graph.nodes[target_id].get("node_type") == "topology" else source_id,
                    "edge_type": edge_type,
                    "relation": attrs.get("relation"),
                    "semantic_level": attrs.get("semantic_level"),
                }
            )
        elif edge_type == "contains_semantic_region":
            parent_id = source_id
            child_id = target_id
            if graph.nodes[source_id].get("node_type") == "semantic_region" and graph.nodes[target_id].get("node_type") == "semantic_region":
                source_level = int(graph.nodes[source_id].get("semantic_level", 0) or 0)
                target_level = int(graph.nodes[target_id].get("semantic_level", 0) or 0)
                if source_level < target_level:
                    parent_id, child_id = target_id, source_id
            semantic_region_edges.append(
                {
                    "parent_semantic_region_id": parent_id,
                    "child_semantic_region_id": child_id,
                    "edge_type": edge_type,
                    "relation": attrs.get("relation"),
                    "semantic_level": attrs.get("semantic_level"),
                }
            )

    topology_waypoint_edges.sort(key=lambda item: (str(item["topology_id"]), str(item["waypoint_id"])))
    waypoint_waypoint_edges.sort(
        key=lambda item: (
            str(graph.nodes[item["source_waypoint_id"]].get("sort_key", "")),
            str(graph.nodes[item["target_waypoint_id"]].get("sort_key", "")),
            str(item["source_waypoint_id"]),
            str(item["target_waypoint_id"]),
        )
    )
    waypoint_object_edges.sort(key=lambda item: (str(item["waypoint_id"]), str(item["object_id"])))
    semantic_topology_edges.sort(key=lambda item: (str(item["semantic_region_id"]), str(item["topology_id"])))
    semantic_region_edges.sort(
        key=lambda item: (
            str(item["parent_semantic_region_id"]),
            str(item["child_semantic_region_id"]),
        )
    )

    waypoint_ids = sorted_waypoint_ids(graph)
    topology_topology_edges = infer_topology_connections(graph, waypoint_waypoint_edges)
    previous_map: Dict[str, Optional[str]] = {}
    next_map: Dict[str, Optional[str]] = {}
    for index, waypoint_id in enumerate(waypoint_ids):
        previous_map[waypoint_id] = waypoint_ids[index - 1] if index > 0 else None
        next_map[waypoint_id] = waypoint_ids[index + 1] if index + 1 < len(waypoint_ids) else None

    topologies: List[Dict[str, object]] = []
    for topology_id, attrs in nav.topology_nodes(graph):
        waypoint_members = [
            item["waypoint_id"]
            for item in topology_waypoint_edges
            if item["topology_id"] == topology_id
        ]
        waypoint_members.sort(
            key=lambda node_id: (
                str(graph.nodes[node_id].get("sort_key", "")),
                node_id,
            )
        )
        topologies.append(
            {
                "topology_id": topology_id,
                "topology_key": attrs.get("topology_key"),
                "label": attrs.get("label"),
                "centroid": attrs.get("centroid"),
                "waypoint_count": attrs.get("waypoint_count"),
                "object_count": attrs.get("object_count"),
                "object_ids": list(attrs.get("object_ids", [])),
                "object_names": list(attrs.get("object_names", [])),
                "object_name_counts": dict(attrs.get("object_name_counts", {})),
                "nav_object_names": list(attrs.get("nav_object_names", [])),
                "dynamic_object_names": list(attrs.get("dynamic_object_names", [])),
                "semantic_region_path": list(attrs.get("semantic_region_path", [])),
                "first_seen": attrs.get("first_seen"),
                "last_seen": attrs.get("last_seen"),
                "waypoint_ids": waypoint_members,
            }
        )
    topologies.sort(key=lambda item: (str(item["topology_key"]), str(item["topology_id"])))

    waypoints = [
        serialize_waypoint(
            graph=graph,
            waypoint_id=waypoint_id,
            sequence_index=index,
            previous_waypoint_id=previous_map[waypoint_id],
            next_waypoint_id=next_map[waypoint_id],
        )
        for index, waypoint_id in enumerate(waypoint_ids)
    ]

    objects = [serialize_object(graph, object_id) for object_id, _ in nav.object_nodes(graph)]
    objects.sort(key=lambda item: (str(item["class_name"]), str(item["object_id"])))
    semantic_regions = [
        serialize_semantic_region(graph, semantic_region_id)
        for semantic_region_id, _ in nav.semantic_region_nodes(graph)
    ]
    semantic_regions.sort(
        key=lambda item: (
            int(item.get("semantic_level", 0) or 0),
            str(item.get("semantic_region_id", "")),
        )
    )

    payload = {
        "graph_path": str(graph_path or ""),
        "stats": nav.graph_stats(graph),
        "semantic_hierarchy": graph.graph.get("semantic_hierarchy", {}),
        "waypoint_sequence": waypoint_ids,
        "topologies": topologies,
        "waypoints": waypoints,
        "objects": objects,
        "semantic_regions": semantic_regions,
        "edges": {
            "topology_waypoint": topology_waypoint_edges,
            "topology_topology_inferred": topology_topology_edges,
            "waypoint_waypoint": waypoint_waypoint_edges,
            "waypoint_object": waypoint_object_edges,
            "semantic_topology": semantic_topology_edges,
            "semantic_region": semantic_region_edges,
        },
    }

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def nice_step(span: float, target_lines: int = 8) -> float:
    """Choose a human-friendly grid spacing for the map background."""

    if span <= 0:
        return 1.0
    rough = span / max(1, target_lines)
    base = 10 ** math.floor(math.log10(rough))
    ratio = rough / base
    if ratio <= 1:
        return base
    if ratio <= 2:
        return 2 * base
    if ratio <= 5:
        return 5 * base
    return 10 * base


def topology_palette(index: int) -> str:
    """Return a stable display color for one topology."""

    colors = [
        "#2563eb",
        "#0891b2",
        "#059669",
        "#65a30d",
        "#d97706",
        "#dc2626",
        "#9333ea",
        "#7c3aed",
        "#db2777",
        "#0f766e",
    ]
    return colors[index % len(colors)]


def semantic_region_palette(level: int) -> str:
    """Return a stable display color for one semantic hierarchy level."""

    colors = {
        1: "#f59e0b",
        2: "#8b5cf6",
        3: "#0ea5e9",
    }
    return colors.get(level, "#475569")


def hex_to_rgba(color: str, alpha: float) -> str:
    """Convert a hex color like #2563eb into an rgba(...) string."""

    if isinstance(color, str) and len(color) == 7 and color.startswith("#"):
        red = int(color[1:3], 16)
        green = int(color[3:5], 16)
        blue = int(color[5:7], 16)
        return f"rgba({red},{green},{blue},{alpha:.3f})"
    return color


def export_graph_visualization(content_payload: Dict[str, object], svg_path: Path, html_path: Path) -> None:
    """Export a 2D coordinate map based on real x/y positions."""

    topologies = content_payload.get("topologies", [])
    waypoints = content_payload.get("waypoints", [])
    objects = content_payload.get("objects", [])
    semantic_regions = content_payload.get("semantic_regions", [])
    edges = content_payload.get("edges", {})
    if not isinstance(topologies, list):
        topologies = []
    if not isinstance(waypoints, list):
        waypoints = []
    if not isinstance(objects, list):
        objects = []
    if not isinstance(semantic_regions, list):
        semantic_regions = []
    if not isinstance(edges, dict):
        edges = {}

    topology_color_map: Dict[str, str] = {}
    for index, topology in enumerate(sorted(topologies, key=lambda item: str(item.get("topology_id", "")))):
        topology_color_map[str(topology.get("topology_id"))] = topology_palette(index)

    world_points: List[tuple[float, float]] = []
    for topology in topologies:
        xy = extract_xy(topology.get("centroid"))
        if xy is not None:
            world_points.append(xy)
    for waypoint in waypoints:
        xy = extract_xy(waypoint.get("position"))
        if xy is not None:
            world_points.append(xy)
    for obj in objects:
        xy = extract_xy(obj.get("centroid"))
        if xy is not None:
            world_points.append(xy)
    for semantic_region in semantic_regions:
        xy = extract_xy(semantic_region.get("centroid"))
        if xy is not None:
            world_points.append(xy)

    if not world_points:
        world_points = [(0.0, 0.0)]

    min_x = min(point[0] for point in world_points)
    max_x = max(point[0] for point in world_points)
    min_y = min(point[1] for point in world_points)
    max_y = max(point[1] for point in world_points)
    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)
    pad_x = max(1.0, span_x * 0.08)
    pad_y = max(1.0, span_y * 0.08)
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y
    span_x = max_x - min_x
    span_y = max_y - min_y

    width = 1320
    height = 980
    left = 88
    right = 40
    top = 96
    bottom = 84
    draw_w = width - left - right
    draw_h = height - top - bottom
    scale = min(draw_w / span_x, draw_h / span_y)
    offset_x = left + (draw_w - span_x * scale) / 2.0
    offset_y = top + (draw_h - span_y * scale) / 2.0

    def map_xy(world_x: float, world_y: float) -> tuple[float, float]:
        screen_x = offset_x + (world_x - min_x) * scale
        screen_y = top + draw_h - ((world_y - min_y) * scale + (draw_h - span_y * scale) / 2.0)
        return screen_x, screen_y

    grid_step = nice_step(max(span_x, span_y))
    grid_parts: List[str] = []
    label_parts: List[str] = []

    x_tick = math.floor(min_x / grid_step) * grid_step
    while x_tick <= max_x + 1e-9:
        screen_x, _ = map_xy(x_tick, min_y)
        grid_parts.append(
            f'<line x1="{screen_x:.2f}" y1="{top}" x2="{screen_x:.2f}" y2="{top + draw_h}" '
            f'stroke="#dbe4f0" stroke-width="1"/>'
        )
        label_parts.append(
            f'<text x="{screen_x:.2f}" y="{top + draw_h + 24}" font-size="12" text-anchor="middle" '
            f'fill="#64748b">{x_tick:.1f}</text>'
        )
        x_tick += grid_step

    y_tick = math.floor(min_y / grid_step) * grid_step
    while y_tick <= max_y + 1e-9:
        _, screen_y = map_xy(min_x, y_tick)
        grid_parts.append(
            f'<line x1="{left}" y1="{screen_y:.2f}" x2="{left + draw_w}" y2="{screen_y:.2f}" '
            f'stroke="#dbe4f0" stroke-width="1"/>'
        )
        label_parts.append(
            f'<text x="{left - 12}" y="{screen_y + 4:.2f}" font-size="12" text-anchor="end" '
            f'fill="#64748b">{y_tick:.1f}</text>'
        )
        y_tick += grid_step

    semantic_region_area_parts: List[str] = []
    semantic_region_link_parts: List[str] = []
    semantic_region_parts: List[str] = []
    topology_parts: List[str] = []
    topology_attachment_parts: List[str] = []
    object_parts: List[str] = []
    waypoint_parts: List[str] = []
    path_parts: List[str] = []

    waypoints_by_id = {str(item.get("waypoint_id")): item for item in waypoints}
    objects_by_id = {str(item.get("object_id")): item for item in objects}
    topologies_by_id = {str(item.get("topology_id")): item for item in topologies}
    semantic_regions_by_id = {
        str(item.get("semantic_region_id")): item
        for item in semantic_regions
    }

    semantic_topology_edges = edges.get("semantic_topology", [])
    semantic_region_edges = edges.get("semantic_region", [])
    if not isinstance(semantic_topology_edges, list):
        semantic_topology_edges = []
    if not isinstance(semantic_region_edges, list):
        semantic_region_edges = []

    semantic_region_positions: Dict[str, tuple[float, float]] = {}
    for semantic_region in semantic_regions:
        semantic_region_id = str(semantic_region.get("semantic_region_id"))
        centroid_xy = extract_xy(semantic_region.get("centroid"))
        if centroid_xy is not None:
            semantic_region_positions[semantic_region_id] = centroid_xy

    for edge in semantic_region_edges:
        if not isinstance(edge, dict):
            continue
        parent_id = str(edge.get("parent_semantic_region_id", ""))
        child_id = str(edge.get("child_semantic_region_id", ""))
        parent_xy = semantic_region_positions.get(parent_id)
        child_xy = semantic_region_positions.get(child_id)
        if parent_xy is None or child_xy is None:
            continue
        parent_screen = map_xy(*parent_xy)
        child_screen = map_xy(*child_xy)
        level = int(semantic_regions_by_id.get(parent_id, {}).get("semantic_level", 0) or 0)
        color = semantic_region_palette(level)
        semantic_region_link_parts.append(
            f'<line x1="{parent_screen[0]:.2f}" y1="{parent_screen[1]:.2f}" '
            f'x2="{child_screen[0]:.2f}" y2="{child_screen[1]:.2f}" '
            f'stroke="{hex_to_rgba(color, 0.58)}" stroke-width="3" stroke-dasharray="8 6"/>'
        )

    for edge in semantic_topology_edges:
        if not isinstance(edge, dict):
            continue
        semantic_region_id = str(edge.get("semantic_region_id", ""))
        topology_id = str(edge.get("topology_id", ""))
        region_xy = semantic_region_positions.get(semantic_region_id)
        topology = topologies_by_id.get(topology_id)
        topology_xy = extract_xy(topology.get("centroid")) if topology is not None else None
        if region_xy is None or topology_xy is None:
            continue
        region_screen = map_xy(*region_xy)
        topology_screen = map_xy(*topology_xy)
        level = int(semantic_regions_by_id.get(semantic_region_id, {}).get("semantic_level", 0) or 0)
        color = semantic_region_palette(level)
        semantic_region_link_parts.append(
            f'<line x1="{region_screen[0]:.2f}" y1="{region_screen[1]:.2f}" '
            f'x2="{topology_screen[0]:.2f}" y2="{topology_screen[1]:.2f}" '
            f'stroke="{hex_to_rgba(color, 0.34)}" stroke-width="2.2" stroke-dasharray="3 7"/>'
        )
    for semantic_region in semantic_regions:
        semantic_region_id = str(semantic_region.get("semantic_region_id"))
        centroid_xy = extract_xy(semantic_region.get("centroid"))
        if centroid_xy is None:
            continue

        member_points: List[tuple[float, float]] = []
        for topology_id in semantic_region.get("topology_ids", []):
            topology = topologies_by_id.get(str(topology_id))
            if topology is None:
                continue
            topology_xy = extract_xy(topology.get("centroid"))
            if topology_xy is not None:
                member_points.append(topology_xy)

        level = int(semantic_region.get("semantic_level", 0) or 0)
        color = semantic_region_palette(level)
        screen_x, screen_y = map_xy(*centroid_xy)
        radius = 18.0 + level * 2.0
        if member_points:
            max_distance = max(
                math.hypot(point[0] - centroid_xy[0], point[1] - centroid_xy[1])
                for point in member_points
            )
            radius = max(radius, max_distance * scale + 22.0 + level * 8.0)

        semantic_region_area_parts.append(
            f'<circle cx="{screen_x:.2f}" cy="{screen_y:.2f}" r="{radius:.2f}" '
            f'fill="{hex_to_rgba(color, 0.05)}" stroke="{hex_to_rgba(color, 0.42)}" '
            f'stroke-width="{1.8 + level * 0.6:.1f}" stroke-dasharray="{6 + level * 2} {5 + level}"/>'
        )

        label = str(semantic_region.get("region_name", semantic_region_id))
        summary = str(semantic_region.get("region_summary", ""))
        topology_count = semantic_region.get("topology_count", 0)
        object_names = semantic_region.get("object_names", [])
        if not isinstance(object_names, list):
            object_names = []
        keywords = "、".join(str(name) for name in object_names[:5]) or "-"
        tooltip = (
            f"{semantic_region_id}\n"
            f"level: {level}\n"
            f"name: {label}\n"
            f"topologies: {topology_count}\n"
            f"objects: {keywords}\n"
            f"{summary}"
        )
        semantic_region_parts.append(
            (
                f'<circle cx="{screen_x:.2f}" cy="{screen_y:.2f}" r="{7 + level * 2:.2f}" '
                f'fill="{color}" stroke="#ffffff" stroke-width="2.2">'
                f'<title>{html.escape(tooltip)}</title></circle>'
            )
        )
        semantic_region_parts.append(
            f'<text x="{screen_x + 12:.2f}" y="{screen_y + 4 - level * 8:.2f}" font-size="{12 + level:.0f}" '
            f'font-weight="700" fill="{color}">L{level} {html.escape(label)}</text>'
        )

    for topology in topologies:
        topology_id = str(topology.get("topology_id"))
        centroid_xy = extract_xy(topology.get("centroid"))
        if centroid_xy is None:
            continue
        screen_x, screen_y = map_xy(*centroid_xy)
        member_waypoint_ids = topology.get("waypoint_ids", [])
        if not isinstance(member_waypoint_ids, list):
            member_waypoint_ids = []

        member_points = []
        for waypoint_id in member_waypoint_ids:
            waypoint = waypoints_by_id.get(str(waypoint_id))
            if waypoint is None:
                continue
            waypoint_xy = extract_xy(waypoint.get("position"))
            if waypoint_xy is not None:
                member_points.append(waypoint_xy)
                waypoint_screen_x, waypoint_screen_y = map_xy(*waypoint_xy)
                topology_attachment_parts.append(
                    f'<line x1="{screen_x:.2f}" y1="{screen_y:.2f}" x2="{waypoint_screen_x:.2f}" y2="{waypoint_screen_y:.2f}" '
                    f'stroke="{topology_color_map.get(topology_id, "#64748b")}" stroke-opacity="0.28" '
                    f'stroke-width="1.6" stroke-dasharray="4 5"/>'
                )

        radius = 14.0
        if member_points:
            max_distance = max(math.hypot(point[0] - centroid_xy[0], point[1] - centroid_xy[1]) for point in member_points)
            radius = max(14.0, max_distance * scale + 10.0)

        fill = topology_color_map.get(topology_id, "#64748b")
        topology_tooltip = "\n".join(
            [
                f"topology: {topology.get('topology_key', topology_id)}",
                f"semantic name: {topology.get('region_name', '-')}",
                f"semantic summary: {topology.get('region_summary', '-')}",
                f"semantic keywords: {'、'.join(str(item) for item in topology.get('region_keywords', [])[:5]) or '-'}",
                f"semantic path: {' -> '.join(str(item) for item in topology.get('semantic_region_path', []) if item) or '-'}",
            ]
        )
        topology_parts.append(
            f'<circle cx="{screen_x:.2f}" cy="{screen_y:.2f}" r="{radius:.2f}" fill="{fill}" fill-opacity="0.08" '
            f'stroke="{fill}" stroke-opacity="0.35" stroke-width="2"/>'
        )
        topology_parts.append(
            (
                f'<path d="M {screen_x:.2f} {screen_y - 11:.2f} L {screen_x + 11:.2f} {screen_y:.2f} '
                f'L {screen_x:.2f} {screen_y + 11:.2f} L {screen_x - 11:.2f} {screen_y:.2f} Z" '
                f'fill="{fill}" stroke="#ffffff" stroke-width="2">'
                f'<title>{html.escape(topology_tooltip)}</title>'
                f'</path>'
            )
        )
        topology_parts.append(
            f'<text x="{screen_x + 14:.2f}" y="{screen_y - 12:.2f}" font-size="13" font-weight="700" '
            f'fill="{fill}">{html.escape(str(topology.get("topology_key", topology_id)))}</text>'
        )

    waypoint_sequence = content_payload.get("waypoint_sequence", [])
    if not isinstance(waypoint_sequence, list):
        waypoint_sequence = []
    path_points = []
    for waypoint_id in waypoint_sequence:
        waypoint = waypoints_by_id.get(str(waypoint_id))
        if waypoint is None:
            continue
        waypoint_xy = extract_xy(waypoint.get("position"))
        if waypoint_xy is None:
            continue
        path_points.append(map_xy(*waypoint_xy))

    if len(path_points) >= 2:
        path_parts.append(
            '<polyline points="'
            + " ".join(f"{x:.2f},{y:.2f}" for x, y in path_points)
            + '" fill="none" stroke="#0f4cdd" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" '
            + 'stroke-opacity="0.85"/>'
        )

    for waypoint in waypoints:
        waypoint_id = str(waypoint.get("waypoint_id"))
        waypoint_xy = extract_xy(waypoint.get("position"))
        if waypoint_xy is None:
            continue
        screen_x, screen_y = map_xy(*waypoint_xy)
        topology_id = str(waypoint.get("topology_id"))
        fill = topology_color_map.get(topology_id, "#2563eb")
        sequence_index = int(waypoint.get("sequence_index", 0))
        tooltip = (
            f"{waypoint_id}\n"
            f"topology: {waypoint.get('topology_key')}\n"
            f"seq: {sequence_index}\n"
            f"position: {format_position(waypoint.get('position'))}\n"
            f"observations: {waypoint.get('observation_count', 0)}"
        )
        waypoint_parts.append(
            (
                f'<circle cx="{screen_x:.2f}" cy="{screen_y:.2f}" r="8.5" fill="{fill}" stroke="#ffffff" '
                f'stroke-width="2.2"><title>{html.escape(tooltip)}</title></circle>'
            )
        )
        waypoint_parts.append(
            f'<text x="{screen_x + 12:.2f}" y="{screen_y - 10:.2f}" font-size="12" font-weight="700" '
            f'fill="#0f172a">{sequence_index}</text>'
        )

    for obj in objects:
        object_xy = extract_xy(obj.get("centroid"))
        if object_xy is None:
            continue
        screen_x, screen_y = map_xy(*object_xy)
        is_dynamic = bool(obj.get("is_dynamic"))
        fill = "#dc2626" if is_dynamic else "#16a34a"
        stroke = "#7f1d1d" if is_dynamic else "#166534"
        label = str(obj.get("class_name", obj.get("object_id", "-")))
        tooltip = (
            f"{obj.get('object_id')}\n"
            f"class: {label}\n"
            f"topology: {obj.get('topology_id')}\n"
            f"role: {'dynamic' if is_dynamic else 'navigable'}\n"
            f"centroid: {format_position(obj.get('centroid'))}\n"
            f"observations: {obj.get('observation_count', 0)}"
        )
        object_parts.append(
            (
                f'<rect x="{screen_x - 5:.2f}" y="{screen_y - 5:.2f}" width="10" height="10" '
                f'rx="2.5" ry="2.5" fill="{fill}" stroke="{stroke}" stroke-width="1.5">'
                f'<title>{html.escape(tooltip)}</title></rect>'
            )
        )
        object_parts.append(
            f'<text x="{screen_x + 8:.2f}" y="{screen_y + 4:.2f}" font-size="11" fill="#334155">{html.escape(label)}</text>'
        )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<style>'
        'text { font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; }'
        '</style>'
        '<rect width="100%" height="100%" fill="#f8fafc"/>'
        f'<rect x="{left}" y="{top}" width="{draw_w}" height="{draw_h}" rx="18" ry="18" fill="#ffffff" stroke="#cdd9ea" stroke-width="1.5"/>'
        '<text x="40" y="34" font-size="28" font-weight="700" fill="#0f172a">'
        'Coordinate Navigation Map'
        '</text>'
        '<text x="40" y="58" font-size="14" fill="#475569">'
        'The map uses real x/y coordinates. Blue lines are waypoint-to-waypoint traversal edges.'
        '</text>'
        '<line x1="40" y1="82" x2="80" y2="82" stroke="#0f4cdd" stroke-width="4"/>'
        '<text x="90" y="87" font-size="13" fill="#334155">Waypoint path</text>'
        '<circle cx="210" cy="82" r="8" fill="#2563eb" stroke="#ffffff" stroke-width="2"/>'
        '<text x="226" y="87" font-size="13" fill="#334155">Waypoint</text>'
        '<rect x="330" y="76" width="10" height="10" rx="2" fill="#16a34a" stroke="#166534" stroke-width="1.5"/>'
        '<text x="348" y="87" font-size="13" fill="#334155">Navigable object</text>'
        '<rect x="520" y="76" width="10" height="10" rx="2" fill="#dc2626" stroke="#7f1d1d" stroke-width="1.5"/>'
        '<text x="538" y="87" font-size="13" fill="#334155">Dynamic object</text>'
        '<path d="M 700 71 L 711 82 L 700 93 L 689 82 Z" fill="#d97706" stroke="#ffffff" stroke-width="2"/>'
        '<text x="720" y="87" font-size="13" fill="#334155">Topology centroid</text>'
        '<circle cx="902" cy="82" r="10" fill="none" stroke="#0ea5e9" stroke-width="2.4" stroke-dasharray="8 6"/>'
        '<text x="920" y="87" font-size="13" fill="#334155">Semantic region</text>'
        + "".join(grid_parts)
        + "".join(label_parts)
        + f'<text x="{left + draw_w / 2:.2f}" y="{height - 26}" font-size="13" text-anchor="middle" fill="#475569">world x</text>'
        + f'<text x="22" y="{top + draw_h / 2:.2f}" font-size="13" text-anchor="middle" fill="#475569" transform="rotate(-90 22 {top + draw_h / 2:.2f})">world y</text>'
        + "".join(semantic_region_area_parts)
        + "".join(semantic_region_link_parts)
        + "".join(path_parts)
        + "".join(topology_attachment_parts)
        + "".join(topology_parts)
        + "".join(object_parts)
        + "".join(waypoint_parts)
        + "".join(semantic_region_parts)
        + '</svg>'
    )
    svg_path.write_text(svg, encoding="utf-8")

    html_payload = json.dumps(content_payload, ensure_ascii=False, indent=2)
    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Coordinate Navigation Map</title>
  <style>
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background: linear-gradient(180deg, #f8fafc 0%, #eef4ff 100%);
      color: #0f172a;
    }}
    .page {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 24px;
    }}
    .panel {{
      background: rgba(255, 255, 255, 0.84);
      border: 1px solid #d7e3f4;
      border-radius: 18px;
      padding: 20px;
      box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
      margin-bottom: 20px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 30px;
    }}
    p {{
      margin: 6px 0;
      color: #475569;
    }}
    .meta {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin-top: 12px;
      font-size: 14px;
      color: #334155;
    }}
    .meta span {{
      background: #edf4ff;
      border-radius: 999px;
      padding: 8px 12px;
    }}
    img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 14px;
      background: #fff;
    }}
    details {{
      margin-top: 10px;
    }}
    summary {{
      cursor: pointer;
      font-weight: 600;
    }}
    pre {{
      background: #0f172a;
      color: #e2e8f0;
      padding: 16px;
      border-radius: 14px;
      overflow: auto;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="panel">
      <h1>Coordinate Navigation Map</h1>
      <p>The visualization now uses real coordinates on the x/y plane instead of a layered card graph.</p>
      <div class="meta">
        <span>topology: {content_payload["stats"]["topology_nodes"]}</span>
        <span>waypoint: {content_payload["stats"]["waypoint_nodes"]}</span>
        <span>object: {content_payload["stats"]["object_nodes"]}</span>
        <span>semantic regions: {content_payload["stats"].get("semantic_region_nodes", 0)}</span>
        <span>nav objects: {content_payload["stats"]["nav_objects"]}</span>
        <span>dynamic objects: {content_payload["stats"]["dynamic_objects"]}</span>
      </div>
    </div>
    <div class="panel">
      <img src="{svg_path.name}" alt="coordinate navigation map">
    </div>
    <div class="panel">
      <details>
        <summary>Expand graph JSON</summary>
        <pre>{html.escape(html_payload)}</pre>
      </details>
    </div>
  </div>
</body>
</html>
"""
    html_path.write_text(html_doc, encoding="utf-8")


def export_graph_visualization_3d(content_payload: Dict[str, object], html_path: Path) -> None:
    """Export an interactive 3D HTML view using real x/y/z coordinates."""

    topologies = content_payload.get("topologies", [])
    waypoints = content_payload.get("waypoints", [])
    objects = content_payload.get("objects", [])
    semantic_regions = content_payload.get("semantic_regions", [])
    edges = content_payload.get("edges", {})
    if not isinstance(topologies, list):
        topologies = []
    if not isinstance(waypoints, list):
        waypoints = []
    if not isinstance(objects, list):
        objects = []
    if not isinstance(semantic_regions, list):
        semantic_regions = []
    if not isinstance(edges, dict):
        edges = {}

    topology_color_map: Dict[str, str] = {}
    for index, topology in enumerate(sorted(topologies, key=lambda item: str(item.get("topology_id", "")))):
        topology_color_map[str(topology.get("topology_id"))] = topology_palette(index)

    waypoints_by_id = {str(item.get("waypoint_id")): item for item in waypoints}
    topologies_by_id = {str(item.get("topology_id")): item for item in topologies}
    semantic_regions_by_id = {
        str(item.get("semantic_region_id")): item
        for item in semantic_regions
    }
    topology_links = edges.get("topology_topology_inferred", [])
    semantic_topology_edges = edges.get("semantic_topology", [])
    semantic_region_edges = edges.get("semantic_region", [])
    if not isinstance(topology_links, list):
        topology_links = []
    if not isinstance(semantic_topology_edges, list):
        semantic_topology_edges = []
    if not isinstance(semantic_region_edges, list):
        semantic_region_edges = []

    world_z_values: List[float] = []
    for waypoint in waypoints:
        xyz = extract_xyz(waypoint.get("position"))
        if xyz is not None:
            world_z_values.append(xyz[2])
    for obj in objects:
        xyz = extract_xyz(obj.get("centroid"))
        if xyz is not None:
            world_z_values.append(xyz[2])
    for topology in topologies:
        xyz = extract_xyz(topology.get("centroid"))
        if xyz is not None:
            world_z_values.append(xyz[2])
    for semantic_region in semantic_regions:
        xyz = extract_xyz(semantic_region.get("centroid"))
        if xyz is not None:
            world_z_values.append(xyz[2])

    if not world_z_values:
        world_z_values = [0.0]

    min_world_z = min(world_z_values)
    max_world_z = max(world_z_values)
    world_z_span = max(1.0, max_world_z - min_world_z)
    topology_layer_z = max_world_z + max(1.5, world_z_span * 0.45)
    semantic_level_gap = max(1.2, world_z_span * 0.35)
    semantic_level_z = {
        1: topology_layer_z + semantic_level_gap,
        2: topology_layer_z + semantic_level_gap * 2.0,
        3: topology_layer_z + semantic_level_gap * 3.0,
    }

    path_x: List[float] = []
    path_y: List[float] = []
    path_z: List[float] = []
    path_text: List[str] = []
    waypoint_marker_colors: List[str] = []

    waypoint_sequence = content_payload.get("waypoint_sequence", [])
    if not isinstance(waypoint_sequence, list):
        waypoint_sequence = []

    for waypoint_id in waypoint_sequence:
        waypoint = waypoints_by_id.get(str(waypoint_id))
        if waypoint is None:
            continue
        xyz = extract_xyz(waypoint.get("position"))
        if xyz is None:
            continue
        path_x.append(xyz[0])
        path_y.append(xyz[1])
        path_z.append(xyz[2])
        path_text.append(
            "<br>".join(
                [
                    str(waypoint.get("waypoint_id", "-")),
                    f"seq: {waypoint.get('sequence_index', '-')}",
                    f"topology: {waypoint.get('topology_key', '-')}",
                    f"position: {format_position(waypoint.get('position'))}",
                    f"observations: {waypoint.get('observation_count', 0)}",
                ]
            )
        )
        waypoint_marker_colors.append(
            topology_color_map.get(str(waypoint.get("topology_id")), "#2563eb")
        )

    topology_x: List[float] = []
    topology_y: List[float] = []
    topology_z: List[float] = []
    topology_text: List[str] = []
    topology_marker_colors: List[str] = []
    topology_display_positions: Dict[str, tuple[float, float, float]] = {}
    attachment_traces: List[Dict[str, object]] = []
    show_attachment_legend = True

    for topology in topologies:
        topology_id = str(topology.get("topology_id"))
        xyz = extract_xyz(topology.get("centroid"))
        if xyz is None:
            continue
        color = topology_color_map.get(topology_id, "#d97706")
        topology_x.append(xyz[0])
        topology_y.append(xyz[1])
        topology_z.append(topology_layer_z)
        topology_marker_colors.append(color)
        topology_display_positions[topology_id] = (xyz[0], xyz[1], topology_layer_z)
        topology_text.append(
            "<br>".join(
                [
                    str(topology.get("topology_key", topology_id)),
                    f"id: {topology_id}",
                    f"semantic name: {topology.get('region_name', '-')}",
                    f"semantic summary: {topology.get('region_summary', '-')}",
                    f"semantic keywords: {'、'.join(str(item) for item in topology.get('region_keywords', [])[:5]) or '-'}",
                    f"semantic path: {' -> '.join(str(item) for item in topology.get('semantic_region_path', []) if item) or '-'}",
                    f"waypoints: {topology.get('waypoint_count', 0)}",
                    f"objects: {topology.get('object_count', 0)}",
                    f"object zones: {len(topology.get('semantic_object_zones', []) if isinstance(topology.get('semantic_object_zones', []), list) else [])}",
                    f"centroid: {format_position(topology.get('centroid'))}",
                    f"display layer z: {topology_layer_z:.2f}",
                ]
            )
        )

        waypoint_ids = topology.get("waypoint_ids", [])
        if not isinstance(waypoint_ids, list):
            waypoint_ids = []
        attach_x: List[float | None] = []
        attach_y: List[float | None] = []
        attach_z: List[float | None] = []
        for waypoint_id in waypoint_ids:
            waypoint = waypoints_by_id.get(str(waypoint_id))
            if waypoint is None:
                continue
            waypoint_xyz = extract_xyz(waypoint.get("position"))
            if waypoint_xyz is None:
                continue
            attach_x.extend([xyz[0], waypoint_xyz[0], None])
            attach_y.extend([xyz[1], waypoint_xyz[1], None])
            attach_z.extend([topology_layer_z, waypoint_xyz[2], None])
        if attach_x:
            attachment_traces.append(
                {
                    "type": "scatter3d",
                    "mode": "lines",
                    "name": "Topology Attachments",
                    "x": attach_x,
                    "y": attach_y,
                    "z": attach_z,
                    "hoverinfo": "skip",
                    "showlegend": show_attachment_legend,
                    "line": {"color": hex_to_rgba(color, 0.78), "width": 5},
                }
            )
            show_attachment_legend = False

    semantic_display_positions: Dict[str, tuple[float, float, float]] = {}
    semantic_level_nodes: Dict[int, Dict[str, List[object]]] = {
        1: {"x": [], "y": [], "z": [], "text": [], "hovertext": []},
        2: {"x": [], "y": [], "z": [], "text": [], "hovertext": []},
        3: {"x": [], "y": [], "z": [], "text": [], "hovertext": []},
    }
    for semantic_region in semantic_regions:
        semantic_region_id = str(semantic_region.get("semantic_region_id"))
        xyz = extract_xyz(semantic_region.get("centroid"))
        if xyz is None:
            continue
        level = int(semantic_region.get("semantic_level", 0) or 0)
        display_z = semantic_level_z.get(level, topology_layer_z + semantic_level_gap * max(1, level))
        semantic_display_positions[semantic_region_id] = (xyz[0], xyz[1], display_z)
        label = str(semantic_region.get("region_name", semantic_region_id))
        object_names = semantic_region.get("object_names", [])
        if not isinstance(object_names, list):
            object_names = []
        keywords = "、".join(str(name) for name in object_names[:5]) or "-"
        hovertext = "<br>".join(
            [
                semantic_region_id,
                f"level: {level}",
                f"name: {label}",
                f"topologies: {semantic_region.get('topology_count', 0)}",
                f"objects: {keywords}",
                str(semantic_region.get("region_summary", "")),
                f"display layer z: {display_z:.2f}",
            ]
        )
        semantic_level_nodes.setdefault(level, {"x": [], "y": [], "z": [], "text": [], "hovertext": []})
        semantic_level_nodes[level]["x"].append(xyz[0])
        semantic_level_nodes[level]["y"].append(xyz[1])
        semantic_level_nodes[level]["z"].append(display_z)
        semantic_level_nodes[level]["text"].append(label)
        semantic_level_nodes[level]["hovertext"].append(hovertext)

    topology_link_x: List[float | None] = []
    topology_link_y: List[float | None] = []
    topology_link_z: List[float | None] = []
    for link in topology_links:
        if not isinstance(link, dict):
            continue
        source_topology_id = str(link.get("source_topology_id", ""))
        target_topology_id = str(link.get("target_topology_id", ""))
        source_display = topology_display_positions.get(source_topology_id)
        target_display = topology_display_positions.get(target_topology_id)
        if source_display is None or target_display is None:
            continue
        topology_link_x.extend([source_display[0], target_display[0], None])
        topology_link_y.extend([source_display[1], target_display[1], None])
        topology_link_z.extend([source_display[2], target_display[2], None])

    semantic_topology_link_x: List[float | None] = []
    semantic_topology_link_y: List[float | None] = []
    semantic_topology_link_z: List[float | None] = []
    for edge in semantic_topology_edges:
        if not isinstance(edge, dict):
            continue
        semantic_region_id = str(edge.get("semantic_region_id", ""))
        topology_id = str(edge.get("topology_id", ""))
        semantic_display = semantic_display_positions.get(semantic_region_id)
        topology_display = topology_display_positions.get(topology_id)
        if semantic_display is None or topology_display is None:
            continue
        semantic_topology_link_x.extend([semantic_display[0], topology_display[0], None])
        semantic_topology_link_y.extend([semantic_display[1], topology_display[1], None])
        semantic_topology_link_z.extend([semantic_display[2], topology_display[2], None])

    semantic_region_link_x: List[float | None] = []
    semantic_region_link_y: List[float | None] = []
    semantic_region_link_z: List[float | None] = []
    for edge in semantic_region_edges:
        if not isinstance(edge, dict):
            continue
        parent_id = str(edge.get("parent_semantic_region_id", ""))
        child_id = str(edge.get("child_semantic_region_id", ""))
        parent_display = semantic_display_positions.get(parent_id)
        child_display = semantic_display_positions.get(child_id)
        if parent_display is None or child_display is None:
            continue
        semantic_region_link_x.extend([parent_display[0], child_display[0], None])
        semantic_region_link_y.extend([parent_display[1], child_display[1], None])
        semantic_region_link_z.extend([parent_display[2], child_display[2], None])

    nav_obj_x: List[float] = []
    nav_obj_y: List[float] = []
    nav_obj_z: List[float] = []
    nav_obj_text: List[str] = []
    dyn_obj_x: List[float] = []
    dyn_obj_y: List[float] = []
    dyn_obj_z: List[float] = []
    dyn_obj_text: List[str] = []

    for obj in objects:
        xyz = extract_xyz(obj.get("centroid"))
        if xyz is None:
            continue
        tooltip = "<br>".join(
            [
                str(obj.get("object_id", "-")),
                f"class: {obj.get('class_name', '-')}",
                f"topology: {obj.get('topology_id', '-')}",
                f"centroid: {format_position(obj.get('centroid'))}",
                f"observations: {obj.get('observation_count', 0)}",
                f"waypoints: {obj.get('waypoint_count', 0)}",
                f"role: {'dynamic' if obj.get('is_dynamic') else 'navigable'}",
            ]
        )
        if obj.get("is_dynamic"):
            dyn_obj_x.append(xyz[0])
            dyn_obj_y.append(xyz[1])
            dyn_obj_z.append(xyz[2])
            dyn_obj_text.append(tooltip)
        else:
            nav_obj_x.append(xyz[0])
            nav_obj_y.append(xyz[1])
            nav_obj_z.append(xyz[2])
            nav_obj_text.append(tooltip)

    traces = [
        {
            "type": "scatter3d",
            "mode": "lines+markers+text",
            "name": "Waypoint Path",
            "x": path_x,
            "y": path_y,
            "z": path_z,
            "text": [str(index) for index in range(len(path_x))],
            "textposition": "top center",
            "hovertext": path_text,
            "hovertemplate": "%{hovertext}<extra></extra>",
            "line": {"color": "#0f4cdd", "width": 6},
            "marker": {
                "size": 5,
                "color": waypoint_marker_colors,
                "line": {"color": "#ffffff", "width": 1},
            },
        },
    ]

    if topology_link_x:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": "Topology Links",
                "x": topology_link_x,
                "y": topology_link_y,
                "z": topology_link_z,
                "hoverinfo": "skip",
                "showlegend": True,
                "line": {"color": "rgba(30,41,59,0.56)", "width": 5},
            }
        )

    if semantic_topology_link_x:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": "Semantic -> Topology",
                "x": semantic_topology_link_x,
                "y": semantic_topology_link_y,
                "z": semantic_topology_link_z,
                "hoverinfo": "skip",
                "showlegend": True,
                "line": {"color": "rgba(245,158,11,0.34)", "width": 4},
            }
        )

    if semantic_region_link_x:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": "Semantic Hierarchy",
                "x": semantic_region_link_x,
                "y": semantic_region_link_y,
                "z": semantic_region_link_z,
                "hoverinfo": "skip",
                "showlegend": True,
                "line": {"color": "rgba(14,165,233,0.42)", "width": 5},
            }
        )

    traces.extend(attachment_traces)
    for level in (1, 2, 3):
        nodes = semantic_level_nodes.get(level, {})
        if not nodes or not nodes.get("x"):
            continue
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers+text",
                "name": f"Semantic L{level}",
                "x": nodes["x"],
                "y": nodes["y"],
                "z": nodes["z"],
                "text": nodes["text"],
                "textposition": "top center",
                "hovertext": nodes["hovertext"],
                "hovertemplate": "%{hovertext}<extra></extra>",
                "marker": {
                    "size": 6 + level * 2,
                    "symbol": "circle-open",
                    "color": semantic_region_palette(level),
                    "line": {"color": "#ffffff", "width": 1.6},
                },
            }
        )
    traces.extend(
        [
        {
            "type": "scatter3d",
            "mode": "markers+text",
            "name": "Topology Centroids",
            "x": topology_x,
            "y": topology_y,
            "z": topology_z,
            "text": [str(item.get("topology_key", item.get("topology_id", "-"))) for item in topologies if extract_xyz(item.get("centroid")) is not None],
            "textposition": "top center",
            "hovertext": topology_text,
            "hovertemplate": "%{hovertext}<extra></extra>",
            "marker": {
                "size": 7,
                "symbol": "diamond",
                "color": topology_marker_colors,
                "line": {"color": "#ffffff", "width": 1.5},
            },
        },
        {
            "type": "scatter3d",
            "mode": "markers",
            "name": "Navigable Objects",
            "x": nav_obj_x,
            "y": nav_obj_y,
            "z": nav_obj_z,
            "hovertext": nav_obj_text,
            "hovertemplate": "%{hovertext}<extra></extra>",
            "marker": {
                "size": 4,
                "symbol": "square",
                "color": "#16a34a",
                "line": {"color": "#166534", "width": 1},
            },
        },
        {
            "type": "scatter3d",
            "mode": "markers",
            "name": "Dynamic Objects",
            "x": dyn_obj_x,
            "y": dyn_obj_y,
            "z": dyn_obj_z,
            "hovertext": dyn_obj_text,
            "hovertemplate": "%{hovertext}<extra></extra>",
            "marker": {
                "size": 4,
                "symbol": "square",
                "color": "#dc2626",
                "line": {"color": "#7f1d1d", "width": 1},
            },
        },
        ]
    )

    layout = {
        "paper_bgcolor": "#f8fafc",
        "plot_bgcolor": "#ffffff",
        "margin": {"l": 0, "r": 0, "b": 0, "t": 10},
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
            "bgcolor": "rgba(255,255,255,0.72)",
        },
        "scene": {
            "xaxis": {"title": "x", "backgroundcolor": "#f8fbff", "gridcolor": "#d8e2ef", "zerolinecolor": "#bcccdc"},
            "yaxis": {"title": "y", "backgroundcolor": "#f8fbff", "gridcolor": "#d8e2ef", "zerolinecolor": "#bcccdc"},
            "zaxis": {"title": "z", "backgroundcolor": "#f8fbff", "gridcolor": "#d8e2ef", "zerolinecolor": "#bcccdc"},
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.6, "y": 1.45, "z": 1.15}},
        },
    }

    html_payload = json.dumps(content_payload, ensure_ascii=False, indent=2)
    traces_json = json.dumps(traces, ensure_ascii=False)
    layout_json = json.dumps(layout, ensure_ascii=False)
    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>3D Navigation Graph</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background: linear-gradient(180deg, #f8fafc 0%, #eef4ff 100%);
      color: #0f172a;
    }}
    .page {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }}
    .panel {{
      background: rgba(255, 255, 255, 0.84);
      border: 1px solid #d7e3f4;
      border-radius: 18px;
      padding: 20px;
      box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
      margin-bottom: 20px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 30px;
    }}
    p {{
      margin: 6px 0;
      color: #475569;
    }}
    .meta {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin-top: 12px;
      font-size: 14px;
      color: #334155;
    }}
    .meta span {{
      background: #edf4ff;
      border-radius: 999px;
      padding: 8px 12px;
    }}
    #plot {{
      width: 100%;
      height: 860px;
    }}
    details {{
      margin-top: 10px;
    }}
    summary {{
      cursor: pointer;
      font-weight: 600;
    }}
    pre {{
      background: #0f172a;
      color: #e2e8f0;
      padding: 16px;
      border-radius: 14px;
      overflow: auto;
      font-size: 13px;
    }}
    .note {{
      font-size: 13px;
      color: #64748b;
    }}
  </style>
</head>
  <body>
  <div class="page">
    <div class="panel">
      <h1>3D Navigation Graph</h1>
      <p>This view uses real x/y coordinates. Topology nodes are lifted onto a separate display layer in z, and topology links are inferred from adjacent waypoint connections.</p>
      <div class="meta">
        <span>topology: {content_payload["stats"]["topology_nodes"]}</span>
        <span>waypoint: {content_payload["stats"]["waypoint_nodes"]}</span>
        <span>object: {content_payload["stats"]["object_nodes"]}</span>
        <span>semantic regions: {content_payload["stats"].get("semantic_region_nodes", 0)}</span>
        <span>temporal edges: {content_payload["stats"]["temporal_edges"]}</span>
        <span>topology links: {len(topology_links)}</span>
        <span>topology layer z: {topology_layer_z:.2f}</span>
        <span>semantic top z: {semantic_level_z.get(3, topology_layer_z):.2f}</span>
      </div>
      <p class="note">This 3D page loads Plotly from the official CDN when opened.</p>
    </div>
    <div class="panel">
      <div id="plot"></div>
    </div>
    <div class="panel">
      <details>
        <summary>Expand graph JSON</summary>
        <pre>{html.escape(html_payload)}</pre>
      </details>
    </div>
  </div>
  <script>
    const traces = {traces_json};
    const layout = {layout_json};
    Plotly.newPlot('plot', traces, layout, {{
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['select2d', 'lasso2d']
    }});
  </script>
</body>
</html>
"""
    html_path.write_text(html_doc, encoding="utf-8")


def object3d_axis_aligned_corners(center: object, extent: object) -> Optional[List[List[float]]]:
    """Build 8 axis-aligned 3D box corners from object3d_engine center/extent."""

    center_xyz = extract_xyz(center)
    extent_xyz = extract_xyz(extent)
    if center_xyz is None or extent_xyz is None:
        return None

    half_x = max(abs(extent_xyz[0]) / 2.0, 0.03)
    half_y = max(abs(extent_xyz[1]) / 2.0, 0.03)
    half_z = max(abs(extent_xyz[2]) / 2.0, 0.03)
    cx, cy, cz = center_xyz
    return [
        [cx - half_x, cy - half_y, cz - half_z],
        [cx - half_x, cy - half_y, cz + half_z],
        [cx - half_x, cy + half_y, cz - half_z],
        [cx + half_x, cy - half_y, cz - half_z],
        [cx + half_x, cy + half_y, cz + half_z],
        [cx + half_x, cy + half_y, cz - half_z],
        [cx + half_x, cy - half_y, cz + half_z],
        [cx - half_x, cy + half_y, cz + half_z],
    ]


def object3d_bbox_line_payload(corners: List[List[float]]) -> tuple[List[float | None], List[float | None], List[float | None]]:
    edge_indexes = (
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 6),
        (1, 7),
        (2, 5),
        (2, 7),
        (3, 5),
        (3, 6),
        (4, 5),
        (4, 6),
        (4, 7),
    )
    xs: List[float | None] = []
    ys: List[float | None] = []
    zs: List[float | None] = []
    for start, end in edge_indexes:
        xs.extend([corners[start][0], corners[end][0], None])
        ys.extend([corners[start][1], corners[end][1], None])
        zs.extend([corners[start][2], corners[end][2], None])
    return xs, ys, zs


def extract_bbox_corners(value: object) -> Optional[List[List[float]]]:
    if not isinstance(value, list) or len(value) < 8:
        return None

    corners: List[List[float]] = []
    for item in value[:8]:
        xyz = extract_xyz(item)
        if xyz is None:
            return None
        corners.append([xyz[0], xyz[1], xyz[2]])
    return corners


def extract_point_sample(value: object, max_points: int = 700) -> List[List[float]]:
    if not isinstance(value, list):
        return []

    points: List[List[float]] = []
    for item in value:
        xyz = extract_xyz(item)
        if xyz is None:
            continue
        points.append([xyz[0], xyz[1], xyz[2]])
        if len(points) >= max_points:
            break
    return points


def export_object3d_global_visualization(content_payload: Dict[str, object], html_path: Path) -> None:
    """Export an object-level 3D map using object3d_engine bbox and point samples."""

    waypoints = content_payload.get("waypoints", [])
    objects = content_payload.get("objects", [])
    if not isinstance(waypoints, list):
        waypoints = []
    if not isinstance(objects, list):
        objects = []

    waypoint_sequence = content_payload.get("waypoint_sequence", [])
    if not isinstance(waypoint_sequence, list):
        waypoint_sequence = []
    waypoints_by_id = {str(item.get("waypoint_id")): item for item in waypoints if isinstance(item, dict)}

    path_x: List[float] = []
    path_y: List[float] = []
    path_z: List[float] = []
    path_hover: List[str] = []
    for waypoint_id in waypoint_sequence:
        waypoint = waypoints_by_id.get(str(waypoint_id))
        if waypoint is None:
            continue
        xyz = extract_xyz(waypoint.get("position"))
        if xyz is None:
            continue
        path_x.append(xyz[0])
        path_y.append(xyz[1])
        path_z.append(xyz[2])
        path_hover.append(
            "<br>".join(
                [
                    str(waypoint.get("waypoint_id", "-")),
                    f"seq: {waypoint.get('sequence_index', '-')}",
                    f"topology: {waypoint.get('topology_key', '-')}",
                    f"position: {format_position(waypoint.get('position'))}",
                    f"observations: {waypoint.get('observation_count', 0)}",
                ]
            )
        )

    traces: List[Dict[str, object]] = []
    if path_x:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines+markers",
                "name": "camera / waypoint trajectory",
                "x": path_x,
                "y": path_y,
                "z": path_z,
                "hovertext": path_hover,
                "hovertemplate": "%{hovertext}<extra></extra>",
                "line": {"color": "#202020", "width": 5},
                "marker": {"size": 4, "color": "#202020"},
            }
        )

    object_count_with_bbox = 0
    object_count_with_points = 0
    for index, obj in enumerate(item for item in objects if isinstance(item, dict)):
        centroid = extract_xyz(obj.get("centroid"))
        if centroid is None:
            continue
        color = topology_palette(index)
        object_id = str(obj.get("object_id", f"object:{index + 1}"))
        class_name = str(obj.get("class_name", "unknown"))
        hover = "<br>".join(
            [
                f"{object_id} {class_name}",
                f"centroid: {format_position(obj.get('centroid'))}",
                f"bbox center: {format_position(obj.get('bbox_3d_center'))}",
                f"bbox extent: {format_position(obj.get('bbox_3d_extent'))}",
                f"observations: {obj.get('observation_count', 0)}",
                f"method: {obj.get('global_position_method', '-')}",
                f"lifecycle: {obj.get('object3d_lifecycle_state', '-')}",
                f"first seen: {obj.get('object3d_first_seen_timestamp', obj.get('first_seen', '-'))}",
                f"last seen: {obj.get('object3d_last_seen_timestamp', obj.get('last_seen', '-'))}",
                f"duration(s): {obj.get('object3d_observed_duration_s', '-')}",
                f"support points(avg): {obj.get('mean_support_point_count', '-')}",
                f"mask area(avg): {obj.get('mean_mask_area', '-')}",
            ]
        )

        points = extract_point_sample(obj.get("global_points_sample"))
        if points:
            object_count_with_points += 1
            traces.append(
                {
                    "type": "scatter3d",
                    "mode": "markers",
                    "name": f"{object_id} points",
                    "x": [point[0] for point in points],
                    "y": [point[1] for point in points],
                    "z": [point[2] for point in points],
                    "hoverinfo": "skip",
                    "marker": {"size": 2.4, "color": color, "opacity": 0.42},
                    "showlegend": False,
                }
            )

        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers+text",
                "name": f"{object_id} {class_name}",
                "x": [centroid[0]],
                "y": [centroid[1]],
                "z": [centroid[2]],
                "text": [class_name],
                "textposition": "top center",
                "hovertext": [hover],
                "hovertemplate": "%{hovertext}<extra></extra>",
                "marker": {"size": 7, "symbol": "diamond", "color": color},
            }
        )

        bbox_center = obj.get("bbox_3d_center") or obj.get("centroid")
        corners = extract_bbox_corners(obj.get("bbox_3d_corners"))
        if corners is None:
            corners = object3d_axis_aligned_corners(bbox_center, obj.get("bbox_3d_extent"))
        if corners is None:
            continue
        object_count_with_bbox += 1
        bbox_x, bbox_y, bbox_z = object3d_bbox_line_payload(corners)
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": f"{object_id} bbox",
                "x": bbox_x,
                "y": bbox_y,
                "z": bbox_z,
                "hoverinfo": "skip",
                "line": {"color": color, "width": 5},
                "showlegend": False,
            }
        )

    layout = {
        "title": "code_made object3d_engine global object map",
        "paper_bgcolor": "#f8fafc",
        "plot_bgcolor": "#ffffff",
        "margin": {"l": 0, "r": 0, "b": 0, "t": 50},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        "scene": {
            "xaxis": {"title": "world x (m)", "backgroundcolor": "#f8fbff", "gridcolor": "#d8e2ef"},
            "yaxis": {"title": "world y (m)", "backgroundcolor": "#f8fbff", "gridcolor": "#d8e2ef"},
            "zaxis": {"title": "world z (m)", "backgroundcolor": "#f8fbff", "gridcolor": "#d8e2ef"},
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.65, "y": 1.45, "z": 1.05}},
        },
    }

    traces_json = json.dumps(traces, ensure_ascii=False)
    layout_json = json.dumps(layout, ensure_ascii=False)
    html_payload = json.dumps(
        {
            "graph_path": content_payload.get("graph_path"),
            "waypoint_count": len(path_x),
            "object_count": len(objects),
            "raw_object_count": content_payload.get("raw_object_count"),
            "postprocess": content_payload.get("postprocess"),
            "objects_with_bbox": object_count_with_bbox,
            "objects_with_point_sample": object_count_with_points,
        },
        ensure_ascii=False,
        indent=2,
    )
    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Object3D Global Map</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background: #f8fafc;
      color: #0f172a;
    }}
    .page {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }}
    .panel {{
      background: rgba(255, 255, 255, 0.88);
      border: 1px solid #d7e3f4;
      border-radius: 18px;
      padding: 18px 20px;
      box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
      margin-bottom: 18px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    p {{ margin: 6px 0; color: #475569; }}
    #plot {{ width: 100%; height: 850px; }}
    pre {{
      background: #0f172a;
      color: #e2e8f0;
      padding: 14px;
      border-radius: 12px;
      overflow: auto;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="panel">
      <h1>Object3D Global Map</h1>
      <p>该视图使用 code_made 中已经替换后的 object3d_engine 结果：轨迹、物体中心、深度感知 bbox mask 生成的 3DBox，以及每个物体保留的少量全局点云采样。</p>
      <pre>{html.escape(html_payload)}</pre>
    </div>
    <div class="panel"><div id="plot"></div></div>
  </div>
  <script>
    const traces = {traces_json};
    const layout = {layout_json};
    Plotly.newPlot('plot', traces, layout, {{
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['select2d', 'lasso2d']
    }});
  </script>
</body>
</html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_doc, encoding="utf-8")


def export_object3d_tracking_visualization(tracking_payload: Dict[str, object], html_path: Path) -> None:
    """Export object-level 3D HTML from object3d_engine tracking summary."""

    camera_positions = tracking_payload.get("camera_positions", [])
    if not isinstance(camera_positions, list):
        camera_positions = []

    waypoints: List[Dict[str, object]] = []
    waypoint_sequence: List[str] = []
    for index, position in enumerate(camera_positions):
        xyz = extract_xyz(position)
        if xyz is None:
            continue
        waypoint_id = f"camera:{index}"
        waypoint_sequence.append(waypoint_id)
        waypoints.append(
            {
                "waypoint_id": waypoint_id,
                "sequence_index": index,
                "topology_key": "object3d_tracking",
                "position": [xyz[0], xyz[1], xyz[2]],
                "observation_count": 0,
            }
        )

    objects = tracking_payload.get("objects", [])
    if not isinstance(objects, list):
        objects = []
    normalized_objects: List[Dict[str, object]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        normalized = dict(obj)
        if not normalized.get("class_name"):
            normalized["class_name"] = normalized.get("dominant_class_name", "unknown")
        normalized.setdefault("global_position_method", "object3d_engine_tracking")
        normalized_objects.append(normalized)

    content_payload = {
        "graph_path": tracking_payload.get("summary_path", ""),
        "waypoint_sequence": waypoint_sequence,
        "waypoints": waypoints,
        "objects": normalized_objects,
        "raw_object_count": tracking_payload.get("raw_object_count"),
        "postprocess": tracking_payload.get("postprocess"),
    }
    export_object3d_global_visualization(content_payload, html_path)
