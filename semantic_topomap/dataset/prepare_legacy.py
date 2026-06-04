#!/usr/bin/env python3
"""Prepare dataset_test3 in the layout expected by run_semantic_topomap.py."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


VLP16_VERTICAL_ANGLES_DEG = np.array(
    [-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="/home/zyf/Desktop/dataset_test3")
    parser.add_argument("--output", default="/home/zyf/code_made/core_content/dataset_test3_cuvslam")
    parser.add_argument("--limit", type=int, default=24, help="Rows to adapt for a quick test; <=0 keeps all rows")
    parser.add_argument(
        "--slam-trajectory",
        default="",
        help="Optional cuVSLAM trajectory.csv used to replace placeholder poses",
    )
    parser.add_argument(
        "--slam-max-dt-ms",
        type=float,
        default=50.0,
        help="Maximum allowed nearest-neighbor timestamp difference for SLAM pose matching",
    )
    parser.add_argument(
        "--slam-axis-map",
        choices=["zed_to_code", "xyz"],
        default="zed_to_code",
        help="Axis mapping from cuVSLAM/ZED pose to code_made world pose",
    )
    return parser.parse_args()


def ensure_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.resolve() == src.resolve():
            return
        dst.unlink()
    dst.symlink_to(src)


def decode_vlp16_packets(packet_bytes: bytes) -> np.ndarray:
    """Decode common 1206-byte Velodyne packets to x/y/z/intensity float32 points.

    This is intentionally a lightweight decoder for compatibility testing. It
    assumes VLP-16 firing layout and keeps the original sensor frame.
    """
    packet_size = 1206
    usable_len = (len(packet_bytes) // packet_size) * packet_size
    if usable_len == 0:
        return np.empty((0, 4), dtype=np.float32)

    points: list[tuple[float, float, float, float]] = []
    vertical = np.deg2rad(VLP16_VERTICAL_ANGLES_DEG)
    sin_v = np.sin(vertical)
    cos_v = np.cos(vertical)

    for packet_start in range(0, usable_len, packet_size):
        packet = packet_bytes[packet_start : packet_start + packet_size]
        for block_idx in range(12):
            block_start = block_idx * 100
            if packet[block_start : block_start + 2] != b"\xff\xee":
                continue

            azimuth_raw = int.from_bytes(packet[block_start + 2 : block_start + 4], "little")
            next_block_start = (block_idx + 1) * 100
            if block_idx < 11 and packet[next_block_start : next_block_start + 2] == b"\xff\xee":
                next_azimuth_raw = int.from_bytes(packet[next_block_start + 2 : next_block_start + 4], "little")
                azimuth_delta = (next_azimuth_raw - azimuth_raw) % 36000
            else:
                azimuth_delta = 0

            data_start = block_start + 4
            for firing in range(2):
                firing_azimuth_raw = (azimuth_raw + firing * azimuth_delta / 2.0) % 36000
                azimuth = math.radians(firing_azimuth_raw / 100.0)
                cos_a = math.cos(azimuth)
                sin_a = math.sin(azimuth)
                for laser in range(16):
                    offset = data_start + (firing * 16 + laser) * 3
                    distance_raw = int.from_bytes(packet[offset : offset + 2], "little")
                    if distance_raw == 0:
                        continue
                    distance_m = distance_raw * 0.002
                    intensity = float(packet[offset + 2])
                    xy = distance_m * float(cos_v[laser])
                    x = xy * sin_a
                    y = xy * cos_a
                    z = distance_m * float(sin_v[laser])
                    points.append((x, y, z, intensity))

    if not points:
        return np.empty((0, 4), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def load_slam_inputs(source: Path, slam_trajectory_path: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if not slam_trajectory_path:
        return None, None

    trajectory_path = Path(slam_trajectory_path).expanduser().resolve()
    if not trajectory_path.exists():
        raise FileNotFoundError(trajectory_path)

    frames_path = source / "front_camera" / "frames.csv"
    if not frames_path.exists():
        raise FileNotFoundError(frames_path)

    frames = pd.read_csv(frames_path)
    trajectory = pd.read_csv(trajectory_path)
    required_frame_cols = {"front_frame_id", "zed_image_timestamp_ns"}
    required_traj_cols = {"frame_id", "timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw"}
    if missing := sorted(required_frame_cols - set(frames.columns)):
        raise ValueError(f"front_camera/frames.csv missing required columns: {missing}")
    if missing := sorted(required_traj_cols - set(trajectory.columns)):
        raise ValueError(f"SLAM trajectory missing required columns: {missing}")

    frames = frames.copy()
    trajectory = trajectory.copy()
    frames["front_frame_id"] = frames["front_frame_id"].astype(int)
    frames["zed_image_timestamp_ns"] = frames["zed_image_timestamp_ns"].astype(np.int64)
    trajectory["timestamp_ns"] = trajectory["timestamp_ns"].astype(np.int64)
    trajectory = trajectory.sort_values("timestamp_ns").reset_index(drop=True)
    return frames, trajectory


def remap_slam_pose(pose_row: pd.Series, axis_map: str) -> dict[str, float]:
    position = np.array([pose_row["tx"], pose_row["ty"], pose_row["tz"]], dtype=np.float64)
    quat = np.array([pose_row["qx"], pose_row["qy"], pose_row["qz"], pose_row["qw"]], dtype=np.float64)

    if axis_map == "xyz":
        mapped_position = position
        mapped_quat = quat
    elif axis_map == "zed_to_code":
        # ZED/cuVSLAM uses x-right, y-down, z-forward. The mapping pipeline
        # treats x/y as the ground plane and z as vertical: x-right, y-forward, z-up.
        basis = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )
        mapped_position = basis @ position
        mapped_rotation = basis @ Rotation.from_quat(quat).as_matrix() @ basis.T
        mapped_quat = Rotation.from_matrix(mapped_rotation).as_quat()
    else:
        raise ValueError(f"Unsupported slam axis map: {axis_map}")

    return {
        "tx": float(mapped_position[0]),
        "ty": float(mapped_position[1]),
        "tz": float(mapped_position[2]),
        "qx": float(mapped_quat[0]),
        "qy": float(mapped_quat[1]),
        "qz": float(mapped_quat[2]),
        "qw": float(mapped_quat[3]),
    }


def match_slam_pose(
    *,
    front_frame_id: int,
    frames_by_front_id: dict[int, pd.Series],
    trajectory: pd.DataFrame,
    max_dt_ns: int,
    axis_map: str,
) -> dict[str, float | int | str]:
    frame_row = frames_by_front_id.get(int(front_frame_id))
    if frame_row is None:
        raise KeyError(f"front_frame_id {front_frame_id} not found in front_camera/frames.csv")

    target_ts = int(frame_row["zed_image_timestamp_ns"])
    timestamps = trajectory["timestamp_ns"].to_numpy(dtype=np.int64)
    insert_at = int(np.searchsorted(timestamps, target_ts))
    candidates = []
    if insert_at < len(timestamps):
        candidates.append(insert_at)
    if insert_at > 0:
        candidates.append(insert_at - 1)
    if not candidates:
        raise ValueError("SLAM trajectory is empty")

    best_index = min(candidates, key=lambda idx: abs(int(timestamps[idx]) - target_ts))
    pose_row = trajectory.iloc[best_index]
    dt_ns = abs(int(pose_row["timestamp_ns"]) - target_ts)
    if dt_ns > max_dt_ns:
        raise ValueError(
            f"SLAM timestamp mismatch for front_frame_id={front_frame_id}: "
            f"target={target_ts}, matched={int(pose_row['timestamp_ns'])}, dt_ns={dt_ns}"
        )

    mapped_pose = remap_slam_pose(pose_row, axis_map)
    mapped_pose.update(
        {
            "pose_source": "cuvslam",
            "slam_frame_id": int(pose_row["frame_id"]),
            "slam_timestamp_ns": int(pose_row["timestamp_ns"]),
            "slam_match_dt_ns": int(dt_ns),
            "zed_image_timestamp_ns": target_ts,
            "slam_axis_map": axis_map,
        }
    )
    return mapped_pose


def main() -> int:
    args = parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    alignment_path = source / "sync" / "alignment.csv"
    if not alignment_path.exists():
        raise FileNotFoundError(alignment_path)

    rows = pd.read_csv(alignment_path)
    if args.limit > 0:
        rows = rows.head(args.limit).copy()

    slam_frames, slam_trajectory = load_slam_inputs(source, args.slam_trajectory)
    frames_by_front_id = (
        {int(row["front_frame_id"]): row for _, row in slam_frames.iterrows()} if slam_frames is not None else {}
    )
    max_dt_ns = int(float(args.slam_max_dt_ms) * 1_000_000)

    output.mkdir(parents=True, exist_ok=True)
    track_rows = []
    for idx, row in rows.iterrows():
        timestamp = str(int(row["timestamp_ns"]))
        front_src = (source / "sync" / str(row["front_image_path"])).resolve()
        rear_src = (source / "sync" / str(row["rear_rgb_path"])).resolve()
        lidar_src = (source / "sync" / str(row["lidar_scan_path"])).resolve()
        depth_src = (source / "depth" / f"{timestamp}.png").resolve()

        ensure_symlink(front_src, output / "front_cam" / f"{timestamp}.png")
        ensure_symlink(rear_src, output / "back_cam" / f"{timestamp}.png")

        points = decode_vlp16_packets(lidar_src.read_bytes())
        if points.size == 0:
            points = np.zeros((1, 4), dtype=np.float32)
        (output / "lidar").mkdir(exist_ok=True)
        points.tofile(output / "lidar" / f"{timestamp}.bin")

        if not depth_src.exists():
            raise FileNotFoundError(f"recorded depth not found: {depth_src}")
        ensure_symlink(depth_src, output / "depth" / f"{timestamp}.png")

        if slam_trajectory is not None:
            pose = match_slam_pose(
                front_frame_id=int(row["front_frame_id"]),
                frames_by_front_id=frames_by_front_id,
                trajectory=slam_trajectory,
                max_dt_ns=max_dt_ns,
                axis_map=args.slam_axis_map,
            )
        else:
            # dataset_test3 does not contain robot/world poses, so this is a
            # deterministic placeholder trajectory for pipeline testing.
            pose = {
                "tx": float(idx) * 0.5,
                "ty": 0.0,
                "tz": 0.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
                "pose_source": "placeholder",
            }

        track_rows.append(
            {
                "timestamp": timestamp,
                "front_cam_ts": timestamp,
                "back_cam_ts": timestamp,
                "lidar_ts": timestamp,
                **pose,
            }
        )

    with (output / "track.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(track_rows[0].keys()))
        writer.writeheader()
        writer.writerows(track_rows)

    summary = {
        "source": str(source),
        "output": str(output),
        "frames": len(track_rows),
        "pose_source": "cuvslam" if slam_trajectory is not None else "placeholder",
        "slam_trajectory": str(Path(args.slam_trajectory).resolve()) if args.slam_trajectory else "",
        "slam_axis_map": args.slam_axis_map if slam_trajectory is not None else "",
        "slam_max_dt_ms": float(args.slam_max_dt_ms) if slam_trajectory is not None else None,
    }
    if slam_trajectory is not None:
        match_dts = [int(row["slam_match_dt_ns"]) for row in track_rows]
        summary.update(
            {
                "slam_match_dt_ns_max": max(match_dts),
                "slam_match_dt_ns_mean": float(np.mean(match_dts)),
                "slam_frame_id_min": min(int(row["slam_frame_id"]) for row in track_rows),
                "slam_frame_id_max": max(int(row["slam_frame_id"]) for row in track_rows),
            }
        )
    (output / "pose_alignment_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"prepared: {output}")
    print(f"frames: {len(track_rows)}")
    print(f"files: {sum(1 for _ in output.rglob('*') if _.is_file() or _.is_symlink())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
