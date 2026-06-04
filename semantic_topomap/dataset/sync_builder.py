from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


AXIS_BASIS = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)

VLP16_VERTICAL_ANGLES_DEG = np.array(
    [-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15],
    dtype=np.float32,
)


def decode_vlp16_packets(packet_bytes: bytes) -> np.ndarray:
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
    return np.asarray(points, dtype=np.float32) if points else np.empty((0, 4), dtype=np.float32)


def zed_to_code_pose(row: pd.Series) -> dict[str, float]:
    position = np.array([row["tx"], row["ty"], row["tz"]], dtype=np.float64)
    quat = np.array([row["qx"], row["qy"], row["qz"], row["qw"]], dtype=np.float64)
    mapped_position = AXIS_BASIS @ position
    mapped_rotation = AXIS_BASIS @ Rotation.from_quat(quat).as_matrix() @ AXIS_BASIS.T
    mapped_quat = Rotation.from_matrix(mapped_rotation).as_quat()
    return {
        "tx": float(mapped_position[0]),
        "ty": float(mapped_position[1]),
        "tz": float(mapped_position[2]),
        "qx": float(mapped_quat[0]),
        "qy": float(mapped_quat[1]),
        "qz": float(mapped_quat[2]),
        "qw": float(mapped_quat[3]),
    }


def _nearest_row_by_timestamp(df: pd.DataFrame, column: str, value: int) -> tuple[pd.Series, int]:
    timestamps = df[column].to_numpy(dtype=np.int64)
    idx = int(np.searchsorted(timestamps, int(value)))
    candidates = []
    if idx < len(timestamps):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    if not candidates:
        raise ValueError(f"empty timestamp table for {column}")
    best = min(candidates, key=lambda i: abs(int(timestamps[i]) - int(value)))
    return df.iloc[best], abs(int(timestamps[best]) - int(value))


def _parse_timestamp_from_name(path: Path) -> int | None:
    try:
        return int(path.stem)
    except Exception:
        return None


def _media_table(path: Path, patterns: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for pattern in patterns:
        for item in sorted(path.glob(pattern)):
            ts = _parse_timestamp_from_name(item)
            if ts is not None:
                rows.append({"timestamp_ns": ts, "path": str(item)})
    return pd.DataFrame(rows).sort_values("timestamp_ns").reset_index(drop=True) if rows else pd.DataFrame()


def _choose_media_timestamp(table: pd.DataFrame, *, jetson_ts: int, zed_ts: int) -> int:
    if table.empty:
        return jetson_ts
    first = int(table["timestamp_ns"].iloc[0])
    return zed_ts if abs(first - int(zed_ts)) < abs(first - int(jetson_ts)) else jetson_ts


def build_track_from_svo_index(
    *,
    dataset: Path,
    extracted_dir: Path,
    cuvslam_dir: Path,
    prepared_dir: Path,
    stride: int = 15,
    max_dt_ms: float = 80.0,
    use_alignment: bool = True,
    max_rows: int = 0,
) -> dict:
    dataset = dataset.resolve()
    extracted_dir = extracted_dir.resolve()
    cuvslam_dir = cuvslam_dir.resolve()
    prepared_dir = prepared_dir.resolve()
    prepared_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("front_cam", "back_cam", "lidar", "depth"):
        (prepared_dir / sub).mkdir(parents=True, exist_ok=True)

    zed_index = pd.read_csv(dataset / "zed" / "zed_svo_index.csv").sort_values("frame_id").reset_index(drop=True)
    trajectory = pd.read_csv(cuvslam_dir / "trajectory.csv").sort_values("timestamp_ns").reset_index(drop=True)
    rear_table = _media_table(dataset / "rear_camera" / "rgb", ("*.png", "*.jpg", "*.jpeg"))
    lidar_table = _media_table(dataset / "lidar" / "scans", ("*.bin", "*.pcd", "*.ply", "*.npy"))

    front_dir = extracted_dir / "front_rgb"
    depth_dir = extracted_dir / "front_depth"
    if not front_dir.exists():
        fallback = dataset / "front_camera" / "rgb"
        if fallback.exists():
            front_dir = fallback
    if not depth_dir.exists():
        fallback = dataset / "depth"
        if fallback.exists():
            depth_dir = fallback
    front_table = _media_table(front_dir, ("*.png", "*.jpg", "*.jpeg"))
    depth_table = _media_table(depth_dir, ("*.png", "*.tiff", "*.tif", "*.npy"))

    alignment_path = dataset / "sync" / "alignment.csv"
    if use_alignment and alignment_path.exists():
        alignment = pd.read_csv(alignment_path)
        if "front_recorded_frame_index" in alignment.columns:
            frame_ids = set(int(v) for v in alignment["front_recorded_frame_index"].dropna().astype(int).tolist())
            selected = zed_index[zed_index["recorded_frame_index"].astype(int).isin(frame_ids)].copy()
        else:
            selected = zed_index.iloc[:: max(1, int(stride))].copy()
    else:
        selected = zed_index.iloc[:: max(1, int(stride))].copy()
    if int(max_rows) > 0:
        selected = selected.head(int(max_rows)).copy()
    max_dt_ns = int(float(max_dt_ms) * 1_000_000)
    rows = []
    skipped = 0
    for _, zed_row in selected.iterrows():
        jetson_ts = int(zed_row["jetson_timestamp_ns"])
        zed_ts = int(zed_row["zed_image_timestamp_ns"])
        frame_id = int(zed_row["frame_id"])
        timestamp = str(jetson_ts)

        front_candidates = [
            front_dir / f"{jetson_ts}.png",
            front_dir / f"{zed_ts}.png",
            front_dir / f"{frame_id}.png",
            front_dir / f"{frame_id:06d}.png",
        ]
        depth_candidates = [
            depth_dir / f"{jetson_ts}.png",
            depth_dir / f"{zed_ts}.png",
            depth_dir / f"{frame_id}.png",
            depth_dir / f"{frame_id:06d}.png",
        ]
        front_src = next((p for p in front_candidates if p.exists()), None)
        if front_src is None and not front_table.empty:
            front_target_ts = _choose_media_timestamp(front_table, jetson_ts=jetson_ts, zed_ts=zed_ts)
            front_row, front_dt = _nearest_row_by_timestamp(front_table, "timestamp_ns", front_target_ts)
            if front_dt <= max_dt_ns:
                front_src = Path(str(front_row["path"]))
        depth_src = next((p for p in depth_candidates if p.exists()), None)
        if depth_src is None and not depth_table.empty:
            depth_target_ts = _choose_media_timestamp(depth_table, jetson_ts=jetson_ts, zed_ts=zed_ts)
            depth_row, depth_dt = _nearest_row_by_timestamp(depth_table, "timestamp_ns", depth_target_ts)
            if depth_dt <= max_dt_ns:
                depth_src = Path(str(depth_row["path"]))
        if front_src is None or depth_src is None:
            skipped += 1
            continue

        pose_row, pose_dt = _nearest_row_by_timestamp(trajectory, "timestamp_ns", zed_ts)
        if pose_dt > max_dt_ns:
            skipped += 1
            continue
        pose = zed_to_code_pose(pose_row)

        rear_src = None
        rear_dt = None
        if not rear_table.empty:
            rear_row, rear_dt = _nearest_row_by_timestamp(rear_table, "timestamp_ns", jetson_ts)
            rear_src = Path(str(rear_row["path"]))
        lidar_src = None
        lidar_dt = None
        if not lidar_table.empty:
            lidar_row, lidar_dt = _nearest_row_by_timestamp(lidar_table, "timestamp_ns", jetson_ts)
            lidar_src = Path(str(lidar_row["path"]))

        front_dst = prepared_dir / "front_cam" / f"{timestamp}.png"
        depth_dst = prepared_dir / "depth" / f"{timestamp}.png"
        rear_dst = prepared_dir / "back_cam" / f"{timestamp}.png"
        lidar_dst = prepared_dir / "lidar" / f"{timestamp}.bin"
        _copy_or_link(front_src, front_dst)
        _copy_or_link(depth_src, depth_dst)
        if rear_src is not None:
            _copy_or_link(rear_src, rear_dst)
        else:
            _copy_or_link(front_src, rear_dst)
        if lidar_src is not None:
            _write_lidar_points(lidar_src, lidar_dst)
        else:
            np.zeros((1, 4), dtype=np.float32).tofile(lidar_dst)

        rows.append(
            {
                "timestamp": timestamp,
                "front_cam_ts": timestamp,
                "back_cam_ts": timestamp,
                "lidar_ts": timestamp,
                "depth_ts": timestamp,
                "tx": pose["tx"],
                "ty": pose["ty"],
                "tz": pose["tz"],
                "qx": pose["qx"],
                "qy": pose["qy"],
                "qz": pose["qz"],
                "qw": pose["qw"],
                "pose_source": "cuvslam",
                "slam_frame_id": int(pose_row["frame_id"]),
                "slam_timestamp_ns": int(pose_row["timestamp_ns"]),
                "slam_match_dt_ns": int(pose_dt),
                "zed_frame_id": frame_id,
                "zed_image_timestamp_ns": zed_ts,
                "jetson_timestamp_ns": jetson_ts,
                "rear_match_dt_ns": "" if rear_dt is None else int(rear_dt),
                "lidar_match_dt_ns": "" if lidar_dt is None else int(lidar_dt),
                "slam_axis_map": "zed_to_code",
            }
        )

    fieldnames = list(rows[0].keys()) if rows else []
    if rows:
        with (prepared_dir / "track.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "prepared_dir": str(prepared_dir),
        "input_zed_rows": int(len(zed_index)),
        "stride": int(stride),
        "use_alignment": bool(use_alignment),
        "max_rows": int(max_rows),
        "selected_rows": int(len(selected)),
        "track_rows": int(len(rows)),
        "skipped_rows": int(skipped),
        "front_source": str(front_dir),
        "depth_source": str(depth_dir),
    }
    (prepared_dir / "prepare_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except Exception:
        shutil.copy2(src, dst)


def _write_lidar_points(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".bin":
        raw = src.read_bytes()
        points = decode_vlp16_packets(raw)
        if points.size == 0:
            # If the file is already a float32 point cloud, keep that format.
            arr = np.frombuffer(raw, dtype=np.float32)
            if arr.size >= 4 and arr.size % 4 == 0:
                points = arr.reshape(-1, 4).astype(np.float32)
        if points.size == 0:
            points = np.zeros((1, 4), dtype=np.float32)
        points.astype(np.float32).tofile(dst)
        return
    _copy_or_link(src, dst)
