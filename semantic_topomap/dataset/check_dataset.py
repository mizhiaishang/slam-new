from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REQUIRED_ZED_INDEX_COLUMNS = {
    "frame_id",
    "jetson_timestamp_ns",
    "zed_image_timestamp_ns",
    "recorded_frame_index",
}


def _count_files(path: Path, patterns: tuple[str, ...]) -> int:
    if not path.exists():
        return 0
    total = 0
    for pattern in patterns:
        total += sum(1 for _ in path.glob(pattern))
    return total


def check_dataset(dataset: Path, *, require_rear: bool = True, require_lidar: bool = True) -> dict:
    dataset = dataset.resolve()
    zed_dir = dataset / "zed"
    svo2 = zed_dir / "zed.svo2"
    zed_index = zed_dir / "zed_svo_index.csv"
    rear_rgb = dataset / "rear_camera" / "rgb"
    lidar_scans = dataset / "lidar" / "scans"

    report: dict = {
        "dataset": str(dataset),
        "ok": True,
        "errors": [],
        "warnings": [],
        "paths": {
            "svo2": str(svo2),
            "zed_svo_index": str(zed_index),
            "rear_rgb": str(rear_rgb),
            "lidar_scans": str(lidar_scans),
        },
    }

    if not svo2.exists():
        report["errors"].append(f"missing SVO2: {svo2}")
    if not zed_index.exists():
        report["errors"].append(f"missing ZED SVO index: {zed_index}")
    else:
        try:
            df = pd.read_csv(zed_index)
            missing = sorted(REQUIRED_ZED_INDEX_COLUMNS - set(df.columns))
            if missing:
                report["errors"].append(f"zed_svo_index.csv missing columns: {missing}")
            report["zed_svo_index"] = {
                "rows": int(len(df)),
                "columns": list(df.columns),
                "first_frame_id": None if df.empty else int(df["frame_id"].iloc[0]),
                "last_frame_id": None if df.empty else int(df["frame_id"].iloc[-1]),
            }
        except Exception as exc:
            report["errors"].append(f"failed reading zed_svo_index.csv: {type(exc).__name__}: {exc}")

    rear_count = _count_files(rear_rgb, ("*.png", "*.jpg", "*.jpeg"))
    lidar_count = _count_files(lidar_scans, ("*.bin", "*.pcd", "*.ply", "*.npy"))
    report["rear_rgb_count"] = rear_count
    report["lidar_scan_count"] = lidar_count
    if require_rear and rear_count <= 0:
        report["errors"].append(f"missing rear RGB images: {rear_rgb}")
    if require_lidar and lidar_count <= 0:
        report["errors"].append(f"missing LiDAR scans: {lidar_scans}")

    front_count = _count_files(dataset / "front_camera" / "rgb", ("*.png", "*.jpg", "*.jpeg"))
    depth_count = _count_files(dataset / "depth", ("*.png", "*.tiff", "*.tif", "*.npy"))
    if front_count:
        report["warnings"].append("front RGB directory exists; portable pipeline still treats SVO2 as the primary front source")
    if depth_count:
        report["warnings"].append("depth directory exists; portable pipeline can reuse it until SVO2 depth export is enabled")
    report["existing_front_rgb_count"] = front_count
    report["existing_depth_count"] = depth_count

    report["ok"] = not report["errors"]
    return report


def write_report(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
