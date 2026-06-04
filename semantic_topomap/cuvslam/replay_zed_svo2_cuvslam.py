#!/usr/bin/env python3
"""Replay a ZED SVO/SVO2 recording through PyCuVSLAM and save trajectory CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--svo",
        default="/home/zyf/Desktop/dataset_test3/zed/zed.svo2",
        help="Input ZED .svo/.svo2 recording",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/zyf/imu/cuvslam/results/dataset_test3_zed_svo2",
        help="Directory for trajectory and summary outputs",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames; 0 means full recording")
    parser.add_argument("--enable-slam", action="store_true", help="Enable cuVSLAM SLAM mode")
    parser.add_argument("--visualize", action="store_true", help="Enable Rerun visualization if rerun is installed")
    parser.add_argument("--rich-vis", action="store_true", help="Draw observations/landmarks in Rerun")
    parser.add_argument("--vis-stride", type=int, default=100, help="Visualize every N frames")
    return parser.parse_args()


def require_modules():
    missing = []
    try:
        import cuvslam  # noqa: F401
    except Exception as exc:  # pragma: no cover - runtime environment check
        missing.append(f"cuvslam ({type(exc).__name__}: {exc})")
    try:
        import pyzed.sl as sl  # noqa: F401
    except Exception as exc:  # pragma: no cover - runtime environment check
        missing.append(f"pyzed.sl ({type(exc).__name__}: {exc})")
    if missing:
        raise RuntimeError(
            "Missing runtime dependencies: "
            + "; ".join(missing)
            + ". Build/install PyCuVSLAM and install the ZED SDK Python API first."
        )


def create_cuvslam_camera_from_zed_params(cuvslam, zed_params):
    cu_camera = cuvslam.Camera()
    zed_resolution = zed_params.image_size
    cu_camera.size = [zed_resolution.width, zed_resolution.height]
    cu_camera.principal = [zed_params.cx, zed_params.cy]
    cu_camera.focal = [zed_params.fx, zed_params.fy]
    return cu_camera


def init_cuvslam_from_zed(cuvslam, zed_calibration, *, enable_slam: bool, rich_vis: bool):
    cameras = [
        create_cuvslam_camera_from_zed_params(cuvslam, zed_calibration.left_cam),
        create_cuvslam_camera_from_zed_params(cuvslam, zed_calibration.right_cam),
    ]
    cameras[1].rig_from_camera.translation[0] = zed_calibration.get_camera_baseline()

    odom_cfg = cuvslam.Tracker.OdometryConfig(
        async_sba=False,
        enable_final_landmarks_export=rich_vis,
        enable_observations_export=rich_vis,
        rectified_stereo_camera=True,
        multicam_mode=cuvslam.Tracker.MulticameraMode.Performance,
    )

    slam_cfg = None
    if enable_slam:
        slam_cfg = cuvslam.Tracker.SlamConfig(
            enable_reading_internals=True,
            map_cell_size=2,
            sync_mode=True,
            max_map_size=10000,
        )

    return cuvslam.Tracker(cuvslam.Rig(cameras), odom_cfg, slam_cfg)


def init_zed(sl, svo_path: Path):
    zed = sl.Camera()
    init = sl.InitParameters(coordinate_units=sl.UNIT.METER, depth_mode=sl.DEPTH_MODE.NONE)
    init.set_from_svo_file(str(svo_path))
    settings_dir = Path(
        os.environ.get(
            "ZED_SETTINGS_DIR",
            Path(__file__).resolve().parents[2] / "runtime" / "zed_sdk" / "settings",
        )
    )
    settings_dir.mkdir(parents=True, exist_ok=True)
    init.optional_settings_path = str(settings_dir.resolve()) + "/"
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open SVO file {svo_path}: {err}")
    return zed


def maybe_init_rerun(enable: bool):
    if not enable:
        return None
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except Exception as exc:
        print(f"[warn] Rerun visualization disabled: {type(exc).__name__}: {exc}")
        return None
    default_blueprint = rrb.Blueprint(
        rrb.TimePanel(state="collapsed"),
        rrb.Vertical(
            row_shares=[0.6, 0.4],
            contents=[rrb.Spatial3DView(), rrb.Spatial2DView(origin="rig/cam0")],
        ),
    )
    rr.init("dataset_test3_zed_svo2", strict=True, spawn=True, default_blueprint=default_blueprint)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    return rr


def log_rerun(rr, frame_id, pose, trajectory, image_left, stride: int):
    if rr is None or frame_id % max(1, stride) != 0:
        return
    rr.set_time_sequence("frame", frame_id)
    rr.log("trajectory", rr.LineStrips3D(trajectory))
    rr.log("rig", rr.Transform3D(translation=pose.translation, quaternion=pose.rotation))
    rr.log(
        "rig/cam0",
        rr.Pinhole(image_plane_distance=1, focal_length=100, width=image_left.get_width(), height=image_left.get_height()),
    )
    rr.log("rig/cam0/image", rr.Image(image_left).compress(jpeg_quality=80))


def main() -> int:
    args = parse_args()
    svo_path = Path(args.svo).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not svo_path.exists():
        raise FileNotFoundError(svo_path)

    require_modules()
    import cuvslam
    import pyzed.sl as sl

    zed = init_zed(sl, svo_path)
    tracker = init_cuvslam_from_zed(
        cuvslam,
        zed.get_camera_information().camera_configuration.calibration_parameters,
        enable_slam=args.enable_slam,
        rich_vis=args.rich_vis,
    )
    rr = maybe_init_rerun(args.visualize)

    image_left = sl.Mat()
    image_right = sl.Mat()
    runtime = sl.RuntimeParameters()
    trajectory = []
    rows = []
    failed_frames = 0
    frame_id = 0

    print(f"Starting SVO playback: {svo_path}")
    while zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        if args.max_frames > 0 and frame_id >= args.max_frames:
            break

        zed.retrieve_image(image_left, sl.VIEW.LEFT_GRAY)
        zed.retrieve_image(image_right, sl.VIEW.RIGHT_GRAY)
        timestamp_ns = int(zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())

        pose_estimate, slam_pose = tracker.track(timestamp_ns, [image_left.get_data(), image_right.get_data()])
        frame_id += 1

        if pose_estimate.world_from_rig is None:
            failed_frames += 1
            print(f"[warn] Failed to track frame {frame_id}")
            continue

        pose = slam_pose if args.enable_slam and slam_pose is not None else pose_estimate.world_from_rig.pose
        trajectory.append(pose.translation)
        rows.append(
            {
                "frame_id": frame_id,
                "timestamp_ns": timestamp_ns,
                "tx": pose.translation[0],
                "ty": pose.translation[1],
                "tz": pose.translation[2],
                "qx": pose.rotation[0],
                "qy": pose.rotation[1],
                "qz": pose.rotation[2],
                "qw": pose.rotation[3],
            }
        )
        if frame_id % 100 == 0:
            print(f"frame={frame_id} tracked={len(rows)} failed={failed_frames} fps={zed.get_current_fps()}")
        log_rerun(rr, frame_id, pose, trajectory, image_left, args.vis_stride)

    zed.close()

    trajectory_csv = output_dir / "trajectory.csv"
    with trajectory_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_id", "timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "svo": str(svo_path),
        "output_dir": str(output_dir),
        "processed_frames": frame_id,
        "tracked_frames": len(rows),
        "failed_frames": failed_frames,
        "enable_slam": bool(args.enable_slam),
        "trajectory_csv": str(trajectory_csv),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
