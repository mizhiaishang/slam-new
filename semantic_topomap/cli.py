from __future__ import annotations

from semantic_topomap.runtime_env import desired_environment, ensure_runtime_environment

ensure_runtime_environment(reexec=True)

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from semantic_topomap.dataset.check_dataset import check_dataset, write_report
from semantic_topomap.dataset.sync_builder import build_track_from_svo_index
from semantic_topomap.dataset.zed_svo2_exporter import export_svo2_front_data
from semantic_topomap.doctor import build_doctor_report, write_doctor_report
from semantic_topomap.utils.command import run_command
from semantic_topomap.utils.paths import ensure_dir, project_root, resolve_under_root


def load_config(path: Path | None) -> dict:
    cfg_path = path or project_root() / "configs" / "default.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data["_config_path"] = str(cfg_path)
    return data


def _model_path(cfg: dict, key: str) -> Path:
    return resolve_under_root(cfg["models"][key])


def _third_party_path(cfg: dict, key: str) -> Path:
    return resolve_under_root(cfg["third_party"][key])


def cmd_check(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    report = check_dataset(
        Path(args.dataset),
        require_rear=bool(cfg.get("runtime", {}).get("require_rear", True)),
        require_lidar=bool(cfg.get("runtime", {}).get("require_lidar", True)),
    )
    output = Path(args.output) if args.output else Path(args.dataset) / "semantic_topomap_check.json"
    write_report(report, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 2


def cmd_doctor(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    dataset = Path(args.dataset) if args.dataset else None
    report = build_doctor_report(cfg, dataset=dataset)
    if args.output:
        write_doctor_report(report, Path(args.output))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 2


def cmd_export_svo2(args: argparse.Namespace) -> int:
    report = export_svo2_front_data(
        dataset=Path(args.dataset),
        output=ensure_dir(Path(args.output) / "extracted"),
        max_frames=int(args.max_frames),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_run_cuvslam(args: argparse.Namespace) -> int:
    output = ensure_dir(Path(args.output) / "cuvslam")
    script = project_root() / "semantic_topomap" / "cuvslam" / "export_cuvslam_pointcloud_replay.py"
    cmd = [
        sys.executable,
        str(script),
        "--svo",
        str(Path(args.dataset) / "zed" / "zed.svo2"),
        "--output-dir",
        str(output),
        "--snapshot-stride",
        str(args.snapshot_stride),
        "--max-snapshot-points",
        str(args.max_snapshot_points),
    ]
    if args.max_frames > 0:
        cmd.extend(["--max-frames", str(args.max_frames)])
    if args.enable_slam:
        cmd.append("--enable-slam")
    run_command(cmd, cwd=project_root(), env=desired_environment())
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    summary = build_track_from_svo_index(
        dataset=Path(args.dataset),
        extracted_dir=Path(args.output) / "extracted",
        cuvslam_dir=Path(args.output) / "cuvslam",
        prepared_dir=Path(args.output) / "prepared",
        stride=int(args.stride),
        max_dt_ms=float(args.max_dt_ms),
        use_alignment=not bool(getattr(args, "ignore_alignment", False)),
        max_rows=int(getattr(args, "max_rows", 0)),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_build_map(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    workspace = Path(args.output).resolve()
    prepared = workspace / "prepared"
    semantic = ensure_dir(workspace / "semantic_map")
    legacy = project_root() / "semantic_topomap" / "build_map_legacy.py"

    env = desired_environment()
    pythonpath = [
        str(project_root() / "runtime" / "python"),
        str(project_root() / "third_party"),
        str(project_root() / "third_party" / "nav_tools"),
        str(project_root()),
    ]
    env["PYTHONPATH"] = os.pathsep.join(pythonpath + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))

    cmd = [
        sys.executable,
        str(legacy),
        "--mode",
        "build",
        "--database-track-dir",
        str(prepared),
        "--data-base-path",
        str(prepared),
        "--model-config-path",
        str(_model_path(cfg, "opr_config")),
        "--weights-path",
        str(_model_path(cfg, "opr_weights")),
        "--opr-root",
        str(_third_party_path(cfg, "opr_root")),
        "--depth-anything-root",
        str(_third_party_path(cfg, "depth_anything_root")),
        "--yolo-weights",
        str(_model_path(cfg, "yolo_weights")),
        "--depth-model-path",
        str(_model_path(cfg, "depth_weights")),
        "--output-dir",
        str(semantic),
        "--device",
        str(cfg.get("runtime", {}).get("device", "cpu")),
        "--nav-graph-root",
        str(_third_party_path(cfg, "nav_graph_root")),
        "--object3d-engine-root",
        str(_third_party_path(cfg, "object3d_engine_root")),
        "--sam-checkpoint",
        str(_model_path(cfg, "sam_checkpoint")),
        "--sam-model-type",
        str(cfg["models"].get("sam_model_type", "vit_b")),
        "--recorded-depth-dir-name",
        "depth",
        "--enable-waypoint-sampling",
        "--object3d-min-consecutive-frames",
        str(cfg["object3d"].get("min_consecutive_frames", 2)),
        "--object3d-overlap-iou-threshold",
        str(cfg["object3d"].get("overlap_iou_threshold", 0.05)),
        "--object3d-overlap-min-ratio-threshold",
        str(cfg["object3d"].get("overlap_min_ratio_threshold", 0.35)),
        "--object3d-motion-filter-classes",
        str(cfg["object3d"].get("motion_filter_classes", "person")),
        "--object3d-motion-min-consecutive-observations",
        str(cfg["object3d"].get("motion_min_consecutive_observations", 2)),
        "--object3d-motion-static-max-center-span-m",
        str(cfg["object3d"].get("motion_static_max_center_span_m", 1.0)),
        "--object3d-motion-static-max-median-step-m",
        str(cfg["object3d"].get("motion_static_max_median_step_m", 0.35)),
        "--object3d-motion-static-max-single-step-m",
        str(cfg["object3d"].get("motion_static_max_single_step_m", 1.2)),
        "--object3d-disappearance-max-observation-distance-m",
        str(cfg["object3d"].get("disappearance_max_observation_distance_m", 3.0)),
        "--object3d-disappearance-position-tolerance-m",
        str(cfg["object3d"].get("disappearance_position_tolerance_m", 1.0)),
        "--object3d-disappearance-match-distance-m",
        str(cfg["object3d"].get("disappearance_match_distance_m", 1.0)),
        "--object3d-disappearance-min-visible-misses",
        str(cfg["object3d"].get("disappearance_min_visible_misses", 2)),
        "--object3d-disappearance-fov-margin-deg",
        str(cfg["object3d"].get("disappearance_fov_margin_deg", 8.0)),
        "--waypoint-min-distance-m",
        str(cfg["waypoint"].get("min_distance_m", 0.8)),
        "--waypoint-min-yaw-deg",
        str(cfg["waypoint"].get("min_yaw_deg", 25.0)),
        "--feature-distance-threshold",
        str(cfg["topology"].get("feature_distance_threshold", 1.8)),
        "--coord-distance-threshold",
        str(cfg["topology"].get("coord_distance_threshold", 2.5)),
        "--new-node-distance-threshold",
        str(cfg["topology"].get("new_node_distance_threshold", 2.7)),
        "--hfov",
        str(cfg["topology"].get("hfov", 91.0)),
        "--vfov",
        str(cfg["topology"].get("vfov", 65.0)),
    ]
    if cfg.get("runtime", {}).get("yolo_device"):
        cmd.extend(["--yolo-device", str(cfg["runtime"]["yolo_device"])])
    if cfg.get("runtime", {}).get("sam_device"):
        cmd.extend(["--sam-device", str(cfg["runtime"]["sam_device"])])
    if int(cfg.get("runtime", {}).get("max_frames", -1)) > 0:
        cmd.extend(["--max-frames", str(cfg["runtime"]["max_frames"])])
    if not bool(cfg["object3d"].get("overlap_filter_enabled", True)):
        cmd.append("--disable-object3d-overlap-filter")
    if not bool(cfg["object3d"].get("motion_filter_enabled", True)):
        cmd.append("--disable-object3d-motion-filter")
    if not bool(cfg["object3d"].get("motion_unknown_filter_enabled", True)):
        cmd.append("--disable-object3d-motion-unknown-filter")
    if not bool(cfg["object3d"].get("disappearance_filter_enabled", True)):
        cmd.append("--disable-object3d-disappearance-filter")
    if not bool(cfg["waypoint"].get("sampling_enabled", True)):
        cmd.append("--disable-waypoint-sampling")
    if not bool(cfg["waypoint"].get("keep_first_last", True)):
        cmd.append("--disable-waypoint-keep-first-last")
    if not bool(cfg["waypoint"].get("keep_topology_change", True)):
        cmd.append("--disable-waypoint-keep-topology-change")
    run_command(cmd, cwd=project_root(), env=env)
    return 0


def cmd_export_rerun(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    workspace = Path(args.output).resolve()
    script = project_root() / "semantic_topomap" / "visualization" / "export_rerun_legacy.py"
    rerun_cfg = cfg.get("rerun", {})
    output_rrd = workspace / "rerun" / "semantic_topomap_replay.rrd"
    output_rrd.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script),
        "--result-dir",
        str(workspace / "semantic_map"),
        "--dataset-root",
        str(workspace / "prepared"),
        "--cuvslam-dir",
        str(workspace / "cuvslam"),
        "--output",
        str(output_rrd),
        "--max-frames",
        str(rerun_cfg.get("max_frames", -1)),
        "--cloud-points",
        str(rerun_cfg.get("cloud_points", 0)),
        "--image-width",
        str(rerun_cfg.get("image_width", 640)),
        "--max-growing-cloud-points",
        str(rerun_cfg.get("max_growing_cloud_points", 100000)),
        "--cloud-color",
        str(rerun_cfg.get("cloud_color", "40,230,255")),
        "--cloud-radius",
        str(rerun_cfg.get("cloud_radius", 0.0275)),
        "--cloud-secondary-color",
        str(rerun_cfg.get("cloud_secondary_color", "255,255,255")),
        "--cloud-secondary-radius",
        str(rerun_cfg.get("cloud_secondary_radius", 0.009)),
        "--object-label-mode",
        str(rerun_cfg.get("object_label_mode", "none")),
        "--hierarchy-label-color-mode",
        str(rerun_cfg.get("hierarchy_label_color_mode", "bright")),
        "--topology-z",
        str(rerun_cfg.get("topology_z", 5.5)),
        "--semantic-level-z-step",
        str(rerun_cfg.get("semantic_level_z_step", 2.6)),
    ]
    for flag, name in [
        ("--include-removed", "include_removed"),
        ("--grow-pointcloud", "grow_pointcloud"),
        ("--show-hierarchy", "show_hierarchy"),
        ("--black-background", "black_background"),
    ]:
        if bool(rerun_cfg.get(name, False)):
            cmd.append(flag)
    run_command(cmd, cwd=project_root(), env=desired_environment())
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    output = ensure_dir(Path(args.output).resolve())
    check_args = argparse.Namespace(dataset=args.dataset, output=str(output / "reports" / "dataset_check.json"), config=args.config)
    status = cmd_check(check_args)
    if status != 0 and not args.force:
        return status
    cmd_export_svo2(argparse.Namespace(dataset=args.dataset, output=str(output), max_frames=args.max_frames))
    cmd_run_cuvslam(
        argparse.Namespace(
            dataset=args.dataset,
            output=str(output),
            snapshot_stride=args.snapshot_stride,
            max_snapshot_points=args.max_snapshot_points,
            max_frames=args.max_frames,
            enable_slam=args.enable_slam,
        )
    )
    cmd_prepare(
        argparse.Namespace(
            dataset=args.dataset,
            output=str(output),
            stride=args.stride,
            max_dt_ms=args.max_dt_ms,
            ignore_alignment=bool(args.ignore_alignment or args.max_frames > 0),
            max_rows=int(args.max_frames if args.max_frames > 0 else 0),
        )
    )
    cmd_build_map(argparse.Namespace(output=str(output), config=args.config))
    cmd_export_rerun(argparse.Namespace(output=str(output), config=args.config))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="semantic-topomap")
    parser.add_argument("--config", type=Path, default=None, help="Portable pipeline config yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("doctor")
    p.add_argument("--dataset", default=None)
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("export-svo2")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-frames", type=int, default=0)
    p.set_defaults(func=cmd_export_svo2)

    p = sub.add_parser("run-cuvslam")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--snapshot-stride", type=int, default=30)
    p.add_argument("--max-snapshot-points", type=int, default=2500)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--enable-slam", action="store_true")
    p.set_defaults(func=cmd_run_cuvslam)

    p = sub.add_parser("prepare")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stride", type=int, default=15)
    p.add_argument("--max-dt-ms", type=float, default=80.0)
    p.add_argument("--ignore-alignment", action="store_true", help="Debug mode: sample SVO index directly instead of using alignment.csv")
    p.add_argument("--max-rows", type=int, default=0, help="Debug mode: keep at most N selected SVO rows before matching")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("build-map")
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_build_map)

    p = sub.add_parser("export-rerun")
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_export_rerun)

    p = sub.add_parser("run")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stride", type=int, default=15)
    p.add_argument("--max-dt-ms", type=float, default=80.0)
    p.add_argument("--snapshot-stride", type=int, default=30)
    p.add_argument("--max-snapshot-points", type=int, default=2500)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--ignore-alignment", action="store_true", help="Debug mode: sample SVO index directly during prepare")
    p.add_argument("--enable-slam", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
