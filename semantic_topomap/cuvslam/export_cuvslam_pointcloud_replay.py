#!/usr/bin/env python3
"""Replay a ZED SVO/SVO2 file with PyCuVSLAM and export sparse point-cloud replay files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svo", default="/home/zyf/Desktop/dataset_test3/zed/zed.svo2")
    parser.add_argument("--output-dir", default="/home/zyf/imu/cuvslam/results/dataset_test3_zed_svo2_pointcloud_replay")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means full recording")
    parser.add_argument("--snapshot-stride", type=int, default=30, help="Export one current-landmark snapshot every N frames")
    parser.add_argument("--max-snapshot-points", type=int, default=2500, help="Limit current landmarks per replay snapshot")
    parser.add_argument("--enable-slam", action="store_true", help="Also run cuVSLAM SLAM layer")
    return parser.parse_args()


def color_from_id(identifier: int) -> list[int]:
    return [(identifier * 17) % 256, (identifier * 31) % 256, (identifier * 47) % 256]


def create_cuvslam_camera_from_zed_params(cuvslam, zed_params):
    cu_camera = cuvslam.Camera()
    zed_resolution = zed_params.image_size
    cu_camera.size = [zed_resolution.width, zed_resolution.height]
    cu_camera.principal = [zed_params.cx, zed_params.cy]
    cu_camera.focal = [zed_params.fx, zed_params.fy]
    return cu_camera


def init_cuvslam_from_zed(cuvslam, zed_calibration, *, enable_slam: bool):
    cameras = [
        create_cuvslam_camera_from_zed_params(cuvslam, zed_calibration.left_cam),
        create_cuvslam_camera_from_zed_params(cuvslam, zed_calibration.right_cam),
    ]
    cameras[1].rig_from_camera.translation[0] = zed_calibration.get_camera_baseline()

    odom_cfg = cuvslam.Tracker.OdometryConfig(
        async_sba=False,
        enable_observations_export=True,
        enable_landmarks_export=True,
        enable_final_landmarks_export=True,
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


def finite_vec3(values: Iterable[float]) -> list[float] | None:
    vec = [float(v) for v in values]
    if len(vec) != 3 or not all(math.isfinite(v) for v in vec):
        return None
    return vec


def write_ply(path: Path, points: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for item in points:
            x, y, z = item["xyz"]
            r, g, b = item["color"]
            f.write(f"{x:.7f} {y:.7f} {z:.7f} {int(r)} {int(g)} {int(b)}\n")


def write_html(path: Path, payload: dict) -> None:
    payload_json = json.dumps(payload, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>cuVSLAM 点云回放</title>
  <style>
    :root {{
      --bg: #101418;
      --panel: #182026;
      --ink: #edf3ef;
      --muted: #91a199;
      --accent: #7fd2a4;
      --hot: #f4ba6f;
      --grid: rgba(255,255,255,.08);
    }}
    body {{
      margin: 0;
      background: radial-gradient(circle at 18% 12%, #26352e 0, transparent 32%), var(--bg);
      color: var(--ink);
      font-family: "Aptos", "Segoe UI", sans-serif;
    }}
    header {{
      padding: 18px 22px 10px;
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: end;
    }}
    h1 {{ margin: 0; font-size: 24px; }}
    .sub {{ color: var(--muted); font-size: 13px; margin-top: 6px; }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 330px;
      gap: 16px;
      padding: 10px 18px 18px;
      box-sizing: border-box;
      height: calc(100vh - 82px);
    }}
    .stage, aside {{
      background: color-mix(in srgb, var(--panel) 88%, black);
      border: 1px solid rgba(255,255,255,.08);
      border-radius: 18px;
      box-shadow: 0 18px 60px rgba(0,0,0,.26);
      overflow: hidden;
    }}
    .stage {{ position: relative; }}
    canvas {{ width: 100%; height: 100%; display: block; }}
    aside {{ padding: 16px; overflow: auto; }}
    button, input[type=range] {{ width: 100%; }}
    button {{
      border: 0;
      border-radius: 12px;
      padding: 11px 14px;
      background: var(--accent);
      color: #0b1510;
      font-weight: 700;
      cursor: pointer;
    }}
    .metric {{
      display: grid;
      grid-template-columns: 1fr auto;
      padding: 9px 0;
      border-bottom: 1px solid rgba(255,255,255,.07);
      gap: 12px;
    }}
    .metric span:first-child {{ color: var(--muted); }}
    .legend {{ display: flex; gap: 10px; margin-top: 13px; color: var(--muted); font-size: 13px; }}
    .dot {{ width: 11px; height: 11px; border-radius: 50%; display: inline-block; margin-right: 5px; }}
    .hint {{
      position: absolute;
      left: 14px;
      bottom: 12px;
      color: rgba(237,243,239,.75);
      font-size: 12px;
      background: rgba(0,0,0,.28);
      padding: 7px 10px;
      border-radius: 999px;
    }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; height: auto; }}
      .stage {{ height: 65vh; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>cuVSLAM 点云回放</h1>
      <div class="sub">轨迹 + 当前帧 landmarks + 最终累计稀疏点云。拖动进度条查看回放。</div>
    </div>
    <div class="sub">输出：trajectory.csv / final_landmarks.ply / replay_snapshots.json</div>
  </header>
  <main>
    <section class="stage">
      <canvas id="map"></canvas>
      <div class="hint">鼠标滚轮缩放，拖拽平移；绿色=轨迹，橙色=当前帧点云，浅色=最终点云</div>
    </section>
    <aside>
      <button id="play">播放</button>
      <p><input id="slider" type="range" min="0" max="0" value="0"></p>
      <div class="metric"><span>帧</span><b id="mFrame">-</b></div>
      <div class="metric"><span>时间戳</span><b id="mTime">-</b></div>
      <div class="metric"><span>轨迹点</span><b id="mTraj">-</b></div>
      <div class="metric"><span>当前 landmarks</span><b id="mCurr">-</b></div>
      <div class="metric"><span>最终 landmarks</span><b id="mFinal">-</b></div>
      <div class="legend"><span><i class="dot" style="background:#7fd2a4"></i>轨迹</span><span><i class="dot" style="background:#f4ba6f"></i>当前点云</span></div>
      <p class="sub">说明：这是 cuVSLAM 的稀疏特征点地图，不是 ZED 深度图生成的稠密点云。它适合看 SLAM 追踪点/地图点分布。</p>
    </aside>
  </main>
  <script>
    const payload = {payload_json};
    const canvas = document.getElementById('map');
    const ctx = canvas.getContext('2d');
    const slider = document.getElementById('slider');
    const playBtn = document.getElementById('play');
    let frameIdx = 0, playing = false, timer = null;
    let scale = 1, panX = 0, panY = 0, dragging = false, last = [0,0];

    const snapshots = payload.snapshots || [];
    slider.max = Math.max(0, snapshots.length - 1);

    function allPoints() {{
      const pts = [];
      for (const p of payload.trajectory || []) pts.push(p);
      for (const p of payload.final_landmarks_sample || []) pts.push(p.xyz);
      for (const s of snapshots) for (const p of s.landmarks || []) pts.push(p.xyz);
      return pts;
    }}
    const pts = allPoints();
    const xs = pts.map(p => p[0]), zs = pts.map(p => p[2]);
    const minX = Math.min(...xs, -1), maxX = Math.max(...xs, 1);
    const minZ = Math.min(...zs, -1), maxZ = Math.max(...zs, 1);
    const cx = (minX + maxX) / 2, cz = (minZ + maxZ) / 2;
    const span = Math.max(maxX - minX, maxZ - minZ, 1);

    function resize() {{
      const r = canvas.getBoundingClientRect();
      canvas.width = Math.max(600, Math.floor(r.width * devicePixelRatio));
      canvas.height = Math.max(400, Math.floor(r.height * devicePixelRatio));
      draw();
    }}
    function project(p) {{
      const base = Math.min(canvas.width, canvas.height) * 0.82 / span * scale;
      return [
        canvas.width / 2 + (p[0] - cx) * base + panX,
        canvas.height / 2 - (p[2] - cz) * base + panY
      ];
    }}
    function dot(p, color, r) {{
      const [x,y] = project(p);
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
    }}
    function line(points, color, width) {{
      if (points.length < 2) return;
      ctx.beginPath();
      points.forEach((p, i) => {{
        const [x,y] = project(p);
        if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
      }});
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke();
    }}
    function drawGrid() {{
      ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--grid');
      ctx.lineWidth = 1;
      for (let i = -20; i <= 20; i++) {{
        line([[i,0,-20],[i,0,20]], 'rgba(255,255,255,.055)', 1);
        line([[-20,0,i],[20,0,i]], 'rgba(255,255,255,.055)', 1);
      }}
    }}
    function draw() {{
      ctx.clearRect(0,0,canvas.width,canvas.height);
      drawGrid();
      const snap = snapshots[frameIdx] || snapshots[0] || {{landmarks: []}};
      for (const p of payload.final_landmarks_sample || []) dot(p.xyz, 'rgba(210,225,216,.23)', 1.7);
      const traj = (payload.trajectory || []).slice(0, Math.max(1, snap.trajectory_index || 1));
      line(traj, '#7fd2a4', 3);
      for (const p of snap.landmarks || []) dot(p.xyz, '#f4ba6f', 2.7);
      if (traj.length) dot(traj[traj.length - 1], '#ffffff', 5);
      document.getElementById('mFrame').textContent = snap.frame_id ?? '-';
      document.getElementById('mTime').textContent = snap.timestamp_ns ?? '-';
      document.getElementById('mTraj').textContent = traj.length;
      document.getElementById('mCurr').textContent = (snap.landmarks || []).length;
      document.getElementById('mFinal').textContent = (payload.final_landmarks_count || 0);
      slider.value = frameIdx;
    }}
    function step() {{
      frameIdx = (frameIdx + 1) % Math.max(1, snapshots.length);
      draw();
    }}
    slider.addEventListener('input', e => {{ frameIdx = Number(e.target.value); draw(); }});
    playBtn.addEventListener('click', () => {{
      playing = !playing;
      playBtn.textContent = playing ? '暂停' : '播放';
      if (timer) clearInterval(timer);
      if (playing) timer = setInterval(step, 180);
    }});
    canvas.addEventListener('wheel', e => {{
      e.preventDefault();
      scale *= e.deltaY < 0 ? 1.12 : 0.89;
      scale = Math.min(20, Math.max(.2, scale));
      draw();
    }}, {{passive:false}});
    canvas.addEventListener('mousedown', e => {{ dragging = true; last = [e.clientX, e.clientY]; }});
    window.addEventListener('mouseup', () => dragging = false);
    window.addEventListener('mousemove', e => {{
      if (!dragging) return;
      panX += (e.clientX - last[0]) * devicePixelRatio;
      panY += (e.clientY - last[1]) * devicePixelRatio;
      last = [e.clientX, e.clientY];
      draw();
    }});
    window.addEventListener('resize', resize);
    resize();
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> int:
    args = parse_args()
    svo_path = Path(args.svo).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not svo_path.exists():
        raise FileNotFoundError(svo_path)

    import cuvslam
    import pyzed.sl as sl

    zed = init_zed(sl, svo_path)
    tracker = init_cuvslam_from_zed(
        cuvslam,
        zed.get_camera_information().camera_configuration.calibration_parameters,
        enable_slam=bool(args.enable_slam),
    )

    image_left = sl.Mat()
    image_right = sl.Mat()
    runtime = sl.RuntimeParameters()
    trajectory: list[list[float]] = []
    rows: list[dict] = []
    snapshots: list[dict] = []
    failed_frames = 0
    frame_id = 0

    print(f"Starting cuVSLAM point-cloud replay export: {svo_path}")
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
        pos = finite_vec3(pose.translation)
        if pos is None:
            failed_frames += 1
            continue

        trajectory.append(pos)
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

        if frame_id == 1 or frame_id % max(1, args.snapshot_stride) == 0:
            current_points = []
            for lm in tracker.get_last_landmarks():
                xyz = finite_vec3(lm.coords)
                if xyz is None:
                    continue
                current_points.append({"id": int(lm.id), "xyz": xyz, "color": color_from_id(int(lm.id))})
                if len(current_points) >= max(1, args.max_snapshot_points):
                    break
            snapshots.append(
                {
                    "frame_id": frame_id,
                    "timestamp_ns": timestamp_ns,
                    "trajectory_index": len(trajectory),
                    "pose": pos,
                    "landmarks": current_points,
                }
            )

        if frame_id % 100 == 0:
            print(
                f"frame={frame_id} tracked={len(rows)} snapshots={len(snapshots)} "
                f"failed={failed_frames} fps={zed.get_current_fps()}"
            )

    zed.close()

    final_points = []
    for lm_id, coords in tracker.get_final_landmarks().items():
        xyz = finite_vec3(coords)
        if xyz is None:
            continue
        final_points.append({"id": int(lm_id), "xyz": xyz, "color": color_from_id(int(lm_id))})

    trajectory_csv = output_dir / "trajectory.csv"
    with trajectory_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_id", "timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        writer.writeheader()
        writer.writerows(rows)

    final_ply = output_dir / "final_landmarks.ply"
    write_ply(final_ply, final_points)

    snapshots_json = output_dir / "replay_snapshots.json"
    snapshots_json.write_text(json.dumps({"snapshots": snapshots}, ensure_ascii=False), encoding="utf-8")

    final_sample = final_points
    if len(final_sample) > 10000:
        step = max(1, len(final_sample) // 10000)
        final_sample = final_sample[::step][:10000]

    html_payload = {
        "svo": str(svo_path),
        "trajectory": trajectory,
        "snapshots": snapshots,
        "final_landmarks_count": len(final_points),
        "final_landmarks_sample": final_sample,
    }
    html_path = output_dir / "cuvslam_pointcloud_replay.html"
    write_html(html_path, html_payload)

    summary = {
        "svo": str(svo_path),
        "output_dir": str(output_dir),
        "processed_frames": frame_id,
        "tracked_frames": len(rows),
        "failed_frames": failed_frames,
        "snapshot_stride": int(args.snapshot_stride),
        "snapshot_count": len(snapshots),
        "final_landmarks_count": len(final_points),
        "trajectory_csv": str(trajectory_csv),
        "final_landmarks_ply": str(final_ply),
        "replay_snapshots_json": str(snapshots_json),
        "html_replay": str(html_path),
        "enable_slam": bool(args.enable_slam),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
