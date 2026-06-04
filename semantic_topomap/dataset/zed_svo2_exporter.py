from __future__ import annotations

import json
import os
from pathlib import Path

from semantic_topomap.utils.paths import project_root


def export_svo2_front_data(*, dataset: Path, output: Path, max_frames: int = 0) -> dict:
    """Export front RGB/depth/intrinsics from SVO2.

    This function intentionally keeps a small, explicit surface. On machines
    with the ZED SDK Python API installed, it extracts left RGB and depth maps.
    If the SDK is unavailable, callers may reuse existing dataset front/depth
    folders; the report makes that fallback visible.
    """
    dataset = dataset.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    svo2 = dataset / "zed" / "zed.svo2"
    report = {
        "svo2": str(svo2),
        "output": str(output),
        "front_rgb_dir": str(output / "front_rgb"),
        "front_depth_dir": str(output / "front_depth"),
        "camera_intrinsics": str(output / "camera_intrinsics.json"),
        "used_zed_sdk": False,
        "fallback": None,
    }
    try:
        import cv2
        import numpy as np
        import pyzed.sl as sl
    except Exception as exc:
        report["fallback"] = f"ZED SDK Python API unavailable: {type(exc).__name__}: {exc}"
        (output / "svo2_export_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    if not svo2.exists():
        raise FileNotFoundError(svo2)

    front_dir = output / "front_rgb"
    depth_dir = output / "front_depth"
    front_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    zed = sl.Camera()
    init = sl.InitParameters(coordinate_units=sl.UNIT.METER, depth_mode=sl.DEPTH_MODE.ULTRA)
    init.set_from_svo_file(str(svo2))
    settings_dir = Path(os.environ.get("ZED_SETTINGS_DIR", project_root() / "runtime" / "zed_sdk" / "settings"))
    settings_dir.mkdir(parents=True, exist_ok=True)
    init.optional_settings_path = str(settings_dir.resolve()) + "/"
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"failed to open SVO2 {svo2}: {err}")

    calib = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
    intrinsics = {
        "width": int(calib.image_size.width),
        "height": int(calib.image_size.height),
        "fx": float(calib.fx),
        "fy": float(calib.fy),
        "cx": float(calib.cx),
        "cy": float(calib.cy),
    }
    (output / "camera_intrinsics.json").write_text(json.dumps(intrinsics, ensure_ascii=False, indent=2), encoding="utf-8")

    image = sl.Mat()
    depth = sl.Mat()
    runtime = sl.RuntimeParameters()
    count = 0
    while zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        if max_frames > 0 and count >= max_frames:
            break
        ts = int(zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        rgb = image.get_data()
        if rgb.shape[-1] == 4:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGRA2BGR)
        depth_m = depth.get_data().astype(np.float32)
        depth_mm = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        depth_mm = np.clip(depth_mm * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(front_dir / f"{ts}.png"), rgb)
        cv2.imwrite(str(depth_dir / f"{ts}.png"), depth_mm)
        count += 1
    zed.close()

    report.update({"used_zed_sdk": True, "frames": count, "fallback": None})
    (output / "svo2_export_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
