from __future__ import annotations

import importlib
import json
from pathlib import Path

from semantic_topomap.dataset.check_dataset import check_dataset
from semantic_topomap.utils.paths import resolve_under_root


REQUIRED_MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "cv2": "opencv-python",
    "scipy": "scipy",
    "yaml": "pyyaml",
    "torch": "torch",
    "ultralytics": "ultralytics",
    "segment_anything": "segment-anything",
}

OPTIONAL_RUNTIME_MODULES = {
    "rerun": "rerun-sdk，导出 Rerun 回放需要",
    "pyzed.sl": "ZED SDK Python API，从 SVO2 导出前视图像/深度需要",
    "cuvslam": "PyCuVSLAM，从 SVO2 运行 cuVSLAM 位姿和点云需要",
}


def _check_import(module: str) -> dict:
    try:
        importlib.import_module(module)
        return {"module": module, "ok": True}
    except Exception as exc:
        return {
            "module": module,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _check_path(path: Path | None) -> dict:
    if path is None:
        return {"path": None, "exists": False}
    return {"path": str(path), "exists": path.exists()}


def build_doctor_report(cfg: dict, dataset: Path | None = None) -> dict:
    required_modules = []
    optional_modules = []
    for module, package in REQUIRED_MODULES.items():
        item = _check_import(module)
        item["package"] = package
        required_modules.append(item)
    for module, note in OPTIONAL_RUNTIME_MODULES.items():
        item = _check_import(module)
        item["note"] = note
        optional_modules.append(item)

    model_checks = {}
    for name, value in (cfg.get("models") or {}).items():
        if name == "sam_model_type":
            continue
        model_checks[name] = _check_path(resolve_under_root(value))

    third_party_checks = {}
    for name, value in (cfg.get("third_party") or {}).items():
        third_party_checks[name] = _check_path(resolve_under_root(value))

    report = {
        "ok": True,
        "required_python_modules": required_modules,
        "optional_runtime_modules": optional_modules,
        "models": model_checks,
        "third_party": third_party_checks,
        "dataset": None,
        "errors": [],
        "warnings": [],
    }

    for item in required_modules:
        if not item["ok"]:
            report["errors"].append(f"缺少必需 Python 模块：{item['module']}，建议安装 {item['package']}")
    for name, item in model_checks.items():
        if not item["exists"]:
            report["errors"].append(f"缺少模型文件：{name} -> {item['path']}")
    for name, item in third_party_checks.items():
        if not item["exists"]:
            report["errors"].append(f"缺少第三方代码目录：{name} -> {item['path']}")

    for item in optional_modules:
        if not item["ok"]:
            report["warnings"].append(f"{item['note']}；当前未安装 {item['module']} ({item['error']})")

    if dataset is not None:
        dataset_report = check_dataset(
            dataset,
            require_rear=bool(cfg.get("runtime", {}).get("require_rear", True)),
            require_lidar=bool(cfg.get("runtime", {}).get("require_lidar", True)),
        )
        report["dataset"] = dataset_report
        if not dataset_report["ok"]:
            report["errors"].extend(dataset_report["errors"])
        report["warnings"].extend(dataset_report.get("warnings", []))

    report["ok"] = not report["errors"]
    return report


def write_doctor_report(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
