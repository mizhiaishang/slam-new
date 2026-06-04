from __future__ import annotations

import os
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def runtime_python_path(root: Path | None = None) -> Path:
    return (root or project_root()) / "runtime" / "python"


def runtime_library_paths(root: Path | None = None) -> list[Path]:
    root = root or project_root()
    paths = [
        root / "runtime" / "python" / "cuvslam",
        root / "runtime" / "zed_sdk" / "lib",
        root / "runtime" / "local_libs" / "lib",
    ]
    for cuda_lib in (Path("/usr/local/cuda-13.2/lib64"), Path("/usr/local/cuda/lib64")):
        if cuda_lib.exists():
            paths.append(cuda_lib)
            break
    return paths


def configure_python_path(root: Path | None = None) -> None:
    root = root or project_root()
    paths = [
        runtime_python_path(root),
        root,
        root / "third_party",
        root / "third_party" / "nav_tools",
    ]
    for path in reversed(paths):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def desired_environment(root: Path | None = None) -> dict[str, str]:
    root = root or project_root()
    env = os.environ.copy()
    configure_python_path(root)

    python_paths = [
        str(runtime_python_path(root)),
        str(root),
        str(root / "third_party"),
        str(root / "third_party" / "nav_tools"),
    ]
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = ":".join(python_paths + ([existing_pythonpath] if existing_pythonpath else []))

    ld_paths = [str(p) for p in runtime_library_paths(root) if p.exists()]
    existing_ld = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = ":".join(ld_paths + ([existing_ld] if existing_ld else []))
    env["ZED_SDK_ROOT_DIR"] = str(root / "runtime" / "zed_sdk")
    env["ZED_SETTINGS_DIR"] = str(root / "runtime" / "zed_sdk" / "settings")
    return env


def ensure_runtime_environment(*, reexec: bool = False) -> None:
    root = project_root()
    configure_python_path(root)
    env = desired_environment(root)
    if not reexec:
        os.environ.update(env)
        return

    if os.environ.get("SEMANTIC_TOPOMAP_ENV_BOOTSTRAPPED") == "1":
        os.environ.update(env)
        return

    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    required_ld = [str(p) for p in runtime_library_paths(root) if p.exists()]
    missing_ld = [p for p in required_ld if p not in current_ld.split(":")]
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    required_python = str(runtime_python_path(root))
    missing_python = required_python not in current_pythonpath.split(":")
    if not missing_ld and not missing_python:
        os.environ["SEMANTIC_TOPOMAP_ENV_BOOTSTRAPPED"] = "1"
        os.environ.update(env)
        return

    env["SEMANTIC_TOPOMAP_ENV_BOOTSTRAPPED"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)
