#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/env.sh

python - <<'PY'
mods = [
    "numpy", "pandas", "cv2", "scipy", "yaml", "torch",
    "ultralytics", "rerun",
]
for m in mods:
    try:
        __import__(m)
        print(f"[ok] {m}")
    except Exception as e:
        print(f"[missing] {m}: {type(e).__name__}: {e}")
for m in ["pyzed.sl", "cuvslam", "segment_anything"]:
    try:
        __import__(m)
        print(f"[ok] {m}")
    except Exception as e:
        print(f"[warn] {m}: {type(e).__name__}: {e}")
PY
