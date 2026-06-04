#!/usr/bin/env bash
# Source this file before running the portable pipeline directly:
#   source scripts/env.sh

SLAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH="$SLAM_ROOT/runtime/python:$SLAM_ROOT:$SLAM_ROOT/third_party:$SLAM_ROOT/third_party/nav_tools${PYTHONPATH:+:$PYTHONPATH}"

_SLAM_LD_PATHS=(
  "$SLAM_ROOT/runtime/python/cuvslam"
  "$SLAM_ROOT/runtime/zed_sdk/lib"
  "$SLAM_ROOT/runtime/local_libs/lib"
)

if [[ -d /usr/local/cuda-13.2/lib64 ]]; then
  _SLAM_LD_PATHS+=("/usr/local/cuda-13.2/lib64")
elif [[ -d /usr/local/cuda/lib64 ]]; then
  _SLAM_LD_PATHS+=("/usr/local/cuda/lib64")
fi

_SLAM_LD_JOINED="$(IFS=:; echo "${_SLAM_LD_PATHS[*]}")"
export LD_LIBRARY_PATH="$_SLAM_LD_JOINED${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export ZED_SDK_ROOT_DIR="$SLAM_ROOT/runtime/zed_sdk"
export ZED_SETTINGS_DIR="$SLAM_ROOT/runtime/zed_sdk/settings"

unset _SLAM_LD_PATHS
unset _SLAM_LD_JOINED
