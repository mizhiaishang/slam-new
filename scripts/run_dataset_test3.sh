#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/env.sh

DATASET="${1:-/home/zyf/Desktop/dataset_test3}"
OUTPUT="${2:-/home/zyf/Public/slam/outputs/dataset_test3}"
CONFIG="${3:-configs/default.yaml}"

python -m semantic_topomap.cli \
  --config "$CONFIG" \
  run \
  --dataset "$DATASET" \
  --output "$OUTPUT"
