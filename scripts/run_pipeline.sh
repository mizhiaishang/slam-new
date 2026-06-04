#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/run_pipeline.sh <dataset_dir> <output_dir> [config_yaml]"
  exit 1
fi

cd "$(dirname "$0")/.."
source scripts/env.sh

DATASET="$1"
OUTPUT="$2"
CONFIG="${3:-configs/default.yaml}"

python -m semantic_topomap.cli \
  --config "$CONFIG" \
  run \
  --dataset "$DATASET" \
  --output "$OUTPUT"
