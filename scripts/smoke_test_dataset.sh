#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/smoke_test_dataset.sh <dataset_dir> <output_dir> [config_yaml]"
  exit 1
fi

cd "$(dirname "$0")/.."
source scripts/env.sh

DATASET="$1"
OUTPUT="$2"
CONFIG="${3:-configs/smoke.yaml}"

python -m semantic_topomap.cli --config "$CONFIG" doctor \
  --dataset "$DATASET" \
  --output "$OUTPUT/reports/doctor.json"

python -m semantic_topomap.cli --config "$CONFIG" run \
  --dataset "$DATASET" \
  --output "$OUTPUT" \
  --max-frames 3 \
  --ignore-alignment \
  --stride 1 \
  --snapshot-stride 2 \
  --max-snapshot-points 50

echo "Smoke test complete: $OUTPUT"
