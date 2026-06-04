#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m pip install -U pip
python -m pip install -r requirements.txt
if compgen -G "runtime/wheels/*.whl" > /dev/null; then
  python -m pip install runtime/wheels/*.whl
fi
python -m pip install -e .
source scripts/env.sh
echo "Installed semantic-topomap. Run: semantic-topomap --help"
