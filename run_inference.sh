#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON:-python}"
"${PYTHON_BIN}" infer.py \
  --image-dir data/images \
  --output-dir outputs/inference_run
