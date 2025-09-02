#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_score_local.sh 2025-09-01 90
DT=${1:-$(date +%F)}
H=${2:-90}

# Activate venv if exists
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

export PYTHONPATH=src:${PYTHONPATH:-}

python -m src.pipelines.batch_score \
  --dt "${DT}" \
  --horizon "${H}" \
  --model_name "lgbm"
