#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <train_start_dt> <train_end_dt> <valid_dt> [--compare-baselines]"
  echo "ex   : $0 2025-08-01 2025-08-30 2025-08-31 --compare-baselines"
  exit 1
fi

export PYTHONPATH="$(pwd)"

echo "[info] bucket     : ${S3_BUCKET:-kdt3-preprocessing-data}"
echo "[info] feature px : ${S3_FEATURE_PREFIX:-features/}   (TODO: 추후 수정)"
echo "[info] model px   : ${S3_MODEL_PREFIX:-models/lgbm/}  (TODO: 추후 수정)"
echo "[info] model ver  : ${MODEL_VERSION:-lgbm_v1.0_shaTODO} (TODO: 추후 수정)"

python -m src.pipelines.batch_train "$@"
