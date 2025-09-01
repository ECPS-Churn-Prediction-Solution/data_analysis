#!/usr/bin/env bash
set -euo pipefail

# 날짜 인자 필수
if [[ $# -lt 1 ]]; then
  echo "usage: $0 YYYY-MM-DD"
  exit 1
fi

DT="$1"
export PYTHONPATH="$(pwd)"

echo "[info] bucket     : ${S3_BUCKET:-kdt3-preprocessing-data}"
echo "[info] feature px : ${S3_FEATURE_PREFIX:-features/} (TODO: 추후 수정)"
echo "[info] partition  : ${S3_FEATURE_PARTITION_FMT:-dt=%Y-%m-%d} (TODO: 추후 수정)"
echo "[info] model px   : ${S3_MODEL_PREFIX:-models/lgbm/} (TODO: 추후 수정)"
echo "[info] model ver  : ${MODEL_VERSION:-lgbm_v1.0_shaTODO} (TODO: 추후 수정)"

python -m src.model.predict_lgbm "$DT"
