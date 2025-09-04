# -*- coding: utf-8 -*-
"""
배치 스코어링 엔트리포인트:
- 입력 피처: FEATURE_PREDICT_URI 환경변수로 지정 (s3://... 또는 로컬 파일)
- 모델: S3(models)에서 자동 로드 (registry.py)
- 출력: S3(predictions) + DB(churn_scores)
"""
from __future__ import annotations
import os
from typing import Optional

from ..model.predict_lgbm import predict_uri as _predict_uri
from ..pipelines.prediction_export import export_predictions


def main(predict_uri: Optional[str] = None):
    uri = predict_uri or os.getenv("FEATURE_PREDICT_URI")
    if not uri:
        raise SystemExit(
            "FEATURE_PREDICT_URI가 비어 있습니다. 예) "
            "FEATURE_PREDICT_URI=s3://ecps-event-log/features/dt=2025-09-04/part-00000-....parquet"
        )

    df = _predict_uri(uri)  # columns: user_id, churn_score
    result = export_predictions(df, source_uri=uri)

    print(
        f"[batch_score] rows={result['rows']} "
        f"s3://{result['bucket']}/{result['s3_key']}" if result["s3_key"] else f"[batch_score] rows={result['rows']}"
    )


if __name__ == "__main__":
    main()
