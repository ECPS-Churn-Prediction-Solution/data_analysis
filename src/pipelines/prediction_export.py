# -*- coding: utf-8 -*-
from __future__ import annotations
import io
from datetime import datetime
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
import boto3

from ..common.settings import CFG
from ..common.io import s3_join

def _pred_prefix(dt: str, model_name: str, horizon: int) -> str:
    """
    s3 key prefix:
    {S3_PREDICTION_PREFIX}/model={model_name}/model_version={MODEL_VERSION}/
    feature_version={FEATURE_VERSION}/horizon={horizon}/dt={dt}/
    """
    return s3_join(
        CFG.S3_PREDICTION_PREFIX.rstrip("/"),
        f"model={model_name}",
        f"model_version={CFG.MODEL_VERSION}",
        f"feature_version={CFG.FEATURE_VERSION}",
        f"horizon={int(horizon)}",
        f"dt={dt}",
    )

def export_predictions_to_s3(df: pd.DataFrame, *, dt: str, model_name: str, horizon: int) -> str:
    table = pa.Table.from_pandas(df, preserve_index=False)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    buf.seek(0)

    key = s3_join(
        _pred_prefix(dt, model_name, horizon),
        f"part-{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.parquet",
    )

    extra = {}
    if CFG.S3_SSE:
        extra["ServerSideEncryption"] = CFG.S3_SSE
        if CFG.S3_KMS_KEY_ID:
            extra["SSEKMSKeyId"] = CFG.S3_KMS_KEY_ID

    boto3.client("s3", region_name=CFG.AWS_REGION).put_object(
    Bucket=CFG.S3_PREDICTION_BUCKET, Key=key, Body=buf.getvalue(), **extra
    )
    return f"s3://{CFG.S3_PREDICTION_BUCKET}/{key}"

