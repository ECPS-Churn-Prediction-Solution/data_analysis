# -*- coding: utf-8 -*-
"""
스코어링 결과를 S3와 DB에 적재.
"""
from __future__ import annotations
import uuid
from datetime import datetime, timezone, date
from typing import Optional, Dict, Any

import pandas as pd

from ..common.settings import CFG
from ..common.io import s3_join, write_parquet_s3
from ..db.writer import ensure_schema_and_table, insert_scores


def _extract_source_dt_from_uri(uri: Optional[str]) -> Optional[date]:
    """s3://.../dt=YYYY-MM-DD/ 를 만나면 그 날짜를 추출."""
    if not uri:
        return None
    import re
    m = re.search(r"dt=(\d{4}-\d{2}-\d{2})", uri)
    if not m:
        return None
    try:
        return date.fromisoformat(m.group(1))
    except Exception:
        return None


def export_predictions(
    df_scores: pd.DataFrame,
    *,
    source_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """
    입력: df_scores (columns: user_id, churn_score)
    동작:
      1) 메타 컬럼 추가 (model_version, feature_version, scored_at, source_dt)
      2) S3에 parquet 저장 (옵션: CFG.WRITE_PREDICTIONS_TO_S3)
      3) DB 적재 (public.churn_scores)
    반환: {"rows": n, "s3_key": "...", "bucket": "..."}
    """
    if df_scores.empty:
        return {"rows": 0, "s3_key": None, "bucket": None}

    now = datetime.now(CFG.tz_utc)
    out = df_scores.copy()
    out["model_version"] = CFG.MODEL_VERSION
    out["feature_version"] = CFG.FEATURE_VERSION
    out["scored_at"] = now
    out["source_dt"] = _extract_source_dt_from_uri(source_uri)

    # 1) S3 저장 (옵션)
    s3_key = None
    if CFG.WRITE_PREDICTIONS_TO_S3:
        ds = now.strftime("%Y-%m-%d")
        s3_key = s3_join(
            CFG.S3_PREDICTION_PREFIX,
            f"dt={ds}",
            f"model={CFG.MODEL_VERSION}",
            f"pred_{uuid.uuid4().hex}.parquet",
        )
        write_parquet_s3(s3_key, out, bucket=CFG.S3_PREDICTION_BUCKET)

    # 2) DB 적재
    with CFG.connect_db() as conn:
        ensure_schema_and_table(conn)
        n = insert_scores(conn, out)

    return {"rows": int(n), "s3_key": s3_key, "bucket": CFG.S3_PREDICTION_BUCKET}
