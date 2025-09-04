# -*- coding: utf-8 -*-
"""
배치 스코어링 결과를 analytics.prediction_user_churn에 업서트.
- FEATURE_PREDICT_URI 읽어서 예측 실행
- scoring_policy + action_recommendations 적용
- percentile, risk_band, 유효기간, threshold_dt 계산
"""
from __future__ import annotations
import os
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone, date, timedelta

import numpy as np
import pandas as pd

from ..common.settings import CFG
from ..model.predict_lgbm import predict_uri as _predict_uri
from ..utils.policy import get_scoring_policy, get_action_map, now_utc
from ..db.analytics_writer import upsert_rows, INFTY

def _extract_source_dt_from_uri(uri: Optional[str]) -> Optional[date]:
    if not uri:
        return None
    import re
    m = re.search(r"dt=(\d{4}-\d{2}-\d{2})", uri)
    if not m:
        return None
    return date.fromisoformat(m.group(1))

def _risk_band(prob: float, vh: float, h: float, m: float) -> str:
    if prob >= vh: return "VH"
    if prob >= h:  return "H"
    if prob >= m:  return "M"
    return "L"

def main(predict_uri: Optional[str] = None):
    uri = predict_uri or os.getenv("FEATURE_PREDICT_URI")
    if not uri:
        raise SystemExit("FEATURE_PREDICT_URI가 비었습니다.")
    # 1) 예측
    df = _predict_uri(uri)  # user_id, churn_score
    if df.empty:
        print("[analytics_upsert] no rows")
        return

    # 2) 배치 메타
    n = len(df)
    probs = df["churn_score"].astype(float)
    pct = probs.rank(pct=True, method="average") * 100.0

    source_dt = _extract_source_dt_from_uri(uri)  # dt=YYYY-MM-DD
    if source_dt is None:
        # 없으면 오늘(UTC)로 스냅샷
        source_dt = now_utc().date()

    reference_dt = datetime(source_dt.year, source_dt.month, source_dt.day, tzinfo=timezone.utc)
    scored_at = now_utc()
    horizon = int(CFG.CHURN_HORIZON_DAYS)
    churn_threshold_dt = reference_dt - timedelta(days=horizon)
    data_cutoff_at = reference_dt  # 피처 컷오프를 reference_dt로 동치 처리

    with CFG.connect_db() as conn:
        th_default, cut_vh, cut_h, cut_m = get_scoring_policy(conn)
        action_map = get_action_map(conn)
        # 3) 행 생성
        rows: List[Tuple[Any, ...]] = []
        pipeline_run_id = os.getenv("PIPELINE_RUN_ID") or os.getenv("CI_PIPELINE_ID") or os.getenv("GITHUB_RUN_ID") or "manual"

        for (user_id, p, pctl) in zip(df["user_id"].tolist(), probs.tolist(), pct.tolist()):
            band = _risk_band(float(p), cut_vh, cut_h, cut_m)
            policy_id, action_code = action_map.get(band, (None, "NONE"))
            rows.append((
                int(user_id),                     # user_id
                scored_at,                        # scored_at
                CFG.MODEL_NAME,                   # model_name
                CFG.MODEL_VERSION,                # model_version
                CFG.FEATURE_VERSION,              # feature_version
                data_cutoff_at,                   # data_cutoff_at
                reference_dt,                     # reference_dt
                horizon,                          # churn_horizon_days
                churn_threshold_dt,               # churn_threshold_dt
                float(p),                         # churn_probability_raw
                band,                             # risk_band
                float(pctl),                      # score_percentile
                None, None, None, None, None, None,   # top1~top3 (SHAP 없으므로 NULL)
                None, None, None, None,               # order/tx/login/cart/recency... (없으면 NULL)
                None, None, None, None, None, None,   # demographic/behavioral NULL
                action_code,                          # action_code_suggested
                0, 0,                                  # imputations_count, anomalies_count
                str(pipeline_run_id),                  # pipeline_run_id
                scored_at, INFTY                       # valid_from, valid_until
            ))

        # 4) 업서트
        affected = upsert_rows(conn, rows)
        print(f"[analytics_upsert] upserted rows={affected} into analytics.prediction_user_churn (ref_dt={reference_dt.date()}, horizon={horizon})")

if __name__ == "__main__":
    main()
