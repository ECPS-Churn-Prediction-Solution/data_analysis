# -*- coding: utf-8 -*-
"""
analytics.prediction_user_churn 업서트 (SCD2: valid_from/valid_until)
- 기존 current(=valid_until=∞) 행을 닫고(valid_until=now), 새 행 insert
"""
from __future__ import annotations
from typing import Iterable, Dict, Any, List, Tuple
from datetime import datetime, timezone, date

from ..common.settings import CFG

INFTY = "infinity"

def _close_current_sql() -> str:
    return """
    UPDATE analytics.prediction_user_churn
       SET valid_until = NOW()
     WHERE user_id = %s AND reference_dt = %s
       AND churn_horizon_days = %s AND model_version = %s
       AND valid_until = 'infinity'
    """

def _insert_sql() -> str:
    return """
    INSERT INTO analytics.prediction_user_churn (
        user_id, scored_at, model_name, model_version, feature_version,
        data_cutoff_at, reference_dt, churn_horizon_days, churn_threshold_dt,
        churn_probability_raw, risk_band, score_percentile,
        top1_feature, top1_shap, top2_feature, top2_shap, top3_feature, top3_shap,
        order_count, total_spend, avg_order_value, avg_days_between_orders,
        login_count, cart_count, recency_days, rfm_sum, age, gender, age_group,
        used_coupon, avg_cart_per_login, category_diversity, rfm_bucket, kmeans_cluster,
        action_code_suggested, imputations_count, anomalies_count, pipeline_run_id,
        valid_from, valid_until
    )
    VALUES (
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s,
        %s, %s, %s,
        %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s,
        %s, %s
    )
    ON CONFLICT (user_id, scored_at, model_version, churn_horizon_days)
    DO NOTHING
    """

def upsert_rows(conn, rows: List[Tuple[Any, ...]]) -> int:
    """
    rows: 위 INSERT 문 순서에 맞춘 튜플 목록
    """
    affected = 0
    with conn.cursor() as cur:
        for r in rows:
            user_id = r[0]
            reference_dt = r[6]
            horizon = r[7]
            model_version = r[3]
            cur.execute(_close_current_sql(), (user_id, reference_dt, horizon, model_version))
            cur.execute(_insert_sql(), r)
            affected += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
    conn.commit()
    return affected
