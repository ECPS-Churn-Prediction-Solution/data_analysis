# -*- coding: utf-8 -*-
"""
analytics.prediction_user_churn 업서트 (SCD2: valid_from/valid_until)
- 기본: 동일 (user_id, reference_dt, horizon)의 '현재행(∞)'을 모든 버전에서 닫고 → 새 행 INSERT
- AB 테스트 유지가 목적이면 CLOSE_ALL_CURRENT=0 로 두어 버전별 현재행을 병행 유지
"""
from __future__ import annotations
from typing import Any, List, Tuple
import os

from ..common.settings import CFG

INFTY = "infinity"
CLOSE_ALL_CURRENT = os.getenv("CLOSE_ALL_CURRENT", "1") == "1"

def _close_current_sql() -> str:
    if CLOSE_ALL_CURRENT:
        # 모든 버전의 현재행을 닫음 (운영 권장)
        return """
        UPDATE analytics.prediction_user_churn
           SET valid_until = NOW()
         WHERE user_id = %s
           AND reference_dt = %s
           AND churn_horizon_days = %s
           AND valid_until = 'infinity'::timestamptz
        """
    else:
        # 해당 model_version의 현재행만 닫음 (AB 테스트용)
        return """
        UPDATE analytics.prediction_user_churn
           SET valid_until = NOW()
         WHERE user_id = %s
           AND reference_dt = %s
           AND churn_horizon_days = %s
           AND model_version = %s
           AND valid_until = 'infinity'::timestamptz
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
        %s, %s::timestamptz
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
            user_id       = r[0]
            reference_dt  = r[6]
            horizon       = r[7]
            model_version = r[3]

            sql = _close_current_sql()
            if CLOSE_ALL_CURRENT:
                cur.execute(sql, (user_id, reference_dt, horizon))
            else:
                cur.execute(sql, (user_id, reference_dt, horizon, model_version))

            cur.execute(_insert_sql(), r)
            # INSERT가 실제 반영된 행 수만 카운트
            affected += (cur.rowcount or 0)
    conn.commit()
    return affected
