# -*- coding: utf-8 -*-
"""
analytics.scoring_policy 업서트 유틸
- 같은 (model_name, model_version, feature_version, horizon_days)의 현재행(effective_until='infinity')
  을 종료한 뒤 새 정책을 삽입.
"""
from __future__ import annotations
from typing import Optional
import datetime
import psycopg2
from psycopg2.extras import execute_values

def upsert_scoring_policy(
    conn,
    *,
    model_name: str,
    model_version: str,
    feature_version: str,
    churn_horizon_days: int,
    threshold_default: float,
    cutpoint_vh: float,
    cutpoint_h: float,
    cutpoint_m: float,
    effective_from: Optional[datetime.datetime] = None,  # None이면 NOW()
):
    # 방어적 클리핑
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    threshold_default = _clip01(threshold_default)
    cutpoint_vh = _clip01(cutpoint_vh)
    cutpoint_h  = _clip01(cutpoint_h)
    cutpoint_m  = _clip01(cutpoint_m)

    with conn, conn.cursor() as cur:
        # 1) 기존 current 종료
        cur.execute("""
            UPDATE analytics.scoring_policy sp
               SET effective_until = NOW()
             WHERE sp.effective_until = 'infinity'::timestamptz
               AND sp.model_name = %s
               AND sp.model_version = %s
               AND sp.feature_version = %s
               AND sp.horizon_days = %s
        """, (model_name, model_version, feature_version, churn_horizon_days))

        # 2) 신규 삽입
        if effective_from is None:
            cur.execute("""
                INSERT INTO analytics.scoring_policy
                (model_name, model_version, feature_version, horizon_days,
                 threshold_default, cutpoint_vh, cutpoint_h, cutpoint_m,
                 effective_from, effective_until, created_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s, NOW(), 'infinity'::timestamptz, NOW())
            """, (model_name, model_version, feature_version, churn_horizon_days,
                  threshold_default, cutpoint_vh, cutpoint_h, cutpoint_m))
        else:
            cur.execute("""
                INSERT INTO analytics.scoring_policy
                (model_name, model_version, feature_version, horizon_days,
                 threshold_default, cutpoint_vh, cutpoint_h, cutpoint_m,
                 effective_from, effective_until, created_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'infinity'::timestamptz,NOW())
            """, (model_name, model_version, feature_version, churn_horizon_days,
                  threshold_default, cutpoint_vh, cutpoint_h, cutpoint_m, effective_from))
