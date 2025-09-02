# -*- coding: utf-8 -*-
"""
analytics.scoring_policy upsert utility.
"""
from __future__ import annotations
from typing import Optional
import datetime


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
    effective_from: Optional[datetime.datetime] = None,
):
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    threshold_default = _clip01(threshold_default)
    cutpoint_vh = _clip01(cutpoint_vh)
    cutpoint_h  = _clip01(cutpoint_h)
    cutpoint_m  = _clip01(cutpoint_m)

    with conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE analytics.scoring_policy sp
               SET effective_until = NOW()
             WHERE sp.effective_until = 'infinity'::timestamptz
               AND sp.model_name = %s
               AND sp.model_version = %s
               AND sp.feature_version = %s
               AND sp.horizon_days = %s
            """,
            (model_name, model_version, feature_version, churn_horizon_days),
        )

        if effective_from is None:
            cur.execute(
                """
                INSERT INTO analytics.scoring_policy
                (model_name, model_version, feature_version, horizon_days,
                 threshold_default, cutpoint_vh, cutpoint_h, cutpoint_m,
                 effective_from, effective_until, created_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s, NOW(), 'infinity'::timestamptz, NOW())
                """,
                (model_name, model_version, feature_version, churn_horizon_days,
                 threshold_default, cutpoint_vh, cutpoint_h, cutpoint_m),
            )
        else:
            cur.execute(
                """
                INSERT INTO analytics.scoring_policy
                (model_name, model_version, feature_version, horizon_days,
                 threshold_default, cutpoint_vh, cutpoint_h, cutpoint_m,
                 effective_from, effective_until, created_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'infinity'::timestamptz,NOW())
                """,
                (model_name, model_version, feature_version, churn_horizon_days,
                 threshold_default, cutpoint_vh, cutpoint_h, cutpoint_m, effective_from),
            )
