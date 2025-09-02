# -*- coding: utf-8 -*-
"""
Utilities to populate mart.daily_churn_prediction_aggr from analytics.prediction_user_churn.

For each (ref_date, horizon):
  - mean_probability  : avg(churn_probability_raw)
  - predicted_rate    : share of users with prob >= threshold_default
  - band shares       : VH/H/M/L shares

Writes a single row per date into mart.daily_churn_prediction_aggr.
If your table has churn30/churn1 only, we map horizons accordingly and leave others NULL.
"""
from __future__ import annotations
from typing import Iterable, Tuple, Optional
import psycopg2


def _compute_daily_stats(cur, ref_date: str, model_version: str, feature_version: str, horizon_days: int
                         ) -> Tuple[Optional[float], Optional[float], float, float, float, float, int]:
    """
    Returns: (mean_prob, predicted_rate, share_vh, share_h, share_m, share_l, n)
    """
    cur.execute(
        """
        WITH curr AS (
          SELECT p.user_id, p.churn_probability_raw, p.risk_band
          FROM analytics.prediction_user_churn p
          WHERE p.reference_dt::date = %s::date
            AND p.model_version = %s
            AND p.feature_version = %s
            AND p.churn_horizon_days = %s
            AND p.valid_until = 'infinity'::timestamptz
        )
        SELECT
          CASE WHEN COUNT(*)=0 THEN NULL ELSE AVG(churn_probability_raw)::float END AS mean_prob,
          NULL::float AS predicted_rate,  -- filled below after threshold lookup
          CASE WHEN COUNT(*)=0 THEN 0 ELSE AVG((risk_band='VH')::int)::float END AS share_vh,
          CASE WHEN COUNT(*)=0 THEN 0 ELSE AVG((risk_band='H')::int)::float  END AS share_h,
          CASE WHEN COUNT(*)=0 THEN 0 ELSE AVG((risk_band='M')::int)::float  END AS share_m,
          CASE WHEN COUNT(*)=0 THEN 0 ELSE AVG((risk_band='L')::int)::float  END AS share_l,
          COUNT(*) AS n
        FROM curr
        """,
        (ref_date, model_version, feature_version, horizon_days),
    )
    mean_prob, predicted_rate, share_vh, share_h, share_m, share_l, n = cur.fetchone()

    if n == 0:
        return None, None, 0.0, 0.0, 0.0, 0.0, 0

    # fetch threshold_default
    cur.execute(
        """
        SELECT threshold_default
        FROM analytics.scoring_policy
        WHERE model_version=%s AND feature_version=%s AND horizon_days=%s
          AND effective_until='infinity'::timestamptz
        ORDER BY effective_from DESC
        LIMIT 1
        """,
        (model_version, feature_version, horizon_days),
    )
    row = cur.fetchone()
    threshold = row[0] if row else None

    if threshold is None:
        return mean_prob, None, share_vh, share_h, share_m, share_l, n

    cur.execute(
        """
        WITH curr AS (
          SELECT churn_probability_raw
          FROM analytics.prediction_user_churn p
          WHERE p.reference_dt::date = %s::date
            AND p.model_version = %s
            AND p.feature_version = %s
            AND p.churn_horizon_days = %s
            AND p.valid_until = 'infinity'::timestamptz
        )
        SELECT AVG((churn_probability_raw >= %s)::int)::float
        FROM curr
        """,
        (ref_date, model_version, feature_version, horizon_days, threshold),
    )
    predicted_rate = cur.fetchone()[0]
    return mean_prob, predicted_rate, share_vh, share_h, share_m, share_l, n


def update_mart_daily_aggregates(conn, *, ref_date: str, model_version: str, feature_version: str,
                                 horizons: Iterable[int]) -> None:
    """
    Upsert one row in mart.daily_churn_prediction_aggr for ref_date.

    Mapping:
      - If horizon 30 present → write to `churn30`
      - If horizon 1 present  → write to `churn1`
      - VH/H/M/L columns reflect the first horizon in the list that has data.
    """
    with conn.cursor() as cur:
        results = []
        for h in horizons:
            stats = _compute_daily_stats(cur, ref_date, model_version, feature_version, h)
            results.append((h,) + stats)  # (h, mean, predicted, vh,h,m,l, n)

        # band shares from first non-empty
        band = next(((vh, h, m, l) for (_h, _mp, _pr, vh, h, m, l, n) in results if n > 0),
                    (0.0, 0.0, 0.0, 0.0))
        churn30 = next((_pr for (_h, _mp, _pr, vh, h, m, l, n) in results if _h == 30), None)
        churn1  = next((_pr for (_h, _mp, _pr, vh, h, m, l, n) in results if _h == 1), None)

        cur.execute("SELECT 1 FROM mart.daily_churn_prediction_aggr WHERE date=%s", (ref_date,))
        exists = cur.fetchone() is not None
        if not exists:
            cur.execute(
                """
                INSERT INTO mart.daily_churn_prediction_aggr
                  (date, churn30, churn1, VH, H, M, L, created_at, modified_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                """,
                (ref_date, churn30, churn1, band[0], band[1], band[2], band[3]),
            )
        else:
            cur.execute(
                """
                UPDATE mart.daily_churn_prediction_aggr
                   SET churn30 = COALESCE(%s, churn30),
                       churn1  = COALESCE(%s, churn1),
                       VH = %s, H = %s, M = %s, L = %s,
                       modified_at = NOW()
                 WHERE date=%s
                """,
                (churn30, churn1, band[0], band[1], band[2], band[3], ref_date),
            )
