# -*- coding: utf-8 -*-
"""
analytics.v_prediction_user_churn_current 를 읽어 mart.* 집계 테이블 적재
- 필수: churn_risk_distribution, high_risk_users, daily_churn_kpi
- (SHAP/세그먼트 없으면 해당 테이블은 skip 또는 0건)
"""
from __future__ import annotations
import os
from datetime import date, datetime, timezone

from ..common.settings import CFG

def _as_date(s: str) -> date:
    return date.fromisoformat(s)

def main(report_dt: str|None=None, horizon_days: str|None=None):
    # 파라미터
    if not report_dt:
        report_dt = os.getenv("REPORT_DT") or datetime.now(timezone.utc).date().isoformat()
    if not horizon_days:
        horizon_days = os.getenv("HORIZON_DAYS") or str(int(CFG.CHURN_HORIZON_DAYS))
    rd = _as_date(report_dt)
    hz = int(horizon_days)

    with CFG.connect_db() as conn, conn.cursor() as cur:
        # 1) 위험 밴드 분포
        cur.execute("""
        WITH base AS (
          SELECT risk_band FROM analytics.v_prediction_user_churn_current
          WHERE report_dt_kst = %s AND churn_horizon_days = %s
        ), tot AS ( SELECT COUNT(*)::bigint AS n FROM base )
        INSERT INTO mart.churn_risk_distribution (report_dt, horizon_days, risk_band, user_count, ratio)
        SELECT %s, %s, risk_band, COUNT(*)::bigint AS user_count,
               CASE WHEN t.n>0 THEN ROUND(COUNT(*)::numeric / t.n, 4) ELSE 0 END AS ratio
        FROM base, tot t
        GROUP BY risk_band, t.n
        ON CONFLICT (report_dt, horizon_days, risk_band)
        DO UPDATE SET user_count = EXCLUDED.user_count, ratio = EXCLUDED.ratio
        """, (rd, hz, rd, hz))

        # 2) 고위험 사용자 목록 (VH/H) + 정책 추천
        cur.execute("""
        WITH curp AS (
          SELECT p.user_id, p.risk_band, p.churn_probability_raw,
                 p.top1_feature, p.top1_shap, p.top2_feature, p.top2_shap, p.top3_feature, p.top3_shap
          FROM analytics.v_prediction_user_churn_current p
          WHERE p.report_dt_kst = %s AND p.churn_horizon_days = %s AND p.risk_band IN ('VH','H')
        ),
        amap AS (
          SELECT risk_band, policy_id, action_code
          FROM analytics.action_recommendations
          WHERE is_active = true AND effective_until='infinity'
        )
        INSERT INTO mart.high_risk_users
          (report_dt, horizon_days, user_id, risk_band, churn_probability, policy_id, action_code,
           top1_feature, top1_shap, top2_feature, top2_shap, top3_feature, top3_shap)
        SELECT %s, %s, c.user_id, c.risk_band, c.churn_probability_raw,
               a.policy_id, a.action_code,
               c.top1_feature, c.top1_shap, c.top2_feature, c.top2_shap, c.top3_feature, c.top3_shap
        FROM curp c LEFT JOIN amap a USING (risk_band)
        ON CONFLICT (report_dt, horizon_days, user_id)
        DO UPDATE SET
            risk_band = EXCLUDED.risk_band,
            churn_probability = EXCLUDED.churn_probability,
            policy_id = EXCLUDED.policy_id,
            action_code = EXCLUDED.action_code,
            top1_feature = EXCLUDED.top1_feature, top1_shap = EXCLUDED.top1_shap,
            top2_feature = EXCLUDED.top2_feature, top2_shap = EXCLUDED.top2_shap,
            top3_feature = EXCLUDED.top3_feature, top3_shap = EXCLUDED.top3_shap
        """, (rd, hz, rd, hz))

        # 3) Daily KPI (threshold_default 기준)
        cur.execute("""
        WITH base AS (
          SELECT p.churn_probability_raw
          FROM analytics.v_prediction_user_churn_current p
          WHERE p.report_dt_kst = %s AND p.churn_horizon_days = %s
        ),
        policy AS (
          SELECT threshold_default
          FROM analytics.scoring_policy
          WHERE model_name=%s AND model_version=%s AND feature_version=%s
            AND churn_horizon_days=%s AND effective_until='infinity'
          ORDER BY effective_from DESC LIMIT 1
        ),
        tot AS ( SELECT COUNT(*)::bigint AS n FROM base ),
        churned AS (
          SELECT COUNT(*)::bigint AS n
          FROM base b, policy pol
          WHERE b.churn_probability_raw >= pol.threshold_default
        )
        INSERT INTO mart.daily_churn_kpi
          (report_dt, horizon_days, customers_total, churn_rate, retention_rate)
        SELECT %s, %s,
               t.n,
               CASE WHEN t.n>0 THEN ROUND(c.n::numeric / t.n, 4) ELSE 0 END AS churn_rate,
               CASE WHEN t.n>0 THEN ROUND(1 - c.n::numeric / t.n, 4) ELSE 0 END AS retention_rate
        FROM tot t, churned c
        ON CONFLICT (report_dt, horizon_days)
        DO UPDATE SET
          customers_total = EXCLUDED.customers_total,
          churn_rate = EXCLUDED.churn_rate,
          retention_rate = EXCLUDED.retention_rate,
          modified_at = NOW()
        """, (rd, hz, CFG.MODEL_NAME, CFG.MODEL_VERSION, CFG.FEATURE_VERSION, hz, rd, hz))

        conn.commit()

    print(f"[mart_refresh] done: report_dt={rd}, horizon_days={hz}")

if __name__ == "__main__":
    main()
