# -*- coding: utf-8 -*-
from __future__ import annotations
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras as extras

from ..common.settings import CFG

def utcnow(): return datetime.now(timezone.utc)

def percentile_ranks(probs: np.ndarray) -> np.ndarray:
    n = len(probs)
    if n <= 1: return np.array([100.0] * n, dtype=float)
    order = np.argsort(probs)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    _, inv, counts = np.unique(probs, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    avg_ranks = sums / counts
    ranks = avg_ranks[inv]
    return (ranks - 1) / (n - 1) * 100.0

def _fetch_current_scoring_policy(conn, model_name, model_version, feature_version, horizon_days: int):
    with conn.cursor(cursor_factory=extras.DictCursor) as cur:
        cur.execute("""
            SELECT threshold_default, cutpoint_vh, cutpoint_h, cutpoint_m
            FROM analytics.scoring_policy
            WHERE model_name=%s AND model_version=%s AND feature_version=%s
              AND horizon_days=%s AND effective_until='infinity'::timestamptz
            ORDER BY effective_from DESC LIMIT 1
        """, (model_name, model_version, feature_version, horizon_days))
        row = cur.fetchone()
        if not row:
            raise RuntimeError("no current scoring_policy")
        return dict(row)

def _fetch_active_actions(conn) -> Dict[str, str]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT risk_band, action_code
            FROM analytics.action_recommendations
            WHERE is_active = true AND effective_until='infinity'::timestamptz
        """)
        return {rb: ac for rb, ac in cur.fetchall()}

def _risk_band(prob: float, vh: float, h: float, m: float) -> str:
    if prob >= vh: return "VH"
    if prob >= h:  return "H"
    if prob >= m:  return "M"
    return "L"

def build_rows_from_predictions(
    df_features: pd.DataFrame,
    probs: np.ndarray,
    shap_values: Optional[np.ndarray],
    feature_names: Optional[List[str]],
    *,
    model_version: str,
    feature_version: str,
    churn_horizon_days: int = 30,
    model_name: str = "lgbm",
    pipeline_run_id: Optional[str] = None,
    reference_dt: Optional[datetime] = None,
    data_cutoff_at: Optional[datetime] = None,
) -> pd.DataFrame:
    if pipeline_run_id is None:
        pipeline_run_id = f"{model_name}_{int(time.time())}"
    scored_at = utcnow()
    if reference_dt is None: reference_dt = scored_at
    if data_cutoff_at is None: data_cutoff_at = reference_dt

    probs = np.asarray(probs, dtype=float)
    pct = percentile_ranks(probs)

    # row별 SHAP top3 (절대값 기준)
    top1_f = [None]*len(df_features); top2_f = [None]*len(df_features); top3_f = [None]*len(df_features)
    top1_s = [None]*len(df_features); top2_s = [None]*len(df_features); top3_s = [None]*len(df_features)
    if shap_values is not None and feature_names is not None and len(df_features) == shap_values.shape[0]:
        sv = shap_values
        if sv.ndim == 3: sv = sv[:, 1, :]  # 이진분류일 때 양성 클래스
        for i in range(len(df_features)):
            contrib = np.abs(sv[i])
            idx = np.argsort(-contrib)[:3]
            names = [feature_names[j] for j in idx]
            vals  = [float(sv[i, j]) for j in idx]
            if len(names) > 0: top1_f[i], top1_s[i] = names[0], vals[0]
            if len(names) > 1: top2_f[i], top2_s[i] = names[1], vals[1]
            if len(names) > 2: top3_f[i], top3_s[i] = names[2], vals[2]

    out = pd.DataFrame({
        "user_id": df_features["user_id"].astype("int64"),
        "scored_at": scored_at,
        "model_name": model_name,
        "model_version": model_version,
        "feature_version": feature_version,
        "data_cutoff_at": data_cutoff_at,
        "reference_dt": reference_dt,
        "churn_horizon_days": int(churn_horizon_days),
        "churn_threshold_dt": reference_dt - timedelta(days=churn_horizon_days),
        "churn_probability_raw": probs,
        "score_percentile": pct,
        "top1_feature": top1_f, "top1_shap": top1_s,
        "top2_feature": top2_f, "top2_shap": top2_s,
        "top3_feature": top3_f, "top3_shap": top3_s,
        # 패스스루 (없으면 NULL)
        "order_count": df_features.get("order_count"),
        "total_spend": df_features.get("total_spend"),
        "avg_order_value": df_features.get("avg_order_value"),
        "avg_days_between_orders": df_features.get("avg_days_between_orders"),
        "login_count": df_features.get("login_count"),
        "cart_count": df_features.get("cart_count"),
        "recency_days": df_features.get("recency_days"),
        "rfm_sum": df_features.get("rfm_sum"),
        "age": df_features.get("age"),
        "gender": df_features.get("gender"),
        "age_group": df_features.get("age_group"),
        "used_coupon": df_features.get("used_coupon"),
        "avg_cart_per_login": df_features.get("avg_cart_per_login"),
        "category_diversity": df_features.get("category_diversity"),
        "rfm_bucket": df_features.get("rfm_bucket"),
        "kmeans_cluster": df_features.get("kmeans_cluster"),
        "imputations_count": df_features.get("imputations_count", 0),
        "anomalies_count": df_features.get("anomalies_count", 0),
        "pipeline_run_id": pipeline_run_id,
        "action_code_suggested": None,
    })
    return out

def upsert_prediction_user_churn(conn, df_rows: pd.DataFrame,
                                 *, model_name: str, model_version: str, feature_version: str,
                                 churn_horizon_days: int):
    pol = _fetch_current_scoring_policy(conn, model_name, model_version, feature_version, churn_horizon_days)
    actions = _fetch_active_actions(conn)

    rb = []
    ac = []
    for p in df_rows["churn_probability_raw"].tolist():
        band = _risk_band(float(p), pol["cutpoint_vh"], pol["cutpoint_h"], pol["cutpoint_m"])
        rb.append(band); ac.append(actions.get(band))
    df_rows = df_rows.copy()
    df_rows["risk_band"] = rb
    df_rows["action_code_suggested"] = ac

    cols = [
        "user_id","scored_at","model_name","model_version","feature_version","data_cutoff_at",
        "reference_dt","churn_horizon_days","churn_threshold_dt","churn_probability_raw",
        "risk_band","score_percentile","top1_feature","top1_shap","top2_feature","top2_shap",
        "top3_feature","top3_shap","order_count","total_spend","avg_order_value",
        "avg_days_between_orders","login_count","cart_count","recency_days","rfm_sum",
        "age","gender","age_group","used_coupon","avg_cart_per_login","category_diversity",
        "rfm_bucket","kmeans_cluster","action_code_suggested","imputations_count","anomalies_count",
        "pipeline_run_id"
    ]

    with conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TEMP TABLE tmp_pred_churn ON COMMIT DROP AS
            SELECT * FROM analytics.prediction_user_churn WHERE 1=0;
        """)
        cur.execute("ALTER TABLE tmp_pred_churn DROP COLUMN valid_from;")
        cur.execute("ALTER TABLE tmp_pred_churn DROP COLUMN valid_until;")
        cur.execute("ALTER TABLE tmp_pred_churn DROP COLUMN report_dt_kst;")

        values = [tuple(df_rows[c].tolist()[i] for c in cols) for i in range(len(df_rows))]
        extras.execute_values(cur, f"""
            INSERT INTO tmp_pred_churn ({', '.join(cols)}) VALUES %s
        """, values)

        cur.execute("""
            WITH incoming AS (
              SELECT DISTINCT user_id, reference_dt, churn_horizon_days, model_version
              FROM tmp_pred_churn
            )
            UPDATE analytics.prediction_user_churn t
               SET valid_until = NOW()
             WHERE t.valid_until = 'infinity'::timestamptz
               AND (t.user_id, t.reference_dt, t.churn_horizon_days, t.model_version)
                   IN (SELECT user_id, reference_dt, churn_horizon_days, model_version FROM incoming);
        """)

        cur.execute(f"""
            INSERT INTO analytics.prediction_user_churn ({', '.join(cols)}, valid_from, valid_until)
            SELECT {', '.join(cols)}, NOW(), 'infinity'::timestamptz
            FROM tmp_pred_churn;
        """)
