# -*- coding: utf-8 -*-
"""
Batch scoring orchestrator:
- (1) Load features from S3 per dt
- (2) Predict (proba, optional SHAP)
- (3) Build rows and upsert to analytics.prediction_user_churn (SCD2)
- (4) Update mart.daily_churn_prediction_aggr for the same dt

CLI:
  python -m src.pipelines.batch_score --dt 2025-09-01 --horizon 90

Env:
  PG_DSN or (PG_HOST, PG_DB, PG_USER, PG_PASSWORD[, PG_PORT])
  MODEL_VERSION / FEATURE_VERSION come from CFG

Exit codes:
  0: success (even if no features found)
"""
from __future__ import annotations
import argparse
import os
from datetime import datetime, date

import psycopg2

from ..model.predict_lgbm import predict_for_date
from ..db.writer import build_rows_from_predictions, upsert_prediction_user_churn
from ..db.mart_writer import update_mart_daily_aggregates
from ..db.seed_policy import ensure_scoring_policy
from ..common.settings import CFG


def _connect_db():
    dsn = os.getenv("PG_DSN")
    if dsn:
        return psycopg2.connect(dsn)
    return psycopg2.connect(
        host=os.getenv("PG_HOST", getattr(CFG, "PG_HOST", None)),
        dbname=os.getenv("PG_DB", getattr(CFG, "PG_DATABASE", None)),
        user=os.getenv("PG_USER", getattr(CFG, "PG_USER", None)),
        password=os.getenv("PG_PASSWORD", getattr(CFG, "PG_PASSWORD", None)),
        port=int(os.getenv("PG_PORT", getattr(CFG, "PG_PORT", 5432) or 5432)),
        sslmode=os.getenv("PG_SSLMODE", getattr(CFG, "PG_SSLMODE", "prefer")),
    )


def _parse_dt(s: str | None) -> str:
    if not s:
        return date.today().isoformat()
    s = s.strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    datetime.strptime(s, "%Y-%m-%d")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", default=None, help="Partition date (YYYY-MM-DD or YYYYMMDD)")
    ap.add_argument("--horizon", type=int, default=int(os.getenv("CHURN_HORIZON_DAYS", 90)),
                    help="Churn horizon days (e.g., 30/60/90)")
    ap.add_argument("--model_name", default=os.getenv("MODEL_NAME", "lgbm"))
    ap.add_argument("--pipeline_run_id", default=os.getenv("PIPELINE_RUN_ID"))
    args = ap.parse_args()

    dt = _parse_dt(args.dt)
    horizon = int(args.horizon)

    # (1) Predict
    df, proba, shap_values, feature_names = predict_for_date(dt)
    if df.empty:
        print(f"[batch_score] No features for dt={dt}. Nothing to do.")
        return 0

    # (2) Ensure scoring policy exists for this (model, version, feature, horizon)
    with _connect_db() as conn:
        ensure_scoring_policy(
            conn,
            model_name=args.model_name,
            model_version=CFG.MODEL_VERSION,
            feature_version=CFG.FEATURE_VERSION,
            churn_horizon_days=horizon,
        )

    # (3) Build output rows
    out = build_rows_from_predictions(
        df,
        proba,
        shap_values,
        feature_names,
        model_version=CFG.MODEL_VERSION,
        feature_version=CFG.FEATURE_VERSION,
        churn_horizon_days=horizon,
        model_name=args.model_name,
        pipeline_run_id=args.pipeline_run_id,
    )

    # (4) Upsert analytics and update mart
    with _connect_db() as conn:
        upsert_prediction_user_churn(
            conn,
            out,
            model_name=args.model_name,
            model_version=CFG.MODEL_VERSION,
            feature_version=CFG.FEATURE_VERSION,
            churn_horizon_days=horizon,
        )
        update_mart_daily_aggregates(
            conn,
            ref_date=dt,
            model_version=CFG.MODEL_VERSION,
            feature_version=CFG.FEATURE_VERSION,
            horizons=[horizon],  # pass multiple if you compute multiple horizons same day
        )
        conn.commit()

    print(
        f"[batch_score] dt={dt} wrote {len(out)} rows to analytics.prediction_user_churn "
        f"and refreshed mart.daily_churn_prediction_aggr."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
