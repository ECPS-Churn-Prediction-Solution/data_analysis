# -*- coding: utf-8 -*-
"""
Batch scoring orchestrator:
- (1) Load features from S3 per dt
- (2) Predict (proba, optional SHAP)
- (3) Build rows and upsert to analytics.prediction_user_churn (SCD2)
- (4) Update mart.daily_churn_prediction_aggr for the same dt
- (5) (optional) Export raw predictions to S3/parquet

Notes:
- writer.upsert_prediction_user_churn() internally maps risk_band -> action_code
  using analytics.action_recommendations and snapshots it into action_code_suggested.
- This script pins reference_dt/data_cutoff_at to the given dt (UTC midnight)
  so mart aggregation aligns with the same date.

CLI:
  python -m src.pipelines.batch_score --dt 2025-09-01 --horizon 30 \
    [--write_s3] [--skip_mart]

Env:
  PG_DSN or (PG_HOST, PG_DB, PG_USER, PG_PASSWORD[, PG_PORT])
  MODEL_VERSION / FEATURE_VERSION come from CFG
  WRITE_PREDICTIONS_TO_S3=1 to enable S3 export by default

Exit codes:
  0: success (even if no features found)
"""
from __future__ import annotations
import argparse
import os
from datetime import datetime, date, timezone

import psycopg2

from ..model.predict_lgbm import predict_for_date
from ..db.writer import build_rows_from_predictions, upsert_prediction_user_churn
from ..db.mart_writer import update_mart_daily_aggregates
from ..db.seed_policy import ensure_scoring_policy
from ..common.settings import CFG

# (NEW) optional S3 export
try:
    from .prediction_export import export_predictions_to_s3
except Exception:  # pragma: no cover
    export_predictions_to_s3 = None  # type: ignore


# -------- DB helpers --------
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


def _utc_midnight(dt_str: str) -> datetime:
    # reference_dt/data_cutoff_at을 파티션 날짜의 UTC 00:00으로 고정
    return datetime.fromisoformat(dt_str + "T00:00:00").replace(tzinfo=timezone.utc)


def _ensure_actions_and_log(conn) -> None:
    """Pre-flight: 활성 액션이 밴드별로 있는지 확인하고 로그만 남김.
    매핑 자체는 writer.upsert_prediction_user_churn()에서 수행됨."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT risk_band, action_code
            FROM analytics.action_recommendations
            WHERE is_active = true
              AND effective_until = 'infinity'::timestamptz
            """
        )
        rows = cur.fetchall()
        m = {rb: ac for rb, ac in rows}
        missing = [b for b in ("VH", "H", "M", "L") if b not in m]
        print(f"[batch_score] active actions: {m or '(none)'}")
        if missing:
            print(
                f"[batch_score][WARN] missing active actions for bands: {missing}. "
                f"writer will store NULL for action_code_suggested on those bands."
            )


# -------- main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", default=None, help="Partition date (YYYY-MM-DD or YYYYMMDD)")
    ap.add_argument(
        "--horizon",
        type=int,
        default=int(os.getenv("CHURN_HORIZON_DAYS", 30)),
        help="Churn horizon days (e.g., 30/60/90)",
    )
    ap.add_argument("--model_name", default=os.getenv("MODEL_NAME", "lgbm"))
    ap.add_argument("--pipeline_run_id", default=os.getenv("PIPELINE_RUN_ID"))
    ap.add_argument("--skip_mart", action="store_true", help="Skip mart aggregation step")

    # (NEW) also write raw predictions to S3 parquet
    ap.add_argument(
        "--write_s3",
        action="store_true",
        default=os.getenv("WRITE_PREDICTIONS_TO_S3", "0") not in {"0", "false", "False"},
        help="Also export raw predictions to S3 (parquet) under S3_PREDICTION_PREFIX.",
    )

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
        # 액션 존재 사전 점검(매핑은 writer에서 처리)
        _ensure_actions_and_log(conn)

    # (3) Build output rows (reference_dt/data_cutoff_at = dt 자정 UTC 고정)
    ref_dt = _utc_midnight(dt)
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
        reference_dt=ref_dt,
        data_cutoff_at=ref_dt,
    )

    # (3.5) (NEW) Export raw predictions to S3 if requested
    if args.write_s3:
        if export_predictions_to_s3 is None:
            print("[batch_score][WARN] export_predictions_to_s3 not available; skipping S3 export.")
        else:
            try:
                uri = export_predictions_to_s3(
                    out, dt=dt, model_name=args.model_name, horizon=horizon
                )
                print(f"[batch_score] exported raw predictions to {uri}")
            except Exception as e:  # pragma: no cover
                print(f"[batch_score][WARN] failed to export predictions to S3: {e!r}")

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
        if not args.skip_mart:
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
        f"{'(mart skipped)' if args.skip_mart else 'and refreshed mart.daily_churn_prediction_aggr.'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
