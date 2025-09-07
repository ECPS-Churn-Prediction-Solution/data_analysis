# -*- coding: utf-8 -*-
"""
배치 스코어링 결과를 analytics.prediction_user_churn에 업서트.
- FEATURE_PREDICT_URI 읽어서 예측 실행 (predict_lgbm.predict_uri)
- scoring_policy + action_recommendations 적용
- percentile, risk_band, 유효기간, threshold_dt 계산
- (가능하면) 피처 칼럼 매핑하여 NULL 대신 값 채움
- 업서트 직후 mart.* 집계 리프레시
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
from .mart_refresh import main as mart_refresh_main

# ---- helpers -----------------------------------------------------------------

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

def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _to_number(x) -> Optional[float]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        return float(x)
    except Exception:
        try:
            return float(str(x).replace(",", ""))
        except Exception:
            return None

def _to_int(x) -> Optional[int]:
    v = _to_number(x)
    if v is None:
        return None
    try:
        return int(round(v))
    except Exception:
        return None

def _gender_label(x) -> Optional[str]:
    """
    DDL 제약: 'MALE' | 'FEMALE'
    가능한 입력 형태를 안전하게 매핑. 불명확하면 None (제약 위반 회피)
    """
    if x is None:
        return None
    s = str(x).strip().upper()
    if s in ("MALE", "M"):
        return "MALE"
    if s in ("FEMALE", "F"):
        return "FEMALE"
    # 숫자 인코딩 케이스(권장 매핑: 1=MALE, 2=FEMALE / 그 외는 None)
    try:
        n = int(float(s))
        if n == 1:
            return "MALE"
        if n == 2:
            return "FEMALE"
    except Exception:
        pass
    return None

def _age_group_of(age: Optional[int]) -> Optional[str]:
    if age is None:
        return None
    try:
        a = int(age)
    except Exception:
        return None
    if a <= 24:  return "<25"
    if a <= 34:  return "25-34"
    if a <= 44:  return "35-44"
    return "45+"

def _maybe_load_features(uri: str) -> Optional[pd.DataFrame]:
    """
    예측 함수가 피처 칼럼을 돌려주지 않는 환경을 대비해,
    동일 URI를 파케로 읽어 피처만 로드(가능한 경우).
    프로젝트 공용 I/O가 있으면 우선 시도, 실패 시 pandas 직접 읽기.
    """
    # 1) 공용 I/O 유틸이 있다면 사용
    try:
        from ..common.io import read_df  # 프로젝트에 있을 가능성 높음
        fe = read_df(uri)  # s3://.../dt=YYYY-MM-DD/part-*.parquet 지원
        if isinstance(fe, pd.DataFrame) and not fe.empty:
            return fe
    except Exception:
        pass
    # 2) pandas 직접 읽기 (s3fs/pyarrow가 세팅되어 있어야 함)
    try:
        fe = pd.read_parquet(uri)
        if isinstance(fe, pd.DataFrame) and not fe.empty:
            return fe
    except Exception:
        pass
    return None

def _build_row(
    rec: Dict[str, Any],
    scored_at: datetime,
    reference_dt: datetime,
    horizon: int,
    churn_threshold_dt: datetime,
    pipeline_run_id: str,
    model_name: str,
    model_version: str,
    feature_version: str,
) -> Tuple[Any, ...]:
    """analytics.prediction_user_churn INSERT 컬럼 순서에 맞춰 튜플 구성"""

    # ====== 기본 값/파생 ======
    user_id = int(rec["user_id"])
    p       = float(rec["churn_score"])
    pctl    = float(rec["score_percentile"])
    band    = str(rec["risk_band"])

    # 피처 매핑(있으면 사용, 없으면 None)
    # ───────── 주문/사용/액션 쪽 ─────────
    order_count  = _to_int(rec.get("orders_last_90d")) \
                   or _to_int(rec.get("order_count_last_90d")) \
                   or _to_int(rec.get("frequency_last_90d")) \
                   or _to_int(rec.get("orders_last_30d")) \
                   or _to_int(rec.get("frequency_last_30d"))

    avg_order_value = _to_number(rec.get("avg_order_value")) \
                      or _to_number(rec.get("monetary_avg_order"))

    avg_days_between_orders = _to_number(rec.get("days_between_orders"))

    login_count = _to_int(rec.get("login_count_last_30d")) \
                  or _to_int(rec.get("session_count_last_30d")) \
                  or _to_int(rec.get("logins_last_30d"))

    cart_count  = _to_int(rec.get("cart_count_last_30d")) \
                  or _to_int(rec.get("cart_additions_last_30d"))

    # recency: 가능한 후보명 중 가장 먼저 존재하는 값 사용
    recency_days = _to_int(rec.get("days_since_last_order")) \
                   or _to_int(rec.get("days_since_last_purchase")) \
                   or _to_int(rec.get("days_since_last_activity")) \
                   or _to_int(rec.get("days_since_last_session"))

    # 총 지출 추정: 명시 total이 없으면 "avg_order_value * order_count"
    total_spend = _to_number(rec.get("total_spend_last_90d")) \
                  or _to_number(rec.get("monetary_total_90d"))
    if total_spend is None and (avg_order_value is not None) and (order_count is not None):
        total_spend = float(avg_order_value) * int(order_count)

    # ───────── 인구통계/행동 ─────────
    age    = _to_int(rec.get("age"))
    gender = _gender_label(rec.get("gender"))
    age_group = _age_group_of(age)

    used_coupon = None
    cup_rate = _to_number(rec.get("coupon_usage_rate"))
    if cup_rate is not None:
        used_coupon = bool(cup_rate > 0)

    category_diversity = _to_number(rec.get("category_diversity")) \
                         or _to_number(rec.get("num_interests"))

    # 평균 카트/로그인
    avg_cart_per_login = None
    if cart_count is not None and login_count is not None and login_count > 0:
        avg_cart_per_login = float(cart_count) / float(login_count)

    # RFM 파생
    r_score = _to_int(rec.get("recency_score"))
    f_score = _to_int(rec.get("frequency_score"))
    m_score = _to_int(rec.get("monetary_score"))
    rfm_sum = None
    if r_score is not None and f_score is not None and m_score is not None:
        rfm_sum = int(r_score + f_score + m_score)

    rfm_bucket = rec.get("rfm_bucket")
    kmeans_cluster = rec.get("kmeans_cluster")
    if rfm_bucket is not None:
        rfm_bucket = str(rfm_bucket)
    if kmeans_cluster is not None:
        kmeans_cluster = str(kmeans_cluster)

    # SHAP (없으면 NULL 유지)
    top1_feature = rec.get("top1_feature")
    top1_shap    = _to_number(rec.get("top1_shap"))
    top2_feature = rec.get("top2_feature")
    top2_shap    = _to_number(rec.get("top2_shap"))
    top3_feature = rec.get("top3_feature")
    top3_shap    = _to_number(rec.get("top3_shap"))

    # 결과 튜플(DDL 순서와 정확히 동일)
    return (
        user_id,                     # user_id
        scored_at,                   # scored_at
        model_name,                  # model_name  (DDL 제약: 'lgbm')
        model_version,               # model_version
        feature_version,             # feature_version
        reference_dt,                # data_cutoff_at (== reference_dt)
        reference_dt,                # reference_dt
        int(horizon),                # churn_horizon_days
        churn_threshold_dt,          # churn_threshold_dt
        p,                           # churn_probability_raw
        band,                        # risk_band
        pctl,                        # score_percentile
        top1_feature, top1_shap,     # top1
        top2_feature, top2_shap,     # top2
        top3_feature, top3_shap,     # top3
        order_count,                 # order_count
        _to_int(total_spend),        # total_spend (정수형으로 정의되어 있어 int 캐스팅)
        _to_number(avg_order_value), # avg_order_value
        _to_number(avg_days_between_orders), # avg_days_between_orders
        login_count,                 # login_count
        cart_count,                  # cart_count
        recency_days,                # recency_days
        rfm_sum,                     # rfm_sum
        age,                         # age
        gender,                      # gender ('MALE'|'FEMALE' 또는 NULL)
        age_group,                   # age_group
        used_coupon,                 # used_coupon
        _to_number(avg_cart_per_login), # avg_cart_per_login
        _to_number(category_diversity), # category_diversity
        rfm_bucket,                  # rfm_bucket
        kmeans_cluster,              # kmeans_cluster
        str(rec.get("action_code_suggested")),  # action_code_suggested
        int(rec.get("imputations_count", 0)),   # imputations_count
        int(rec.get("anomalies_count", 0)),     # anomalies_count
        str(rec.get("pipeline_run_id")),        # pipeline_run_id
        scored_at,                   # valid_from
        INFTY                        # valid_until
    )

# ---- main --------------------------------------------------------------------

def main(predict_uri: Optional[str] = None):
    uri = predict_uri or os.getenv("FEATURE_PREDICT_URI")
    if not uri:
        raise SystemExit("FEATURE_PREDICT_URI가 비었습니다.")

    # 1) 예측 실행 → 최소 {user_id, churn_score}
    df = _predict_uri(uri)
    if df is None or df.empty:
        print("[analytics_upsert] no rows")
        return

    # 2) 기준일/메타 계산
    probs = df["churn_score"].astype(float)
    pct   = probs.rank(pct=True, method="average") * 100.0

    source_dt = _extract_source_dt_from_uri(uri) or now_utc().date()
    reference_dt = datetime(source_dt.year, source_dt.month, source_dt.day, tzinfo=timezone.utc)
    scored_at    = now_utc()
    horizon      = int(CFG.CHURN_HORIZON_DAYS)
    churn_threshold_dt = reference_dt - timedelta(days=horizon)
    data_cutoff_at     = reference_dt  # 동치 처리

    # 3) (가능하면) 피처 프레임 로드 후 조인
    feat_df = None
    # df 안에 피처가 이미 있을 수도 있으니 우선 df 컬럼으로 시도
    has_feature_cols = len(set(df.columns) & {
        "age","gender","tenure_days","recency_score","frequency_score","monetary_score",
        "monetary_avg_order","avg_items_per_order","frequency_last_30d","frequency_last_90d",
        "days_between_orders","coupon_usage_rate","days_since_last_order","days_since_last_purchase",
        "days_since_last_activity","days_since_last_session","session_count_last_30d",
        "cart_additions_last_30d","category_diversity","num_interests","rfm_bucket","kmeans_cluster",
        "total_spend_last_90d","monetary_total_90d","avg_order_value","order_count_last_90d",
        "orders_last_90d","orders_last_30d","login_count_last_30d","logins_last_30d","cart_count_last_30d"
    }) > 0

    if not has_feature_cols:
        feat_df = _maybe_load_features(uri)
        if feat_df is not None and "user_id" in feat_df.columns:
            # 필요한 컬럼만(전부 가져와도 무방) user_id 기준 left join
            df = df.merge(feat_df, on="user_id", how="left")

    # 4) 정책/액션 로드
    with CFG.connect_db() as conn:
        _, cut_vh, cut_h, cut_m = get_scoring_policy(conn)
        action_map = get_action_map(conn)

    # 5) 각 행 빌드
    rows: List[Tuple[Any, ...]] = []
    pipeline_run_id = os.getenv("PIPELINE_RUN_ID") or os.getenv("CI_PIPELINE_ID") or os.getenv("GITHUB_RUN_ID") or "manual"

    # score_percentile 계산 결과를 df에 부착
    df = df.copy()
    df["score_percentile"] = pct.values

    # risk_band / action_code_suggested 부착
    df["risk_band"] = [ _risk_band(float(p), cut_vh, cut_h, cut_m) for p in probs.tolist() ]
    df["action_code_suggested"] = df["risk_band"].map(lambda b: action_map.get(b, (None,"NONE"))[1])
    df["pipeline_run_id"] = pipeline_run_id

    # dict row로 변환하며 INSERT 튜플 생성
    for rec in df.to_dict(orient="records"):
        rows.append(_build_row(
            rec=rec,
            scored_at=scored_at,
            reference_dt=reference_dt,
            horizon=horizon,
            churn_threshold_dt=churn_threshold_dt,
            pipeline_run_id=pipeline_run_id,
            model_name=CFG.MODEL_NAME,          # DDL 제약: 반드시 'lgbm'
            model_version=CFG.MODEL_VERSION,
            feature_version=CFG.FEATURE_VERSION,
        ))

    # 6) 업서트 실행(SCD2: 현재행 닫고 → 새행 삽입)
    with CFG.connect_db() as conn:
        affected = upsert_rows(conn, rows)
    print(f"[analytics_upsert] upserted rows={affected} into analytics.prediction_user_churn (ref_dt={reference_dt.date()}, horizon={horizon})")

    # 7) mart 집계 리프레시(해당 일자/지평만)
    mart_refresh_main(report_dt=str(reference_dt.date()), horizon_days=str(horizon))

if __name__ == "__main__":
    main()
