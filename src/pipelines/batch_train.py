# -*- coding: utf-8 -*-
"""
배치 학습 파이프라인 (프로덕션용)
- S3에서 일자 파티션 피처 로드 (features/dt=YYYY-MM-DD/)
- 비교용 로지스틱/랜덤포레스트(선택) 학습 및 검증 지표 산출 (차트 저장 없음)
- [핵심] LGBM 학습 -> 모델 아티팩트 S3 저장 (registry.save_model)
- 검증 셋에서 임계값 스윕(0..1)으로 'Balanced Accuracy' 최대 임계값 채택
- 리스크 밴드 컷포인트는 검증 예측확률의 분위수로 산정 (기본 90/70/40%)  # TODO(추후 수정)
- 위 결과를 analytics.scoring_policy 에 업서트

실행 예:
  python -m src.pipelines.batch_train 2025-08-01 2025-08-30 2025-08-31 \
    --horizon 90 --model-version lgbm_v1.0_shaTODO --feature-version feat_v1.0
"""
from __future__ import annotations
import os
import sys
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd

# LGBM 필수 (프로덕션 추론 경로가 LGBM 전용)
try:
    import lightgbm as lgb
except Exception as e:
    raise RuntimeError("LightGBM이 필요합니다. `pip install lightgbm` 후 다시 실행하세요.") from e

# 비교용(있으면 사용, 없어도 OK)
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, matthews_corrcoef,
    balanced_accuracy_score, confusion_matrix
)

from ..common.settings import CFG
from ..common.io import read_parquet_s3, s3_join
from ..common.log import get_logger
from ..model.registry import save_model
from ..db.policy import upsert_scoring_policy

log = get_logger("batch_train")

# ----------------------------
# S3 경로 헬퍼
# ----------------------------
def _features_prefix(dt: str) -> str:
    # TODO(추후 수정): 파티션 규약 변경 시 중앙집중 수정
    return s3_join(CFG.S3_FEATURE_PREFIX, f"dt={dt}")

# ----------------------------
# 데이터 로드/병합
# ----------------------------
def load_range(dates: List[str]) -> pd.DataFrame:
    frames = []
    for d in dates:
        df = read_parquet_s3(_features_prefix(d))
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ----------------------------
# 훈련/검증 분리 (유저 기준 랜덤)
# ----------------------------
def make_train_valid(df: pd.DataFrame, valid_dt: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 단순: valid_dt 파티션은 검증, 나머지는 학습. (데이터 누수 방지)
    df_valid = read_parquet_s3(_features_prefix(valid_dt))
    if df_valid.empty:
        raise ValueError(f"검증 파티션이 비었습니다: dt={valid_dt}")
    if df.empty:
        raise ValueError("학습 파티션이 비었습니다.")

    # 공통 스키마로 정렬
    cols = sorted(set(df.columns) | set(df_valid.columns))
    df = df.reindex(columns=cols)
    df_valid = df_valid.reindex(columns=cols)

    return df, df_valid

# ----------------------------
# 피처 선택/정제
# ----------------------------
def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    if "churn" not in df.columns:
        raise ValueError("입력 피처에 'churn' 라벨이 필요합니다.")
    y = df["churn"].astype(int).values

    # 사용 가능한 수치형 피처 전부 사용하되, 명백한 ID/라벨 제외
    drop_cols = {"user_id", "churn"}
    X = df.drop(columns=[c for c in df.columns if c in drop_cols], errors="ignore")

    # 범주형이 섞여 있다면 간단 처리: 문자열/범주 → 존재하면 더미화 대신 빈값 처리(0.0)  # TODO(추후 수정: 원-핫)
    # 여기서는 수치형만 필터링
    X = X.select_dtypes(include=["number"]).copy()

    # 결측/무한대 처리
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # 열 순서 고정
    feature_names = list(X.columns)
    return X, y, feature_names

# ----------------------------
# 임계값 스윕 & 요약
# ----------------------------
def _metrics_at(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict:
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return dict(
        thr=thr,
        ACC=accuracy_score(y_true, y_pred),
        BAL=balanced_accuracy_score(y_true, y_pred),
        F1=f1_score(y_true, y_pred, zero_division=0),
        MCC=matthews_corrcoef(y_true, y_pred),
        TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn)
    )

def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[Dict, pd.DataFrame]:
    grid = np.linspace(0.0, 1.0, 2001)  # 0.0005 step
    rows = [_metrics_at(y_true, y_prob, t) for t in grid]
    df = pd.DataFrame(rows)
    best_bal = df.loc[df["BAL"].idxmax()].to_dict()
    # 보조 지표
    best_mcc = df.loc[df["MCC"].idxmax()].to_dict()
    best_acc = df.loc[df["ACC"].idxmax()].to_dict()
    best_f1  = df.loc[df["F1"].idxmax()].to_dict()
    summary = {
        "best_bal": best_bal,
        "best_mcc": best_mcc,
        "best_acc": best_acc,
        "best_f1":  best_f1,
        "base_05":  df.iloc[(np.abs(df["thr"] - 0.5)).idxmin()].to_dict()
    }
    return summary, df

# ----------------------------
# 밴드 컷포인트 산정 (기본: 분위수 기반)
# ----------------------------
def compute_band_cutpoints(prob: np.ndarray, q_vh=0.90, q_h=0.70, q_m=0.40) -> Tuple[float,float,float]:
    """
    검증 예측확률 분포의 분위수로 컷포인트 설정.
    - VH: 상위 10% 이상
    - H : 상위 30% 이상
    - M : 상위 60% 이상
    L 은 그 미만
    # TODO(추후 수정): 비즈니스 목표(캠페인 예산/규모)에 맞게 조정
    """
    prob = np.asarray(prob, dtype=float)
    if prob.size == 0:
        return 0.8, 0.6, 0.4
    cut_vh = float(np.quantile(prob, q_vh))
    cut_h  = float(np.quantile(prob, q_h))
    cut_m  = float(np.quantile(prob, q_m))
    # 정렬 보장
    cut_vh = max(cut_vh, cut_h)
    cut_h  = max(cut_h,  cut_m)
    return cut_vh, cut_h, cut_m

# ----------------------------
# 메인 파이프라인
# ----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("train_start_dt", type=str, help="학습 시작 dt=YYYY-MM-DD")
    parser.add_argument("train_end_dt",   type=str, help="학습 끝   dt=YYYY-MM-DD (포함/단일 가능)")
    parser.add_argument("valid_dt",       type=str, help="검증 dt=YYYY-MM-DD")
    parser.add_argument("--horizon", type=int, default=CFG.HORIZON_DAYS, help="churn_horizon_days")
    parser.add_argument("--model-version", type=str, default=CFG.MODEL_VERSION, help="저장할 모델 버전 # TODO(추후 수정)")
    parser.add_argument("--feature-version", type=str, default=CFG.FEATURE_VERSION, help="피처 버전 # TODO(추후 수정)")
    parser.add_argument("--compare-baselines", action="store_true", help="로지/랜포/XGB 비교 지표 출력(저장 X)")
    args = parser.parse_args()

    # 1) 로드
    train_dates = [args.train_start_dt]
    if args.train_end_dt != args.train_start_dt:
        train_dates.append(args.train_end_dt)
    df_train = load_range(train_dates)
    df_valid = read_parquet_s3(_features_prefix(args.valid_dt))
    if df_train.empty or df_valid.empty:
        raise ValueError("학습/검증 데이터가 비었습니다. S3 경로/파티션을 확인하세요.")

    # 2) X, y 준비
    X_tr, y_tr, feats_tr = prepare_xy(df_train)
    X_va, y_va, feats_va = prepare_xy(df_valid)
    # 피처 정렬(이름 기준)
    common_feats = sorted(set(feats_tr) & set(feats_va))
    X_tr = X_tr.reindex(columns=common_feats)
    X_va = X_va.reindex(columns=common_feats)

    # 3) (선택) 베이스라인 비교 (로지/랜포/XGB)
    if args.compare_baselines:
        # Logistic
        logit = Pipeline(steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(
                solver="lbfgs", max_iter=2000, class_weight="balanced", n_jobs=None
            ))
        ])
        logit.fit(X_tr, y_tr)
        p_logit = logit.predict_proba(X_va)[:, 1]
        log.info(f"[baseline] logistic AUC={roc_auc_score(y_va, p_logit):.4f}")

        # RF
        rf = RandomForestClassifier(
            n_estimators=500, max_depth=None,
            min_samples_split=2, min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=42, n_jobs=-1
        )
        rf.fit(X_tr, y_tr)
        p_rf = rf.predict_proba(X_va)[:, 1]
        log.info(f"[baseline] rf       AUC={roc_auc_score(y_va, p_rf):.4f}")

        # XGB (있으면)
        if _HAS_XGB:
            xgb = XGBClassifier(
                n_estimators=600, max_depth=5,
                subsample=0.8, colsample_bytree=0.8,
                learning_rate=0.05, reg_lambda=1.0,
                objective="binary:logistic", eval_metric="auc",
                n_jobs=-1, random_state=42, tree_method="hist"
            )
            xgb.fit(X_tr, y_tr)
            p_xgb = xgb.predict_proba(X_va)[:, 1]
            log.info(f"[baseline] xgb     AUC={roc_auc_score(y_va, p_xgb):.4f}")
        else:
            log.info("[baseline] xgb not installed (skipped)")

    # 4) LGBM 학습 (프로덕션 저장 대상)
    # conf/model.yaml 있으면 읽고, 없으면 디폴트 사용
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 50,
        "verbose": -1,
        # TODO(추후 수정): conf/model.yaml로 외부화
    }
    try:
        import yaml, pathlib
        cfg_path = pathlib.Path("conf/model.yaml")
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                yml = yaml.safe_load(f)
            if isinstance(yml, dict):
                params.update({k: yml[k] for k in yml.keys()})
                log.info("[model] conf/model.yaml 로드 완료")
    except Exception as e:
        log.info(f"[model] conf/model.yaml 로드 생략: {e}")

    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_va, label=y_va)
    booster = lgb.train(
        params,
        dtr,
        valid_sets=[dtr, dva],
        num_boost_round=params.get("num_boost_round", 800),
        early_stopping_rounds=params.get("early_stopping_rounds", 50),
        verbose_eval=False
    )

    # 5) 검증 예측 & 임계값 튜닝
    p_va = booster.predict(X_va, num_iteration=booster.best_iteration or -1)
    auc = roc_auc_score(y_va, p_va)
    log.info(f"[lgbm] valid AUC={auc:.4f} (features={len(common_feats)})")

    thr_sum, thr_df = tune_thresholds(y_va, p_va)
    # 기본값: Balanced Accuracy 최대 임계값
    thr_default = float(thr_sum["best_bal"]["thr"])

    # 6) 밴드 컷포인트 산정(검증 확률 분위수)  # TODO(추후 수정: 비즈니스 기준에 맞게 q 조정)
    cut_vh, cut_h, cut_m = compute_band_cutpoints(p_va, q_vh=0.90, q_h=0.70, q_m=0.40)
    # 방어적 정렬
    cut_vh = max(cut_vh, cut_h); cut_h = max(cut_h, cut_m)

    log.info(f"[policy] threshold_default={thr_default:.4f}, cut_vh={cut_vh:.4f}, cut_h={cut_h:.4f}, cut_m={cut_m:.4f}")

    # 7) 모델 저장(S3, 버전/피처 포함) + 정책 업서트(DB)
    out = save_model(booster, feature_names=common_feats, extra_meta={
        "valid_auc": float(auc),
        "horizon_days": int(args.horizon),
        "notes": "auto-trained via batch_train.py"
    })
    log.info(f"[saved] model_uri={out['model']} meta_uri={out['meta']}")

    # 정책 반영(DB)
    from ..common.settings import CFG as _CFG
    with _CFG.connect_db() as conn:
        upsert_scoring_policy(
            conn,
            model_name="lgbm",
            model_version=args.model_version,     # TODO(추후 수정)
            feature_version=args.feature_version, # TODO(추후 수정)
            churn_horizon_days=int(args.horizon),
            threshold_default=thr_default,
            cutpoint_vh=cut_vh,
            cutpoint_h=cut_h,
            cutpoint_m=cut_m,
            effective_from=None,  # NOW()
        )
    log.info("[policy] analytics.scoring_policy upsert 완료")

if __name__ == "__main__":
    main()
