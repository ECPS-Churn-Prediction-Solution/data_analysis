# -*- coding: utf-8 -*-
"""
LightGBM 학습/예측 유틸
- train_model(train_uri, valid_uri)
- predict_uri(predict_uri)
- predict_for_date(dt)  # features/<partition>에서 일자 파티션 읽기
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import lightgbm as lgb

# SHAP은 선택적
try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:
    shap = None  # type: ignore
    _HAS_SHAP = False

from ..common.settings import CFG
from ..common.io import read_parquet_s3, read_features, s3_join
from ..features.prepare import prepare_features
from .registry import save_model, load_model

# ----------------------------
# Helpers
# ----------------------------
def _features_prefix_for_date(dt_str: str) -> str:
    """
    날짜 문자열(YYYY-MM-DD)로 파티션 경로 생성.
    기본: <S3_FEATURE_PREFIX>/dt=YYYY-MM-DD/
    CFG.S3_FEATURE_PARTITION_FMT가 strftime 포맷이면 적용.
    """
    part_fmt = CFG.S3_FEATURE_PARTITION_FMT
    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%d")
        if any(ch in part_fmt for ch in ("%Y", "%m", "%d")):
            part = dt.strftime(part_fmt)
        else:
            part = part_fmt
    except Exception:
        # dt_str이 이미 파티션 문자열일 수도 있음
        if "%Y-%m-%d" in part_fmt:
            part = part_fmt.replace("%Y-%m-%d", dt_str)
        else:
            part = f"dt={dt_str}"
    return s3_join(CFG.S3_FEATURE_PREFIX, part)

def _align_features(
    X: pd.DataFrame,
    feature_names: List[str],
    categorical_features: List[str],
) -> pd.DataFrame:
    """
    - 학습 시점 feature_names 순서로 정렬
    - 누락 컬럼 보정:
        * 범주형: 'unknown' 카테고리
        * 수치형: 0.0
    - 여분 컬럼은 드랍
    """
    missing = [c for c in feature_names if c not in X.columns]
    if missing:
        n = len(X)
        cat_set = set(categorical_features)
        for c in missing:
            if c in cat_set:
                X[c] = pd.Series(["unknown"] * n, dtype="category")
            else:
                X[c] = 0.0
    X = X.reindex(columns=feature_names)
    return X

def _predict_in_batches(booster: lgb.Booster, X: pd.DataFrame, batch_size: int) -> np.ndarray:
    n = len(X)
    if n == 0:
        return np.array([], dtype=float)
    probs = np.empty(n, dtype=float)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        probs[start:stop] = booster.predict(
            X.iloc[start:stop, :],
            num_iteration=getattr(booster, "best_iteration", None) or -1,
        )
    return probs

def _shap_in_batches(booster: lgb.Booster, X: pd.DataFrame, batch_size: int, approximate: bool) -> np.ndarray:
    if not _HAS_SHAP or len(X) == 0:
        return np.empty((0, X.shape[1]), dtype=float)
    explainer = shap.TreeExplainer(booster)  # type: ignore
    rows: List[np.ndarray] = []
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        chunk = X.iloc[start:stop, :]
        try:
            sv = explainer.shap_values(chunk, approximate=approximate) if approximate else explainer.shap_values(chunk)
        except TypeError:
            sv = explainer.shap_values(chunk)
        if isinstance(sv, list):
            sv = sv[1]  # binary: positive class
        rows.append(np.asarray(sv, dtype=float))
    return np.vstack(rows) if rows else np.empty((0, X.shape[1]), dtype=float)

# ----------------------------
# Train / Predict APIs
# ----------------------------
def train_model(train_uri: str, valid_uri: Optional[str] = None, params: Optional[Dict] = None):
    df_tr = read_features(train_uri)
    X_tr, y_tr = prepare_features(df_tr)

    cat_cols = list(X_tr.select_dtypes(include="category").columns)
    lgb_tr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)

    valid_sets = [lgb_tr]
    if valid_uri:
        df_va = read_features(valid_uri)
        X_va, y_va = prepare_features(df_va)
        lgb_va = lgb.Dataset(X_va, label=y_va, reference=lgb_tr, categorical_feature=cat_cols)
        valid_sets.append(lgb_va)

    _params = dict(
        objective="binary",
        metric=["auc", "binary_logloss"],
        boosting_type="gbdt",
        num_leaves=63,
        learning_rate=0.05,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        verbose=-1,
    )
    if params:
        _params.update(params)

    booster = lgb.train(
        _params,
        lgb_tr,
        valid_sets=valid_sets,
        num_boost_round=200,
        early_stopping_rounds=30 if len(valid_sets) > 1 else None,
    )

    meta = {
        "feature_names": list(X_tr.columns),
        "categorical_features": cat_cols,
        "model_name": CFG.MODEL_NAME,
        "feature_version": CFG.FEATURE_VERSION,
        "horizon_days": int(CFG.CHURN_HORIZON_DAYS),
    }
    save_model(
        booster,
        feature_names=meta["feature_names"],
        categorical_features=meta["categorical_features"],
        extra_meta={k: meta[k] for k in ["model_name", "feature_version", "horizon_days"]},
    )
    return booster, meta

def predict_uri(predict_uri: str) -> pd.DataFrame:
    booster, meta = load_model()
    feature_names: List[str] = meta.get("feature_names") or []
    categorical_features: List[str] = meta.get("categorical_features") or []

    df = read_features(predict_uri)
    X, _ = prepare_features(df)

    X = _align_features(X, feature_names, categorical_features)
    scores = booster.predict(X, num_iteration=getattr(booster, "best_iteration", None) or -1)
    out = pd.DataFrame({
        "user_id": df["user_id"].astype("Int64"),
        "churn_score": scores,
    })
    return out

def predict_for_date(dt: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    지정 일자(dt=YYYY-MM-DD)의 피처를 읽어 예측/SHAP을 반환.
    반환: (원본 DF, 예측확률, shap_values, feature_names)
    - shap_values shape: (n_samples, n_features) 또는 빈 배열
    """
    booster, meta = load_model()
    feature_names: List[str] = meta.get("feature_names") or []
    categorical_features: List[str] = meta.get("categorical_features") or []

    if not feature_names:
        raise RuntimeError("meta.json에 feature_names가 없습니다. 학습/저장 파이프라인을 확인하세요.")

    prefix = _features_prefix_for_date(dt)
    df = read_parquet_s3(prefix)

    if df.empty:
        return df, np.array([]), np.empty((0, len(feature_names))), feature_names

    X_raw, _ = prepare_features(df)  # ID/LABEL 제외된 피처 프레임
    X = _align_features(X_raw, feature_names, categorical_features)

    # 예측
    proba = _predict_in_batches(booster, X, CFG.PREDICT_BATCH_SIZE)

    # SHAP
    if CFG.PREDICT_COMPUTE_SHAP and _HAS_SHAP:
        shap_values = _shap_in_batches(booster, X, CFG.SHAP_BATCH_SIZE, CFG.PREDICT_SHAP_APPROX)
    else:
        shap_values = np.empty((0, X.shape[1]), dtype=float)

    return df, np.asarray(proba), shap_values, feature_names
