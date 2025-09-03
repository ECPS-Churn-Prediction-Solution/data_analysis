# -*- coding: utf-8 -*-
"""
S3에서 피처 로드 → 전처리(표준화) → 예측/SHAP 계산
- 버킷: CFG.S3_BUCKET (또는 직접 URI 지정)
- 프리픽스/파티션: CFG.S3_FEATURE_PREFIX + CFG.S3_FEATURE_PARTITION_FMT
- 운영 안전성:
  * 대용량 배치 추론/SHAP
  * SHAP 토글/근사치 옵션(중앙설정 CFG 사용)
  * meta.feature_names 기준 엄격 정렬 + 누락 컬럼 보정(카테고리는 'unknown')
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd

# SHAP은 선택적. 없으면 빈 배열 반환 → writer가 Top-K 스킵
try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:
    shap = None  # type: ignore
    _HAS_SHAP = False

from ..common.settings import CFG
from ..common.io import read_parquet_s3, s3_join
from ..features.prepare import prepare_features
from .registry import load_model


def _features_prefix_for_date(dt_str: str) -> str:
    """
    날짜 문자열(YYYY-MM-DD)로 파티션 경로 생성.
    기본: <S3_FEATURE_PREFIX>/dt=YYYY-MM-DD/
    CFG.S3_FEATURE_PARTITION_FMT가 strftime 포맷이면 적용.
    """
    part_fmt = CFG.S3_FEATURE_PARTITION_FMT
    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%d")
        part = dt.strftime(part_fmt) if any(ch in part_fmt for ch in ("%Y", "%m", "%d")) else part_fmt
    except Exception:
        # dt_str이 이미 파티션 문자열일 수도 있음
        part = part_fmt.replace("%Y-%m-%d", dt_str)
        if part == part_fmt:
            # 포맷 치환 실패 시 fallback
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
        * 범주형: 'unknown' 카테고리로 채움
        * 수치형: 0.0
    - 여분 컬럼은 드랍
    """
    # 누락 컬럼 생성
    missing = [c for c in feature_names if c not in X.columns]
    if missing:
        n = len(X)
        for c in missing:
            if c in categorical_features:
                X[c] = pd.Series(["unknown"] * n, dtype="category")
            else:
                X[c] = 0.0
    # 순서 강제 + 여분 드랍
    X = X.reindex(columns=feature_names)
    return X


def _predict_in_batches(booster, X: pd.DataFrame, batch_size: int) -> np.ndarray:
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


def _shap_in_batches(booster, X: pd.DataFrame, batch_size: int, approximate: bool) -> np.ndarray:
    """
    LightGBM 전용 TreeExplainer로 배치 SHAP. 이진 분류일 때 클래스 축(list) 처리 포함.
    approximate=True일 때 빠르지만 약간의 오차 허용.
    """
    if not _HAS_SHAP or len(X) == 0:
        return np.empty((0, X.shape[1]), dtype=float)

    explainer = shap.TreeExplainer(booster)
    out_rows: list[np.ndarray] = []
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        chunk = X.iloc[start:stop, :]
        try:
            sv = explainer.shap_values(chunk, approximate=approximate) if approximate else explainer.shap_values(chunk)
        except TypeError:
            sv = explainer.shap_values(chunk)
        if isinstance(sv, list):
            sv = sv[1]  # binary: positive class
        out_rows.append(np.asarray(sv, dtype=float))
    return np.vstack(out_rows) if out_rows else np.empty((0, X.shape[1]), dtype=float)


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
    # prefix 아래의 모든 parquet(part-*)을 읽어 하나의 DF로 합치는 유틸
    df = read_parquet_s3(prefix)

    if df.empty:
        return df, np.array([]), np.empty((0, len(feature_names))), feature_names

    # === 전처리: 표준화 (alias/dtype/결측/카테고리) ===
    X_raw, _ = prepare_features(df)  # ID/LABEL 제외된 피처 프레임

    # === 정렬/보정: 학습 feature_names 기준 ===
    X = _align_features(X_raw, feature_names, categorical_features)

    # === 예측 ===
    proba = _predict_in_batches(booster, X, CFG.PREDICT_BATCH_SIZE)

    # === SHAP(옵션) ===
    if CFG.PREDICT_COMPUTE_SHAP and _HAS_SHAP:
        shap_values = _shap_in_batches(booster, X, CFG.SHAP_BATCH_SIZE, CFG.PREDICT_SHAP_APPROX)
    else:
        shap_values = np.empty((0, X.shape[1]), dtype=float)

    return df, np.asarray(proba), shap_values, feature_names
