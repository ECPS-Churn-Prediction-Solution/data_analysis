# -*- coding: utf-8 -*-
"""
S3에서 피처 로드 → 예측/SHAP 계산
- 버킷: kdt3-preprocessing-data (고정)
- 프리픽스/파티션: TODO(추후 수정)
- 운영 안전성:
  * 대용량 처리: 배치 추론/배치 SHAP 지원
  * SHAP 토글/근사치 옵션(환경변수)
  * 메타 feature_names 기준 엄격 정렬 + 결측/무한대 처리
"""
from __future__ import annotations
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

# SHAP은 선택적. 없으면 None 반환 → writer가 Top3 스킵
try:
    import shap
    _HAS_SHAP = True
except Exception:
    shap = None  # type: ignore
    _HAS_SHAP = False

from ..common.settings import CFG
from ..common.io import read_parquet_s3, s3_join
from .registry import load_model

# --------- 환경 변수 토글 ---------
# SHAP 계산 여부(기본 1=계산). 0이면 계산 안 함
_PREDICT_COMPUTE_SHAP = os.getenv("PREDICT_COMPUTE_SHAP", "1") not in {"0", "false", "False"}

# SHAP 근사치 사용 여부(기본 1=근사 on). 정확도 <-> 속도 트레이드오프
_PREDICT_SHAP_APPROX = os.getenv("PREDICT_SHAP_APPROX", "1") not in {"0", "false", "False"}

# 예측 배치 크기(행 수). 메모리 상황에 맞게 조정
_PREDICT_BATCH_SIZE = int(os.getenv("PREDICT_BATCH_SIZE", "50000"))

# SHAP 배치 크기(행 수). 메모리 상황에 맞게 조정
_SHAP_BATCH_SIZE = int(os.getenv("SHAP_BATCH_SIZE", "20000"))

def _features_prefix_for_date(dt_str: str) -> str:
    """
    날짜 문자열(YYYY-MM-DD)로 파티션 경로 생성.
    기본: <S3_FEATURE_PREFIX>/dt=YYYY-MM-DD/
    # TODO(추후 수정): 파티션 규약 변경 시 여기만 수정
    """
    return s3_join(CFG.S3_FEATURE_PREFIX, f"dt={dt_str}")

def _ensure_numeric_matrix(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """
    meta.feature_names 기준으로 열 재정렬 + 수치화 + 결측/무한대 처리
    """
    X = df.reindex(columns=feature_names)
    # 수치형 변환 시도(문자열 섞여있을 수 있음)
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
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
            num_iteration=booster.best_iteration or -1
        )
    return probs

def _shap_in_batches(booster, X: pd.DataFrame, batch_size: int, approximate: bool) -> np.ndarray:
    """
    LightGBM 전용 TreeExplainer로 배치 SHAP. 이진분류일 때 클래스 축(list) 처리 포함.
    approximate=True 일 때 빠르지만 약간의 오차 허용.
    """
    if not _HAS_SHAP or len(X) == 0:
        return np.empty((0, X.shape[1]), dtype=float)

    explainer = shap.TreeExplainer(booster)
    rows = []
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        chunk = X.iloc[start:stop, :]
        try:
            sv = explainer.shap_values(chunk, approximate=approximate) if approximate else explainer.shap_values(chunk)
        except TypeError:
            # SHAP 버전에 따라 approximate 인자가 없을 수 있음
            sv = explainer.shap_values(chunk)
        # 이진 분류의 경우 리스트 반환
        if isinstance(sv, list):
            sv = sv[1]
        rows.append(np.asarray(sv, dtype=float))
    return np.vstack(rows) if rows else np.empty((0, X.shape[1]), dtype=float)

def predict_for_date(dt: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    지정 일자(dt=YYYY-MM-DD)의 피처를 읽어 예측/SHAP을 반환.
    반환: (원본 DF, 예측확률, shap_values, feature_names)
    - shap_values shape: (n_samples, n_features) 또는 빈 배열
    """
    booster, meta = load_model()
    feature_names = meta.get("feature_names")
    if not feature_names:
        raise RuntimeError("meta.json에 feature_names가 없습니다. 학습/저장 파이프라인을 확인하세요.")

    prefix = _features_prefix_for_date(dt)
    df = read_parquet_s3(prefix)

    if df.empty:
        # TODO(추후 수정): 운영 정책(빈 데이터일 때) 결정
        return df, np.array([]), np.empty((0, len(feature_names))), feature_names

    # user_id는 원본 df에 유지 (writer에서 사용)
    X = _ensure_numeric_matrix(df, feature_names)

    # 예측(배치)
    proba = _predict_in_batches(booster, X, _PREDICT_BATCH_SIZE)

    # SHAP (옵션)
    if _PREDICT_COMPUTE_SHAP and _HAS_SHAP:
        shap_values = _shap_in_batches(booster, X, _SHAP_BATCH_SIZE, _PREDICT_SHAP_APPROX)
    else:
        shap_values = np.empty((0, X.shape[1]), dtype=float)

    return df, np.asarray(proba), shap_values, feature_names
