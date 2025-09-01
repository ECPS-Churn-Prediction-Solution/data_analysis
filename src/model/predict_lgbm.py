# -*- coding: utf-8 -*-
"""
S3에서 피처 로드 → 예측/SHAP 계산
- 버킷: kdt3-preprocessing-data (고정)
- 프리픽스/파티션: TODO(추후 수정)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import shap

from ..common.settings import CFG
from ..common.io import read_parquet_s3, s3_join
from .registry import load_model

def _features_prefix_for_date(dt_str: str) -> str:
    """
    날짜 문자열(YYYY-MM-DD)로 파티션 경로 생성.
    기본: <S3_FEATURE_PREFIX>/dt=YYYY-MM-DD/
    """
    # TODO(추후 수정): 파티션 규약 변경 시 여기만 수정
    return s3_join(CFG.S3_FEATURE_PREFIX, f"dt={dt_str}")

def predict_for_date(dt: str):
    """
    지정 일자(dt=YYYY-MM-DD)의 피처를 읽어 예측/SHAP을 반환.
    """
    booster, meta = load_model()
    feature_names = meta.get("feature_names")

    prefix = _features_prefix_for_date(dt)
    df = read_parquet_s3(prefix)

    if df.empty:
        # TODO(추후 수정): 운영 정책(빈 데이터일 때) 결정
        return df, np.array([]), np.empty((0, len(feature_names) if feature_names else 0)), feature_names

    # 사용자 파이프라인에 맞춰 전처리/정렬
    if feature_names:
        X = df.reindex(columns=feature_names)
    else:
        # 모델 메타에 feature_names가 없을 경우, 수치 컬럼만 사용(임시)
        X = df.select_dtypes(include=["number"])

    X = X.fillna(0.0)
    proba = booster.predict(X, num_iteration=booster.best_iteration or -1)

    # SHAP
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):   # 이진분류일 때 클래스축 포함 가능
        shap_values = shap_values[1]

    return df, np.asarray(proba), shap_values, feature_names or list(X.columns)
