# -*- coding: utf-8 -*-
"""
Parquet 피처셋 표준화:
- 열 이름 alias 정리
- dtype 강제(카테고리/수치/ID/라벨)
- 결측치 처리
- 학습/예측 공용 prepare_features()
"""

from typing import Tuple, Optional
import pandas as pd

# ====== 컬럼 메타 ======
ID_COL = "user_id"
LABEL_COL = "churn"
CAT_COLS = ["gender"]            # LightGBM categorical_feature
EXCLUDE_COLS = [ID_COL, LABEL_COL]

# 이미지가 잘려 보였던 케이스 등을 위한 alias 보정
ALIAS = {
    "days_since_last_s": "days_since_last_order",
    "freq_last_30d": "frequency_last_30d",
    "freq_last_90d": "frequency_last_90d",
    # 필요 시 추가
}

# 0~1 범위가 자연스러운 비율 컬럼 후보
RATIO_COLS = [
    "coupon_usage_rate",
]

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    for src, dst in ALIAS.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})
    return df

def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    # ID
    if ID_COL in df.columns:
        df[ID_COL] = pd.to_numeric(df[ID_COL], errors="coerce").astype("Int64")

    # 라벨 (있을 때만)
    if LABEL_COL in df.columns:
        df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(0).astype(int)

    # 카테고리
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("unknown").astype("category")

    # 수치형 (ID/라벨/카테고리 제외)
    num_cols = [c for c in df.columns if c not in (CAT_COLS + [ID_COL, LABEL_COL])]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if num_cols:
        df[num_cols] = df[num_cols].fillna(0)

    # 비율 컬럼 clip
    for c in RATIO_COLS:
        if c in df.columns:
            df[c] = df[c].clip(lower=0, upper=1)

    return df

def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    y = df[LABEL_COL] if LABEL_COL in df.columns else None
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].copy()
    return X, y

def prepare_features(df: pd.DataFrame):
    df = standardize_columns(df)
    df = coerce_dtypes(df)
    X, y = split_xy(df)
    return X, y
