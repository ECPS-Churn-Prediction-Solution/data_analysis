# -*- coding: utf-8 -*-
"""
공통 설정 로더.
- 버킷명은 사용자 요청에 따라 'kdt3-preprocessing-data'로 기본값 고정
- 피처/모델/예측 저장 위치와 배치/SHAP 토글 등을 중앙집중화
"""
import os
from dataclasses import dataclass
from typing import Optional
from datetime import timezone
from contextlib import contextmanager

import boto3
from botocore.config import Config as BotoConfig
from dotenv import load_dotenv
import psycopg2

load_dotenv(override=True)


@dataclass(frozen=True)
class _Config:
    # ===== AWS / S3 =====
    AWS_REGION: str = os.getenv("AWS_REGION", "ap-northeast-2")
    AWS_PROFILE: Optional[str] = os.getenv("AWS_PROFILE")  # 로컬 프로필 사용 시 지정
    S3_ENDPOINT_URL: Optional[str] = os.getenv("S3_ENDPOINT_URL")  # MinIO/프록시 시 사용(없으면 None)

    # [기본값 고정] 사용자 제공 버킷명 (환경변수로 덮어쓸 수는 있음)
    S3_BUCKET: str = os.getenv("S3_BUCKET", "kdt3-preprocessing-data")

    # ===== Feature Dataset (Parquet) =====
    # 예: s3://{S3_BUCKET}/{S3_FEATURE_PREFIX}/dt=YYYY-MM-DD/part-*.parquet
    S3_FEATURE_PREFIX: str = os.getenv("S3_FEATURE_PREFIX", "features/")
    # 파티션 디렉터리 포맷 (strftime 사용): dt=%Y-%m-%d, dt=%Y%m%d 등
    S3_FEATURE_PARTITION_FMT: str = os.getenv("S3_FEATURE_PARTITION_FMT", "dt=%Y-%m-%d")

    # (선택) 직접 파일 경로로 학습/검증/예측을 지정하고 싶을 때 사용
    FEATURE_TRAIN_URI: str = os.getenv("FEATURE_TRAIN_URI", "")
    FEATURE_VALID_URI: str = os.getenv("FEATURE_VALID_URI", "")
    FEATURE_PREDICT_URI: str = os.getenv("FEATURE_PREDICT_URI", "")

    # ===== Model Artifacts =====
    # 모델 키 프리픽스: 예) models/lgbm/
    S3_MODEL_PREFIX: str = os.getenv("S3_MODEL_PREFIX", "models/lgbm/")
    # 모델 전용 버킷 (미지정 시 S3_BUCKET 사용)
    S3_MODEL_BUCKET: str = os.getenv("S3_MODEL_BUCKET", os.getenv("S3_BUCKET", "kdt3-preprocessing-data"))
    # 모델/피처 버전
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "lgbm_v1.0_shaTODO")
    FEATURE_VERSION: str = os.getenv("FEATURE_VERSION", "feat_v1.0")
    HORIZON_DAYS: int = int(os.getenv("HORIZON_DAYS", "30"))

    # ===== Prediction Exports =====
    S3_PREDICTION_PREFIX: str = os.getenv("S3_PREDICTION_PREFIX", "predictions/")
    S3_PREDICTION_BUCKET: str = os.getenv("S3_PREDICTION_BUCKET", os.getenv("S3_BUCKET", "kdt3-preprocessing-data"))
    # 예측 결과를 S3에 저장할지 여부(0/1)
    WRITE_PREDICTIONS_TO_S3: bool = os.getenv("WRITE_PREDICTIONS_TO_S3", "1") not in {"0", "false", "False"}

    # ===== Server-Side Encryption / Request Payer =====
    S3_SSE: Optional[str] = os.getenv("S3_SSE")           # '', 'AES256', 'aws:kms'
    S3_KMS_KEY_ID: Optional[str] = os.getenv("S3_KMS_KEY_ID")
    S3_REQUEST_PAYER: Optional[str] = os.getenv("S3_REQUEST_PAYER")  # 'requester'

    # ===== Batch & SHAP toggles =====
    # (중앙집중 토글: 개별 모듈에서 os.getenv 쓰지 말고 CFG를 사용)
    PREDICT_COMPUTE_SHAP: bool = os.getenv("PREDICT_COMPUTE_SHAP", "1") not in {"0", "false", "False"}
    PREDICT_SHAP_APPROX: bool = os.getenv("PREDICT_SHAP_APPROX", "1") not in {"0", "false", "False"}
    PREDICT_BATCH_SIZE: int = int(os.getenv("PREDICT_BATCH_SIZE", "50000"))
    SHAP_BATCH_SIZE: int = int(os.getenv("SHAP_BATCH_SIZE", "20000"))

    # ===== DB (적재용) =====
    PGHOST: str = os.getenv("PGHOST", "localhost")
    PGPORT: int = int(os.getenv("PGPORT", "5432"))
    PGDATABASE: str = os.getenv("PGDATABASE", "postgres")
    PGUSER: str = os.getenv("PGUSER", "postgres")
    PGPASSWORD: str = os.getenv("PGPASSWORD", "")

    @property
    def tz_utc(self):
        return timezone.utc

    def boto3_session(self):
        if self.AWS_PROFILE:
            return boto3.session.Session(profile_name=self.AWS_PROFILE, region_name=self.AWS_REGION)
        return boto3.session.Session(region_name=self.AWS_REGION)

    def s3_client(self):
        return self.boto3_session().client(
            "s3",
            endpoint_url=self.S3_ENDPOINT_URL,
            config=BotoConfig(
                retries={"max_attempts": 10, "mode": "standard"},
                connect_timeout=10,
                read_timeout=120,
            ),
        )

    @contextmanager
    def connect_db(self):
        conn = psycopg2.connect(
            host=self.PGHOST, port=self.PGPORT, dbname=self.PGDATABASE, user=self.PGUSER, password=self.PGPASSWORD
        )
        try:
            yield conn
        finally:
            conn.close()


CFG = _Config()
