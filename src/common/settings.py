# -*- coding: utf-8 -*-
"""
공통 설정 로더.
- 버킷명은 사용자 요청에 따라 'kdt3-preprocessing-data'로 고정
- 나머지 프리픽스/포맷 등은 TODO(추후 수정)로 표시
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

    # [고정] 사용자 제공 버킷명
    S3_BUCKET: str = os.getenv("S3_BUCKET", "kdt3-preprocessing-data")

    # TODO(추후 수정): 기능 확정 시 프리픽스/파티션 포맷 조정
    S3_FEATURE_PREFIX: str = os.getenv("S3_FEATURE_PREFIX", "features/")                 # 예: features/
    S3_FEATURE_PARTITION_FMT: str = os.getenv("S3_FEATURE_PARTITION_FMT", "dt=%Y-%m-%d") # 예: dt=YYYY-MM-DD
    S3_MODEL_PREFIX = os.getenv("S3_MODEL_PREFIX", "models/lgbm/")
    # 모델 전용 버킷: 지정 없으면 S3_BUCKET을 재사용
    S3_MODEL_BUCKET = os.getenv("S3_MODEL_BUCKET", S3_BUCKET)

    S3_PREDICTION_PREFIX: str = os.getenv("S3_PREDICTION_PREFIX", "predictions/")
    # 예측 결과 전용 버킷: 지정 없으면 S3_BUCKET 재사용
    S3_PREDICTION_BUCKET: str = os.getenv("S3_PREDICTION_BUCKET", os.getenv("S3_BUCKET"))

    # 선택: 서버사이드 암호화
    S3_SSE: Optional[str] = os.getenv("S3_SSE")              # '', 'AES256', 'aws:kms'
    S3_KMS_KEY_ID: Optional[str] = os.getenv("S3_KMS_KEY_ID")  # aws:kms일 때만
    S3_REQUEST_PAYER: Optional[str] = os.getenv("S3_REQUEST_PAYER")  # 'requester' 사용 시

    # ===== 모델/피처 버전 =====
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "lgbm_v1.0_shaTODO")  # TODO(추후 수정)
    FEATURE_VERSION: str = os.getenv("FEATURE_VERSION", "feat_v1.0")      # TODO(추후 수정)
    HORIZON_DAYS: int = int(os.getenv("HORIZON_DAYS", "30"))

    # ===== DB (적재용) =====
    PGHOST: str = os.getenv("PGHOST", "localhost")          # TODO(추후 수정)
    PGPORT: int = int(os.getenv("PGPORT", "5432"))
    PGDATABASE: str = os.getenv("PGDATABASE", "postgres")   # TODO(추후 수정)
    PGUSER: str = os.getenv("PGUSER", "postgres")           # TODO(추후 수정)
    PGPASSWORD: str = os.getenv("PGPASSWORD", "")           # TODO(추후 수정)

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
            endpoint_url=self.S3_ENDPOINT_URL,  # None이면 AWS 기본 엔드포인트
            config=BotoConfig(
                retries={"max_attempts": 10, "mode": "standard"},
                connect_timeout=10,
                read_timeout=120,
                # TODO(추후 수정): 네트워크 정책/프록시 요구 시 조정
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
