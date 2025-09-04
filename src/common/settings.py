# -*- coding: utf-8 -*-
"""
공통 설정 로더 (.env / 환경변수)
- S3 버킷/프리픽스: (Features) ecps-event-log / features/ (dt=%Y-%m-%d)
                   (Models)   ecps-models     / lgbm/
                   (Predicts) ecps-prediction / predictions/
- DB: PG_HOST/PG_PORT/PG_DB/PG_USER/PG_PASSWORD/PG_SSLMODE
- SHAP/배치 토글 포함
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional
from datetime import timezone
from contextlib import contextmanager

import boto3
from botocore.config import Config as BotoConfig
from dotenv import load_dotenv
import psycopg2

try:
    import psycopg2  # type: ignore
except Exception:
    psycopg2 = None  # type: ignore

try:
    import psycopg  # psycopg3
except Exception:
    psycopg = None  # type: ignore

load_dotenv(override=True)


@dataclass(frozen=True)
class _Config:
    # ===== AWS / S3 =====
    AWS_REGION: str = os.getenv("AWS_REGION", "ap-northeast-2")
    AWS_PROFILE: Optional[str] = os.getenv("AWS_PROFILE")  # 'default' 등
    S3_ENDPOINT_URL: Optional[str] = os.getenv("S3_ENDPOINT_URL")  # MinIO 등 사용 시

    # ===== Features (Glue 출력과 일치) =====
    # 예: s3://{S3_BUCKET}/{S3_FEATURE_PREFIX}/dt=YYYY-MM-DD/*.parquet
    S3_BUCKET: str = os.getenv("S3_BUCKET", "ecps-event-log")
    S3_FEATURE_PREFIX: str = os.getenv("S3_FEATURE_PREFIX", "features/")
    S3_FEATURE_PARTITION_FMT: str = os.getenv("S3_FEATURE_PARTITION_FMT", "dt=%Y-%m-%d")

    # ===== Models =====
    # 최종 경로: s3://{S3_MODEL_BUCKET}/{S3_MODEL_PREFIX}{MODEL_VERSION}/{model.txt, meta.json}
    S3_MODEL_BUCKET: str = os.getenv("S3_MODEL_BUCKET", os.getenv("S3_BUCKET", "ecps-event-log"))
    S3_MODEL_PREFIX: str = os.getenv("S3_MODEL_PREFIX", "lgbm/")
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "lgbm_v1.0_shaTODO")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "lgbm")

    FEATURE_TRAIN_URI: str = os.getenv("FEATURE_TRAIN_URI", "")
    FEATURE_VALID_URI: str = os.getenv("FEATURE_VALID_URI", "")
    FEATURE_PREDICT_URI: str = os.getenv("FEATURE_PREDICT_URI", "")

    # ===== DB Target (predictions sink) =====
    PG_TARGET_SCHEMA: str = os.getenv("PG_TARGET_SCHEMA", "analytics")
    PG_TARGET_TABLE: str = os.getenv("PG_TARGET_TABLE", "churn_scores")

    # ===== Features/Churn meta =====
    FEATURE_VERSION: str = os.getenv("FEATURE_VERSION", "feat_v1.0")
    # 과거 명칭 호환: CHURN_HORIZON_DAYS 우선, 없으면 HORIZON_DAYS
    CHURN_HORIZON_DAYS: int = int(os.getenv("CHURN_HORIZON_DAYS", os.getenv("HORIZON_DAYS", "30")))

    # ===== Predictions export =====
    S3_PREDICTION_BUCKET: str = os.getenv("S3_PREDICTION_BUCKET", os.getenv("S3_BUCKET", "ecps-event-log"))
    S3_PREDICTION_PREFIX: str = os.getenv("S3_PREDICTION_PREFIX", "predictions/")
    WRITE_PREDICTIONS_TO_S3: bool = os.getenv("WRITE_PREDICTIONS_TO_S3", "1") not in {"0", "false", "False"}

    # ===== Server-Side Encryption / Request Payer =====
    S3_SSE: Optional[str] = os.getenv("S3_SSE")           # '', 'AES256', 'aws:kms'
    S3_KMS_KEY_ID: Optional[str] = os.getenv("S3_KMS_KEY_ID")
    S3_REQUEST_PAYER: Optional[str] = os.getenv("S3_REQUEST_PAYER")  # 'requester'

    # ===== SHAP/Batch (토글) =====
    PREDICT_COMPUTE_SHAP: bool = os.getenv("PREDICT_COMPUTE_SHAP", "0") not in {"0", "false", "False"}
    PREDICT_SHAP_APPROX: bool = os.getenv("PREDICT_SHAP_APPROX", "1") not in {"0", "false", "False"}
    PREDICT_BATCH_SIZE: int = int(os.getenv("PREDICT_BATCH_SIZE", "50000"))
    SHAP_BATCH_SIZE: int = int(os.getenv("SHAP_BATCH_SIZE", "20000"))

    # ===== DB (적재용) =====
    # 새 규약(PG_HOST 등)과 과거 규약(PGHOST 등) 양쪽 호환
    PGHOST: str = os.getenv("PG_HOST", os.getenv("PGHOST", "localhost"))
    PGPORT: int = int(os.getenv("PG_PORT", os.getenv("PGPORT", "5432")))
    PGDATABASE: str = os.getenv("PG_DB", os.getenv("PGDATABASE", "postgres"))
    PGUSER: str = os.getenv("PG_USER", os.getenv("PGUSER", "postgres"))
    PGPASSWORD: str = os.getenv("PG_PASSWORD", os.getenv("PGPASSWORD", ""))
    PGSSLMODE: str = os.getenv("PG_SSLMODE", "prefer")  # require / verify-ca / verify-full 등

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
        """
        우선 psycopg2로 연결 시도 → 실패(특히 UnicodeDecodeError) 시 psycopg3로 폴백.
        두 드라이버 모두에서 동일하게 cursor.execute / executemany 사용 가능하도록 반환.
        """
        last_err = None

        # 1) psycopg2 우선 시도 (옵션/인코딩 강제)
        if psycopg2 is not None:
            try:
                conn = psycopg2.connect(
                    host=self.PGHOST,
                    port=self.PGPORT,
                    dbname=self.PGDATABASE,
                    user=self.PGUSER,
                    password=self.PGPASSWORD,
                    sslmode=self.PGSSLMODE,
                    options="-c lc_messages=C -c client_encoding=UTF8",
                )
                try:
                    yield conn
                finally:
                    conn.close()
                return
            except UnicodeDecodeError as e:
                last_err = e  # 폴백 시도
            except Exception as e:
                # 다른 이유로 실패해도 폴백 시도
                last_err = e

        # 2) psycopg3 (psycopg) 폴백
        if psycopg is None:
            # psycopg3 미설치인 경우: 친절한 에러 메시지
            raise RuntimeError(
                f"DB 연결 실패(psycopg2): {last_err}. "
                "폴백 드라이버(psycopg)가 설치되어 있지 않습니다. "
                "다음 명령으로 설치 후 재시도하세요: pip install \"psycopg[binary]\""
            )

        # psycopg3 연결 (영문 메시지)
        conn = psycopg.connect(
            host=self.PGHOST,
            port=self.PGPORT,
            dbname=self.PGDATABASE,
            user=self.PGUSER,
            password=self.PGPASSWORD,
            sslmode=self.PGSSLMODE,  # 'require' 그대로 사용 가능
            options="-c lc_messages=C",
            autocommit=False,
        )
        # 클라이언트 인코딩 UTF8 보장
        try:
            with conn.cursor() as cur:
                cur.execute("SET CLIENT_ENCODING TO 'UTF8'")
        except Exception:
            pass

        try:
            yield conn
        finally:
            conn.close()


CFG = _Config()
