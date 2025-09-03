# -*- coding: utf-8 -*-
"""
모델 아티팩트 저장/로드 (LightGBM Booster 기준).
- s3://{S3_MODEL_BUCKET}/{S3_MODEL_PREFIX}/{MODEL_VERSION}/model.txt, meta.json
- meta.json에 feature_names, categorical_features 등 저장
- Requester Pays/SSE/KMS 옵션 처리
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional

import lightgbm as lgb
from botocore.exceptions import ClientError

from ..common.settings import CFG
from ..common.io import s3_join


def _model_base_prefix() -> str:
    """
    모델 저장 베이스 prefix 생성.
    예) models/lgbm/lgbm_v1.0_shaTODO/
    """
    return s3_join(CFG.S3_MODEL_PREFIX, CFG.MODEL_VERSION)


def save_model(
    model: lgb.Booster,
    *,
    feature_names: Optional[list] = None,
    categorical_features: Optional[list] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    LightGBM Booster를 S3(모델 버킷)에 저장.
    - model.txt : Booster 텍스트 포맷
    - meta.json : feature_names, categorical_features, created_at, version 등
    저장 위치:
      s3://{CFG.S3_MODEL_BUCKET}/{CFG.S3_MODEL_PREFIX}/{MODEL_VERSION}/(model.txt, meta.json)
    """
    s3 = CFG.s3_client()
    base = _model_base_prefix()           # 예: models/lgbm/lgbm_v1.0_shaTODO/
    key_model = base + "model.txt"
    key_meta  = base + "meta.json"

    # 공통 put 옵션 (SSE/KMS)
    put_opts: Dict[str, Any] = {}
    if getattr(CFG, "S3_SSE", None):
        put_opts["ServerSideEncryption"] = CFG.S3_SSE
        if getattr(CFG, "S3_KMS_KEY_ID", None):
            put_opts["SSEKMSKeyId"] = CFG.S3_KMS_KEY_ID

    # model.txt 직렬화 (문자열로)
    model_text: str = model.model_to_string(num_iteration=getattr(model, "best_iteration", None) or -1)
    s3.put_object(
        Bucket=CFG.S3_MODEL_BUCKET,
        Key=key_model,
        Body=model_text.encode("utf-8"),
        **put_opts,
    )

    # 메타 확충
    params = getattr(model, "params", {}) or {}
    meta = {
        "feature_names": feature_names or model.feature_name() or [],
        "categorical_features": categorical_features or [],  # 학습 파이프라인에서 주입
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_version": CFG.MODEL_VERSION,
        "feature_version": CFG.FEATURE_VERSION,
        "aws_region": CFG.AWS_REGION,
        "bucket": CFG.S3_MODEL_BUCKET,
        "best_iteration": int(getattr(model, "best_iteration", None) or -1),
        "num_features": int(len(feature_names or model.feature_name() or [])),
        "objective": params.get("objective"),
        **(extra_meta or {}),
    }
    s3.put_object(
        Bucket=CFG.S3_MODEL_BUCKET,
        Key=key_meta,
        Body=json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"),
        **put_opts,
    )

    return {
        "model": f"s3://{CFG.S3_MODEL_BUCKET}/{key_model}",
        "meta":  f"s3://{CFG.S3_MODEL_BUCKET}/{key_meta}",
    }


def load_model() -> Tuple[lgb.Booster, Dict[str, Any]]:
    """
    S3(모델 버킷)에서 Booster/메타를 로드.
    - Requester Pays 환경에서도 동작하도록 옵션 처리
    - 키가 없으면 친절한 에러 메시지 제공
    """
    s3 = CFG.s3_client()
    base = _model_base_prefix()
    key_model = base + "model.txt"
    key_meta  = base + "meta.json"

    req_opts: Dict[str, Any] = {}
    if getattr(CFG, "S3_REQUEST_PAYER", None):
        req_opts["RequestPayer"] = CFG.S3_REQUEST_PAYER

    # model.txt 다운로드
    try:
        resp = s3.get_object(Bucket=CFG.S3_MODEL_BUCKET, Key=key_model, **req_opts)
    except ClientError as e:
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다: s3://{CFG.S3_MODEL_BUCKET}/{key_model} "
            f"(MODEL_VERSION={CFG.MODEL_VERSION})"
        ) from e
    model_text = resp["Body"].read().decode("utf-8")
    booster = lgb.Booster(model_str=model_text)

    # meta.json 다운로드
    try:
        meta_resp = s3.get_object(Bucket=CFG.S3_MODEL_BUCKET, Key=key_meta, **req_opts)
    except ClientError as e:
        raise FileNotFoundError(
            f"메타 파일을 찾을 수 없습니다: s3://{CFG.S3_MODEL_BUCKET}/{key_meta} "
            f"(MODEL_VERSION={CFG.MODEL_VERSION})"
        ) from e
    meta = json.loads(meta_resp["Body"].read().decode("utf-8"))

    return booster, meta
