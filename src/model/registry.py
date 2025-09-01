# -*- coding: utf-8 -*-
"""
모델 아티팩트 저장/로드 (LightGBM Booster 기준).
- s3://kdt3-preprocessing-data/<S3_MODEL_PREFIX>/<MODEL_VERSION>/model.txt, meta.json
- feature_names 포함한 메타를 별도로 저장
- Requester Pays/SSE 등은 common.io의 put옵션을 사용(저장), load는 여기서 직접 처리
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional

import lightgbm as lgb
from botocore.exceptions import ClientError

from ..common.settings import CFG
from ..common.io import write_bytes_s3, write_json_s3, s3_join

def _model_base_prefix() -> str:
    """
    모델 저장 베이스 prefix 생성.
    예) models/lgbm/lgbm_v1.0_shaTODO/
    # TODO(추후 수정): 모델 버전/구조 확정되면 경로 정책 업데이트
    """
    return s3_join(CFG.S3_MODEL_PREFIX, CFG.MODEL_VERSION)

def save_model(
    model: lgb.Booster,
    *,
    feature_names: Optional[list] = None,
    extra_meta: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    LightGBM Booster를 S3에 저장.
    - model.txt : Booster 텍스트 포맷
    - meta.json : feature_names, created_at, version 등
    """
    base = _model_base_prefix()           # 예: models/lgbm/lgbm_v1.0_shaTODO/
    key_model = base + "model.txt"
    key_meta  = base + "meta.json"

    # model.txt 직렬화 (파일 없이 문자열로)
    model_text: str = model.model_to_string(num_iteration=model.best_iteration or -1)
    write_bytes_s3(key_model, model_text.encode("utf-8"))

    # 메타 확충: best_iteration, objective, num_features
    params = getattr(model, "params", {}) or {}
    meta = {
        "feature_names": feature_names or model.feature_name(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_version": CFG.MODEL_VERSION,
        "feature_version": CFG.FEATURE_VERSION,
        "aws_region": CFG.AWS_REGION,
        "bucket": CFG.S3_BUCKET,
        "best_iteration": int(model.best_iteration or -1),
        "num_features": int(len(feature_names or model.feature_name() or [])),
        "objective": params.get("objective"),
        **(extra_meta or {}),
    }
    write_json_s3(key_meta, meta)

    return {
        "model": f"s3://{CFG.S3_BUCKET}/{key_model}",
        "meta":  f"s3://{CFG.S3_BUCKET}/{key_meta}",
    }

def load_model() -> Tuple[lgb.Booster, Dict[str, Any]]:
    """
    S3에서 Booster/메타를 로드.
    - Requester Pays 환경에서도 동작하도록 옵션 처리
    - 키가 없으면 친절한 에러 메시지 제공
    """
    s3 = CFG.s3_client()
    base = _model_base_prefix()
    key_model = base + "model.txt"
    key_meta  = base + "meta.json"

    req_opts = {}
    if CFG.S3_REQUEST_PAYER:
        req_opts["RequestPayer"] = CFG.S3_REQUEST_PAYER

    # model.txt 다운로드
    try:
        resp = s3.get_object(Bucket=CFG.S3_BUCKET, Key=key_model, **req_opts)
    except ClientError as e:
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다: s3://{CFG.S3_BUCKET}/{key_model} "
            f"(MODEL_VERSION={CFG.MODEL_VERSION})"
        ) from e
    model_text = resp["Body"].read().decode("utf-8")
    booster = lgb.Booster(model_str=model_text)

    # meta.json 다운로드
    try:
        meta_resp = s3.get_object(Bucket=CFG.S3_BUCKET, Key=key_meta, **req_opts)
    except ClientError as e:
        raise FileNotFoundError(
            f"메타 파일을 찾을 수 없습니다: s3://{CFG.S3_BUCKET}/{key_meta} "
            f"(MODEL_VERSION={CFG.MODEL_VERSION})"
        ) from e
    meta = json.loads(meta_resp["Body"].read().decode("utf-8"))

    return booster, meta
