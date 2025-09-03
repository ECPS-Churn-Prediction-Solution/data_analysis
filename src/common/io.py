# -*- coding: utf-8 -*-
"""
S3 입출력 유틸.
- list_objects_v2 페이지네이션 처리
- parquet/CSV 읽기/쓰기
- SSE/KMS/RequesterPays 옵션 반영
"""
from __future__ import annotations
import io
import json
import os
import re
from typing import Iterable, List, Optional, Dict

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

from .settings import CFG

# ----------------------------
# Path helpers
# ----------------------------
def s3_join(*parts: str) -> str:
    """
    Join S3-style path fragments without forcing a trailing slash.
    Example:
        s3_join("models/lgbm", "v1", "model.txt") -> "models/lgbm/v1/model.txt"
    """
    cleaned = [p.strip('/') for p in parts if p is not None and p != '']
    return '/'.join(cleaned)

def _put_opts() -> Dict:
    """공통 PutObject 옵션 구성 (SSE/KMS/RequesterPays 대응)."""
    opts: Dict = {}
    if CFG.S3_SSE:
        if CFG.S3_SSE.lower() == "aws:kms":
            opts["ServerSideEncryption"] = "aws:kms"
            if CFG.S3_KMS_KEY_ID:
                opts["SSEKMSKeyId"] = CFG.S3_KMS_KEY_ID
        else:
            opts["ServerSideEncryption"] = "AES256"
    if CFG.S3_REQUEST_PAYER:
        opts["RequestPayer"] = CFG.S3_REQUEST_PAYER
    return opts

def _req_opts() -> Dict:
    """공통 GetObject/ListObjects 옵션 (RequesterPays 등)."""
    opts: Dict = {}
    if CFG.S3_REQUEST_PAYER:
        opts["RequestPayer"] = CFG.S3_REQUEST_PAYER
    return opts

# ----------------------------
# S3 listing / reading
# ----------------------------
def list_keys(prefix: str, suffix: Optional[str] = None, *, bucket: Optional[str] = None) -> List[str]:
    """
    주어진 prefix 하위 객체 키 목록을 반환. (suffix가 있으면 필터링)
    """
    s3 = CFG.s3_client()
    bkt = bucket or CFG.S3_BUCKET
    keys: List[str] = []
    token: Optional[str] = None
    while True:
        params = {"Bucket": bkt, "Prefix": prefix}
        params.update(_req_opts())
        if token:
            params["ContinuationToken"] = token
        resp = s3.list_objects_v2(**params)
        for obj in resp.get("Contents", []):
            k = obj["Key"]
            if suffix and not k.endswith(suffix):
                continue
            keys.append(k)
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return keys

def read_parquet_s3(prefix_or_key: str, columns: Optional[List[str]] = None, *, bucket: Optional[str]=None) -> pd.DataFrame:
    """
    prefix_or_key가 디렉터리 프리픽스면 하위의 모든 *.parquet 읽어서 concat,
    단일 파일 키면 해당 파일만 읽음.
    """
    bkt = bucket or CFG.S3_BUCKET
    s3 = CFG.s3_client()
    # 디렉터리인지 파일인지 단순 판별
    if re.search(r'\.parquet$', prefix_or_key, re.IGNORECASE):
        keys = [prefix_or_key]
    else:
        keys = list_keys(prefix_or_key, suffix=".parquet", bucket=bkt)

    frames: List[pd.DataFrame] = []
    for k in keys:
        obj = s3.get_object(Bucket=bkt, Key=k, **_req_opts())
        buf = io.BytesIO(obj["Body"].read())
        table = pq.read_table(buf, columns=columns)
        frames.append(table.to_pandas())

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def read_csv_s3(prefix_or_key: str, encoding: str = "utf-8", **read_csv_kwargs) -> pd.DataFrame:
    """
    CSV 읽기 (단일 파일 혹은 프리픽스 하위 *.csv 모두).
    """
    bkt = CFG.S3_BUCKET
    s3 = CFG.s3_client()
    if re.search(r'\.csv(\.gz)?$', prefix_or_key, re.IGNORECASE):
        keys = [prefix_or_key]
    else:
        keys = list_keys(prefix_or_key, suffix=".csv", bucket=bkt)
    frames: List[pd.DataFrame] = []
    for k in keys:
        obj = s3.get_object(Bucket=bkt, Key=k, **_req_opts())
        body = obj["Body"].read()
        if k.endswith(".gz"):
            import gzip
            buf = io.BytesIO(body)
            with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
                frames.append(pd.read_csv(gz, encoding=encoding, **read_csv_kwargs))
        else:
            frames.append(pd.read_csv(io.BytesIO(body), encoding=encoding, **read_csv_kwargs))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

# ----------------------------
# Writers
# ----------------------------
def write_bytes_s3(key: str, data: bytes, *, bucket: Optional[str] = None, extra: Optional[dict] = None) -> None:
    s3 = CFG.s3_client()
    s3.put_object(Bucket=(bucket or CFG.S3_BUCKET), Key=key, Body=data, **(extra or {}))

def write_json_s3(key: str, obj: dict, *, bucket: Optional[str] = None, extra: Optional[dict] = None) -> None:
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    write_bytes_s3(key, data, bucket=bucket, extra=extra)

def write_parquet_s3(key: str, df: pd.DataFrame, *, bucket: Optional[str] = None) -> None:
    """DataFrame을 Parquet로 직렬화해 S3에 업로드."""
    table = pa.Table.from_pandas(df)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    buf.seek(0)
    write_bytes_s3(key, buf.read(), bucket=bucket, extra=_put_opts())

# ----------------------------
# Local/S3 transparent reader
# ----------------------------
def read_features(path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    통합 Parquet 로더.
    - s3://bucket/key.parquet  또는 s3://bucket/prefix/ (환경 변수가 지정된 버킷과 다르면 path의 버킷 사용)
    - 로컬 경로도 지원.
    """
    if path.startswith("s3://"):
        # s3://bucket/...
        # 버킷이 CFG.S3_BUCKET과 달라도 정상 읽기
        _, rest = path[5:].split("/", 1)
        bucket, key = rest.split("/", 1)
        return read_parquet_s3(key, columns=columns, bucket=bucket)
    else:
        return pd.read_parquet(path, columns=columns)
