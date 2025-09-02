# -*- coding: utf-8 -*-
"""
S3 입출력 유틸.
- list_objects_v2 페이지네이션 처리
- parquet/CSV 읽기
- SSE/KMS/RequesterPays 옵션 반영
"""
from __future__ import annotations
import io
from typing import Iterable, List, Optional, Dict

import pandas as pd
import pyarrow.parquet as pq

from .settings import CFG

# 공통 PutObject 옵션 구성 (SSE/KMS/RequesterPays 대응)
def _put_opts() -> Dict:
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

# 공통 GetObject/ListObjects 옵션 (RequesterPays 등)
def _req_opts() -> Dict:
    opts: Dict = {}
    if CFG.S3_REQUEST_PAYER:
        opts["RequestPayer"] = CFG.S3_REQUEST_PAYER
    return opts

def s3_join(prefix: str, *parts: str) -> str:
    key = (prefix or "").rstrip("/")
    for p in parts:
        if not p:
            continue
        key += "/" + p.strip("/")
    return key + ("/" if key and not key.endswith("/") else "")

def list_keys(prefix: str, suffix: Optional[str] = None) -> List[str]:
    """prefix 하위의 모든 객체 key를 반환(페이지네이션 처리). suffix가 있으면 해당 접미사만 필터."""
    s3 = CFG.s3_client()
    keys: List[str] = []
    token = None
    while True:
        params = {"Bucket": CFG.S3_BUCKET, "Prefix": prefix}
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

def read_parquet_s3(prefix: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """prefix 하위의 모든 parquet 파일을 읽어 단일 DF로 결합."""
    s3 = CFG.s3_client()
    keys = list_keys(prefix, suffix=".parquet")
    frames: List[pd.DataFrame] = []
    for k in keys:
        obj = s3.get_object(Bucket=CFG.S3_BUCKET, Key=k, **_req_opts())
        buf = io.BytesIO(obj["Body"].read())
        table = pq.read_table(buf, columns=columns)
        frames.append(table.to_pandas())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def read_csv_s3(prefix: str, *, encoding: str = "utf-8", **read_csv_kwargs) -> pd.DataFrame:
    """prefix 하위의 모든 .csv(.gz) 파일을 읽어 결합."""
    s3 = CFG.s3_client()
    keys = [*list_keys(prefix, suffix=".csv"), *list_keys(prefix, suffix=".csv.gz")]
    frames: List[pd.DataFrame] = []
    for k in keys:
        obj = s3.get_object(Bucket=CFG.S3_BUCKET, Key=k, **_req_opts())
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

def write_json_s3(key: str, obj: dict, *, bucket: str | None = None, extra: dict | None = None) -> None:
    data = json.dumps(obj).encode("utf-8")
    write_bytes_s3(key, data, bucket=bucket, extra=extra)

def write_bytes_s3(key: str, data: bytes, *, bucket: str | None = None, extra: dict | None = None) -> None:
    s3 = CFG.s3_client()
    s3.put_object(Bucket=(bucket or CFG.S3_BUCKET), Key=key, Body=data, **(extra or {}))

def write_parquet_s3(key: str, df: pd.DataFrame) -> None:
    """DataFrame을 Parquet로 직렬화해 S3에 업로드."""
    import pyarrow as pa
    table = pa.Table.from_pandas(df)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")  # TODO(추후 수정): 압축 코덱 정책
    buf.seek(0)
    write_bytes_s3(key, buf.read())
