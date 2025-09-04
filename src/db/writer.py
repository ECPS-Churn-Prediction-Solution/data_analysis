# -*- coding: utf-8 -*-
"""
예측 결과를 Postgres에 적재 (psycopg2/psycopg 공용).
- 대상: {CFG.PG_TARGET_SCHEMA}.{CFG.PG_TARGET_TABLE}  (기본값: analytics.churn_scores)
- 중복 방지: (user_id, model_version, source_dt) UNIQUE + UPSERT
"""
from __future__ import annotations
import pandas as pd
from ..common.settings import CFG

def _q_ident(s: str) -> str:
    return '"' + s.replace('"', '""') + '"'

def _fqtn() -> str:
    return f'{_q_ident(CFG.PG_TARGET_SCHEMA)}.{_q_ident(CFG.PG_TARGET_TABLE)}'

def _uniq_name() -> str:
    return f'uq_{CFG.PG_TARGET_SCHEMA}_{CFG.PG_TARGET_TABLE}'.lower()

def ensure_schema_and_table(conn) -> None:
    tbl = _fqtn()
    uq  = _uniq_name()

    TABLE_SQL = f"""
    CREATE SCHEMA IF NOT EXISTS {_q_ident(CFG.PG_TARGET_SCHEMA)};
    CREATE TABLE IF NOT EXISTS {tbl} (
        user_id         BIGINT,
        churn_score     DOUBLE PRECISION NOT NULL,
        model_version   TEXT             NOT NULL,
        feature_version TEXT             NOT NULL,
        scored_at       TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
        source_dt       DATE             NULL
    );
    """
    INDEX_SQL = f"""
    CREATE INDEX IF NOT EXISTS idx_{CFG.PG_TARGET_SCHEMA}_{CFG.PG_TARGET_TABLE}_dt_user
        ON {tbl} (source_dt, user_id);
    """
    UNIQUE_SQL = f"""
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint WHERE conname = '{uq}'
        ) THEN
            ALTER TABLE {tbl}
            ADD CONSTRAINT {uq} UNIQUE (user_id, model_version, source_dt);
        END IF;
    END $$;
    """

    with conn.cursor() as cur:
        cur.execute(TABLE_SQL)
        cur.execute(INDEX_SQL)
        cur.execute(UNIQUE_SQL)
    conn.commit()

def insert_scores(conn, df: pd.DataFrame) -> int:
    cols = ["user_id", "churn_score", "model_version", "feature_version", "scored_at", "source_dt"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"insert_scores: missing columns: {missing}")

    def to_py_int(x):
        try:
            import pandas as pd  # local import to avoid global dependency
            return None if pd.isna(x) else int(x)
        except Exception:
            return None

    records = [
        (
            to_py_int(r["user_id"]),
            float(r["churn_score"]),
            str(r["model_version"]),
            str(r["feature_version"]),
            r["scored_at"],
            r.get("source_dt", None),
        )
        for _, r in df[cols].iterrows()
    ]

    sql = f"""
    INSERT INTO {_fqtn()}
        (user_id, churn_score, model_version, feature_version, scored_at, source_dt)
    VALUES
        (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (user_id, model_version, source_dt)
    DO UPDATE SET
        churn_score     = EXCLUDED.churn_score,
        feature_version = EXCLUDED.feature_version,
        scored_at       = EXCLUDED.scored_at
    """
    with conn.cursor() as cur:
        cur.executemany(sql, records)
    conn.commit()
    return len(records)
