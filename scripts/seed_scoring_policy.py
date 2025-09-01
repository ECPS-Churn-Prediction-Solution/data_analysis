# -*- coding: utf-8 -*-
"""
conf/scoring_policy.seed.yaml을 읽어 analytics.scoring_policy에 upsert.
- 같은 (model_name, model_version, feature_version, churn_horizon_days)에 대해
  현재행(effective_until='infinity')이 있으면 종료시킨 뒤 새로 시드 삽입.
"""
import os, sys, yaml, datetime
import psycopg2
from psycopg2.extras import execute_values

# 로컬 실행 전제: PYTHONPATH에 프로젝트 루트 추가
sys.path.append(os.getcwd())
from src.common.settings import CFG

def main(path: str = "conf/scoring_policy.seed.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    mn = y["model_name"]
    mv = y["model_version"]
    fv = y["feature_version"]
    eff_from = y.get("effective_from", "now")
    if eff_from == "now":
        eff_from = None  # DB NOW() 사용

    rows = []
    for p in y["policies"]:
        rows.append((
            mn, mv, fv,
            int(p["churn_horizon_days"]),
            float(p["threshold_default"]),
            float(p["cutpoint_vh"]),
            float(p["cutpoint_h"]),
            float(p["cutpoint_m"]),
        ))

    sql_close = """
        WITH t AS (
          SELECT DISTINCT model_name, model_version, feature_version, churn_horizon_days
          FROM (VALUES %s) AS v(model_name, model_version, feature_version, churn_horizon_days,
                                threshold_default, cutpoint_vh, cutpoint_h, cutpoint_m)
        )
        UPDATE analytics.scoring_policy sp
           SET effective_until = NOW()
         WHERE sp.effective_until = 'infinity'::timestamptz
           AND (sp.model_name, sp.model_version, sp.feature_version, sp.horizon_days)
               IN (SELECT model_name, model_version, feature_version, churn_horizon_days FROM t);
    """

    sql_insert = """
        INSERT INTO analytics.scoring_policy
        (model_name, model_version, feature_version, horizon_days,
         threshold_default, cutpoint_vh, cutpoint_h, cutpoint_m,
         effective_from, effective_until, created_at)
        VALUES %s
    """

    values_insert = []
    now_func = "NOW()"  # effective_from 미지정 시 DB now()를 쓰기 위한 플레이스홀더
    for r in rows:
        (mn, mv, fv, hd, thr, vh, h, m) = r
        values_insert.append((
            mn, mv, fv, hd, thr, vh, h, m,
            None if eff_from is None else eff_from,  # 파라미터로 넘김
            datetime.datetime.max.replace(tzinfo=datetime.timezone.utc),  # 'infinity' 대체
            None  # created_at은 DEFAULT NOW()이지만 여기선 명시적 컬럼 사용
        ))

    with CFG.connect_db() as conn, conn.cursor() as cur:
        # 1) 기존 current 종료
        execute_values(cur, sql_close, [(r[0], r[1], r[2], r[3], 0,0,0,0) for r in rows])

        # 2) 새 정책 삽입
        #   effective_from을 NOW()로 넣으려면 파라미터 바인딩 대신 문자열 치환이 필요.
        #   간단히 VALUES %s로 넣되, eff_from is None이면 cur.execute로 한 줄씩 처리.
        if eff_from is None:
            for (mn, mv, fv, hd, thr, vh, h, m, _, eff_until, _) in values_insert:
                cur.execute(
                    """
                    INSERT INTO analytics.scoring_policy
                    (model_name, model_version, feature_version, horizon_days,
                     threshold_default, cutpoint_vh, cutpoint_h, cutpoint_m,
                     effective_from, effective_until, created_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s, NOW(), %s, NOW())
                    """,
                    (mn, mv, fv, hd, thr, vh, h, m, eff_until)
                )
        else:
            execute_values(cur, sql_insert, values_insert)

    print("[seed] scoring_policy updated")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "conf/scoring_policy.seed.yaml"
    main(path)
