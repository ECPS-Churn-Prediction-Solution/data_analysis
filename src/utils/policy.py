# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Dict, Tuple
from datetime import datetime, timezone

from ..common.settings import CFG

def get_scoring_policy(conn) -> Tuple[float, float, float, float]:
    """
    analytics.scoring_policy에서 현재(유효-until=∞) 정책을 1건 가져온다.
    반환: (threshold_default, cutpoint_vh, cutpoint_h, cutpoint_m)
    """
    sql = """
    SELECT threshold_default, cutpoint_vh, cutpoint_h, cutpoint_m
    FROM analytics.scoring_policy
    WHERE model_name=%s AND model_version=%s AND feature_version=%s
      AND churn_horizon_days=%s AND effective_until='infinity'
    ORDER BY effective_from DESC
    LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, (CFG.MODEL_NAME, CFG.MODEL_VERSION, CFG.FEATURE_VERSION, int(CFG.CHURN_HORIZON_DAYS)))
        row = cur.fetchone()
    if not row:
        # 폴백: 정책이 없으면 보수적으로 분할(0.75/0.5/0.25)
        return (0.5, 0.75, 0.5, 0.25)
    return tuple(float(x) for x in row)  # type: ignore

def get_action_map(conn) -> Dict[str, Tuple[Optional[int], str]]:
    """
    analytics.action_recommendations에서 현재 활성 정책을 밴드별로 얻는다.
    반환: { 'VH': (policy_id, action_code), ... }
    """
    sql = """
    SELECT policy_id, risk_band, action_code
    FROM analytics.action_recommendations
    WHERE is_active = true AND effective_until='infinity'
    """
    out: Dict[str, Tuple[Optional[int], str]] = {}
    with conn.cursor() as cur:
        cur.execute(sql)
        for pid, band, code in cur.fetchall():
            out[str(band)] = (pid, str(code))
    # 기본값 보강
    for band in ("VH", "H", "M", "L"):
        out.setdefault(band, (None, "NONE"))
    return out

def now_utc():
    return datetime.now(timezone.utc)
