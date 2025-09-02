# -*- coding: utf-8 -*-
"""
Ensure a current scoring_policy row exists for (model, version, feature, horizon).
- If not found, try seeding from conf/scoring_policy.seed.yaml.
- If YAML missing or no match, insert a conservative default.

YAML format accepted (single mapping or list):
  - model_name: lgbm
    model_version: lgbm_v1
    feature_version: f_v1
    churn_horizon_days: 30
    threshold_default: 0.5
    cutpoint_vh: 0.8
    cutpoint_h: 0.6
    cutpoint_m: 0.4
"""
from __future__ import annotations
import os

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# reuse your helpers
from .writer import _fetch_current_scoring_policy  # raises if not found
from .scoring_policy_upsert import upsert_scoring_policy  # make sure this path exists


def _try_seed_from_yaml(conn, *, model_name: str, model_version: str, feature_version: str,
                        churn_horizon_days: int) -> bool:
    if yaml is None:
        return False
    # project root / conf / scoring_policy.seed.yaml
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    path = os.path.join(root, "conf", "scoring_policy.seed.yaml")
    if not os.path.exists(path):
        return False

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    candidates = data if isinstance(data, list) else [data]
    for c in candidates:
        if (
            c.get("model_name") == model_name and
            c.get("model_version") == model_version and
            c.get("feature_version") == feature_version and
            int(c.get("churn_horizon_days")) == int(churn_horizon_days)
        ):
            upsert_scoring_policy(
                conn,
                model_name=model_name,
                model_version=model_version,
                feature_version=feature_version,
                churn_horizon_days=churn_horizon_days,
                threshold_default=float(c.get("threshold_default", 0.5)),
                cutpoint_vh=float(c.get("cutpoint_vh", 0.8)),
                cutpoint_h=float(c.get("cutpoint_h", 0.6)),
                cutpoint_m=float(c.get("cutpoint_m", 0.4)),
            )
            return True
    return False


def ensure_scoring_policy(conn, *, model_name: str, model_version: str, feature_version: str,
                          churn_horizon_days: int) -> None:
    try:
        _fetch_current_scoring_policy(conn, model_name, model_version, feature_version, churn_horizon_days)
        return  # already exists
    except Exception:
        pass

    # seed from YAML if possible; otherwise insert default
    seeded = _try_seed_from_yaml(
        conn,
        model_name=model_name,
        model_version=model_version,
        feature_version=feature_version,
        churn_horizon_days=churn_horizon_days,
    )
    if not seeded:
        upsert_scoring_policy(
            conn,
            model_name=model_name,
            model_version=model_version,
            feature_version=feature_version,
            churn_horizon_days=churn_horizon_days,
            threshold_default=0.5,
            cutpoint_vh=0.8,
            cutpoint_h=0.6,
            cutpoint_m=0.4,
        )
