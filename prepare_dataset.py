# -*- coding: utf-8 -*-
"""
Step 1 (COHORT): 20~30대 남성 중심 스냅샷 & 90일 라벨 생성

- 기준일(ref) 이전 데이터만으로 피처 스냅샷 생성
- 라벨: ref 이후 90일 동안 '주문 없음' → churn_90d=1
- '코호트(20~30대 남성)' 플래그를 만들어
  · 전체 스냅샷(full)과
  · 코호트 전용 스냅샷(cohort) 둘 다 저장

입력: raw_data_current/ (users, orders, user_logins, cart_items, [order_items])
출력: eda_output/predictive_churn_20s30s_male/step1_prepare/
      - train_full.csv / test_full.csv
      - train_cohort.csv / test_cohort.csv
      - charts/*.png
      - report.txt (요약 + 분포 + 코호트 비율)
"""

import os
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------
# 설정
# ---------------------------------
INPUT_DIR = "raw_data_current"
OUT_DIR   = Path("eda_output/predictive_churn_20s30s_male/step1_prepare")
CHART_DIR = OUT_DIR / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

CHURN_HORIZON_DAYS = 90

# 코호트 정의: 20~30대 남성 (만 나이 대략화: 출생연도 기준)
COHORT_GENDERS = {"MALE", "male", "Male", "M", "m"}
COHORT_AGE_MIN = 20
COHORT_AGE_MAX = 39   # 20~39세 → 20~30대

# ---------------------------------
# 유틸
# ---------------------------------
def load_csv(name, parse_dates=None, dtypes=None):
    path = os.path.join(INPUT_DIR, f"{name}.csv")
    return pd.read_csv(path, parse_dates=parse_dates, dtype=dtypes, encoding="utf-8-sig")

def section(t): 
    return "\n" + "="*70 + f"\n{t}\n" + "="*70 + "\n"

def df_to_text(df, max_rows=120, floatfmt=6):
    if df is None or len(df)==0: return "(no data)\n"
    with pd.option_context('display.max_rows', max_rows,
                           'display.max_columns', 200,
                           'display.width', 150,
                           'display.float_format', lambda x: f"{x:.{floatfmt}f}"):
        return df.to_string() + "\n"

def save_hist(series, title, fname):
    s = pd.Series(series).dropna()
    plt.figure()
    plt.hist(s, bins=30)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(CHART_DIR / fname); plt.close()

# ---------------------------------
# 로드
# ---------------------------------
users       = load_csv("users", parse_dates=["birthdate", "created_at"])
orders      = load_csv("orders", parse_dates=["order_date"])
user_logins = load_csv("user_logins", parse_dates=["login_at"])
cart_items  = load_csv("cart_items", parse_dates=["added_at"])

try:
    order_items = load_csv("order_items")
except Exception:
    order_items = None

# ---------------------------------
# 기준일(ref) 결정
# ---------------------------------
cands = []
for df, col in [(orders,"order_date"), (user_logins,"login_at"), (cart_items,"added_at"), (users,"created_at")]:
    if df is not None and not df.empty:
        cands.append(pd.to_datetime(df[col]).max())
max_date = max(cands) if cands else pd.Timestamp.today()

# 실무 친화적 시차: Train= max-180d, Test= max-90d
train_ref = max_date - timedelta(days=180)
test_ref  = max_date - timedelta(days=90)

# ---------------------------------
# 스냅샷 피처 생성
# ---------------------------------
def snapshot_features(ref):
    # ref 이전 데이터만 사용
    ord_hist   = orders[orders["order_date"] <= ref] if not orders.empty else orders
    login_hist = user_logins[user_logins["login_at"] <= ref] if not user_logins.empty else user_logins
    cart_hist  = cart_items[cart_items["added_at"] <= ref] if not cart_items.empty else cart_items

    # 주문 집계
    if ord_hist is not None and not ord_hist.empty:
        ord_sum = (ord_hist.groupby("user_id", as_index=False)
                   .agg(order_count=("order_id","count"),
                        total_spend=("total_amount","sum"),
                        last_order=("order_date","max")))
        aov = (ord_hist.groupby("user_id")["total_amount"].mean()
               .rename("avg_order_value").reset_index())
        tmp = ord_hist.sort_values(["user_id","order_date"]).copy()
        tmp["prev"] = tmp.groupby("user_id")["order_date"].shift(1)
        tmp["gap"]  = (tmp["order_date"] - tmp["prev"]).dt.days
        avg_gap = (tmp.groupby("user_id")["gap"].mean()
                   .rename("avg_days_between_orders").reset_index())
    else:
        ord_sum = pd.DataFrame(columns=["user_id","order_count","total_spend","last_order"])
        aov     = pd.DataFrame(columns=["user_id","avg_order_value"])
        avg_gap = pd.DataFrame(columns=["user_id","avg_days_between_orders"])

    # 활동 집계
    login_cnt = (login_hist.groupby("user_id").size().rename("login_count").reset_index()
                 if login_hist is not None and not login_hist.empty else pd.DataFrame(columns=["user_id","login_count"]))
    cart_cnt  = (cart_hist.groupby("user_id").size().rename("cart_count").reset_index()
                 if cart_hist is not None and not cart_hist.empty else pd.DataFrame(columns=["user_id","cart_count"]))

    # 결합
    feat = users[["user_id","birthdate","created_at","gender"]].copy()
    feat = (feat.merge(ord_sum, on="user_id", how="left")
                .merge(aov, on="user_id", how="left")
                .merge(avg_gap, on="user_id", how="left")
                .merge(login_cnt, on="user_id", how="left")
                .merge(cart_cnt, on="user_id", how="left"))

    # 결측 채움
    for col, v in [("order_count",0), ("total_spend",0.0), ("avg_order_value",0.0),
                   ("avg_days_between_orders", np.nan), ("login_count",0), ("cart_count",0)]:
        if col in feat.columns: feat[col] = feat[col].fillna(v)

    # 나이/연령대
    feat["age"] = ref.year - pd.to_datetime(feat["birthdate"]).dt.year
    bins = [0,24,34,44,200]; labels = ["<25","25-34","35-44","45+"]
    feat["age_group"] = pd.cut(feat["age"], bins=bins, labels=labels, include_lowest=True, right=True)

    # 비율/파생
    feat["avg_cart_per_login"] = np.where(feat["login_count"]>0, feat["cart_count"]/feat["login_count"], 0.0)

    # 코호트 플래그 (20~39세 & 남성)
    g = feat["gender"].astype(str)
    feat["is_male_20s30s"] = (
        g.isin(list(COHORT_GENDERS)) &
        feat["age"].between(COHORT_AGE_MIN, COHORT_AGE_MAX, inclusive="both")
    ).astype(int)

    feat["snapshot_ref_date"] = ref
    return feat

# ---------------------------------
# 라벨 생성 (ref 이후 90일 무주문 → 1)
# ---------------------------------
def label_90d(ref):
    end = ref + timedelta(days=CHURN_HORIZON_DAYS)
    if orders is None or orders.empty:
        lab = users[["user_id"]].copy(); lab["churn_90d"] = 1; return lab
    fut = orders[(orders["order_date"] > ref) & (orders["order_date"] <= end)]
    has = fut.groupby("user_id").size().rename("has_future_order").reset_index()
    has["has_future_order"] = True
    lab = users[["user_id"]].merge(has, on="user_id", how="left")
    lab["churn_90d"] = np.where(lab["has_future_order"].fillna(False), 0, 1)
    return lab[["user_id","churn_90d"]]

# ---------------------------------
# 생성 & 저장 (전체 + 코호트)
# ---------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)

def build_and_save(tag, ref):
    feat = snapshot_features(ref).merge(label_90d(ref), on="user_id", how="left")

    # 전체
    full_path = OUT_DIR / f"{tag}_full.csv"
    feat.to_csv(full_path, index=False, encoding="utf-8-sig")

    # 코호트
    cohort_df = feat[feat["is_male_20s30s"]==1].copy()
    cohort_path = OUT_DIR / f"{tag}_cohort.csv"
    cohort_df.to_csv(cohort_path, index=False, encoding="utf-8-sig")

    # 차트(코호트 중심 + 비교)
    for c in ["order_count","login_count","avg_days_between_orders","age"]:
        if c in feat.columns:
            save_hist(cohort_df[c], f"{tag.upper()} COHORT - {c}", f"{tag}_cohort_hist_{c}.png")
            save_hist(feat[c],       f"{tag.upper()} FULL   - {c}", f"{tag}_full_hist_{c}.png")

    return feat, cohort_df, full_path, cohort_path

train_full, train_cohort, train_full_path, train_cohort_path = build_and_save("train", train_ref)
test_full,  test_cohort,  test_full_path,  test_cohort_path  = build_and_save("test",  test_ref)

# ---------------------------------
# 리포트
# ---------------------------------
txt = OUT_DIR / "report.txt"
with open(txt, "w", encoding="utf-8") as f:
    f.write(section("SNAPSHOT SETUP"))
    f.write(f"Max date in data: {max_date}\n")
    f.write(f"Train ref: {train_ref} | Test ref: {test_ref}\n")
    f.write(f"Churn horizon (days): {CHURN_HORIZON_DAYS}\n")

    def add_block(name, df_all, df_cohort):
        f.write(section(f"{name.upper()} – SHAPE & LABEL"))
        f.write(f"Full   : {df_all.shape} | churn_90d={df_all['churn_90d'].mean():.4f}\n")
        f.write(f"Cohort : {df_cohort.shape} | churn_90d={df_cohort['churn_90d'].mean():.4f}\n")
        # 코호트 비율
        share = df_cohort.shape[0] / max(1, df_all.shape[0])
        f.write(f"Cohort share: {share*100:.2f}% of {name}\n")
        # 간단한 분포(중앙값)
        for col in ["order_count","login_count","avg_days_between_orders","total_spend","age"]:
            if col in df_all.columns:
                f.write(f"- {col:>24} | full median={df_all[col].median():.2f} | cohort median={df_cohort[col].median():.2f}\n")

    add_block("train", train_full, train_cohort)
    add_block("test",  test_full,  test_cohort)

    f.write(section("FILES"))
    f.write(f"- {train_full_path.name}\n- {train_cohort_path.name}\n- {test_full_path.name}\n- {test_cohort_path.name}\n")

    f.write(section("SAVED CHARTS"))
    for p in sorted(CHART_DIR.glob("*.png")):
        f.write(f"- {p.name}\n")

print("✅ Step1 (cohort) complete")
print(f"- Train(full):  {train_full_path}")
print(f"- Train(cohort):{train_cohort_path}")
print(f"- Test(full):   {test_full_path}")
print(f"- Test(cohort): {test_cohort_path}")
