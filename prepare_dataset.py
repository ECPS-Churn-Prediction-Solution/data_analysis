# -*- coding: utf-8 -*-
"""
Step 1: Prepare snapshot datasets & labels for churn prediction.
- Snapshot features up to a reference date (train_ref / test_ref)
- Label = no purchase in the next 90 days after reference → churn_90d = 1
- Writes train/test CSVs + charts + report.txt

Inputs :
  raw_data_current/users.csv (user_id, birthdate, created_at, gender, ...)
  raw_data_current/orders.csv (order_id, user_id, order_date, total_amount, used_coupon_code?, ...)
  raw_data_current/order_items.csv (optional)
  raw_data_current/user_logins.csv (user_id, login_at)
  raw_data_current/cart_items.csv (user_id, added_at)

Outputs:
  eda_output/predictive_churn/step1_prepare/train_snapshot.csv
  eda_output/predictive_churn/step1_prepare/test_snapshot.csv
  eda_output/predictive_churn/step1_prepare/charts/*.png
  eda_output/predictive_churn/step1_prepare/report.txt
"""
import os
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Settings
# -------------------------
INPUT_DIR = "raw_data_current"
OUT_DIR = Path("eda_output/predictive_churn/step1_prepare")
CHART_DIR = OUT_DIR / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

CHURN_HORIZON_DAYS = 90  # future window to label churn

# -------------------------
# Utils
# -------------------------
def load_csv(name, parse_dates=None, dtypes=None):
    path = os.path.join(INPUT_DIR, f"{name}.csv")
    return pd.read_csv(path, parse_dates=parse_dates, dtype=dtypes, encoding="utf-8-sig")

def section(title: str) -> str:
    return "\n" + "="*70 + f"\n{title}\n" + "="*70 + "\n"

def df_to_text(df: pd.DataFrame, max_rows=80, floatfmt=6) -> str:
    if df is None or len(df)==0: return "(no data)\n"
    with pd.option_context('display.max_rows', max_rows,
                           'display.max_columns', 200,
                           'display.width', 140,
                           'display.float_format', lambda x: f"{x:.{floatfmt}f}"):
        return df.to_string() + "\n"

def save_histogram(s, title, fname):
    plt.figure()
    s = pd.Series(s).dropna()
    plt.hist(s, bins=30)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(CHART_DIR / fname); plt.close()

# -------------------------
# Load
# -------------------------
users = load_csv("users", parse_dates=["birthdate", "created_at"])
orders = load_csv("orders", parse_dates=["order_date"])
order_items = None
try:
    order_items = load_csv("order_items")
except Exception:
    pass
user_logins = load_csv("user_logins", parse_dates=["login_at"])
cart_items = load_csv("cart_items", parse_dates=["added_at"])

# -------------------------
# Reference dates (max activity in data)
# -------------------------
candidates = []
for df, col in [(orders,"order_date"), (user_logins,"login_at"), (cart_items,"added_at"), (users,"created_at")]:
    if df is not None and not df.empty:
        candidates.append(pd.to_datetime(df[col]).max())
max_date = max(candidates) if candidates else pd.Timestamp.today()
train_ref = max_date - timedelta(days=180)
test_ref  = max_date - timedelta(days=90)

# -------------------------
# Feature builder
# -------------------------
def build_snapshot_features(ref_date: pd.Timestamp) -> pd.DataFrame:
    # activity up to ref_date
    ord_hist = orders[orders["order_date"] <= ref_date] if not orders.empty else orders
    login_hist = user_logins[user_logins["login_at"] <= ref_date] if not user_logins.empty else user_logins
    cart_hist = cart_items[cart_items["added_at"] <= ref_date] if not cart_items.empty else cart_items

    # aggregates from orders
    if ord_hist is not None and not ord_hist.empty:
        ord_sum = (ord_hist.groupby("user_id", as_index=False)
                   .agg(order_count=("order_id","count"),
                        total_spend=("total_amount","sum"),
                        last_order=("order_date","max")))
        aov = (ord_hist.groupby("user_id", as_index=False)["total_amount"]
               .mean().rename(columns={"total_amount":"avg_order_value"}))
        tmp = ord_hist.sort_values(["user_id","order_date"]).copy()
        tmp["prev"] = tmp.groupby("user_id")["order_date"].shift(1)
        tmp["gap"] = (tmp["order_date"] - tmp["prev"]).dt.days
        avg_gap = (tmp.groupby("user_id", as_index=False)["gap"]
                   .mean().rename(columns={"gap":"avg_days_between_orders"}))
    else:
        ord_sum = pd.DataFrame(columns=["user_id","order_count","total_spend","last_order"])
        aov = pd.DataFrame(columns=["user_id","avg_order_value"])
        avg_gap = pd.DataFrame(columns=["user_id","avg_days_between_orders"])

    # user activity aggregates
    login_cnt = (login_hist.groupby("user_id").size().rename("login_count").reset_index()
                 if login_hist is not None and not login_hist.empty else pd.DataFrame(columns=["user_id","login_count"]))
    cart_cnt  = (cart_hist.groupby("user_id").size().rename("cart_count").reset_index()
                 if cart_hist is not None and not cart_hist.empty else pd.DataFrame(columns=["user_id","cart_count"]))

    feat = users[["user_id","birthdate","created_at","gender"]].copy()
    feat = (feat
            .merge(ord_sum, on="user_id", how="left")
            .merge(aov, on="user_id", how="left")
            .merge(avg_gap, on="user_id", how="left")
            .merge(login_cnt, on="user_id", how="left")
            .merge(cart_cnt, on="user_id", how="left"))

    for col, val in [("order_count",0), ("total_spend",0.0), ("avg_order_value",0.0),
                     ("avg_days_between_orders", np.nan), ("login_count",0), ("cart_count",0)]:
        if col in feat.columns: feat[col] = feat[col].fillna(val)

    # demographics
    feat["age"] = ref_date.year - pd.to_datetime(feat["birthdate"]).dt.year
    bins = [0,24,34,44,200]
    labels = ["<25","25-34","35-44","45+"]
    feat["age_group"] = pd.cut(feat["age"], bins=bins, labels=labels, include_lowest=True, right=True)

    # ratios
    feat["avg_cart_per_login"] = np.where(feat["login_count"]>0, feat["cart_count"]/feat["login_count"], 0.0)

    feat["snapshot_ref_date"] = ref_date
    return feat

# -------------------------
# Label builder
# -------------------------
def label_next_90d_churn(ref_date: pd.Timestamp) -> pd.DataFrame:
    # churn=1 if NO orders in (ref_date, ref_date+90]
    horizon_end = ref_date + timedelta(days=CHURN_HORIZON_DAYS)
    if orders is None or orders.empty:
        lab = users[["user_id"]].copy()
        lab["churn_90d"] = 1
        return lab

    future_orders = orders[(orders["order_date"] > ref_date) & (orders["order_date"] <= horizon_end)]
    has_future = future_orders.groupby("user_id").size().rename("has_future_order").reset_index()
    has_future["has_future_order"] = True
    lab = users[["user_id"]].merge(has_future, on="user_id", how="left")
    lab["churn_90d"] = np.where(lab["has_future_order"].fillna(False), 0, 1)
    return lab[["user_id","churn_90d"]]

# -------------------------
# Build & Save
# -------------------------
train_feat = build_snapshot_features(train_ref).merge(label_next_90d_churn(train_ref), on="user_id", how="left")
test_feat  = build_snapshot_features(test_ref).merge(label_next_90d_churn(test_ref), on="user_id", how="left")

OUT_DIR.mkdir(parents=True, exist_ok=True)
train_path = OUT_DIR / "train_snapshot.csv"
test_path  = OUT_DIR / "test_snapshot.csv"
train_feat.to_csv(train_path, index=False, encoding="utf-8-sig")
test_feat.to_csv(test_path, index=False, encoding="utf-8-sig")

# Quick charts
save_histogram(train_feat["order_count"], "Train - order_count", "train_hist_order_count.png")
save_histogram(train_feat["login_count"], "Train - login_count", "train_hist_login_count.png")
save_histogram(train_feat["avg_days_between_orders"], "Train - avg_days_between_orders", "train_hist_avg_gap.png")
save_histogram(train_feat["age"], "Train - age", "train_hist_age.png")

# Report
txt = OUT_DIR / "report.txt"
with open(txt, "w", encoding="utf-8") as f:
    f.write(section("SNAPSHOT SETUP"))
    f.write(f"Max date in data: {max_date}\n")
    f.write(f"Train ref: {train_ref} | Test ref: {test_ref}\n")
    f.write(f"Churn horizon (days): {CHURN_HORIZON_DAYS}\n")
    f.write(section("TRAIN SHAPE & CHURN RATE"))
    f.write(f"{train_feat.shape}\n")
    f.write(f"Train churn rate (90d): {train_feat['churn_90d'].mean():.4f}\n")
    f.write(section("TEST SHAPE & CHURN RATE"))
    f.write(f"{test_feat.shape}\n")
    f.write(f"Test churn rate (90d): {test_feat['churn_90d'].mean():.4f}\n")
    f.write(section("SAVED CHARTS"))
    for p in sorted(CHART_DIR.glob('*.png')):
        f.write(f"- {p.name}\n")

print("✅ Step1 complete")
print(f"- Train CSV: {train_path}")
print(f"- Test  CSV: {test_path}")
