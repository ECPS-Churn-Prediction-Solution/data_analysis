# eda_pipeline.py
import os
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
from pathlib import Path

# =========================
# 설정
# =========================
INPUT_DIR = "raw_data_current"   # ✅ 데이터 폴더
OUTPUT_DIR = "eda_output"
CHURN_INACTIVE_DAYS = 90         # 이탈 정의: 최근 90일 구매 없음
AGE_BINS = [0, 24, 34, 44, 200]
AGE_LABELS = ["<25", "25-34", "35-44", "45+"]

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
CHART_DIR = Path(OUTPUT_DIR) / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 1) 데이터 로드
# =========================
def load_csv(name, parse_dates=None, dtypes=None):
    path = os.path.join(INPUT_DIR, f"{name}.csv")
    df = pd.read_csv(path, parse_dates=parse_dates, dtype=dtypes, encoding="utf-8-sig")
    return df

users = load_csv("users", parse_dates=["birthdate", "created_at"])
orders = load_csv("orders", parse_dates=["order_date"])
order_items = load_csv("order_items")
user_logins = load_csv("user_logins", parse_dates=["login_at"])
cart_items = load_csv("cart_items", parse_dates=["added_at"])
# (선택) 있을 수도 있는 테이블
products = categories = None
for opt in [("products", ["created_at"]), ("categories", None)]:
    try:
        df = load_csv(opt[0], parse_dates=opt[1])
        if opt[0] == "products": products = df
        if opt[0] == "categories": categories = df
    except Exception:
        pass

# =========================
# 2) 기준일 & 이탈(Churn) 정의
# =========================
candidates = []
if not orders.empty:      candidates.append(pd.to_datetime(orders["order_date"]).max())
if not user_logins.empty: candidates.append(pd.to_datetime(user_logins["login_at"]).max())
if not cart_items.empty:  candidates.append(pd.to_datetime(cart_items["added_at"]).max())
if not users.empty:       candidates.append(pd.to_datetime(users["created_at"]).max())
reference_date = max(candidates) if candidates else pd.Timestamp.today()
churn_threshold_date = reference_date - timedelta(days=CHURN_INACTIVE_DAYS)

if not orders.empty:
    last_purchase = (
        orders.groupby("user_id", as_index=False)["order_date"]
        .max()
        .rename(columns={"order_date": "last_purchase_date"})
    )
else:
    last_purchase = pd.DataFrame(columns=["user_id", "last_purchase_date"])

users_feat = users.merge(last_purchase, on="user_id", how="left")
users_feat["churn"] = np.where(
    users_feat["last_purchase_date"].isna(), 1,
    (users_feat["last_purchase_date"] < churn_threshold_date).astype(int)
)

# =========================
# 3) 사용자 단위 피처 (RFM + 활동)
# =========================
if not orders.empty:
    ord_sum = (orders
               .groupby("user_id", as_index=False)
               .agg(order_count=("order_id", "count"),
                    total_spend=("total_amount", "sum"),
                    first_order_date=("order_date", "min")))
    aov = (orders.groupby("user_id", as_index=False)["total_amount"]
           .mean().rename(columns={"total_amount": "avg_order_value"}))
    tmp = orders.sort_values(["user_id", "order_date"]).copy()
    tmp["prev_order_date"] = tmp.groupby("user_id")["order_date"].shift(1)
    tmp["days_between"] = (tmp["order_date"] - tmp["prev_order_date"]).dt.days
    avg_gap = (tmp.groupby("user_id", as_index=False)["days_between"]
               .mean().rename(columns={"days_between": "avg_days_between_orders"}))
else:
    ord_sum = pd.DataFrame(columns=["user_id","order_count","total_spend","first_order_date"])
    aov = pd.DataFrame(columns=["user_id","avg_order_value"])
    avg_gap = pd.DataFrame(columns=["user_id","avg_days_between_orders"])

login_cnt = (user_logins.groupby("user_id").size()
             .rename("login_count").reset_index()) if not user_logins.empty else pd.DataFrame(columns=["user_id","login_count"])
cart_cnt = (cart_items.groupby("user_id").size()
            .rename("cart_count").reset_index()) if not cart_items.empty else pd.DataFrame(columns=["user_id","cart_count"])

if not orders.empty:
    coupon_any = (orders.assign(used_coupon=lambda d: d["used_coupon_code"].notna())
                  .groupby("user_id", as_index=False)["used_coupon"].any())
else:
    coupon_any = pd.DataFrame(columns=["user_id","used_coupon"])

users_feat["recency_days"] = (reference_date - users_feat["last_purchase_date"]).dt.days
users_feat["recency_days"] = users_feat["recency_days"].fillna(CHURN_INACTIVE_DAYS + 9999)

users_feat = (users_feat
              .merge(ord_sum, on="user_id", how="left")
              .merge(aov, on="user_id", how="left")
              .merge(avg_gap, on="user_id", how="left")
              .merge(login_cnt, on="user_id", how="left")
              .merge(cart_cnt, on="user_id", how="left")
              .merge(coupon_any, on="user_id", how="left"))

for col, fillv in [("order_count", 0), ("total_spend", 0.0), ("avg_order_value", 0.0),
                   ("login_count", 0), ("cart_count", 0), ("used_coupon", False),
                   ("avg_days_between_orders", np.nan)]:
    if col in users_feat.columns:
        users_feat[col] = users_feat[col].fillna(fillv)

def qcut_safe(s, q=5, labels=[1,2,3,4,5], ascending=True):
    s = s.copy()
    if s.nunique() < q:
        ranks = s.rank(method="first")
        bins = pd.qcut(ranks, q=min(q, s.nunique()), labels=labels[:min(q, s.nunique())])
        return bins.astype("float").fillna(1).astype(int)
    else:
        if ascending:
            return pd.qcut(s.rank(method="first"), q=q, labels=labels).astype(int)
        else:
            return pd.qcut((-s).rank(method="first"), q=q, labels=labels).astype(int)

users_feat["R_score"] = qcut_safe(users_feat["recency_days"], q=5, labels=[5,4,3,2,1], ascending=False)
users_feat["F_score"] = qcut_safe(users_feat["order_count"], q=5, labels=[1,2,3,4,5], ascending=True)
users_feat["M_score"] = qcut_safe(users_feat["total_spend"], q=5, labels=[1,2,3,4,5], ascending=True)
users_feat["RFM_sum"] = users_feat[["R_score","F_score","M_score"]].sum(axis=1)

users_feat["age"] = pd.Timestamp.today().year - pd.to_datetime(users_feat["birthdate"]).dt.year
users_feat["age_group"] = pd.cut(users_feat["age"], bins=AGE_BINS, labels=AGE_LABELS, right=True, include_lowest=True)

# =========================
# 4) 기초 통계 / 결측치 / 이상치
# =========================
numeric_cols = ["order_count","total_spend","avg_order_value","avg_days_between_orders",
                "login_count","cart_count","recency_days","RFM_sum","age"]
numeric_cols = [c for c in numeric_cols if c in users_feat.columns]

categorical_cols = ["gender","age_group","used_coupon"]
if "signup_source" in users_feat.columns:  # 생성 데이터에 있을 수 있음
    categorical_cols.append("signup_source")

# 요약 통계
desc_num = users_feat[numeric_cols].describe().T

# 결측치 개수/비율
na_counts = users_feat[numeric_cols + categorical_cols + ["churn"]].isna().sum()
na_ratio  = (na_counts / len(users_feat)).round(4)
missing_tbl = pd.DataFrame({"missing_count": na_counts, "missing_ratio": na_ratio})

# IQR 이상치 탐지
def iqr_outlier_count(s):
    s = s.dropna()
    if s.empty: return 0, 0.0, np.nan, np.nan
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
    cnt = ((s < low) | (s > high)).sum()
    return cnt, cnt/len(s), low, high

outlier_rows = []
for c in numeric_cols:
    cnt, ratio, low, high = iqr_outlier_count(users_feat[c].astype(float))
    outlier_rows.append({"column": c, "outlier_count": cnt, "outlier_ratio": round(ratio,4),
                         "lower_bound": round(low,3) if pd.notna(low) else np.nan,
                         "upper_bound": round(high,3) if pd.notna(high) else np.nan})
outlier_tbl = pd.DataFrame(outlier_rows).sort_values("outlier_count", ascending=False)

# =========================
# 5) 분포/관계 시각화 (PNG)
# =========================
def hist_plot(series, title, fname):
    plt.figure()
    s = series.dropna()
    plt.hist(s, bins=30)
    plt.title(title)
    plt.xlabel(series.name); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(CHART_DIR / fname); plt.close()

from pandas.api.types import CategoricalDtype

def count_plot(series, title, fname):
    # 안전한 변환: 카테고리는 문자열로 변환한 뒤 NA 처리
    s = series
    if isinstance(s.dtype, CategoricalDtype):
        s = s.astype("object")
    # 문자열 변환 후 NA 채움
    s = s.astype(str).fillna("NA")
    vc = s.value_counts(dropna=False)

    plt.figure()
    vc.plot(kind="bar")
    plt.title(title)
    plt.xlabel(series.name)
    plt.ylabel("Count")
    plt.tight_layout()
    path = CHART_DIR / fname
    plt.savefig(path)
    plt.close()
    return str(path.name)


# 히스토그램
for c in numeric_cols:
    hist_plot(users_feat[c], f"Histogram - {c}", f"hist_{c}.png")

# 카운트플롯
for c in categorical_cols:
    count_plot(users_feat[c], f"Countplot - {c}", f"count_{c}.png")

# 상관행렬
if len(numeric_cols) >= 2:
    corr = users_feat[numeric_cols].corr()
    plt.figure(figsize=(8,6))
    plt.imshow(corr, interpolation='nearest')
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.columns)
    plt.colorbar()
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "correlation_matrix.png")
    plt.close()
else:
    corr = pd.DataFrame()

# =========================
# 6) 이탈률 집계 + 통계 검정
# =========================
def rate_by(series):
    return users_feat.groupby(series, observed=False)["churn"].mean().rename("churn_rate")

by_gender = rate_by("gender") if "gender" in users_feat.columns else pd.Series(dtype=float)
by_age_group = rate_by("age_group") if "age_group" in users_feat.columns else pd.Series(dtype=float)
by_coupon = rate_by("used_coupon") if "used_coupon" in users_feat.columns else pd.Series(dtype=float)
by_rfm = (users_feat.assign(RFM_bucket=pd.cut(users_feat["RFM_sum"], bins=[2,5,8,11,15],
                                              labels=["Low(3-5)","Mid-Low(6-8)","Mid-High(9-11)","High(12-15)"]))
          .groupby("RFM_bucket", observed=False)["churn"].mean().rename("churn_rate"))

overall_churn_rate = users_feat["churn"].mean()

# 카이제곱 (범주형 ↔ churn)
chi_results = []
for c in categorical_cols:
    ct = pd.crosstab(users_feat[c], users_feat["churn"])
    if ct.shape[0] > 1 and ct.shape[1] > 1:
        chi2, p, dof, _ = chi2_contingency(ct)
        chi_results.append({"variable": c, "chi2_pvalue": p, "dof": dof, "table_shape": ct.shape})
chi_tbl = pd.DataFrame(chi_results)

# t-test & Mann-Whitney (연속형 ↔ churn)
ttest_rows = []
mw_rows = []

for c in numeric_cols:
    a = users_feat.loc[users_feat["churn"]==0, c].dropna().astype(float)
    b = users_feat.loc[users_feat["churn"]==1, c].dropna().astype(float)
    if len(a)>1 and len(b)>1:
        t, p = ttest_ind(a, b, equal_var=False)
        u, p2 = mannwhitneyu(a, b, alternative="two-sided")
        ttest_rows.append({"variable": c, "ttest_pvalue": p, "mean_active": a.mean(), "mean_churned": b.mean()})
        mw_rows.append({"variable": c, "mannwhitney_pvalue": p2, "median_active": a.median(), "median_churned": b.median()})

ttest_tbl = pd.DataFrame(ttest_rows)
mw_tbl = pd.DataFrame(mw_rows)


# 집계 그래프(이탈률)
def save_bar(series, title, xlabel, fname):
    if series.empty: return
    plt.figure()
    series.plot(kind="bar")
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("Churn Rate")
    plt.tight_layout(); plt.savefig(CHART_DIR / fname); plt.close()

save_bar(by_gender, "Churn Rate by Gender", "Gender", "churn_by_gender.png")
save_bar(by_age_group, "Churn Rate by Age Group", "Age Group", "churn_by_age_group.png")
save_bar(by_coupon, "Churn Rate by Coupon Usage", "Used Coupon", "churn_by_coupon.png")
save_bar(by_rfm, "Churn Rate by RFM Bucket", "RFM Bucket", "churn_by_rfm_bucket.png")

# 박스플롯(이탈여부별 분포)
def save_boxplot(col, title, fname):
    if col not in users_feat.columns: return
    plt.figure()
    g0 = users_feat.loc[users_feat["churn"]==0, col].dropna()
    g1 = users_feat.loc[users_feat["churn"]==1, col].dropna()
    if len(g0)==0 or len(g1)==0: return
    plt.boxplot([g0, g1], labels=["Active(0)","Churned(1)"])
    plt.title(title); plt.ylabel(col)
    plt.tight_layout(); plt.savefig(CHART_DIR / fname); plt.close()

for c in ["login_count","total_spend","order_count","avg_order_value","recency_days"]:
    if c in users_feat.columns:
        save_boxplot(c, f"{c} by Churn Status", f"box_{c}_by_churn.png")

# =========================
# 7) TXT 리포트 저장 (표 형식)
# =========================
def section(title):
    return "\n" + "="*70 + f"\n{title}\n" + "="*70 + "\n"

def df_to_text(df, max_rows=50, floatfmt=6):
    if df is None or len(df)==0:
        return "(no data)\n"
    df2 = df.copy()
    with pd.option_context('display.max_rows', max_rows,
                           'display.max_columns', 200,
                           'display.width', 120,
                           'display.float_format', lambda x: f"{x:.{floatfmt}f}"):
        return df2.to_string() + "\n"

txt_path = Path(OUTPUT_DIR) / "report.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(section("DATA SHAPES"))
    f.write(f"users: {users.shape}\norders: {orders.shape}\norder_items: {order_items.shape}\n")
    f.write(f"user_logins: {user_logins.shape}\ncart_items: {cart_items.shape}\n")
    if products is not None: f.write(f"products: {products.shape}\n")
    if categories is not None: f.write(f"categories: {categories.shape}\n")

    f.write(section("REFERENCE & CHURN DEFINITION"))
    f.write(f"Reference date: {reference_date}\n")
    f.write(f"Churn threshold: last purchase < {churn_threshold_date}  (inactive {CHURN_INACTIVE_DAYS} days)\n")
    f.write(f"Overall churn rate: {overall_churn_rate:.4f}\n")

    f.write(section("SUMMARY STATISTICS (NUMERIC)"))
    f.write(df_to_text(desc_num))

    f.write(section("MISSING VALUES (COUNT & RATIO)"))
    f.write(df_to_text(missing_tbl.sort_values("missing_count", ascending=False)))

    f.write(section("OUTLIERS (IQR METHOD)"))
    f.write(df_to_text(outlier_tbl))

    f.write(section("CHURN RATES BY GROUP"))
    if not by_gender.empty: f.write("[By Gender]\n" + df_to_text(by_gender.to_frame()))
    if not by_age_group.empty: f.write("[By Age Group]\n" + df_to_text(by_age_group.to_frame()))
    if not by_coupon.empty: f.write("[By Coupon]\n" + df_to_text(by_coupon.to_frame()))
    f.write("[By RFM Bucket]\n" + df_to_text(by_rfm.to_frame()))

    f.write(section("CHI-SQUARE TESTS (Categorical vs Churn)"))
    f.write(df_to_text(chi_tbl))

    f.write(section("T-TEST (Numeric vs Churn; means)"))
    f.write(df_to_text(ttest_tbl))

    f.write(section("MANN-WHITNEY U TEST (Numeric vs Churn; medians)"))
    f.write(df_to_text(mw_tbl))

    f.write(section("SAVED CHARTS"))
    chart_files = sorted([p.name for p in CHART_DIR.glob("*.png")])
    for c in chart_files:
        f.write(f"- {c}\n")

print("✅ EDA 완료!")
print(f"- TXT 리포트: {txt_path}")
print(f"- 차트 디렉토리: {CHART_DIR}")
