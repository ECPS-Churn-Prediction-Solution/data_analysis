# segmentation_pipeline.py
import os
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

# ======================================================
# 설정
# ======================================================
INPUT_DIR = "raw_data_current"      # EDA와 동일 폴더
OUT_DIR = Path("eda_output/segmentation")
CHART_DIR = OUT_DIR / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

CHURN_INACTIVE_DAYS = 90
AGE_BINS = [0, 24, 34, 44, 200]
AGE_LABELS = ["<25", "25-34", "35-44", "45+"]

# ======================================================
# 0) 유틸
# ======================================================
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

def section(title):
    return "\n" + "="*70 + f"\n{title}\n" + "="*70 + "\n"

def df_to_text(df, max_rows=80, floatfmt=6):
    if df is None or len(df)==0:
        return "(no data)\n"
    with pd.option_context('display.max_rows', max_rows,
                           'display.max_columns', 200,
                           'display.width', 140,
                           'display.float_format', lambda x: f"{x:.{floatfmt}f}"):
        return df.to_string() + "\n"

# ======================================================
# 1) 데이터 로드
# ======================================================
def load_csv(name, parse_dates=None, dtypes=None):
    path = os.path.join(INPUT_DIR, f"{name}.csv")
    df = pd.read_csv(path, parse_dates=parse_dates, dtype=dtypes, encoding="utf-8-sig")
    return df

users = load_csv("users", parse_dates=["birthdate", "created_at"])
orders = load_csv("orders", parse_dates=["order_date"])
order_items = load_csv("order_items")
user_logins = load_csv("user_logins", parse_dates=["login_at"])
cart_items = load_csv("cart_items", parse_dates=["added_at"])
products = load_csv("products", parse_dates=["created_at"])
categories = load_csv("categories")

# 기준일/이탈 정의 (EDA와 동일 로직)
reference_date = max([
    pd.to_datetime(users["created_at"]).max(),
    pd.to_datetime(user_logins["login_at"]).max() if not user_logins.empty else pd.Timestamp.min,
    pd.to_datetime(cart_items["added_at"]).max() if not cart_items.empty else pd.Timestamp.min,
    pd.to_datetime(orders["order_date"]).max() if not orders.empty else pd.Timestamp.min
])
churn_threshold_date = reference_date - timedelta(days=CHURN_INACTIVE_DAYS)

if not orders.empty:
    last_purchase = (orders.groupby("user_id", as_index=False)["order_date"]
                          .max().rename(columns={"order_date":"last_purchase_date"}))
else:
    last_purchase = pd.DataFrame(columns=["user_id","last_purchase_date"])

users_feat = users.merge(last_purchase, on="user_id", how="left")
users_feat["churn"] = np.where(users_feat["last_purchase_date"].isna(), 1,
                               (users_feat["last_purchase_date"] < churn_threshold_date).astype(int))

# 활동/구매 집계
if not orders.empty:
    ord_sum = (orders.groupby("user_id", as_index=False)
                    .agg(order_count=("order_id","count"),
                         total_spend=("total_amount","sum"),
                         first_order_date=("order_date","min")))
    aov = (orders.groupby("user_id", as_index=False)["total_amount"]
                 .mean().rename(columns={"total_amount":"avg_order_value"}))
else:
    ord_sum = pd.DataFrame(columns=["user_id","order_count","total_spend","first_order_date"])
    aov = pd.DataFrame(columns=["user_id","avg_order_value"])

# 주문 간 평균 간격
if not orders.empty:
    tmp = orders.sort_values(["user_id","order_date"]).copy()
    tmp["prev_order_date"] = tmp.groupby("user_id")["order_date"].shift(1)
    tmp["days_between"] = (tmp["order_date"] - tmp["prev_order_date"]).dt.days
    avg_gap = (tmp.groupby("user_id", as_index=False)["days_between"]
                  .mean().rename(columns={"days_between":"avg_days_between_orders"}))
else:
    avg_gap = pd.DataFrame(columns=["user_id","avg_days_between_orders"])

# 로그인/장바구니
login_cnt = (user_logins.groupby("user_id").size().rename("login_count").reset_index()
             if not user_logins.empty else pd.DataFrame(columns=["user_id","login_count"]))
cart_cnt = (cart_items.groupby("user_id").size().rename("cart_count").reset_index()
            if not cart_items.empty else pd.DataFrame(columns=["user_id","cart_count"]))

# 쿠폰 사용 경험
coupon_any = (orders.assign(used_coupon=lambda d: d["used_coupon_code"].notna())
              .groupby("user_id", as_index=False)["used_coupon"].any()
              if not orders.empty else pd.DataFrame(columns=["user_id","used_coupon"]))

# RFM용 recency
users_feat["recency_days"] = (reference_date - users_feat["last_purchase_date"]).dt.days
users_feat["recency_days"] = users_feat["recency_days"].fillna(CHURN_INACTIVE_DAYS + 9999)

# 통합
users_feat = (users_feat
              .merge(ord_sum, on="user_id", how="left")
              .merge(aov, on="user_id", how="left")
              .merge(avg_gap, on="user_id", how="left")
              .merge(login_cnt, on="user_id", how="left")
              .merge(cart_cnt, on="user_id", how="left")
              .merge(coupon_any, on="user_id", how="left"))

for col, fillv in [("order_count",0), ("total_spend",0.0), ("avg_order_value",0.0),
                   ("avg_days_between_orders",np.nan), ("login_count",0),
                   ("cart_count",0), ("used_coupon",False)]:
    if col in users_feat.columns:
        users_feat[col] = users_feat[col].fillna(fillv)

# 나이/연령대
users_feat["age"] = pd.Timestamp.today().year - pd.to_datetime(users_feat["birthdate"]).dt.year
users_feat["age_group"] = pd.cut(users_feat["age"], bins=AGE_BINS, labels=AGE_LABELS, right=True, include_lowest=True)

# ======================================================
# 2) RFM 분석
# ======================================================
# 1~5 점수
users_feat["R_score"] = qcut_safe(users_feat["recency_days"], q=5, labels=[5,4,3,2,1], ascending=False) # 최근일수 낮을수록 점수↑
users_feat["F_score"] = qcut_safe(users_feat["order_count"], q=5, labels=[1,2,3,4,5], ascending=True)
users_feat["M_score"] = qcut_safe(users_feat["total_spend"], q=5, labels=[1,2,3,4,5], ascending=True)
users_feat["RFM_sum"] = users_feat[["R_score","F_score","M_score"]].sum(axis=1)

# 버킷(4단): 3~5 / 6~8 / 9~11 / 12~15
rfm_bucket = pd.cut(users_feat["RFM_sum"], bins=[2,5,8,11,15],
                    labels=["Low(3-5)","Mid-Low(6-8)","Mid-High(9-11)","High(12-15)"])
users_feat["RFM_bucket"] = rfm_bucket

rfm_churn = users_feat.groupby("RFM_bucket", observed=False)["churn"].agg(["mean","count"]).rename(columns={"mean":"churn_rate","count":"n"})
rfm_stats = (users_feat.groupby("RFM_bucket", observed=False)
             .agg(order_count_mean=("order_count","mean"),
                  total_spend_mean=("total_spend","mean"),
                  login_count_mean=("login_count","mean"),
                  recency_days_mean=("recency_days","mean"),
                  churn_rate=("churn","mean"),
                  n=("user_id","count")))

# 시각화: RFM 버킷별 이탈률
plt.figure()
(rfm_churn["churn_rate"]).plot(kind="bar")
plt.title("Churn Rate by RFM Bucket")
plt.ylabel("Churn Rate"); plt.xlabel("RFM Bucket")
plt.tight_layout(); plt.savefig(CHART_DIR / "rfm_bucket_churn.png"); plt.close()

# ======================================================
# 3) 클러스터링용 특징 생성
#    login_count, order_count, avg_cart_per_login, category_diversity
# ======================================================
# 3-1) avg_cart_per_login
users_feat["avg_cart_per_login"] = np.where(users_feat["login_count"]>0,
                                            users_feat["cart_count"]/users_feat["login_count"], 0.0)

# 3-2) category_diversity : 구매한 카테고리의 고유 개수
#       orders -> order_items -> products(category_id) 조인
if not orders.empty and not order_items.empty and not products.empty:
    oi = order_items.merge(orders[["order_id","user_id"]], on="order_id", how="left")
    oi = oi.merge(products[["product_id","category_id"]], on="product_id", how="left")
    cat_div = (oi.dropna(subset=["user_id","category_id"])
                 .groupby("user_id")["category_id"].nunique()
                 .rename("category_diversity").reset_index())
else:
    # fallback: 구매 데이터 없으면 0
    cat_div = pd.DataFrame({"user_id": users_feat["user_id"], "category_diversity": 0})

users_seg = users_feat.merge(cat_div, on="user_id", how="left")
users_seg["category_diversity"] = users_seg["category_diversity"].fillna(0)

features = ["login_count","order_count","avg_cart_per_login","category_diversity"]
X = users_seg[features].copy()

# 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================================================
# 4) K-Means
# ======================================================
# 실루엣 점수 (k=3~6)
sil_rows = []
for k in range(3,7):
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    sil_rows.append({"k":k, "silhouette":sil})

sil_df = pd.DataFrame(sil_rows)

plt.figure()
plt.plot(sil_df["k"], sil_df["silhouette"], marker="o")
plt.title("Silhouette Score by k (KMeans)")
plt.xlabel("k"); plt.ylabel("silhouette")
plt.grid(True)
plt.tight_layout(); plt.savefig(CHART_DIR / "kmeans_silhouette.png"); plt.close()

# 기본 k = 4
k_opt = 4
kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init="auto")
users_seg["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

# 클러스터별 요약
kmeans_summary = (users_seg.groupby("kmeans_cluster")
                  .agg(**{f"{c}_mean":(c,"mean") for c in features},
                       **{f"{c}_median":(c,"median") for c in features},
                       churn_rate=("churn","mean"),
                       size=("user_id","count") )
                  .sort_values("size", ascending=False))

# PCA 2D 시각화
pca = PCA(n_components=2, random_state=42)
XY = pca.fit_transform(X_scaled)
plt.figure()
for lab in sorted(users_seg["kmeans_cluster"].unique()):
    idx = users_seg["kmeans_cluster"]==lab
    plt.scatter(XY[idx,0], XY[idx,1], s=10, alpha=0.6, label=f"C{lab}")
plt.title("KMeans Clusters (PCA 2D)")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.legend(markerscale=2, frameon=False)
plt.tight_layout(); plt.savefig(CHART_DIR / "kmeans_pca_scatter.png"); plt.close()

# 프로파일 막대(표준화 평균)
cluster_profiles = (users_seg.groupby("kmeans_cluster")[features].mean())
cluster_profiles_z = pd.DataFrame(StandardScaler().fit_transform(cluster_profiles),
                                  index=cluster_profiles.index, columns=cluster_profiles.columns)

plt.figure(figsize=(8,4+0.3*len(cluster_profiles_z)))
cluster_profiles_z.plot(kind="bar")
plt.title("KMeans Cluster Profiles (Z-score Means)")
plt.ylabel("Z-score")
plt.tight_layout(); plt.savefig(CHART_DIR / "kmeans_profiles.png"); plt.close()

# ======================================================
# 5) 계층적 군집 (Agglomerative, n=4)
# ======================================================
agg = AgglomerativeClustering(n_clusters=4, linkage="ward")
users_seg["hclust_cluster"] = agg.fit_predict(X_scaled)

hclust_summary = (users_seg.groupby("hclust_cluster")
                  .agg(**{f"{c}_mean":(c,"mean") for c in features},
                       **{f"{c}_median":(c,"median") for c in features},
                       churn_rate=("churn","mean"),
                       size=("user_id","count") )
                  .sort_values("size", ascending=False))

# 덴드로그램(샘플 800명 정도로 축소)
sample_n = 800
sample_idx = np.random.RandomState(42).choice(len(X_scaled), size=min(sample_n, len(X_scaled)), replace=False)
Z = linkage(X_scaled[sample_idx], method="ward")
plt.figure(figsize=(10,4))
dendrogram(Z, no_labels=True, count_sort=True)
plt.title("Hierarchical Clustering Dendrogram (Ward) - sample")
plt.tight_layout(); plt.savefig(CHART_DIR / "hclust_dendrogram.png"); plt.close()

# PCA 2D 산점도
XY2 = pca.fit_transform(X_scaled)
plt.figure()
for lab in sorted(users_seg["hclust_cluster"].unique()):
    idx = users_seg["hclust_cluster"]==lab
    plt.scatter(XY2[idx,0], XY2[idx,1], s=10, alpha=0.6, label=f"C{lab}")
plt.title("Hierarchical Clusters (PCA 2D)")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.legend(markerscale=2, frameon=False)
plt.tight_layout(); plt.savefig(CHART_DIR / "hclust_pca_scatter.png"); plt.close()

# ======================================================
# 6) 결과 TXT 저장
# ======================================================
txt_path = OUT_DIR / "summary.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(section("SETUP"))
    f.write(f"Reference date: {reference_date}\n")
    f.write(f"Churn threshold: {churn_threshold_date} (inactive {CHURN_INACTIVE_DAYS} days)\n")
    f.write(f"Features for clustering: {features}\n")

    f.write(section("RFM ANALYSIS - CHURN BY BUCKET"))
    f.write(df_to_text(rfm_churn))
    f.write("\nRFM Bucket descriptive stats\n")
    f.write(df_to_text(rfm_stats))

    f.write(section("KMEANS CLUSTERING"))
    f.write("Silhouette scores (k=3..6):\n")
    f.write(df_to_text(sil_df))
    f.write("\nKMeans cluster summary (means/medians & churn):\n")
    f.write(df_to_text(kmeans_summary))
    f.write("\nCluster profiles (Z-score means):\n")
    f.write(df_to_text(cluster_profiles_z))

    f.write(section("HIERARCHICAL CLUSTERING (WARD)"))
    f.write("Agglomerative (n=4) cluster summary:\n")
    f.write(df_to_text(hclust_summary))

    f.write(section("SAVED CHARTS"))
    for p in sorted(CHART_DIR.glob("*.png")):
        f.write(f"- {p.name}\n")

print("✅ Segmentation done!")
print(f"- TXT: {txt_path}")
print(f"- Charts: {CHART_DIR}")
