# -*- coding: utf-8 -*-
"""
churn_drivers_pipeline.py

Stage 3: Churn Driver Analysis + Threshold Tuning
- Logistic Regression (odds ratios & coefficient plot)
- Cox Proportional Hazards (survival / time-to-churn)
- Tree-based models (RandomForest; XGBoost optional) for feature importance
- **Threshold search & tuning across multiple objectives**:
  Balanced Accuracy (default), Youden's J, MCC, F1, Accuracy

Inputs  : CSVs under raw_data_current/ (same schema used in EDA/segmentation)
Outputs : eda_output/churn_drivers/ (charts and report.txt)
Notes   : Avoids leakage features (e.g., recency_days) in classification models.
"""

import os
import warnings
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix, classification_report,
    accuracy_score, f1_score, balanced_accuracy_score, matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# lifelines for Cox model
from lifelines import CoxPHFitter

# Try import xgboost (optional)
try:
    from xgboost import XGBClassifier
    _XGB_OK = True
except Exception:
    _XGB_OK = False
    warnings.warn("xgboost not available. Skipping XGBoost feature importance.")

# =========================
# 설정
# =========================
INPUT_DIR = "raw_data_current"          # ✅ same folder as EDA
OUT_DIR = Path("eda_output/churn_drivers")
CHART_DIR = OUT_DIR / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

CHURN_INACTIVE_DAYS = 90                # churn rule: no purchase for 90 days
AGE_BINS = [0, 24, 34, 44, 200]
AGE_LABELS = ["<25", "25-34", "35-44", "45+"]

def section(title: str) -> str:
    return "\n" + "="*70 + f"\n{title}\n" + "="*70 + "\n"

def df_to_text(df: pd.DataFrame, max_rows=120, floatfmt=6) -> str:
    if df is None or len(df)==0:
        return "(no data)\n"
    with pd.option_context('display.max_rows', max_rows,
                           'display.max_columns', 200,
                           'display.width', 140,
                           'display.float_format', lambda x: f"{x:.{floatfmt}f}"):
        return df.to_string() + "\n"

def load_csv(name, parse_dates=None, dtypes=None):
    path = os.path.join(INPUT_DIR, f"{name}.csv")
    return pd.read_csv(path, parse_dates=parse_dates, dtype=dtypes, encoding="utf-8-sig")

# =========================
# 1) 데이터 로드
# =========================
users = load_csv("users", parse_dates=["birthdate", "created_at"])
orders = load_csv("orders", parse_dates=["order_date"])
order_items = load_csv("order_items")
user_logins = load_csv("user_logins", parse_dates=["login_at"])
cart_items = load_csv("cart_items", parse_dates=["added_at"])

# optional
products = categories = None
for opt in [("products", ["created_at"]), ("categories", None)]:
    try:
        df = load_csv(opt[0], parse_dates=opt[1])
        if opt[0] == "products": products = df
        if opt[0] == "categories": categories = df
    except Exception:
        pass

# =========================
# 2) 기준일 & 이탈 정의
# =========================
candidates = []
if not orders.empty:      candidates.append(pd.to_datetime(orders["order_date"]).max())
if not user_logins.empty: candidates.append(pd.to_datetime(user_logins["login_at"]).max())
if not cart_items.empty:  candidates.append(pd.to_datetime(cart_items["added_at"]).max())
if not users.empty:       candidates.append(pd.to_datetime(users["created_at"]).max())
reference_date = max(candidates) if candidates else pd.Timestamp.today()
churn_threshold_date = reference_date - timedelta(days=CHURN_INACTIVE_DAYS)

# last purchase per user
if not orders.empty:
    last_purchase = (orders.groupby("user_id", as_index=False)["order_date"]
                           .max().rename(columns={"order_date": "last_purchase_date"}))
else:
    last_purchase = pd.DataFrame(columns=["user_id","last_purchase_date"])

users_feat = users.merge(last_purchase, on="user_id", how="left")

# churn label: True if last purchase exists and older than threshold, or never purchased
users_feat["churn"] = np.where(
    users_feat["last_purchase_date"].isna(), 1,
    (users_feat["last_purchase_date"] < churn_threshold_date).astype(int)
)

# =========================
# 3) 사용자 활동/구매 특징 (누적)
# =========================
if not orders.empty:
    ord_sum = (orders.groupby("user_id", as_index=False)
                    .agg(order_count=("order_id", "count"),
                         total_spend=("total_amount", "sum"),
                         first_order_date=("order_date", "min")))
    aov = (orders.groupby("user_id", as_index=False)["total_amount"]
           .mean().rename(columns={"total_amount": "avg_order_value"}))
else:
    ord_sum = pd.DataFrame(columns=["user_id","order_count","total_spend","first_order_date"])
    aov = pd.DataFrame(columns=["user_id","avg_order_value"])

# avg days between orders
if not orders.empty:
    tmp = orders.sort_values(["user_id","order_date"]).copy()
    tmp["prev_order_date"] = tmp.groupby("user_id")["order_date"].shift(1)
    tmp["days_between"] = (tmp["order_date"] - tmp["prev_order_date"]).dt.days
    avg_gap = (tmp.groupby("user_id", as_index=False)["days_between"]
               .mean().rename(columns={"days_between":"avg_days_between_orders"}))
else:
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

# last activity date (max of order/login/cart)
last_login = (user_logins.groupby("user_id", as_index=False)["login_at"]
              .max().rename(columns={"login_at":"last_login_at"})) if not user_logins.empty else pd.DataFrame(columns=["user_id","last_login_at"])
last_cart = (cart_items.groupby("user_id", as_index=False)["added_at"]
             .max().rename(columns={"added_at":"last_cart_at"})) if not cart_items.empty else pd.DataFrame(columns=["user_id","last_cart_at"])

users_feat = (users_feat
              .merge(ord_sum, on="user_id", how="left")
              .merge(aov, on="user_id", how="left")
              .merge(avg_gap, on="user_id", how="left")
              .merge(login_cnt, on="user_id", how="left")
              .merge(cart_cnt, on="user_id", how="left")
              .merge(coupon_any, on="user_id", how="left")
              .merge(last_login, on="user_id", how="left")
              .merge(last_cart, on="user_id", how="left"))

for col, fillv in [("order_count",0), ("total_spend",0.0), ("avg_order_value",0.0),
                   ("avg_days_between_orders", np.nan), ("login_count",0),
                   ("cart_count",0), ("used_coupon", False)]:
    if col in users_feat.columns:
        users_feat[col] = users_feat[col].fillna(fillv)

# age & age_group
users_feat["age"] = pd.Timestamp.today().year - pd.to_datetime(users_feat["birthdate"]).dt.year
users_feat["age_group"] = pd.cut(users_feat["age"], bins=AGE_BINS, labels=AGE_LABELS, right=True, include_lowest=True)

# avg_cart_per_login
users_feat["avg_cart_per_login"] = np.where(users_feat["login_count"]>0,
                                            users_feat["cart_count"]/users_feat["login_count"], 0.0)

# category_diversity via order joins
if (products is not None) and (not orders.empty) and (not order_items.empty):
    oi = order_items.merge(orders[["order_id","user_id"]], on="order_id", how="left")
    oi = oi.merge(products[["product_id","category_id"]], on="product_id", how="left")
    cat_div = (oi.dropna(subset=["user_id","category_id"])
                 .groupby("user_id")["category_id"].nunique()
                 .rename("category_diversity").reset_index())
else:
    cat_div = pd.DataFrame({"user_id": users_feat["user_id"], "category_diversity": 0})

users_feat = users_feat.merge(cat_div, on="user_id", how="left")
users_feat["category_diversity"] = users_feat["category_diversity"].fillna(0)

# =========================
# 4) 분류용 데이터셋 (누수 방지: recency/RFM 등 제외)
# =========================
y = users_feat["churn"].astype(int)

numeric_features = [
    "order_count", "total_spend", "avg_order_value",
    "avg_days_between_orders", "login_count", "cart_count",
    "avg_cart_per_login", "category_diversity", "age"
]
numeric_features = [c for c in numeric_features if c in users_feat.columns]

categorical_features = []
if "gender" in users_feat.columns: categorical_features.append("gender")
if "age_group" in users_feat.columns: categorical_features.append("age_group")
if "used_coupon" in users_feat.columns: categorical_features.append("used_coupon")

X = users_feat[numeric_features + categorical_features].copy()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 4-1) Logistic Regression
# =========================
# Preprocess: scale numeric, one-hot categorical
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(with_mean=True, with_std=True), numeric_features),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), categorical_features),
    ],
    remainder="drop"
)

log_reg = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    solver="lbfgs",
    n_jobs=None
)

clf = Pipeline(steps=[("preprocess", preprocess),
                     ("model", log_reg)])

clf.fit(X_train, y_train)

# =========================
# Evaluation (probabilities)
# =========================
y_prob = clf.predict_proba(X_test)[:,1]

# Base curves (no threshold)
auc = roc_auc_score(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)

# --- Helper: compute metrics for a given threshold ---
def metrics_at_threshold(y_true, y_scores, thr):
    y_pred = (y_scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc  = accuracy_score(y_true, y_pred)
    bal  = balanced_accuracy_score(y_true, y_pred)  # (TPR+TNR)/2
    f1   = f1_score(y_true, y_pred, zero_division=0)
    mcc  = matthews_corrcoef(y_true, y_pred)
    tpr  = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tnr  = 1 - fpr
    j    = tpr - fpr  # Youden's J
    return {
        "threshold": thr, "accuracy": acc, "balanced_accuracy": bal,
        "f1": f1, "mcc": mcc, "tpr": tpr, "tnr": tnr, "fpr": fpr, "youden_j": j,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }

# --- Grid search over thresholds ---
grid = np.linspace(0, 1, 2001)  # step 0.0005
rows = [metrics_at_threshold(y_test, y_prob, t) for t in grid]
thr_df = pd.DataFrame(rows)

# Pick best thresholds by objectives
best_bal = thr_df.loc[thr_df["balanced_accuracy"].idxmax()]
best_j   = thr_df.loc[thr_df["youden_j"].idxmax()]
best_mcc = thr_df.loc[thr_df["mcc"].idxmax()]
best_f1  = thr_df.loc[thr_df["f1"].idxmax()]
best_acc = thr_df.loc[thr_df["accuracy"].idxmax()]

# Choose a primary recommendation (robust to imbalance)
BEST_PRIMARY = best_bal  # you can change to best_mcc if preferred

# Also compute metrics at default 0.5
base_05 = thr_df.iloc[(np.abs(thr_df["threshold"]-0.5)).argmin()]

# Save ROC curve with markers
fpr, tpr, roc_thr = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC AUC={auc:.3f}")
plt.plot([0,1],[0,1],'--')
# markers
def mark_on_roc(row, label):
    plt.scatter(row["fpr"], row["tpr"], s=40, label=f"{label} thr={row['threshold']:.3f}")
for lab, r in [("Best BAL", best_bal), ("Best J", best_j), ("Best MCC", best_mcc),
               ("Best F1", best_f1), ("Best ACC", best_acc), ("thr=0.5", base_05)]:
    mark_on_roc(r, lab)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression - ROC Curve (with tuned thresholds)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(CHART_DIR / "roc_with_best_thresholds.png")
plt.close()

# Save PR curve with markers (approximate positions using same thresholds)
prec, rec, pr_thr = precision_recall_curve(y_test, y_prob)
plt.figure()
plt.plot(rec, prec, label=f"AP={ap:.3f}")
def mark_on_pr(row, label):
    # find nearest recall to actual threshold prediction (approximate)
    y_pred = (y_prob >= row["threshold"]).astype(int)
    p = (y_test[y_pred==1].sum() / max((y_pred==1).sum(),1)) if (y_pred==1).sum()>0 else 0.0
    r = (y_test[y_pred==1].sum() / y_test.sum()) if y_test.sum()>0 else 0.0
    plt.scatter(r, p, s=40, label=f"{label} thr={row['threshold']:.3f}")
for lab, r in [("Best BAL", best_bal), ("Best J", best_j), ("Best MCC", best_mcc),
               ("Best F1", best_f1), ("Best ACC", best_acc), ("thr=0.5", base_05)]:
    mark_on_pr(r, lab)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Logistic Regression - PR Curve (with tuned thresholds)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(CHART_DIR / "pr_with_best_thresholds.png")
plt.close()

# Metric curves vs threshold
plt.figure()
plt.plot(thr_df["threshold"], thr_df["balanced_accuracy"], label="Balanced Accuracy")
plt.plot(thr_df["threshold"], thr_df["accuracy"], label="Accuracy")
plt.plot(thr_df["threshold"], thr_df["f1"], label="F1")
plt.plot(thr_df["threshold"], thr_df["mcc"], label="MCC")
plt.plot(thr_df["threshold"], thr_df["tpr"], label="TPR (Recall)")
plt.plot(thr_df["threshold"], thr_df["tnr"], label="TNR (Specificity)")
plt.axvline(0.5, linestyle="--")
plt.axvline(BEST_PRIMARY["threshold"], linestyle=":")
plt.xlabel("Threshold")
plt.ylabel("Metric")
plt.title("Metrics vs Threshold (Logistic Regression)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(CHART_DIR / "metrics_vs_threshold.png")
plt.close()

# Confusion matrices (default 0.5 and tuned)
def save_cm_image(row, fname, title):
    cm = np.array([[row["tn"], row["fp"]],
                   [row["fn"], row["tp"]]], dtype=int)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0,1],[0,1]); plt.yticks([0,1],[0,1])
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, int(v), ha='center', va='center')
    plt.tight_layout()
    plt.savefig(CHART_DIR / fname)
    plt.close()

save_cm_image(base_05, "logreg_confusion_matrix_thr0.5.png",
              f"Confusion Matrix (thr=0.5)")
save_cm_image(BEST_PRIMARY, "logreg_confusion_matrix_tuned.png",
              f"Confusion Matrix (tuned thr={BEST_PRIMARY['threshold']:.3f}, criterion=Balanced Acc)")

# Also keep classic ROC/PR (no markers) for continuity
plt.figure()
plt.plot(fpr, tpr, label=f"ROC AUC={auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Logistic Regression - ROC Curve")
plt.legend()
plt.tight_layout(); plt.savefig(CHART_DIR / "logreg_roc.png"); plt.close()

plt.figure()
plt.plot(rec, prec, label=f"AP={ap:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Logistic Regression - Precision-Recall Curve")
plt.legend()
plt.tight_layout(); plt.savefig(CHART_DIR / "logreg_pr.png"); plt.close()

# Coefficients (map back to columns)
ohe = clf.named_steps["preprocess"].named_transformers_["cat"]
num_names = numeric_features
cat_names = list(ohe.get_feature_names_out(categorical_features)) if categorical_features else []
feat_names = num_names + cat_names

coefs = clf.named_steps["model"].coef_.ravel()
odds = np.exp(coefs)
coef_df = pd.DataFrame({
    "feature": feat_names,
    "coef": coefs,
    "odds_ratio": odds
}).sort_values("coef", ascending=False)

# Save top +/- coefficients plot
top_n = 15
top_pos = coef_df.head(top_n)
top_neg = coef_df.tail(top_n)
plot_df = pd.concat([top_pos, top_neg])

plt.figure(figsize=(8, 0.35*len(plot_df)+2))
plt.barh(plot_df["feature"], plot_df["coef"])
plt.title("Logistic Regression Coefficients (top +/-)")
plt.tight_layout()
plt.savefig(CHART_DIR / "logreg_coefficients.png")
plt.close()

# =========================
# 4-2) Random Forest (Feature Importance)
# =========================
rf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    ))
])
rf.fit(X_train, y_train)
rf_prob = rf.predict_proba(X_test)[:,1]
rf_auc = roc_auc_score(y_test, rf_prob)

# Feature importance (preprocessed space)
rf_model = rf.named_steps["model"]
rf_importances = rf_model.feature_importances_
rf_feat_imp = pd.DataFrame({"feature": feat_names, "importance": rf_importances}) \
                .sort_values("importance", ascending=False)

plt.figure(figsize=(8, 0.3*min(20, len(rf_feat_imp))+2))
top_imp = rf_feat_imp.head(20)
plt.barh(top_imp["feature"], top_imp["importance"])
plt.title("RandomForest Feature Importance (top 20)")
plt.tight_layout()
plt.savefig(CHART_DIR / "rf_feature_importance.png")
plt.close()

# =========================
# 4-3) XGBoost (optional)
# =========================
xgb_result_text = ""
if _XGB_OK:
    xgb = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", XGBClassifier(
            n_estimators=600,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            learning_rate=0.05,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="auc",
            n_jobs=-1,
            random_state=42,
            tree_method="hist"
        ))
    ])
    xgb.fit(X_train, y_train)
    xgb_prob = xgb.predict_proba(X_test)[:,1]
    xgb_auc = roc_auc_score(y_test, xgb_prob)

    model = xgb.named_steps["model"]
    try:
        xgb_importances = model.feature_importances_
        xgb_feat_imp = pd.DataFrame({"feature": feat_names, "importance": xgb_importances}) \
                        .sort_values("importance", ascending=False)
        plt.figure(figsize=(8, 0.3*min(20, len(xgb_feat_imp))+2))
        plt.barh(xgb_feat_imp.head(20)["feature"], xgb_feat_imp.head(20)["importance"])
        plt.title("XGBoost Feature Importance (top 20)")
        plt.tight_layout()
        plt.savefig(CHART_DIR / "xgb_feature_importance.png")
        plt.close()
    except Exception as e:
        xgb_result_text += f"(warn) could not extract xgboost importances: {e}\n"
    xgb_result_text += f"XGBoost AUC: {xgb_auc:.4f}\n"
else:
    xgb_result_text = "XGBoost not installed; skipped.\n"

# =========================
# 5) Cox Proportional Hazards
# =========================
# Survival setup:
# - event = churn (1 if last activity < churn_threshold_date, else 0 [censored])
# - duration = (min(last_activity, churn_threshold_date) - created_at).days, at least 1
last_activity = users_feat[["last_purchase_date","last_login_at","last_cart_at"]].max(axis=1)
users_feat["last_activity_at"] = last_activity

event = (users_feat["last_activity_at"].fillna(pd.Timestamp.min) < churn_threshold_date).astype(int)
duration_days = (
    (pd.concat([users_feat["last_activity_at"], pd.Series([churn_threshold_date]*len(users_feat))], axis=1).min(axis=1) -
     pd.to_datetime(users_feat["created_at"])).dt.days
)
duration_days = duration_days.clip(lower=1).fillna(1)

cox_df = pd.DataFrame({
    "duration": duration_days.astype(int),
    "event": event.astype(int),
})

cox_covariates = [
    "order_count","total_spend","avg_order_value","avg_days_between_orders",
    "login_count","cart_count","avg_cart_per_login","category_diversity","age"
]
for c in cox_covariates:
    if c not in users_feat.columns:
        users_feat[c] = np.nan

cox_df = pd.concat([cox_df, users_feat[cox_covariates]], axis=1)

if "gender" in users_feat.columns:
    cox_df["gender"] = users_feat["gender"].astype("category")
    cox_df = pd.concat([cox_df, pd.get_dummies(cox_df["gender"], prefix="gender", drop_first=True)], axis=1)
    cox_df.drop(columns=["gender"], inplace=True)
if "used_coupon" in users_feat.columns:
    cox_df["used_coupon"] = users_feat["used_coupon"].astype(int)

cox_df = cox_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["duration","event"])

for c in cox_df.columns:
    if c not in ["duration","event"]:
        if cox_df[c].dtype.kind in "biufc":
            cox_df[c] = cox_df[c].fillna(cox_df[c].median())
        else:
            cox_df[c] = cox_df[c].fillna("NA")

cph = CoxPHFitter()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cph.fit(cox_df, duration_col="duration", event_col="event", show_progress=False)

cox_summary = cph.summary  # coef, exp(coef), p, etc.

hr = cox_summary.sort_values("exp(coef)")[["exp(coef)"]]
plt.figure(figsize=(8, 0.35*len(hr)+2))
plt.barh(hr.index.astype(str), hr["exp(coef)"])
plt.title("Cox Model - Hazard Ratios (exp(coef))")
plt.tight_layout()
plt.savefig(CHART_DIR / "cox_hazard_ratios.png")
plt.close()

# =========================
# 6) 결과 TXT 저장
# =========================
txt_path = OUT_DIR / "report.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(section("SETUP"))
    f.write(f"Reference date: {reference_date}\n")
    f.write(f"Churn threshold: last purchase < {churn_threshold_date} (inactive {CHURN_INACTIVE_DAYS} days)\n")
    f.write(f"Input dir: {INPUT_DIR}\n")
    f.write(f"Output dir: {OUT_DIR}\n")

    # LOGISTIC + TUNING
    f.write(section("LOGISTIC REGRESSION (Churn=1)"))
    f.write(f"AUC: {auc:.6f} | Average Precision: {ap:.6f}\n")
    f.write("\n[Threshold tuning grid: 0.000 .. 1.000, step 0.0005]\n")

    def summarize_row(name, row):
        return (f"{name} -> thr={row['threshold']:.4f} | "
                f"ACC={row['accuracy']:.4f} | BAL={row['balanced_accuracy']:.4f} | "
                f"F1={row['f1']:.4f} | MCC={row['mcc']:.4f} | "
                f"TPR={row['tpr']:.4f} | TNR={row['tnr']:.4f} | "
                f"TP={int(row['tp'])}, FP={int(row['fp'])}, TN={int(row['tn'])}, FN={int(row['fn'])}\n")

    f.write(summarize_row("Default(0.5)", base_05))
    f.write(summarize_row("Best Balanced Accuracy", best_bal))
    f.write(summarize_row("Best Youden J", best_j))
    f.write(summarize_row("Best MCC", best_mcc))
    f.write(summarize_row("Best F1", best_f1))
    f.write(summarize_row("Best Accuracy", best_acc))

    f.write("\nTop positive coefficients (push churn up):\n")
    f.write(df_to_text(top_pos[["feature","coef","odds_ratio"]]))
    f.write("\nTop negative coefficients (protect against churn):\n")
    f.write(df_to_text(top_neg[["feature","coef","odds_ratio"]]))
    f.write("\nNOTE: Coefficients are standardized where numeric features were scaled.\n")

    # RF
    f.write(section("RANDOM FOREST"))
    f.write(f"AUC: {rf_auc:.6f}\n")
    f.write("Top-20 Feature Importance:\n")
    f.write(df_to_text(top_imp))

    # XGB
    f.write(section("XGBOOST (optional)"))
    f.write(xgb_result_text)

    # COX
    f.write(section("COX PROPORTIONAL HAZARDS"))
    f.write("Model summary (first 60 rows):\n")
    f.write(df_to_text(cox_summary.head(60)))
    f.write("\nInterpretation tips:\n")
    f.write("- exp(coef) > 1 implies higher hazard (faster churn). < 1 implies protective.\n")
    f.write("- Validate proportional hazards via residuals (not included here for brevity).\n")
    f.write("- Consider time-varying covariates for more robust analysis.\n")

    f.write(section("SAVED CHARTS"))
    for p in sorted(CHART_DIR.glob("*.png")):
        f.write(f"- {p.name}\n")

print("✅ Churn Driver Analysis with Threshold Tuning is ready.")
print(f"- Output dir: {OUT_DIR.resolve()}")
