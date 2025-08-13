# -*- coding: utf-8 -*-
r"""
Churn Prediction Training Script (for churn_features.csv)

- 입력 CSV 스키마(열 이름)는 사용자 제공 더미데이터에 맞춤
- 결과물은 ./model_output 아래에 저장
- 실행 예시:
    python train_model.py                             # ./data/processed/churn_features.csv 읽음
    python train_model.py path/to/your.csv           # 경로 지정
    type churn_features.csv | python train_model.py -  # 표준입력
"""

# ---- Matplotlib: 비대화형 백엔드 강제 (Tk/Tcl 오류 방지) ------------------
import os as _os
_os.environ["MPLBACKEND"] = "Agg"
import matplotlib as _mpl
_mpl.use("Agg")
# -------------------------------------------------------------------------

import sys
import io
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, matthews_corrcoef, balanced_accuracy_score,
    precision_recall_curve, roc_curve, brier_score_loss
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import warnings

# 선택 라이브러리 (없으면 자동 skip)
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    warnings.warn("xgboost not available; skipping.")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
    warnings.warn("lightgbm not available; skipping.")


# =========================
# 유틸
# =========================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_input_csv(path_or_stdin: str) -> pd.DataFrame:
    """
    - 파일 경로를 주면 해당 경로에서 읽음
    - "-" 를 주면 stdin 에서 읽음
    - 인자가 없으면 ./data/processed/churn_features.csv 를 시도
    """
    if path_or_stdin == "-":
        data = sys.stdin.read()
        return pd.read_csv(io.StringIO(data))
    p = Path(path_or_stdin) if path_or_stdin else Path("data/processed/churn_features.csv")
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p.resolve()}")
    return pd.read_csv(p)


def binarize_gender(x: str) -> int:
    # 남성=1, 여성=0 (다른 값은 0으로 처리)
    if isinstance(x, str) and x.upper().strip() == "MALE":
        return 1
    return 0


def safe_float(s):
    try:
        return float(s)
    except Exception:
        return np.nan


def describe_split(y_train, y_valid):
    return {
        "train_size": int(len(y_train)),
        "valid_size": int(len(y_valid)),
        "train_churn_rate": float(np.mean(y_train)),
        "valid_churn_rate": float(np.mean(y_valid)),
    }


def compute_metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    tpr = recall_score(y_true, y_pred)  # 재현율 = TPR
    tnr = tn / (tn + fp) if (tn + fp) else np.nan
    prc = precision_score(y_true, y_pred) if (tp + fp) else 0.0
    roc_auc = roc_auc_score(y_true, y_prob)
    pred_churn_rate = float(np.mean(y_pred))
    actual_churn_rate = float(np.mean(y_true))
    rate_diff_pp = (pred_churn_rate - actual_churn_rate) * 100  # percentage points
    return dict(
        ACC=acc, BAL=bal, F1=f1, MCC=mcc, TPR=tpr, TNR=tnr, PREC=prc,
        ROC_AUC=roc_auc, pred_rate=pred_churn_rate, actual_rate=actual_churn_rate,
        rate_diff_pp=rate_diff_pp, TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn),
        thr=float(thr)
    )


def sweep_thresholds(y_true, y_prob, out_png: Path):
    thresholds = np.unique(np.clip(np.linspace(0.05, 0.95, 181), 0, 1))
    rows = []
    for thr in thresholds:
        rows.append(compute_metrics(y_true, y_prob, thr=thr))
    df = pd.DataFrame(rows)

    # 주요 포인트 선택
    idx_acc = df["ACC"].idxmax()
    idx_mcc = df["MCC"].idxmax()
    idx_bal = df["BAL"].idxmax()
    base_idx = (np.abs(df["thr"] - 0.5)).idxmin()

    # 그래프 저장
    fig = plt.figure(figsize=(9, 6))
    plt.plot(df["thr"], df["ACC"], label="ACC")
    plt.plot(df["thr"], df["BAL"], label="BAL")
    plt.plot(df["thr"], df["F1"], label="F1")
    plt.plot(df["thr"], df["MCC"], label="MCC")
    plt.plot(df["thr"], df["TPR"], label="TPR")
    plt.plot(df["thr"], df["TNR"], label="TNR")
    for name, idx in [("ACC max", idx_acc), ("MCC max", idx_mcc), ("BAL max", idx_bal), ("thr=0.5", base_idx)]:
        plt.scatter([df.loc[idx, "thr"]], [df.loc[idx, "ACC"]], marker="o")
        plt.text(df.loc[idx, "thr"], df.loc[idx, "ACC"] + 0.005, name, fontsize=8)
    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.title("Metrics vs. Threshold (valid)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    # 표로도 반환
    summary = pd.DataFrame({
        "기준": ["기본(0.5)", "Balanced Acc 최대", "MCC 최대", "Accuracy 최대"],
        "임계값": [df.loc[base_idx, "thr"], df.loc[idx_bal, "thr"], df.loc[idx_mcc, "thr"], df.loc[idx_acc, "thr"]],
        "ACC": [df.loc[base_idx, "ACC"], df.loc[idx_bal, "ACC"], df.loc[idx_mcc, "ACC"], df.loc[idx_acc, "ACC"]],
        "BAL": [df.loc[base_idx, "BAL"], df.loc[idx_bal, "BAL"], df.loc[idx_mcc, "BAL"], df.loc[idx_acc, "BAL"]],
        "F1":  [df.loc[base_idx, "F1"],  df.loc[idx_bal, "F1"],  df.loc[idx_mcc, "F1"],  df.loc[idx_acc, "F1"]],
        "MCC": [df.loc[base_idx, "MCC"], df.loc[idx_bal, "MCC"], df.loc[idx_mcc, "MCC"], df.loc[idx_acc, "MCC"]],
        "TPR(재현율)": [df.loc[base_idx, "TPR"], df.loc[idx_bal, "TPR"], df.loc[idx_mcc, "TPR"], df.loc[idx_acc, "TPR"]],
        "TNR(특이도)": [df.loc[base_idx, "TNR"], df.loc[idx_bal, "TNR"], df.loc[idx_mcc, "TNR"], df.loc[idx_acc, "TNR"]],
        "예측 이탈률": [df.loc[base_idx, "pred_rate"], df.loc[idx_bal, "pred_rate"], df.loc[idx_mcc, "pred_rate"], df.loc[idx_acc, "pred_rate"]],
        "실제 이탈률": [df.loc[base_idx, "actual_rate"]]*4,
        "편차(pp)":    [df.loc[base_idx, "rate_diff_pp"], df.loc[idx_bal, "rate_diff_pp"], df.loc[idx_mcc, "rate_diff_pp"], df.loc[idx_acc, "rate_diff_pp"]],
        "TP":  [df.loc[base_idx, "TP"], df.loc[idx_bal, "TP"], df.loc[idx_mcc, "TP"], df.loc[idx_acc, "TP"]],
        "FP":  [df.loc[base_idx, "FP"], df.loc[idx_bal, "FP"], df.loc[idx_mcc, "FP"], df.loc[idx_acc, "FP"]],
        "TN":  [df.loc[base_idx, "TN"], df.loc[idx_bal, "TN"], df.loc[idx_mcc, "TN"], df.loc[idx_acc, "TN"]],
        "FN":  [df.loc[base_idx, "FN"], df.loc[idx_bal, "FN"], df.loc[idx_mcc, "FN"], df.loc[idx_acc, "FN"]],
    })
    for col in ["ACC", "BAL", "F1", "MCC", "TPR(재현율)", "TNR(특이도)"]:
        summary[col] = summary[col].map(lambda v: round(float(v), 3))
    summary["임계값"] = summary["임계값"].map(lambda v: round(float(v), 4))
    summary["예측 이탈률"] = summary["예측 이탈률"].map(lambda v: f"{v*100:.2f}%")
    summary["실제 이탈률"] = summary["실제 이탈률"].map(lambda v: f"{v*100:.2f}%")
    summary["편차(pp)"] = summary["편차(pp)"].map(lambda v: f"{v:+.2f}pp")
    return df, summary


def plot_roc_pr(y_true, y_prob, out_dir: Path, model_name: str):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    fig = plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", alpha=0.5)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC - {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"roc_{model_name}.png", dpi=140)
    plt.close(fig)

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig = plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve - {model_name}")
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"pr_{model_name}.png", dpi=140)
    plt.close(fig)


def reliability_plot(y_true, y_prob, out_png: Path, n_bins: int = 10):
    """간단한 calibration curve + Brier score"""
    brier = brier_score_loss(y_true, y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    prob_mean, frac_pos = [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            prob_mean.append(np.nan)
            frac_pos.append(np.nan)
        else:
            prob_mean.append(np.mean(y_prob[mask]))
            frac_pos.append(np.mean(y_true[mask]))
    fig = plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "--", alpha=0.4, label="Perfect")
    plt.scatter(prob_mean, frac_pos, s=40)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed positive fraction")
    plt.title(f"Reliability (Brier={brier:.4f})")
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return brier


def segment_mask(df: pd.DataFrame):
    """우리 타겟 코호트: 20~34세 & 남성"""
    return (df["age"].between(20, 34)) & (df["gender_bin"] == 1)


# =========================
# 메인
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv",
        nargs="?",
        default="data/processed/churn_features.csv",
        help="입력 CSV 경로 (기본: data/processed/churn_features.csv). 표준입력은 '-'"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="model_output")
    args = parser.parse_args()

    OUT_DIR = Path(args.out)
    ensure_dir(OUT_DIR)

    # 1) 데이터 로드
    df = read_input_csv(args.csv)

    # 2) 컬럼 정리/형 변환
    required = [
        "user_id","age","gender","tenure_days","num_interests",
        "recency_score","frequency_score","monetary_score","monetary_avg_order",
        "avg_items_per_order","frequency_last_30d","frequency_last_90d",
        "days_between_orders","coupon_usage_rate","days_since_last_session",
        "cart_additions_last_30d","churn"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    # 성/타깃 라벨
    df["gender_bin"] = df["gender"].map(binarize_gender).astype(int)
    df["churn"] = df["churn"].astype(int)

    # 숫자형으로 강제 변환
    numeric_cols = [
        "age","tenure_days","num_interests","recency_score","frequency_score",
        "monetary_score","monetary_avg_order","avg_items_per_order",
        "frequency_last_30d","frequency_last_90d","days_between_orders",
        "coupon_usage_rate","days_since_last_session","cart_additions_last_30d",
        "gender_bin"
    ]
    for c in numeric_cols:
        df[c] = df[c].apply(safe_float)

    # 이상치/결측 간단 처리
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    # 3) 학습/검증 분리 (유저 기준 랜덤 분리)
    X = df[numeric_cols]
    y = df["churn"].values.astype(int)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.25, random_state=args.seed, stratify=y
    )

    # 4) 파이프라인 정의
    # 로지스틱: 스케일링 + class_weight balanced
    logit_pipe = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            solver="lbfgs", max_iter=2000, class_weight="balanced", n_jobs=None
        ))
    ])

    # 랜덤포레스트
    rf_clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=-1
    )

    models = {
        "logistic": logit_pipe,
        "rf": rf_clf
    }
    if HAS_XGB:
        models["xgb"] = XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=args.seed,
            tree_method="hist", eval_metric="logloss"
        )
    if HAS_LGBM:
        models["lgbm"] = LGBMClassifier(
            n_estimators=800, max_depth=-1, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=args.seed
        )

    # 5) 학습 + 평가
    meta_report = {}
    thr_tables = {}
    for name, model in models.items():
        print(f"[Train] {name} ...")
        model.fit(X_train, y_train)

        # 예측 확률
        if hasattr(model, "predict_proba"):
            p_valid = model.predict_proba(X_valid)[:, 1]
        elif hasattr(model, "decision_function"):
            z = model.decision_function(X_valid)
            p_valid = 1 / (1 + np.exp(-z))
        else:
            p_valid = model.predict(X_valid).astype(float)

        # 기본 메트릭 (thr=0.5)
        base = compute_metrics(y_valid, p_valid, thr=0.5)

        # 임계값 스윕
        _, summary = sweep_thresholds(y_valid, p_valid, OUT_DIR / f"thr_sweep_{name}.png")
        thr_tables[name] = summary

        # ROC/PR
        plot_roc_pr(y_valid, p_valid, OUT_DIR, name)
        # Calibration
        brier = reliability_plot(y_valid, p_valid, OUT_DIR / f"reliability_{name}.png")

        # 피쳐 중요도 (가능한 모델)
        fi_path = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            fi = pd.DataFrame({
                "feature": X.columns,
                "importance": importances
            }).sort_values("importance", ascending=False)
            fi_path = OUT_DIR / f"feature_importance_{name}.csv"
            fi.to_csv(fi_path, index=False)

            # 그래프
            top = fi.head(20)
            fig = plt.figure(figsize=(8, 6))
            y_pos = np.arange(len(top))
            plt.barh(y_pos, top["importance"].values)
            plt.yticks(y_pos, top["feature"].values)
            plt.gca().invert_yaxis()
            plt.title(f"Feature Importance - {name}")
            plt.xlabel("Importance")
            fig.tight_layout()
            fig.savefig(OUT_DIR / f"feature_importance_{name}.png", dpi=140)
            plt.close(fig)

        # 세그먼트(20~34, MALE) 전용 메트릭
        seg_mask_valid = (X_valid["age"].between(20, 34)) & (X_valid["gender_bin"] == 1)
        if seg_mask_valid.sum() > 0:
            seg_base = compute_metrics(y_valid[seg_mask_valid], p_valid[seg_mask_valid], thr=0.5)
        else:
            seg_base = None

        meta_report[name] = {
            "base@0.5": base,
            "brier": float(brier),
            "split": describe_split(y_train, y_valid),
            "feature_importance_csv": str(fi_path) if fi_path else None,
        }
        if seg_base:
            meta_report[name]["segment_20to34_male@0.5"] = seg_base

    # 6) 모델 비교 테이블 (ROC-AUC / F1@0.5 / ACC@0.5 등)
    rows = []
    for name, info in meta_report.items():
        base = info["base@0.5"]
        rows.append({
            "model": name,
            "ROC_AUC": base["ROC_AUC"],
            "ACC@0.5": base["ACC"],
            "F1@0.5": base["F1"],
            "TPR@0.5": base["TPR"],
            "TNR@0.5": base["TNR"],
            "pred_rate@0.5": base["pred_rate"],
            "actual_rate": base["actual_rate"],
            "Brier": info["brier"],
        })
    compare_df = pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False)
    compare_df_path = OUT_DIR / "model_compare.csv"
    compare_df.to_csv(compare_df_path, index=False)

    # 7) 임계값 요약 표 저장
    for name, table in thr_tables.items():
        table.to_csv(OUT_DIR / f"threshold_summary_{name}.csv", index=False)

    # 8) report.txt 작성
    rep_lines = []
    rep_lines.append("CHURN MODEL – TRAINING REPORT")
    rep_lines.append("=" * 72)
    rep_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rep_lines.append("")
    rep_lines.append("INPUT SUMMARY")
    rep_lines.append("-" * 72)
    rep_lines.append(f"Rows: {len(df):,}")
    rep_lines.append(f"Churn rate (overall): {np.mean(df['churn']):.4f}")
    rep_lines.append(f"Columns: {list(df.columns)}")
    rep_lines.append("")
    rep_lines.append("SPLIT")
    rep_lines.append("-" * 72)
    rep_lines.append(json.dumps(describe_split(y_train, y_valid), indent=2))
    rep_lines.append("")
    rep_lines.append("MODEL COMPARISON (sorted by ROC_AUC)")
    rep_lines.append("-" * 72)
    rep_lines.append(compare_df.to_string(index=False))
    rep_lines.append("")
    rep_lines.append("THRESHOLD RECOMMENDATIONS (per model)")
    rep_lines.append("-" * 72)
    rep_lines.append("기준: 기본(0.5), Accuracy 최대, MCC 최대, Balanced Accuracy 최대")
    for name, table in thr_tables.items():
        rep_lines.append("")
        rep_lines.append(f"[{name}]")
        rep_lines.append(table.to_string(index=False))
    rep_lines.append("")
    rep_lines.append("NOTES")
    rep_lines.append("-" * 72)
    rep_lines.append("• 이탈률(%) 자체를 가장 정확히 추정하려면: 확률 캘리브레이션 + 평균확률 사용 권장.")
    rep_lines.append("• 운영 분류 정확도를 중시하면 MCC 최대 임계값 사용을 권장.")
    rep_lines.append("• 세그먼트(20~34, 남성) 성능도 report에 포함. 더 세분화하면 캠페인 타겟팅 향상.")
    rep_lines.append("• XGBoost/LightGBM은 설치되어 있으면 자동 사용, 없으면 자동 건너뜀.")
    rep = "\n".join(rep_lines)
    report_path = OUT_DIR / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(rep)

    print("\n[완료] 결과가 저장되었습니다 ->", OUT_DIR.resolve())
    print(" - model_compare.csv")
    for name in models.keys():
        print(f" - roc_{name}.png / pr_{name}.png / reliability_{name}.png / threshold_summary_{name}.csv")
    print(" - report.txt")
    print("\n요약:")
    print(compare_df.to_string(index=False))


if __name__ == "__main__":
    main()
