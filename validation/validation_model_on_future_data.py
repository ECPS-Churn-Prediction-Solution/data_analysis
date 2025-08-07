import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- 폰트 설정 (Windows/macOS/Linux 자동 감지) ---
import platform
system_name = platform.system()
if system_name == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif system_name == 'Darwin': # macOS
    plt.rc('font', family='AppleGothic')
else: # Linux, Colab
    # 나눔 폰트가 설치되어 있어야 합니다.
    # Colab: !sudo apt-get install -y fonts-nanum
    plt.rc('font', family='NanumBarunGothic')
plt.rcParams['axes.unicode_minus'] = False


def create_feature_dataset(data_path, snapshot_date):
    """지정된 경로의 Raw 데이터를 불러와 피처 데이터프레임을 생성하는 공통 함수."""
    try:
        users = pd.read_csv(os.path.join(data_path, 'users.csv'), parse_dates=['created_at', 'birthdate'])
        orders = pd.read_csv(os.path.join(data_path, 'orders.csv'), parse_dates=['order_date'])
        order_items = pd.read_csv(os.path.join(data_path, 'order_items.csv'))
        user_interests = pd.read_csv(os.path.join(data_path, 'user_interests.csv'))
        user_logins = pd.read_csv(os.path.join(data_path, 'user_logins.csv'), parse_dates=['login_at'])
        cart_items = pd.read_csv(os.path.join(data_path, 'cart_items.csv'), parse_dates=['added_at'])
    except FileNotFoundError:
        print(f"오류: '{data_path}' 경로에 CSV 파일이 없습니다. 경로를 확인하세요.")
        return None
    
    features_df = users[['user_id', 'gender']].copy()
    features_df['age'] = ((snapshot_date - users['birthdate']).dt.days / 365).astype(int)
    features_df['tenure_days'] = (snapshot_date - users['created_at']).dt.days
    interests_count = user_interests.groupby('user_id')['category_id'].count().rename('num_interests')
    features_df = features_df.merge(interests_count, on='user_id', how='left')
    
    completed_orders = orders[orders['status'] == 'COMPLETED'].copy()
    if not completed_orders.empty:
        rfm_df = completed_orders.groupby('user_id').agg(
            recency_days=('order_date', lambda d: (snapshot_date - d.max()).days),
            frequency_total=('order_id', 'count'),
            monetary_total=('total_amount', 'sum')
        ).reset_index()
        recency_bins = [-1, 7, 14, 30, 60, np.inf]
        recency_labels = [5, 4, 3, 2, 1]
        rfm_df['recency_score'] = pd.cut(rfm_df['recency_days'], bins=recency_bins, labels=recency_labels)
        frequency_bins = [0, 1, 3, 6, 9, np.inf]
        frequency_labels = [1, 2, 3, 4, 5]
        rfm_df['frequency_score'] = pd.cut(rfm_df['frequency_total'], bins=frequency_bins, labels=frequency_labels)
        monetary_bins = [-1, 50000, 150000, 300000, 500000, np.inf]
        monetary_labels = [1, 2, 3, 4, 5]
        rfm_df['monetary_score'] = pd.cut(rfm_df['monetary_total'], bins=monetary_bins, labels=monetary_labels)
        features_df = features_df.merge(rfm_df[['user_id', 'recency_score', 'frequency_score', 'monetary_score']], on='user_id', how='left')
        purchase_behavior = rfm_df[['user_id', 'monetary_total', 'frequency_total']].copy()
        purchase_behavior['monetary_avg_order'] = purchase_behavior['monetary_total'] / purchase_behavior['frequency_total']
        items_per_order = order_items.groupby('order_id')['quantity'].sum().rename('items_in_order').to_frame().merge(orders[['order_id', 'user_id']], on='order_id')
        avg_items = items_per_order.groupby('user_id')['items_in_order'].mean().rename('avg_items_per_order')
        purchase_behavior = purchase_behavior.merge(avg_items, on='user_id', how='left')
        features_df = features_df.merge(purchase_behavior[['user_id', 'monetary_avg_order', 'avg_items_per_order']], on='user_id', how='left')

    last_30d = snapshot_date - pd.to_timedelta(30, 'd')
    last_90d = snapshot_date - pd.to_timedelta(90, 'd')
    freq_30d = completed_orders[completed_orders['order_date'] >= last_30d].groupby('user_id')['order_id'].count().rename('frequency_last_30d')
    features_df = features_df.merge(freq_30d, on='user_id', how='left')
    freq_90d = completed_orders[completed_orders['order_date'] >= last_90d].groupby('user_id')['order_id'].count().rename('frequency_last_90d')
    features_df = features_df.merge(freq_90d, on='user_id', how='left')
    order_dates = completed_orders[['user_id', 'order_date']].sort_values(['user_id', 'order_date'])
    order_dates['days_between_orders'] = order_dates.groupby('user_id')['order_date'].diff().dt.days
    avg_days_between = order_dates.groupby('user_id')['days_between_orders'].mean().rename('days_between_orders')
    features_df = features_df.merge(avg_days_between, on='user_id', how='left')
    merged_items = pd.merge(order_items, completed_orders[['order_id', 'user_id', 'used_coupon_code']], on='order_id')
    total_quantity = merged_items.groupby('user_id')['quantity'].sum()
    coupon_quantity = merged_items[merged_items['used_coupon_code'].notna()].groupby('user_id')['quantity'].sum()
    coupon_usage_rate = (coupon_quantity / total_quantity).rename('coupon_usage_rate')
    features_df = features_df.merge(coupon_usage_rate, on='user_id', how='left')
    last_session = user_logins.groupby('user_id')['login_at'].max().rename('last_session_date')
    features_df = features_df.merge(last_session, on='user_id', how='left')
    features_df['days_since_last_session'] = (snapshot_date - features_df['last_session_date']).dt.days
    cart_30d = cart_items[cart_items['added_at'] >= last_30d].groupby('user_id')['cart_item_id'].count().rename('cart_additions_last_30d')
    features_df = features_df.merge(cart_30d, on='user_id', how='left')
    
    score_cols = ['recency_score', 'frequency_score', 'monetary_score']
    for col in score_cols:
        if col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
            
    final_df = features_df.fillna(0)
    return final_df


def get_actual_churn(data_path, snapshot_date):
    """지정된 경로의 Raw 데이터에 대해 특정 시점의 실제 이탈 여부(정답)를 계산합니다."""
    users = pd.read_csv(os.path.join(data_path, 'users.csv'), parse_dates=['created_at'])
    orders = pd.read_csv(os.path.join(data_path, 'orders.csv'), parse_dates=['order_date'])
    user_logins = pd.read_csv(os.path.join(data_path, 'user_logins.csv'), parse_dates=['login_at'])
    actuals_df = users[['user_id']].copy()
    completed_orders = orders[orders['status'] == 'COMPLETED']
    last_purchase = completed_orders.groupby('user_id')['order_date'].max().rename('last_purchase_date')
    last_session = user_logins.groupby('user_id')['login_at'].max().rename('last_session_date')
    actuals_df = actuals_df.merge(last_purchase, on='user_id', how='left').merge(last_session, on='user_id', how='left')
    recency_days = (snapshot_date - actuals_df['last_purchase_date']).dt.days
    session_days = (snapshot_date - actuals_df['last_session_date']).dt.days
    churn_condition = (session_days > 180) | (recency_days > 365) | (session_days.isnull())
    actuals_df['actual_churn'] = np.where(churn_condition, 1, 0)
    return actuals_df[['user_id', 'actual_churn']]


if __name__ == '__main__':
    # --- 경로 및 시점 정의 ---
    current_raw_path = '../data/raw_current'
    future_raw_path = '../data/raw_after_year'
    
    # 1. 현재 데이터로 모델 학습
    print("--- 1. 현재 데이터로 모델 학습 시작 ---")
    train_snapshot_date = datetime(2025, 8, 1)
    train_features_df = create_feature_dataset(current_raw_path, train_snapshot_date)
    
    if train_features_df is not None:
        train_actuals = get_actual_churn(current_raw_path, train_snapshot_date)
        train_df = pd.merge(train_features_df, train_actuals, on='user_id')

        feature_cols = ['age', 'gender', 'tenure_days', 'num_interests', 'recency_score', 
                        'frequency_score', 'monetary_score', 'monetary_avg_order', 'avg_items_per_order', 
                        'frequency_last_30d', 'frequency_last_90d', 'days_between_orders', 
                        'coupon_usage_rate', 'days_since_last_session', 'cart_additions_last_30d']
        
        X = train_df[feature_cols]
        y = train_df['actual_churn']
        
        X = pd.get_dummies(X, columns=['gender'], drop_first=True, dtype=int)
        
        train_columns = X.columns.tolist()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        scale_pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1
        model = LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight, learning_rate=0.05, max_depth=5, n_estimators=100)
        model.fit(X_scaled, y)
        print("모델 학습 완료.\n")

        # 2. 1년 후 데이터 준비
        print("--- 2. 1년 후 데이터 준비 시작 ---")
        predict_snapshot_date = datetime(2026, 9, 1)
        future_features_df = create_feature_dataset(future_raw_path, predict_snapshot_date)
        
        if future_features_df is not None:
            ground_truth_date = predict_snapshot_date + timedelta(days=180)
            future_actuals_df = get_actual_churn(future_raw_path, ground_truth_date)
            validation_df = pd.merge(future_features_df, future_actuals_df, on='user_id')
            print("1년 후 데이터 준비 완료.\n")
            
            # 3. 1년 후 데이터로 예측 수행
            print("--- 3. 1년 후 데이터로 예측 수행 시작 ---")
            X_future = validation_df.drop(['user_id', 'actual_churn'], axis=1)
            y_future_true = validation_df['actual_churn']
            X_future = pd.get_dummies(X_future, columns=['gender'], drop_first=True, dtype=int)
            X_future = X_future.reindex(columns=train_columns, fill_value=0)
            
            X_future_scaled = scaler.transform(X_future)
            
            y_future_pred = model.predict(X_future_scaled)
            y_future_proba = model.predict_proba(X_future_scaled)[:, 1]
            print("예측 완료.\n")
            
            # 4. Confusion Matrix 및 성능 분석 결과 저장
            print("--- 4. 1년 후 데이터에 대한 모델 성능 검증 ---")
            cm = confusion_matrix(y_future_true, y_future_pred)
            report_str = classification_report(y_future_true, y_future_pred, zero_division=0)
            roc_auc = roc_auc_score(y_future_true, y_future_proba)
            
            results_dir = 'results'
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['비이탈(예측)', '이탈(예측)'], yticklabels=['비이탈(실제)', '이탈(실제)'])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('1년 후 데이터에 대한 Confusion Matrix')
            
            cm_filename = os.path.join(results_dir, 'confusion_matrix_future_validation.png')
            plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"\nConfusion Matrix 이미지가 '{cm_filename}' 경로에 저장되었습니다.")

            summary_text = f"""# =======================================================
# 모델 성능 검증 보고서 (1년 후 데이터 대상)
# =======================================================

# 1. 검증 개요
# -------------------------------------------------------
# - 학습 데이터: '{current_raw_path}'
# - 검증 데이터: '{future_raw_path}'
# - 검증 시점: {ground_truth_date.strftime('%Y-%m-%d')}


# 2. Confusion Matrix 요약
# -------------------------------------------------------
# - True Negative (TN): {cm[0][0]}  (정상 -> 정상)
# - False Positive (FP): {cm[0][1]} (정상 -> 이탈) *Type I Error
# - False Negative (FN): {cm[1][0]} (이탈 -> 정상) *Type II Error
# - True Positive (TP): {cm[1][1]}  (이탈 -> 이탈)


# 3. 성능 지표 (Classification Report)
# -------------------------------------------------------
{report_str}

# 4. ROC AUC Score
# -------------------------------------------------------
# - ROC AUC: {roc_auc:.4f}
# - 해석: 모델이 이탈 고객과 비이탈 고객을 얼마나 잘 구별해내는지를 나타내는 종합적인 성능 지표. 1에 가까울수록 우수함.
"""
            summary_filename = os.path.join(results_dir, 'validation_summary.txt')
            with open(summary_filename, 'w', encoding='utf-8') as f:
                f.write(summary_text)

            print(f"성능 분석 리포트가 '{summary_filename}' 경로에 저장되었습니다.")