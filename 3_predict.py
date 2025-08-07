import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import joblib

# ⭐️⭐️⭐️ [수정된 부분] 함수 정의에 snapshot_date 추가 ⭐️⭐️⭐️
def create_features_from_raw(data_path, snapshot_date):
    """
    지정된 경로의 Raw 데이터를 불러와 예측에 사용할 피처 데이터프레임을 생성합니다.
    """
    print(f"### '{data_path}' 데이터에 대한 피처 생성 시작 ###")
    try:
        users = pd.read_csv(os.path.join(data_path, 'users.csv'), parse_dates=['created_at', 'birthdate'])
        orders = pd.read_csv(os.path.join(data_path, 'orders.csv'), parse_dates=['order_date'])
        order_items = pd.read_csv(os.path.join(data_path, 'order_items.csv'))
        user_interests = pd.read_csv(os.path.join(data_path, 'user_interests.csv'))
        user_logins = pd.read_csv(os.path.join(data_path, 'user_logins.csv'), parse_dates=['login_at'])
        cart_items = pd.read_csv(os.path.join(data_path, 'cart_items.csv'), parse_dates=['added_at'])
        print("8개 CSV 파일 로딩 완료.\n")
    except FileNotFoundError:
        print(f"오류: '{data_path}' 폴더에 CSV 파일이 없습니다. 경로를 확인해주세요.")
        return None
    
    # --- 피처 엔지니어링 로직 ---
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
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
    
    final_df = features_df.fillna(0)
    
    print("피처 생성 완료.\n")
    return final_df

def predict_churn(new_features_df):
    """학습된 모델과 스케일러를 불러와 새로운 데이터에 대한 이탈을 예측합니다."""
    print("--- 저장된 모델과 스케일러 로딩 ---")
    try:
        model = joblib.load('models/best_churn_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
    except FileNotFoundError:
        print("오류: 'models' 폴더에 학습된 모델 파일이 없습니다. 2_train.py를 먼저 실행했는지 확인해주세요.")
        return None

    user_ids = new_features_df['user_id']
    X_new = new_features_df.drop('user_id', axis=1)
    train_columns = ['age', 'tenure_days', 'num_interests', 'recency_score', 'frequency_score', 'monetary_score', 'monetary_avg_order', 'avg_items_per_order', 'frequency_last_30d', 'frequency_last_90d', 'days_between_orders', 'coupon_usage_rate', 'days_since_last_session', 'cart_additions_last_30d', 'gender_MALE']
    X_new = pd.get_dummies(X_new, columns=['gender'], drop_first=True, dtype=int)
    X_new = X_new.reindex(columns=train_columns, fill_value=0)
    X_new_scaled = scaler.transform(X_new)

    print("--- 예측 수행 시작 ---")
    predictions = model.predict(X_new_scaled)
    probabilities = model.predict_proba(X_new_scaled)[:, 1]

    result_df = pd.DataFrame({'user_id': user_ids})
    result_df['prediction_date'] = datetime.now().strftime('%Y-%m-%d')
    result_df['churn_prediction'] = predictions
    result_df['churn_probability'] = np.round(probabilities, 4)
    
    return result_df

# --- 스크립트 실행 ---
if __name__ == '__main__':
    raw_data_path_for_prediction = 'data/raw_after'
    future_snapshot_date = datetime.now() + timedelta(days=60)
    new_features_df = create_features_from_raw(data_path=raw_data_path_for_prediction, snapshot_date=future_snapshot_date)

    if new_features_df is not None:
        predictions_df = predict_churn(new_features_df)
        if predictions_df is not None:
            output_dir = 'predictions'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            output_filename = os.path.join(output_dir, f'predictions_{datetime.now().strftime("%Y%m%d")}.csv')
            predictions_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
            
            print(f"\n--- 예측 결과가 '{output_filename}' 파일로 저장되었습니다. ---")
            print(predictions_df.head(10))
            
            print("\n--- 이탈 확률 높은 고객 TOP 10 ---")
            high_risk_users = predictions_df.sort_values(by='churn_probability', ascending=False).head(10)
            print(high_risk_users)