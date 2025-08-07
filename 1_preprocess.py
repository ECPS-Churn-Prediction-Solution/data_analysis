import pandas as pd
import numpy as np
from datetime import datetime
import os

def create_feature_dataset(data_path, is_s3=False):
    """
    지정된 경로(로컬 또는 S3)에서 Raw 데이터를 불러와
    모델 학습용 피처 데이터프레임을 생성하는 함수.
    """
    print("### 피처 엔지니어링 시작 ###")
    
    # --- 1. 데이터 로딩 ---
    # is_s3 플래그는 현재 사용되지 않지만, S3 연동 시 확장성을 위해 남겨둠
    try:
        print(f"지정된 경로에서 데이터 로딩 중...: {data_path}")
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
    except Exception as e:
        print(f"오류: 파일을 불러오는 데 실패했습니다. S3 경로인 경우 인증 정보를 확인해주세요.")
        print(e)
        return None

    # --- 2. 기준 시점(Snapshot Date) 설정 ---
    snapshot_date = datetime.now()
    print(f"기준 시점: {snapshot_date.strftime('%Y-%m-%d')}\n")

    # --- 3. 피처 엔지니어링 (로직은 이전과 동일) ---
    features_df = users[['user_id', 'gender']].copy()
    
    print("처리 중: 기본 정보 피처...")
    features_df['age'] = ((snapshot_date - users['birthdate']).dt.days / 365).astype(int)
    features_df['tenure_days'] = (snapshot_date - users['created_at']).dt.days
    interests_count = user_interests.groupby('user_id')['category_id'].count().rename('num_interests')
    features_df = features_df.merge(interests_count, on='user_id', how='left')

    print("처리 중: RFM 피처...")
    completed_orders = orders[orders['status'] == 'COMPLETED'].copy()
    if not completed_orders.empty:
        rfm_df = completed_orders.groupby('user_id').agg(
            recency_days=('order_date', lambda d: (snapshot_date - d.max()).days),
            frequency_total=('order_id', 'count'),
            monetary_total=('total_amount', 'sum')
        ).reset_index()
        rfm_df['recency_score'] = pd.qcut(rfm_df['recency_days'], 5, labels=range(5, 0, -1), duplicates='drop').astype(int)
        rfm_df['frequency_score'] = pd.qcut(rfm_df['frequency_total'].rank(method='first'), 5, labels=range(1, 6), duplicates='drop').astype(int)
        rfm_df['monetary_score'] = pd.qcut(rfm_df['monetary_total'], 5, labels=range(1, 6), duplicates='drop').astype(int)
        features_df = features_df.merge(rfm_df[['user_id', 'recency_score', 'frequency_score', 'monetary_score']], on='user_id', how='left')
    
        print("처리 중: 구매 행동 피처...")
        purchase_behavior = rfm_df[['user_id', 'monetary_total', 'frequency_total']].copy()
        purchase_behavior['monetary_avg_order'] = purchase_behavior['monetary_total'] / purchase_behavior['frequency_total']
        items_per_order = order_items.groupby('order_id')['quantity'].sum().rename('items_in_order').to_frame().merge(orders[['order_id', 'user_id']], on='order_id')
        avg_items = items_per_order.groupby('user_id')['items_in_order'].mean().rename('avg_items_per_order')
        purchase_behavior = purchase_behavior.merge(avg_items, on='user_id', how='left')
        features_df = features_df.merge(purchase_behavior[['user_id', 'monetary_avg_order', 'avg_items_per_order']], on='user_id', how='left')
    
    print("처리 중: 추가 활동 피처...")
    last_30d, last_90d = snapshot_date - pd.to_timedelta(30, 'd'), snapshot_date - pd.to_timedelta(90, 'd')
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
    
    print("처리 중: 타겟 변수(churn) 생성...")
    last_purchase = completed_orders.groupby('user_id')['order_date'].max().rename('last_purchase_date')
    features_df = features_df.merge(last_purchase, on='user_id', how='left')
    recency_days = (snapshot_date - features_df['last_purchase_date']).dt.days
    churn_condition = (features_df['days_since_last_session'] > 180) | (recency_days > 365) | (features_df['days_since_last_session'].isnull())
    features_df['churn'] = np.where(churn_condition, 1, 0)
    
    print("\n### 최종 데이터 정리 중... ###")
    final_feature_columns = ['user_id', 'age', 'gender', 'tenure_days', 'num_interests', 'recency_score', 'frequency_score', 'monetary_score', 'monetary_avg_order', 'avg_items_per_order', 'frequency_last_30d', 'frequency_last_90d', 'days_between_orders', 'coupon_usage_rate', 'days_since_last_session', 'cart_additions_last_30d', 'churn']
    final_df = features_df.reindex(columns=final_feature_columns).fillna(0)
    print("피처 엔지니어링 완료!")
    return final_df

# --- 스크립트 실행 ---
if __name__ == '__main__':
    # --- 데이터 소스 경로 설정 ---

    # ⭐️ 1. 현재 사용할 로컬 경로
    # Windows 경로의 \는 r'' 또는 \\ 로 처리해야 합니다.
    input_data_path = r'C:\Users\SYU\Desktop\ECPS\raw_data_1'

    # --- (참고) S3에서 데이터를 불러올 경우 ---
    # 1. pip install boto3 s3fs 실행
    # 2. AWS 자격증명(credentials) 설정
    # 3. 아래 주석을 해제하고 자신의 S3 경로로 변경하여 사용
    # s3_path = 's3://your-bucket-name/your-folder-name/'
    # input_data_path = s3_path
    # ---------------------------------------------

    # 함수 실행
    features_df = create_feature_dataset(data_path=input_data_path)

    if features_df is not None:
        # 최종 결과 확인
        print("\n--- 최종 피처 데이터셋 (상위 5개) ---")
        print(features_df.head())
        print("\n--- 데이터 정보 요약 ---")
        features_df.info()

        # 결과 파일 저장 경로 설정
        output_dir = r'C:\Users\SYU\Desktop\ECPS\churn_feature_1'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"'{output_dir}' 폴더를 새로 생성했습니다.")
        output_filename = os.path.join(output_dir, 'churn_features.csv')
        
        # 지정된 경로에 파일 저장
        features_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"\n최종 피처 데이터가 '{output_filename}' 경로에 저장되었습니다.")