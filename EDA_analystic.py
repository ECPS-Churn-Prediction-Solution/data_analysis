import pandas as pd
import numpy as np

# 데이터 샘플 수 지정
n_samples = 1000

# 재현성을 위한 시드 고정
np.random.seed(42)

# 1. 기본 인구통계 및 관심사 피처 생성 (비즈니스 상황 반영)
# 나이: 20대 후반에 집중된 20~39세 분포
age = np.random.normal(loc=29, scale=5, size=n_samples)
age = np.clip(age, 20, 39).astype(int)

# 성별: 대부분 남성
gender = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.95, 0.05])

# 관심사: 남성 의류 카테고리 내에서 선택
num_interests = np.random.randint(1, 6, size=n_samples)


# 2. RFM 스코어 생성 (고객 그룹별 특성을 반영)
# 고객 유형을 가정: 0:이탈위험, 1:신규/일반, 2:충성/VIP
customer_type = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.5, 0.2])

recency_score = np.zeros(n_samples)
frequency_score = np.zeros(n_samples)
monetary_score = np.zeros(n_samples)

# 이탈위험군 (R 낮음, F 낮음)
recency_score[customer_type == 0] = np.random.choice([1, 2], size=(customer_type == 0).sum(), p=[0.7, 0.3])
frequency_score[customer_type == 0] = np.random.choice([1, 2], size=(customer_type == 0).sum(), p=[0.6, 0.4])
monetary_score[customer_type == 0] = np.random.choice([1, 2, 3], size=(customer_type == 0).sum())

# 신규/일반군 (중간 분포)
recency_score[customer_type == 1] = np.random.choice([2, 3, 4], size=(customer_type == 1).sum())
frequency_score[customer_type == 1] = np.random.choice([2, 3, 4], size=(customer_type == 1).sum())
monetary_score[customer_type == 1] = np.random.choice([2, 3, 4], size=(customer_type == 1).sum())

# 충성/VIP군 (R 높음, F 높음, M 높음)
recency_score[customer_type == 2] = np.random.choice([4, 5], size=(customer_type == 2).sum(), p=[0.4, 0.6])
frequency_score[customer_type == 2] = np.random.choice([4, 5], size=(customer_type == 2).sum(), p=[0.3, 0.7])
monetary_score[customer_type == 2] = np.random.choice([4, 5], size=(customer_type == 2).sum(), p=[0.2, 0.8])


# 3. RFM 스코어와 연관된 행동 피처 생성 (비즈니스 상황 반영)
tenure_days = (frequency_score * 100 + np.random.randint(-50, 200, size=n_samples)).astype(int)
tenure_days = np.clip(tenure_days, 1, None)

# 평균 주문 금액: 남성 의류 현실에 맞게 조정 (예: 5만원 ~ 20만원대)
monetary_avg_order = (monetary_score * 20000 + np.random.normal(50000, 10000, size=n_samples)).astype(int)
monetary_avg_order = np.clip(monetary_avg_order, 30000, None)

# 평균 주문당 상품 수: 1~3개에 집중
avg_items_per_order = 1 + (frequency_score / 3) + np.random.uniform(0, 1.5, size=n_samples)
avg_items_per_order = np.round(np.clip(avg_items_per_order, 1, 5), 1)

frequency_last_90d = (frequency_score * 0.5 + np.random.randint(0, 3, size=n_samples)).astype(int)
frequency_last_90d = np.clip(frequency_last_90d, 0, None)

frequency_last_30d = (recency_score / 5 * frequency_last_90d * np.random.uniform(0.2, 0.8, size=n_samples)).astype(int)
frequency_last_30d = np.minimum(frequency_last_30d, frequency_last_90d)

days_between_orders = (365 / (frequency_score * np.random.uniform(0.8, 1.2, size=n_samples)))
days_between_orders[frequency_score <= 1] = np.nan

coupon_usage_rate = ( (5 - monetary_score) / 5 + np.random.uniform(-0.2, 0.2, size=n_samples) )
coupon_usage_rate = np.round(np.clip(coupon_usage_rate, 0, 1), 2)

days_since_last_session = ((6 - recency_score) * 30 + np.random.randint(-20, 20, size=n_samples)).astype(int)
days_since_last_session = np.clip(days_since_last_session, 1, 365)

cart_additions_last_30d = (recency_score * 1.5 + np.random.randint(0, 5, size=n_samples)).astype(int)
cart_additions_last_30d[recency_score <= 2] = np.random.choice([0, 1], size=(recency_score <= 2).sum())


# 4. 이탈(Churn) 여부 결정
churn_probability = 1 / (1 + np.exp(-(
    -0.8 * recency_score
    -0.5 * frequency_score
    + 0.3 * (days_since_last_session / 30)
    - 0.5 * (cart_additions_last_30d)
    + np.random.normal(0, 0.5, n_samples)
    - 1.5
)))
churn = (np.random.rand(n_samples) < churn_probability).astype(int)


# 5. 데이터프레임으로 합치기
df = pd.DataFrame({
    'age': age,
    'gender': gender,
    'tenure_days': tenure_days,
    'num_interests': num_interests,
    'recency_score': recency_score.astype(int),
    'frequency_score': frequency_score.astype(int),
    'monetary_score': monetary_score.astype(int),
    'monetary_avg_order': monetary_avg_order,
    'avg_items_per_order': avg_items_per_order,
    'frequency_last_30d': frequency_last_30d,
    'frequency_last_90d': frequency_last_90d,
    'days_between_orders': days_between_orders,
    'coupon_usage_rate': coupon_usage_rate,
    'days_since_last_session': days_since_last_session,
    'cart_additions_last_30d': cart_additions_last_30d,
    'churn': churn
})

# 생성된 데이터 확인
print(f"--- 데이터 샘플 (총 {len(df)}건) ---")
print(df.head())
print("\n--- 데이터 정보 요약 ---")
df.info()
print("\n--- 데이터 기술 통계 ---")
print(df.describe())
print(f"\n생성된 이탈 고객 수: {df['churn'].sum()}명 ({df['churn'].mean():.2%})")

# (선택) CSV 파일로 저장하기
# df.to_csv('dummy_mens_apparel_churn_data.csv', index=False)