# data_generator_tuned.py
import os, random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from faker import Faker

# --------------------------
# 설정
# --------------------------
SEED = 2025
N_USERS = 10_000
N_PRODUCTS = 1_000
N_ORDERS = 200_000
OUT_DIR = "raw_data_current"     # ✅ EDA가 읽는 폴더

np.random.seed(SEED)
random.seed(SEED)
fake = Faker('ko_KR')
Faker.seed(SEED)

# --------------------------
# 유틸
# --------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

now = datetime.now()
two_years_ago = now - timedelta(days=730)
ninety_days_ago = now - timedelta(days=90)

os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------
# 1) 카테고리/상품
# --------------------------
categories_data = {
    'category_id': [1, 2, 3, 4, 5, 11, 12, 13, 21, 22, 31, 41],
    'category_name': ['상의','하의','아우터','신발','액세서리','티셔츠','셔츠','니트','청바지','슬랙스','자켓','스니커즈'],
    'parent_id': [None,None,None,None,None,1,1,1,2,2,3,4]
}
categories_df = pd.DataFrame(categories_data)

sub_categories = categories_df[categories_df['parent_id'].notna()]
products = []
for i in range(1, N_PRODUCTS+1):
    row = sub_categories.sample(1).iloc[0]
    products.append({
        'product_id': i,
        'category_id': int(row['category_id']),
        'product_name': f"{row['category_name']} {fake.color_name()} {np.random.choice(['베이직','오버핏','슬림핏'])}",
        'description': fake.catch_phrase(),
        'price': float(round(np.random.uniform(20_000, 200_000), -2)),
        'stock_quantity': np.random.randint(0, 100),
        'created_at': fake.date_time_between(start_date='-1y', end_date='now')
    })
products_df = pd.DataFrame(products)

# --------------------------
# 2) 유저
# --------------------------
signup_sources = ['ads','search','sns','referral','direct']
users = []
for i in range(1, N_USERS+1):
    created_at = fake.date_time_between(start_date=two_years_ago, end_date=now)
    gender = np.random.choice(['MALE','FEMALE'], p=[0.6, 0.4])   # 불균형 약간
    bd = fake.date_of_birth(minimum_age=20, maximum_age=60)
    users.append({
        'user_id': i,
        'email': fake.email(),
        'password_hash': fake.sha256(),
        'user_name': fake.name(),
        'gender': gender,
        'birthdate': bd,
        'phone_number': fake.phone_number(),
        'created_at': created_at,
        'signup_source': np.random.choice(signup_sources, p=[0.25,0.25,0.2,0.15,0.15])
    })
users_df = pd.DataFrame(users)

# 연령/연령대
users_df['age'] = now.year - pd.to_datetime(users_df['birthdate']).dt.year
users_df['age_group'] = pd.cut(users_df['age'], bins=[0,24,34,44,200], labels=['<25','25-34','35-44','45+'], right=True, include_lowest=True)
is_female = (users_df['gender']=='FEMALE').astype(int)
is_45p   = (users_df['age_group']=='45+').astype(int)

# --------------------------
# 3) 로그인/장바구니 (활동 프록시)
#    로그인은 이후 활성도/주문 수에 영향
# --------------------------
# 기본 로그인 평균(포아송) + 개인 무작위
base_login_mu = np.random.uniform(8, 30, size=N_USERS)   # 개별 베이스
login_noise   = np.random.normal(0, 3, size=N_USERS)
login_mu      = np.clip(base_login_mu + login_noise + 3*is_female - 2*is_45p, 1, None)  # 여성 조금 ↑, 45+ 조금 ↓
login_counts  = np.random.poisson(login_mu).astype(int)
login_counts[login_counts<1] = 1

login_rows = []
for uid, cnt in zip(users_df['user_id'], login_counts):
    for _ in range(cnt):
        login_rows.append({
            'login_id': len(login_rows)+1,
            'user_id': uid,
            'login_at': fake.date_time_between(start_date=users_df.loc[uid-1,'created_at'], end_date=now)
        })
user_logins_df = pd.DataFrame(login_rows)

# 장바구니
cart_mu = np.maximum(0, login_counts * np.random.uniform(0.15, 0.35, size=N_USERS))
cart_counts = np.random.poisson(cart_mu).astype(int)

cart_rows = []
cart_id = 1
for uid, cnt in zip(users_df['user_id'], cart_counts):
    for _ in range(cnt):
        cart_rows.append({
            'cart_item_id': cart_id,
            'user_id': uid,
            'product_id': np.random.randint(1, N_PRODUCTS+1),
            'quantity': np.random.randint(1,3),
            'added_at': fake.date_time_between(start_date=users_df.loc[uid-1,'created_at'], end_date=now)
        })
        cart_id += 1
cart_items_df = pd.DataFrame(cart_rows)

# --------------------------
# 4) 쿠폰 사용 경험(used_coupon_any) 먼저 부여
#    활동 많은 유저에게 사용 비율 약간 높임
# --------------------------
login_scaled = (login_counts - login_counts.mean())/ (login_counts.std() + 1e-9)
# 베이스 45% + 로그인 영향 ±, 여성/45+ 약간 보정
coupon_logit = -0.2 + 0.35*login_scaled + 0.10*is_female - 0.15*is_45p
coupon_prob  = sigmoid(coupon_logit)
used_coupon_any = (np.random.rand(N_USERS) < coupon_prob).astype(int)

# --------------------------
# 5) 활성도(=최근 주문 존재) 확률 설계
#    - 베이스 0.7 근처 목표, 계수는 과도한 유의성 방지용으로 보수적
# --------------------------
# 조정 가능한 계수
COEF_BASE      = 0.7      # 베이스 로짓 오프셋 (후에 로짓 변환)
COEF_FEMALE    = 0.20     # 여성 + (활성 ↑ → 이탈 ↓)
COEF_AGE45P    = -0.25    # 45+ - (활성 ↓)
COEF_LOGIN     = 0.55     # 로그인 z-score +
COEF_COUPON    = 0.35     # 쿠폰 경험 +
# 로짓 공간으로 바꿔 계산
base_logit = np.log(COEF_BASE/(1-COEF_BASE))
act_logit = (base_logit
             + COEF_FEMALE*is_female
             + COEF_AGE45P*is_45p
             + COEF_LOGIN*login_scaled
             + COEF_COUPON*used_coupon_any)
p_active = sigmoid(act_logit)   # 최근 90일 내 주문이 있을 확률

is_active_recent = (np.random.rand(N_USERS) < p_active)

# --------------------------
# 6) 주문 수 생성 & 20만으로 정규화
#    - 활성 유저는 더 많이 구매
# --------------------------
# 개인 평균 주문수(포아송 평균) 설계
base_lambda = np.exp(1.2 + 0.6*is_active_recent + 0.3*login_scaled)  # 평균 15~20 근방
orders_per_user = np.random.poisson(base_lambda).astype(int)
orders_per_user[orders_per_user<1] = 1

# 총합을 N_ORDERS로 스케일
total_orders = orders_per_user.sum()
scale = N_ORDERS / total_orders
orders_per_user = np.maximum(1, np.floor(orders_per_user * scale)).astype(int)

# --------------------------
# 7) Orders & Order Items
#    - 활성 유저는 최근에 주문 날짜를 더 배치
#    - 비활성 유저는 마지막 주문이 90일 이전으로 가도록 분포
# --------------------------
orders_rows = []
order_items_rows = []
order_id = 1
order_item_id = 1

for idx, uid in enumerate(users_df['user_id']):
    k = int(orders_per_user[idx])
    if k <= 0: 
        continue

    # 날짜 분포
    if is_active_recent[idx]:
        # 70%는 최근 90일 내, 30%는 과거 2년~최근 사이 섞기
        recent_cnt = int(round(k * 0.7))
        past_cnt   = k - recent_cnt
        recent_dates = [fake.date_time_between(start_date=ninety_days_ago, end_date=now) for _ in range(recent_cnt)]
        past_dates   = [fake.date_time_between(start_date=users_df.loc[uid-1,'created_at'], end_date=ninety_days_ago - timedelta(days=1)) for _ in range(past_cnt)]
        dates = recent_dates + past_dates
    else:
        # 전부 90일 이전
        dates = [fake.date_time_between(start_date=users_df.loc[uid-1,'created_at'], end_date=ninety_days_ago - timedelta(days=1)) for _ in range(k)]

    # 쿠폰 사용 확률(경험자 더 높게)
    p_coupon = 0.15 + 0.35*used_coupon_any[idx]   # 0.15 ~ 0.50

    for d in dates:
        used_coupon_code = np.random.choice([None, 'WELCOME10', 'SUMMER25'], p=[1-p_coupon, p_coupon*0.6, p_coupon*0.4])

        orders_rows.append({
            'order_id': order_id,
            'user_id': uid,
            'order_date': d,
            'total_amount': 0.0,   # 뒤에서 채움
            'status': np.random.choice(['COMPLETED','SHIPPED','PENDING','CANCELLED'], p=[0.72,0.15,0.08,0.05]),
            'shipping_address': fake.address(),
            'used_coupon_code': used_coupon_code
        })

        # 아이템 1~4개
        n_items = np.random.randint(1,5)
        order_total = 0.0
        for _ in range(n_items):
            prod = products_df.sample(1).iloc[0]
            qty = np.random.randint(1,4)
            order_items_rows.append({
                'order_item_id': order_item_id,
                'order_id': order_id,
                'product_id': int(prod['product_id']),
                'quantity': qty,
                'price_per_item': float(prod['price'])
            })
            order_total += qty * float(prod['price'])
            order_item_id += 1

        orders_rows[-1]['total_amount'] = float(order_total)
        order_id += 1

# 만약 스케일링 오차로 주문수가 살짝 넘쳤으면 자르기
if len(orders_rows) > N_ORDERS:
    orders_rows = orders_rows[:N_ORDERS]
    # 관련 order_items도 컷
    valid_ids = set(r['order_id'] for r in orders_rows)
    order_items_rows = [r for r in order_items_rows if r['order_id'] in valid_ids]

orders_df = pd.DataFrame(orders_rows)
order_items_df = pd.DataFrame(order_items_rows)

# --------------------------
# 8) User Interests (간단)
# --------------------------
user_interests_rows = []
for uid in users_df['user_id']:
    n = np.random.randint(1,4)
    cats = np.random.choice(categories_df['category_id'], size=n, replace=False)
    for c in cats:
        user_interests_rows.append({'user_id': int(uid), 'category_id': int(c)})
user_interests_df = pd.DataFrame(user_interests_rows)

# --------------------------
# 9) CSV 저장
# --------------------------
users_out = users_df[['user_id','email','password_hash','user_name','gender','birthdate','phone_number','created_at']]
users_out.to_csv(os.path.join(OUT_DIR, 'users.csv'), index=False, encoding='utf-8-sig')
categories_df.to_csv(os.path.join(OUT_DIR, 'categories.csv'), index=False, encoding='utf-8-sig')
products_df.to_csv(os.path.join(OUT_DIR, 'products.csv'), index=False, encoding='utf-8-sig')
user_interests_df.to_csv(os.path.join(OUT_DIR, 'user_interests.csv'), index=False, encoding='utf-8-sig')
orders_df.to_csv(os.path.join(OUT_DIR, 'orders.csv'), index=False, encoding='utf-8-sig')
order_items_df.to_csv(os.path.join(OUT_DIR, 'order_items.csv'), index=False, encoding='utf-8-sig')
user_logins_df.to_csv(os.path.join(OUT_DIR, 'user_logins.csv'), index=False, encoding='utf-8-sig')
cart_items_df.to_csv(os.path.join(OUT_DIR, 'cart_items.csv'), index=False, encoding='utf-8-sig')

print(f"✅ Done. Saved to '{OUT_DIR}'")
