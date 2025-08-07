import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import os

# Faker 초기화 (한국어 설정)
fake = Faker('ko_KR')

# 재현성을 위한 랜덤 시드 고정
np.random.seed(100)

# --- 생성할 데이터의 시간 범위 설정 ---
start_date_future = datetime.now()
end_date_future = start_date_future + timedelta(days=14)

# --- 생성할 데이터 개수 설정 ---
N_USERS = 1000
N_PRODUCTS = 100
N_ORDERS = 1500

print(f"'{start_date_future.strftime('%Y-%m-%d')}'부터 2주간의 신규 사용자 {N_USERS}명에 대한 Raw 데이터를 생성합니다.\n")

# --- 1. Categories, Products 테이블 생성 ---
# (이전 코드와 동일)
categories_data = {'category_id': [1, 2, 3, 4, 5, 11, 12, 13, 21, 22, 31, 41], 'category_name': ['상의', '하의', '아우터', '신발', '액세서리', '티셔츠', '셔츠', '니트', '청바지', '슬랙스', '자켓', '스니커즈'], 'parent_id': [None, None, None, None, None, 1, 1, 1, 2, 2, 3, 4]}
categories_df = pd.DataFrame(categories_data)
products_data = []
sub_categories = categories_df[categories_df['parent_id'].notna()]
for i in range(1, N_PRODUCTS + 1):
    category_row = sub_categories.sample(1).iloc[0]
    products_data.append({'product_id': i, 'category_id': int(category_row['category_id']), 'product_name': f"{category_row['category_name']} {fake.color_name()} {np.random.choice(['베이직', '오버핏', '슬림핏'])}", 'description': fake.catch_phrase(), 'price': float(round(np.random.uniform(20000, 200000), -2)), 'stock_quantity': np.random.randint(0, 100), 'created_at': fake.date_time_this_year()})
products_df = pd.DataFrame(products_data)


# --- 2. Users 테이블 생성 (신규 사용자, 미래 가입일) ---
# (이전 코드와 동일)
users_data = []
for i in range(1001, 1001 + N_USERS):
    created_time = fake.date_time_between(start_date=start_date_future, end_date=end_date_future)
    users_data.append({'user_id': i, 'email': fake.email(), 'password_hash': fake.sha256(), 'user_name': fake.name(), 'gender': np.random.choice(['MALE', 'FEMALE'], p=[0.9, 0.1]), 'birthdate': fake.date_of_birth(minimum_age=20, maximum_age=40), 'phone_number': fake.phone_number(), 'created_at': created_time})
users_df = pd.DataFrame(users_data)


# --- 3. User_Interests 테이블 생성 ---
# (이전 코드와 동일)
user_interests_data = [{'user_id': user_id, 'category_id': int(category_id)} for user_id in users_df['user_id'] for category_id in np.random.choice(categories_df['category_id'], size=np.random.randint(1, 4), replace=False)]
user_interests_df = pd.DataFrame(user_interests_data)


# --- 4. Orders, Order_Items, User_Logins, Cart_Items 테이블 생성 (⭐️ 수정된 부분) ---
orders_data, order_items_data, user_logins_data, cart_items_data = [], [], [], []
order_item_id_counter, cart_item_id_counter = 1, 1

# ⭐️ 모든 사용자가 최소 1회의 로그인과 장바구니 활동을 하도록 수정
for _, user in users_df.iterrows():
    # 최소 1회 로그인 보장
    for _ in range(np.random.randint(1, 20)):
        user_logins_data.append({'login_id': len(user_logins_data) + 1, 'user_id': user['user_id'], 'login_at': fake.date_time_between(start_date=user['created_at'], end_date=end_date_future)})
    # 최소 1회 장바구니 활동 보장
    for _ in range(np.random.randint(1, 10)):
        cart_items_data.append({'cart_item_id': cart_item_id_counter, 'user_id': user['user_id'], 'product_id': np.random.choice(products_df['product_id']), 'quantity': np.random.randint(1,3), 'added_at': fake.date_time_between(start_date=user['created_at'], end_date=end_date_future)}); cart_item_id_counter += 1

# ⭐️ 모든 사용자가 최소 1회의 주문을 하도록 수정
assigned_users = set()
for order_id in range(1, N_ORDERS + 1):
    # 아직 주문하지 않은 사용자 중에서 우선 할당하거나, 모든 유저가 주문했다면 랜덤 선택
    unassigned_users = set(users_df['user_id']) - assigned_users
    if unassigned_users:
        user_id = np.random.choice(list(unassigned_users))
        assigned_users.add(user_id)
    else:
        user_id = np.random.choice(users_df['user_id'])
    
    user = users_df.loc[users_df['user_id'] == user_id].iloc[0]
    order_date = fake.date_time_between(start_date=user['created_at'], end_date=end_date_future)
    used_coupon = np.random.choice([None, 'WELCOME10'], p=[0.5, 0.5])
    orders_data.append({'order_id': order_id, 'user_id': user_id, 'order_date': order_date, 'total_amount': 0.0, 'status': 'COMPLETED', 'shipping_address': fake.address(), 'used_coupon_code': used_coupon}) # 모두 COMPLETED로 가정
    order_total_amount = 0
    for _ in range(np.random.randint(1, 4)):
        product = products_df.sample(1).iloc[0]
        quantity = np.random.randint(1, 3)
        order_items_data.append({'order_item_id': order_item_id_counter, 'order_id': order_id, 'product_id': product['product_id'], 'quantity': quantity, 'price_per_item': product['price']}); order_total_amount += quantity * product['price']; order_item_id_counter += 1
    orders_data[-1]['total_amount'] = float(order_total_amount)

orders_df = pd.DataFrame(orders_data)
order_items_df = pd.DataFrame(order_items_data)
user_logins_df = pd.DataFrame(user_logins_data)
cart_items_df = pd.DataFrame(cart_items_data)

# --- 5. CSV 파일로 저장 ---
output_dir = 'data/raw_after'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

users_df.to_csv(os.path.join(output_dir, 'users.csv'), index=False, encoding='utf-8-sig')
categories_df.to_csv(os.path.join(output_dir, 'categories.csv'), index=False, encoding='utf-8-sig')
products_df.to_csv(os.path.join(output_dir, 'products.csv'), index=False, encoding='utf-8-sig')
user_interests_df.to_csv(os.path.join(output_dir, 'user_interests.csv'), index=False, encoding='utf-8-sig')
orders_df.to_csv(os.path.join(output_dir, 'orders.csv'), index=False, encoding='utf-8-sig')
order_items_df.to_csv(os.path.join(output_dir, 'order_items.csv'), index=False, encoding='utf-8-sig')
user_logins_df.to_csv(os.path.join(output_dir, 'user_logins.csv'), index=False, encoding='utf-8-sig')
cart_items_df.to_csv(os.path.join(output_dir, 'cart_items.csv'), index=False, encoding='utf-8-sig')

print(f"\n성공적으로 8개의 신규 Raw 데이터 CSV 파일이 '{output_dir}' 폴더에 저장되었습니다.")
print(f"저장된 파일 목록: {os.listdir(output_dir)}")