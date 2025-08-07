import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime
import os

# Faker 초기화 (한국어 설정)
fake = Faker('ko_KR')

# 재현성을 위한 랜덤 시드 고정
np.random.seed(42)

# --- 생성할 데이터 개수 설정 ---
N_USERS = 250
N_PRODUCTS = 100
N_ORDERS = 1000

print(f"사용자 {N_USERS}명, 상품 {N_PRODUCTS}개, 주문 약 {N_ORDERS}건에 대한 Raw 데이터를 생성합니다.\n")

# --- 1. Categories 테이블 생성 ---
categories_data = {
    'category_id': [1, 2, 3, 4, 11, 12, 13, 21, 22],
    'category_name': ['상의', '하의', '아우터', '신발', '티셔츠', '셔츠', '니트', '청바지', '슬랙스'],
    'parent_id': [None, None, None, None, 1, 1, 1, 2, 2]
}
categories_df = pd.DataFrame(categories_data)

# --- 2. Users 테이블 생성 ---
users_data = []
for i in range(1, N_USERS + 1):
    created_time = fake.date_time_this_decade()
    users_data.append({
        'user_id': i,
        'email': fake.email(),
        'password_hash': fake.sha256(),
        'user_name': fake.name(),
        'gender': np.random.choice(['MALE', 'FEMALE'], p=[0.9, 0.1]),
        'birthdate': fake.date_of_birth(minimum_age=20, maximum_age=40),
        'phone_number': fake.phone_number(),
        'created_at': created_time
    })
users_df = pd.DataFrame(users_data)

# --- 3. Products 테이블 생성 ---
products_data = []
sub_categories = categories_df[categories_df['parent_id'].notna()]
for i in range(1, N_PRODUCTS + 1):
    category_row = sub_categories.sample(1).iloc[0]
    category_id = int(category_row['category_id'])
    category_name = category_row['category_name']

    products_data.append({
        'product_id': i,
        'category_id': category_id,
        'product_name': f"{category_name} {fake.color_name()} {np.random.choice(['베이직', '오버핏', '슬림핏'])}",
        'description': fake.catch_phrase(),
        'price': float(round(np.random.uniform(20000, 200000), -2)),
        'stock_quantity': np.random.randint(0, 100),
        'created_at': fake.date_time_this_year()
    })
products_df = pd.DataFrame(products_data)

# --- 4. User_Interests 테이블 생성 ---
user_interests_data = []
for user_id in users_df['user_id']:
    num_interests = np.random.randint(1, 4)
    interested_categories = np.random.choice(categories_df['category_id'], size=num_interests, replace=False)
    for category_id in interested_categories:
        user_interests_data.append({
            'user_id': user_id,
            'category_id': int(category_id)
        })
user_interests_df = pd.DataFrame(user_interests_data)

# --- 5. Orders 및 Order_Items 테이블 생성 ---
orders_data = []
order_items_data = []
order_item_id_counter = 1

for order_id in range(1, N_ORDERS + 1):
    user_id = np.random.choice(users_df['user_id'])
    user_signup_date = users_df.loc[users_df['user_id'] == user_id, 'created_at'].iloc[0]
    order_date = fake.date_time_between(start_date=user_signup_date)

    orders_data.append({
        'order_id': order_id,
        'user_id': user_id,
        'order_date': order_date,
        'total_amount': 0.0,
        'status': np.random.choice(['COMPLETED', 'SHIPPED', 'PENDING', 'CANCELLED'], p=[0.7, 0.15, 0.1, 0.05]),
        'shipping_address': fake.address()
    })

    num_items_in_order = np.random.randint(1, 5)
    order_total_amount = 0

    for _ in range(num_items_in_order):
        product = products_df.sample(1).iloc[0]
        product_id = product['product_id']
        product_price = product['price']
        quantity = np.random.randint(1, 4)

        order_items_data.append({
            'order_item_id': order_item_id_counter,
            'order_id': order_id,
            'product_id': product_id,
            'quantity': quantity,
            'price_per_item': product_price
        })
        order_total_amount += quantity * product_price
        order_item_id_counter += 1

    orders_data[-1]['total_amount'] = float(order_total_amount)

orders_df = pd.DataFrame(orders_data)
order_items_df = pd.DataFrame(order_items_data)

print("데이터 생성이 완료되었습니다. 이제 각 테이블을 CSV 파일로 저장합니다.")

# --- 6. 각 데이터프레임을 별도의 CSV 파일로 저장 ---
output_dir = 'raw_data_csv'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ⭐️⭐️⭐️ 파일 확장자는 .csv, 인코딩은 'utf-8-sig'로 지정 ⭐️⭐️⭐️
users_df.to_csv(os.path.join(output_dir, 'users.csv'), index=False, encoding='utf-8-sig')
categories_df.to_csv(os.path.join(output_dir, 'categories.csv'), index=False, encoding='utf-8-sig')
products_df.to_csv(os.path.join(output_dir, 'products.csv'), index=False, encoding='utf-8-sig')
user_interests_df.to_csv(os.path.join(output_dir, 'user_interests.csv'), index=False, encoding='utf-8-sig')
orders_df.to_csv(os.path.join(output_dir, 'orders.csv'), index=False, encoding='utf-8-sig')
order_items_df.to_csv(os.path.join(output_dir, 'order_items.csv'), index=False, encoding='utf-8-sig')

print(f"\n성공적으로 6개의 CSV 파일이 '{output_dir}' 폴더에 저장되었습니다.")
print(f"저장된 파일 목록: {os.listdir(output_dir)}")