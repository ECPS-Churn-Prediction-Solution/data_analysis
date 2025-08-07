# 파일명: 2_train.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
import joblib # 모델 저장을 위한 라이브러리
import os

print("--- 모델 학습 시작 ---")

# 1. 전처리된 데이터 불러오기
df = pd.read_csv('data/processed/churn_features.csv')

# 2. 데이터 준비 (인코딩, 분할, 스케일링)
X = df.drop(['user_id', 'churn'], axis=1)
y = df['churn']
X = pd.get_dummies(X, columns=['gender'], drop_first=True, dtype=int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 하이퍼파라미터 튜닝 (LightGBM)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
lgbm = LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight)
params = {'n_estimators': [100, 200, 500], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5, 7]}
grid_search = GridSearchCV(lgbm, param_grid=params, cv=3, scoring='f1', n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print(f"\n최적 하이퍼파라미터: {grid_search.best_params_}")

# 4. 테스트 데이터로 최종 성능 평가
y_pred = best_model.predict(X_test_scaled)
print("\n--- 최종 모델 성능 평가 ---")
print(classification_report(y_test, y_pred, zero_division=0))

# 5. 학습된 모델과 스케일러 저장
output_dir = 'models'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

joblib.dump(best_model, os.path.join(output_dir, 'best_churn_model.joblib'))
joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib')) # 스케일러도 반드시 함께 저장!

print(f"\n학습된 모델과 스케일러가 '{output_dir}' 폴더에 저장되었습니다.")