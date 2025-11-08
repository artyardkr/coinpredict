
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import re

print("모델링을 시작합니다: 2024-2025년 데이터, RandomForest, XGBoost")

# 1. 데이터 로드 및 필터링
try:
    df = pd.read_csv('integrated_data_full.csv', parse_dates=['Date'], index_col='Date')
except FileNotFoundError:
    print("오류: integrated_data_full.csv 파일을 찾을 수 없습니다.")
    exit()

df_period = df['2024-01-01':'2025-12-31'].copy()

if df_period.empty:
    print("오류: 2024-2025년 기간에 해당하는 데이터가 없습니다.")
    exit()

# 2. 제외할 변수 목록 정의
# 2.1. Close와 상관관계가 매우 높은 변수
high_corr_cols = [
    'Open', 'High', 'Low', 'cumulative_return', 'bc_market_price', 'bc_market_cap'
]

# 2.2. 기술적 지표 변수 (정규 표현식 및 명시적 목록 사용)
tech_indicator_patterns = [
    '^EMA', '^SMA', '^MACD', '^BB_', '^Stoch', 
    'RSI', 'ATR', 'OBV', 'ADX', 'CCI', 'Williams_R', 'ROC', 'MFI', 'volatility_20d'
]

tech_indicators_to_drop = set()
for pattern in tech_indicator_patterns:
    matches = df_period.filter(regex=pattern).columns.tolist()
    tech_indicators_to_drop.update(matches)

# 2.3. 최종 제외 목록
features_to_drop = set(high_corr_cols)
features_to_drop.update(tech_indicators_to_drop)

# Close는 예측 대상이므로 제외 목록에 추가
features_to_drop.add('Close')

# 데이터프레임에 존재하는 컬럼만 필터링
features_to_drop = [col for col in features_to_drop if col in df_period.columns]

# 3. 피처 및 타겟 변수 정의
X = df_period.drop(columns=features_to_drop)
y = df_period['Close']

# 결측치 처리 (간단하게 평균값으로 대체)
X.fillna(X.mean(), inplace=True)

print(f"모델 학습에 사용될 피처 개수: {X.shape[1]}")
print("제외된 피처 목록(일부):", features_to_drop[:5])

# 4. 훈련/테스트 데이터 분리 (시계열을 고려하여 순차적으로 분리)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 5. 모델 훈련
# 5.1. RandomForest
print("\nRandomForest 모델 훈련 중...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 5.2. XGBoost
print("XGBoost 모델 훈련 중...")
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)

# 6. 모델 평가
rf_predictions = rf_model.predict(X_test)
xgb_predictions = xgb_model.predict(X_test)

print("\n--- 모델 평가 결과 ---")
# RandomForest 평가
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)
print(f"RandomForest - MSE: {rf_mse:.2f}, R-squared: {rf_r2:.4f}")

# XGBoost 평가
xgb_mse = mean_squared_error(y_test, xgb_predictions)
xgb_r2 = r2_score(y_test, xgb_predictions)
print(f"XGBoost      - MSE: {xgb_mse:.2f}, R-squared: {xgb_r2:.4f}")

# 7. 피처 중요도
rf_feature_importances = pd.DataFrame(rf_model.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance', ascending=False)
xgb_feature_importances = pd.DataFrame(xgb_model.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance', ascending=False)

print("\n--- 상위 10개 피처 중요도 ---")
print("--- RandomForest ---")
print(rf_feature_importances.head(10))
print("\n--- XGBoost ---")
print(xgb_feature_importances.head(10))

# 8. 시각화
plt.figure(figsize=(15, 7))
plt.plot(y_test.index, y_test, label='Actual Price', color='blue', linewidth=2)
plt.plot(y_test.index, rf_predictions, label='RandomForest Prediction', color='green', linestyle='--')
plt.plot(y_test.index, xgb_predictions, label='XGBoost Prediction', color='red', linestyle='--')

plt.title('Model Performance (2024-2025)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)

output_filename = 'model_performance_2024_2025.png'
plt.savefig(output_filename)
print(f"\n예측 결과 그래프를 '{output_filename}' 파일로 저장했습니다.")
