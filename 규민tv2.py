import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import warnings

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# Matplotlib 설정
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Malgun Gothic', 'AppleGothic']
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print("--- [통합 실행] 1~5단계 전체 분석 시작 ---")

# --- 1단계: 데이터 전처리 및 준비 ---
print("\n--- 1단계: 데이터 전처리 및 준비 ---")
try:
    file_path = 'integrated_data_full_v2.csv'
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"오류: 원본 파일 '{file_path}'를 찾을 수 없습니다.")
    exit()

# 1-1. 'Date' 처리
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').set_index('Date')
original_df_for_backtest = df.copy() # 백테스팅 시 원본 가격 사용을 위해 복사

# 1-2. 결측치(NaN) 처리
etf_columns = [col for col in df.columns if 'IBIT' in col or 'FBTC' in col or 'GBTC' in col or 'ARKB' in col or 'BITB' in col or 'Total_BTC_ETF_Volume' in col]
etf_start_date = '2024-01-11'
df[etf_columns] = df[etf_columns].fillna(0)
df = df.ffill()
df = df.fillna(0)

# 1-3. 타겟 변수 'y' 생성
df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df = df.dropna(subset=['target'])

# 1-4. 특징(X)과 타겟(y) 분리
X = df.drop('target', axis=1)
y = df['target']
features = X.columns

# 1-5. 훈련/테스트 분리 (shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 1-6. 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1-7. 2단계 분석을 위한 DataFrame 생성
X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=features)
print("1단계 완료. 훈련/테스트 데이터 준비됨.")

# --- 2단계: ETF 도입 전후 요인 분석 (Lasso) ---
print("\n--- 2단계: ETF 도입 전후 요인 분석 (Lasso) ---")
pre_etf_X = X_train_scaled_df[X_train_scaled_df.index < etf_start_date]
pre_etf_y = y_train[y_train.index < etf_start_date]
post_etf_X = X_train_scaled_df[X_train_scaled_df.index >= etf_start_date]
post_etf_y = y_train[y_train.index >= etf_start_date]

print(f"Pre-ETF 훈련 데이터 크기: {pre_etf_X.shape}")
print(f"Post-ETF 훈련 데이터 크기: {post_etf_X.shape}")

# Pre-ETF
lasso_model_pre = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
if not pre_etf_X.empty:
    lasso_model_pre.fit(pre_etf_X, pre_etf_y)
    pre_etf_coef = pd.DataFrame(lasso_model_pre.coef_[0], index=features, columns=['Coefficient'])
    print("\n--- Pre-ETF 기간 주요 영향 요인 (Lasso C=0.1) ---")
    print(pre_etf_coef[pre_etf_coef['Coefficient'] != 0].sort_values(by='Coefficient', ascending=False))
else:
    print("\n--- Pre-ETF 데이터 없음 ---")

# Post-ETF
lasso_model_post = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
if not post_etf_X.empty:
    lasso_model_post.fit(post_etf_X, post_etf_y)
    post_etf_coef = pd.DataFrame(lasso_model_post.coef_[0], index=features, columns=['Coefficient'])
    print("\n--- Post-ETF 기간 주요 영향 요인 (Lasso C=0.1) ---")
    print(post_etf_coef[post_etf_coef['Coefficient'] != 0].sort_values(by='Coefficient', ascending=False))
else:
    print("\n--- Post-ETF 데이터 없음 ---")
print("2단계 완료.")

# --- 3단계: 예측 모델 구축 및 최적화 ---
print("\n--- 3단계: 예측 모델 구축 및 최적화 (Lasso, Ridge, Elastic Net) ---")
tscv = TimeSeriesSplit(n_splits=5)
best_models = {}
best_scores = {}

# 3-1. Lasso
print("\nLasso 최적화 중...")
lasso = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=1000)
grid_lasso = GridSearchCV(lasso, {'C': [0.01, 0.1, 1, 10]}, cv=tscv, scoring='f1', n_jobs=-1)
grid_lasso.fit(X_train_scaled, y_train)
best_models['Lasso'] = grid_lasso.best_estimator_
best_scores['Lasso'] = grid_lasso.best_score_
print(f"Lasso 최고 F1 (CV): {grid_lasso.best_score_:.4f} (Params: {grid_lasso.best_params_})")

# 3-2. Ridge
print("\nRidge 최적화 중...")
ridge = LogisticRegression(penalty='l2', solver='liblinear', random_state=42, max_iter=1000)
grid_ridge = GridSearchCV(ridge, {'C': [0.01, 0.1, 1, 10]}, cv=tscv, scoring='f1', n_jobs=-1)
grid_ridge.fit(X_train_scaled, y_train)
best_models['Ridge'] = grid_ridge.best_estimator_
best_scores['Ridge'] = grid_ridge.best_score_
print(f"Ridge 최고 F1 (CV): {grid_ridge.best_score_:.4f} (Params: {grid_ridge.best_params_})")

# 3-3. ElasticNet
print("\nElasticNet 최적화 중...")
elastic = LogisticRegression(penalty='elasticnet', solver='saga', random_state=42, max_iter=1000)
grid_elastic = GridSearchCV(elastic, {'C': [0.01, 0.1, 1, 10], 'l1_ratio': [0.3, 0.5, 0.7]}, cv=tscv, scoring='f1', n_jobs=-1)
grid_elastic.fit(X_train_scaled, y_train)
best_models['ElasticNet'] = grid_elastic.best_estimator_
best_scores['ElasticNet'] = grid_elastic.best_score_
print(f"ElasticNet 최고 F1 (CV): {grid_elastic.best_score_:.4f} (Params: {grid_elastic.best_params_})")
print("3단계 완료.")

# --- 4단계: 최고 성능 모델 선정 ---
print("\n--- 4단계: 최고 성능 모델 선정 ---")
best_model_name = max(best_scores, key=best_scores.get)
best_model = best_models[best_model_name] # 메모리에서 바로 가져옴

print("모델별 교차 검증 F1 점수:")
for model_name, score in best_scores.items():
    print(f"{model_name}: {score:.4f}")
print(f"\n최고 성능 모델: {best_model_name}")
print(f"최적 모델 정보: {best_model}")
print("4단계 완료.")

# --- 5단계: 백테스팅 및 매매 모델 구현 ---
print("\n--- 5단계: 백테스팅 및 매매 모델 구현 ---")

# 5-1. 테스트 기간 원본 가격 추출
test_period_prices = original_df_for_backtest.loc[y_test.index]['Close']

# 5-2. 모델 예측 (X_test_scaled 사용)
y_pred = best_model.predict(X_test_scaled)

# 5-3. 모델 성능 평가 (***사용자 요청: 예측 정확도***)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- 테스트 세트 예측 성능 ({best_model_name}) ---")
print(f"*** 예측 정확도 (Accuracy): {accuracy * 100:.2f}% ***")
print("\n[혼동 행렬 (Confusion Matrix)]")
print(confusion_matrix(y_test, y_pred))
print("\n[분류 리포트 (Classification Report)]")
print(classification_report(y_test, y_pred, target_names=['Down (0)', 'Up (1)']))

# 5-4. 매매 전략 및 백테스팅
results = pd.DataFrame({
    'Close': test_period_prices,
    'Actual_Direction': y_test,
    'Predicted_Direction': y_pred
}, index=y_test.index)

results['Market_Return'] = results['Close'].pct_change()
results['Signal'] = results['Predicted_Direction'].shift(1) # 어제의 예측
results['Strategy_Return'] = np.where(results['Signal'] == 1, results['Market_Return'], 0)
results = results.fillna(0)

# 5-5. 누적 수익률 계산
results['Cumulative_Market_Return'] = (1 + results['Market_Return']).cumprod()
results['Cumulative_Strategy_Return'] = (1 + results['Strategy_Return']).cumprod()

# 5-6. 최종 결과 요약
final_market_return = results['Cumulative_Market_Return'].iloc[-1]
final_strategy_return = results['Cumulative_Strategy_Return'].iloc[-1]

print("\n--- 최종 누적 수익률 비교 ---")
print(f"Buy-and-Hold (단순 보유) 누적 수익률: {final_market_return:.4f} (즉, {((final_market_return - 1) * 100):.2f}%)")
print(f"ElasticNet 모델 전략 누적 수익률:     {final_strategy_return:.4f} (즉, {((final_strategy_return - 1) * 100):.2f}%)")

if final_strategy_return > final_market_return:
    print("\n결과: 모델 기반 전략이 단순 보유(B&H)보다 우수한 성과를 보였습니다. 📈")
elif final_strategy_return < final_market_return:
    print("\n결과: 모델 기반 전략이 단순 보유(B&H)보다 저조한 성과를 보였습니다. 📉")
else:
    print("\n결과: 모델 기반 전략과 단순 보유(B&H) 성과가 동일합니다.")

# 5-7. 시각화
plt.figure(figsize=(14, 7))
results['Cumulative_Market_Return'].plot(label='Buy-and-Hold (단순 보유)')
results['Cumulative_Strategy_Return'].plot(label=f'{best_model_name} Model Strategy (매매 전략)', linestyle='--')
plt.title('Backtesting: Model Strategy vs. Buy-and-Hold (Test Period)')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns (1 = 100%)')
plt.legend()
plt.grid(True)
plt.savefig('backtesting_results_chart_final.png') # 이미지 파일로 저장
print("\n백테스팅 누적 수익률 비교 차트를 'backtesting_results_chart_final.png'로 저장했습니다.")
print("\n--- [통합 실행] 모든 분석 완료 ---")