import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Extrapolation 문제 해결 (변화율 예측 + Feature 정규화)")
print("=" * 70)

# ===== 1. 데이터 로드 =====
print("\n1. 데이터 로드")
print("-" * 70)

df = pd.read_csv('integrated_data_full.csv', index_col=0, parse_dates=True)
print(f"전체 데이터: {df.shape} ({df.index[0].date()} ~ {df.index[-1].date()})")

# ===== 2. 제거할 특성 정의 =====
print("\n2. 데이터 누수 특성 제거")
print("-" * 70)

exclude_features = [
    # 가격 관련 (직접적 누수)
    'Close', 'High', 'Low', 'Open',
    'cumulative_return',
    'bc_market_price', 'bc_market_cap',

    # EMA/SMA (Close의 이동평균 - 간접적 누수)
    'EMA5_close', 'EMA10_close', 'EMA14_close', 'EMA20_close', 'EMA30_close', 'EMA100_close',
    'SMA5_close', 'SMA10_close', 'SMA20_close', 'SMA30_close',

    # Bollinger Bands (Close 기반)
    'BB_high', 'BB_mid', 'BB_low',
]

print(f"제거할 특성: {len(exclude_features)}개")

# ===== 3. 타겟 생성 (변화율!) =====
print("\n3. 타겟 생성 - 일별 변화율 (%) 예측")
print("-" * 70)

# 다음날 수익률 예측
df['target_return'] = (df['Close'].shift(-1) / df['Close'] - 1) * 100

print(f"타겟 통계 (일별 수익률 %):")
print(f"  평균: {df['target_return'].mean():.3f}%")
print(f"  표준편차: {df['target_return'].std():.3f}%")
print(f"  최소: {df['target_return'].min():.3f}%")
print(f"  최대: {df['target_return'].max():.3f}%")

# ===== 4. Feature Engineering (가격 독립적으로) =====
print("\n4. Feature Engineering - 가격 영향 제거")
print("-" * 70)

# bc_miners_revenue를 정규화
if 'bc_miners_revenue' in df.columns:
    df['miners_revenue_normalized'] = df['bc_miners_revenue'] / df['Close']
    print("✓ miners_revenue_normalized 생성 (채굴 수익 / BTC 가격)")

# 기술적 지표를 변화율로 변환
for col in ['RSI', 'MACD', 'ATR', 'OBV', 'ADX', 'CCI', 'MFI']:
    if col in df.columns:
        df[f'{col}_change'] = df[col].pct_change() * 100

print("✓ 기술적 지표 변화율 생성")

# 매크로 지표 변화율
for col in ['DGS10', 'CPIAUCSL', 'UNRATE', 'M2SL']:
    if col in df.columns and df[col].notna().sum() > 0:
        df[f'{col}_change'] = df[col].pct_change() * 100

print("✓ 매크로 지표 변화율 생성")

# ===== 5. 데이터 정리 =====
df_clean = df.dropna(subset=['target_return']).copy()

# 특성 선택
all_features = [col for col in df_clean.columns
                if col not in exclude_features
                and col != 'target_return'
                and not col.endswith('_change')]  # 변화율 특성은 나중에 추가

# 변화율 특성 추가
change_features = [col for col in df_clean.columns if col.endswith('_change')]
all_features.extend(change_features)

# miners_revenue_normalized 추가
if 'miners_revenue_normalized' in df_clean.columns:
    all_features.append('miners_revenue_normalized')
    all_features.remove('bc_miners_revenue')  # 원본 제거

X = df_clean[all_features].copy()
y = df_clean['target_return'].copy()

# NaN 제거
mask = X.notna().all(axis=1) & y.notna()
X = X[mask]
y = y[mask]

print(f"\n최종 데이터:")
print(f"  특성 수: {len(all_features)}개")
print(f"  샘플 수: {len(X)}개")
print(f"  기간: {X.index[0].date()} ~ {X.index[-1].date()}")

# ===== 6. 데이터 분할 =====
print("\n6. 데이터 분할")
print("-" * 70)

# 80:20 분할
split_idx = int(len(X) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"Train: {len(X_train)}개 ({X_train.index[0].date()} ~ {X_train.index[-1].date()})")
print(f"Test:  {len(X_test)}개 ({X_test.index[0].date()} ~ {X_test.index[-1].date()})")

# 타겟 분포 확인
print(f"\nTrain 타겟 (수익률 %):")
print(f"  평균: {y_train.mean():.3f}%")
print(f"  표준편차: {y_train.std():.3f}%")
print(f"  범위: {y_train.min():.3f}% ~ {y_train.max():.3f}%")

print(f"\nTest 타겟 (수익률 %):")
print(f"  평균: {y_test.mean():.3f}%")
print(f"  표준편차: {y_test.std():.3f}%")
print(f"  범위: {y_test.min():.3f}% ~ {y_test.max():.3f}%")

# ===== 7. 모델 훈련 =====
print("\n" + "=" * 70)
print("7. 모델 훈련 (변화율 예측)")
print("=" * 70)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest
print("\n[Random Forest]")
print("-" * 70)
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_scaled, y_train)

y_train_pred_rf = rf.predict(X_train_scaled)
y_test_pred_rf = rf.predict(X_test_scaled)

rf_train_r2 = r2_score(y_train, y_train_pred_rf)
rf_test_r2 = r2_score(y_test, y_test_pred_rf)
rf_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_rf))
rf_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_rf))

print(f"Train R²: {rf_train_r2:.4f} | RMSE: {rf_train_rmse:.3f}%")
print(f"Test R²:  {rf_test_r2:.4f} | RMSE: {rf_test_rmse:.3f}%")
print(f"R² 차이:  {rf_train_r2 - rf_test_r2:.4f}")

if rf_test_r2 < 0:
    print("⚠️ Test R² 음수 - 모델이 평균보다 못함")
elif rf_train_r2 - rf_test_r2 > 0.1:
    print("⚠️ 과적합 의심")
else:
    print("✅ 정상 범위")

# Feature Importance
rf_importance = pd.DataFrame({
    'feature': all_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n상위 10개 중요 특성:")
for idx, row in rf_importance.head(10).iterrows():
    print(f"  {row['feature']:35s} : {row['importance']:.4f}")

# XGBoost
print("\n[XGBoost]")
print("-" * 70)
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train_scaled, y_train)

y_train_pred_xgb = xgb_model.predict(X_train_scaled)
y_test_pred_xgb = xgb_model.predict(X_test_scaled)

xgb_train_r2 = r2_score(y_train, y_train_pred_xgb)
xgb_test_r2 = r2_score(y_test, y_test_pred_xgb)
xgb_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_xgb))
xgb_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_xgb))

print(f"Train R²: {xgb_train_r2:.4f} | RMSE: {xgb_train_rmse:.3f}%")
print(f"Test R²:  {xgb_test_r2:.4f} | RMSE: {xgb_test_rmse:.3f}%")
print(f"R² 차이:  {xgb_train_r2 - xgb_test_r2:.4f}")

if xgb_test_r2 < 0:
    print("⚠️ Test R² 음수 - 모델이 평균보다 못함")
elif xgb_train_r2 - xgb_test_r2 > 0.1:
    print("⚠️ 과적합 의심")
else:
    print("✅ 정상 범위")

# ===== 8. 실제 가격으로 변환해서 평가 =====
print("\n" + "=" * 70)
print("8. 실제 가격 예측 성능 (변화율 → 가격 변환)")
print("=" * 70)

# Test 기간의 실제 Close 가격
test_dates = y_test.index
actual_prices = df.loc[test_dates, 'Close'].values
predicted_returns_rf = y_test_pred_rf / 100  # % → 비율
predicted_returns_xgb = y_test_pred_xgb / 100

# 다음날 예측 가격 = 오늘 가격 × (1 + 예측 수익률)
predicted_prices_rf = actual_prices * (1 + predicted_returns_rf)
predicted_prices_xgb = actual_prices * (1 + predicted_returns_xgb)

# 실제 다음날 가격
actual_next_prices = df.loc[test_dates, 'Close'].shift(-1).values[:-1]
predicted_prices_rf = predicted_prices_rf[:-1]
predicted_prices_xgb = predicted_prices_xgb[:-1]

# 가격 예측 성능
price_r2_rf = r2_score(actual_next_prices, predicted_prices_rf)
price_r2_xgb = r2_score(actual_next_prices, predicted_prices_xgb)
price_rmse_rf = np.sqrt(mean_squared_error(actual_next_prices, predicted_prices_rf))
price_rmse_xgb = np.sqrt(mean_squared_error(actual_next_prices, predicted_prices_xgb))

print("\n[Random Forest - 가격 예측]")
print(f"R²: {price_r2_rf:.4f} | RMSE: ${price_rmse_rf:,.2f}")

print("\n[XGBoost - 가격 예측]")
print(f"R²: {price_r2_xgb:.4f} | RMSE: ${price_rmse_xgb:,.2f}")

# ===== 9. 시각화 =====
print("\n9. 결과 시각화")
print("-" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 수익률 예측 (Random Forest)
ax1 = axes[0, 0]
ax1.scatter(y_test, y_test_pred_rf, alpha=0.5, s=20)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        'r--', linewidth=2, label='Perfect')
ax1.set_xlabel('Actual Return (%)', fontsize=11)
ax1.set_ylabel('Predicted Return (%)', fontsize=11)
ax1.set_title(f'Random Forest - Return Prediction (R²={rf_test_r2:.4f})',
             fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 수익률 예측 (XGBoost)
ax2 = axes[0, 1]
ax2.scatter(y_test, y_test_pred_xgb, alpha=0.5, s=20, color='orange')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        'r--', linewidth=2, label='Perfect')
ax2.set_xlabel('Actual Return (%)', fontsize=11)
ax2.set_ylabel('Predicted Return (%)', fontsize=11)
ax2.set_title(f'XGBoost - Return Prediction (R²={xgb_test_r2:.4f})',
             fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 가격 예측 (Random Forest)
ax3 = axes[1, 0]
ax3.scatter(actual_next_prices, predicted_prices_rf, alpha=0.5, s=20)
ax3.plot([actual_next_prices.min(), actual_next_prices.max()],
        [actual_next_prices.min(), actual_next_prices.max()],
        'r--', linewidth=2, label='Perfect')
ax3.set_xlabel('Actual Price ($)', fontsize=11)
ax3.set_ylabel('Predicted Price ($)', fontsize=11)
ax3.set_title(f'Random Forest - Price Prediction (R²={price_r2_rf:.4f})',
             fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 가격 예측 (XGBoost)
ax4 = axes[1, 1]
ax4.scatter(actual_next_prices, predicted_prices_xgb, alpha=0.5, s=20, color='orange')
ax4.plot([actual_next_prices.min(), actual_next_prices.max()],
        [actual_next_prices.min(), actual_next_prices.max()],
        'r--', linewidth=2, label='Perfect')
ax4.set_xlabel('Actual Price ($)', fontsize=11)
ax4.set_ylabel('Predicted Price ($)', fontsize=11)
ax4.set_title(f'XGBoost - Price Prediction (R²={price_r2_xgb:.4f})',
             fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('extrapolation_fix_results.png', dpi=300, bbox_inches='tight')
print("✓ extrapolation_fix_results.png")
plt.close()

# ===== 10. 결과 저장 =====
print("\n10. 결과 저장")
print("-" * 70)

results = pd.DataFrame({
    'model': ['Random Forest', 'XGBoost'],
    'return_train_r2': [rf_train_r2, xgb_train_r2],
    'return_test_r2': [rf_test_r2, xgb_test_r2],
    'return_r2_gap': [rf_train_r2 - rf_test_r2, xgb_train_r2 - xgb_test_r2],
    'price_test_r2': [price_r2_rf, price_r2_xgb],
    'price_rmse': [price_rmse_rf, price_rmse_xgb]
})

results.to_csv('extrapolation_fix_results.csv', index=False)
print("✓ extrapolation_fix_results.csv")

rf_importance.to_csv('feature_importance_return_rf.csv', index=False)
print("✓ feature_importance_return_rf.csv")

print("\n" + "=" * 70)
print("Extrapolation 문제 해결 완료!")
print("=" * 70)

print("\n📊 최종 요약:")
print("-" * 70)
print(f"방법: 절대 가격 → 일별 변화율(%) 예측")
print(f"\n수익률 예측 성능:")
for _, row in results.iterrows():
    print(f"  {row['model']:15s}: Test R² = {row['return_test_r2']:7.4f} (Gap: {row['return_r2_gap']:.4f})")

print(f"\n가격 예측 성능 (변화율→가격 변환):")
for _, row in results.iterrows():
    print(f"  {row['model']:15s}: R² = {row['price_test_r2']:7.4f} | RMSE = ${row['price_rmse']:,.2f}")

print("\n💡 개선사항:")
print("-" * 70)
if rf_test_r2 > 0 and xgb_test_r2 > 0:
    print("✅ Test R²가 양수 - Extrapolation 문제 해결!")
else:
    print("⚠️ 여전히 Test R² 음수 - 추가 개선 필요")

if rf_train_r2 - rf_test_r2 < 0.2 and xgb_train_r2 - xgb_test_r2 < 0.2:
    print("✅ 과적합 개선됨 (R² Gap < 0.2)")
else:
    print("⚠️ 여전히 과적합 존재")

print("=" * 70)
