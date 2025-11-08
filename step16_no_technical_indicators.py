import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("기술적 지표 제외 가격 예측 (No Technical Indicators)")
print("=" * 70)

# ===== 1. 데이터 로드 =====
print("\n1. 데이터 로드")
print("-" * 70)

df = pd.read_csv('integrated_data_full.csv', index_col=0, parse_dates=True)
print(f"전체 데이터: {df.shape} ({df.index[0].date()} ~ {df.index[-1].date()})")

# 2021년 데이터만 사용
df_2021 = df[df.index.year == 2021].copy()
print(f"2021년 데이터: {df_2021.shape} ({df_2021.index[0].date()} ~ {df_2021.index[-1].date()})")

# ===== 2. 특성 분류 =====
print("\n2. 특성 분류")
print("-" * 70)

# 기술적 지표 (제거할 것들)
technical_indicators = [
    # 이동평균
    'EMA5_close', 'EMA10_close', 'EMA14_close', 'EMA20_close', 'EMA30_close', 'EMA100_close', 'EMA200_close',
    'SMA5_close', 'SMA10_close', 'SMA20_close', 'SMA30_close',
    'EMA100_volume', 'EMA200_volume',
    'EMA5_marketcap', 'EMA10_marketcap', 'EMA14_marketcap', 'EMA20_marketcap', 'EMA30_marketcap',
    'EMA100_marketcap', 'EMA200_marketcap',
    'SMA5_marketcap', 'SMA10_marketcap', 'SMA20_marketcap', 'SMA30_marketcap',
    'market_cap_approx',
    # 기술적 지표
    'RSI', 'MACD', 'MACD_signal', 'MACD_diff',
    'BB_high', 'BB_low', 'BB_mid', 'BB_width',
    'ATR', 'OBV', 'Stoch_K', 'Stoch_D',
    'ADX', 'CCI', 'Williams_R', 'ROC', 'MFI',
    # 수익률/변동성
    'daily_return', 'cumulative_return', 'volatility_20d', 'volume_change'
]

# 데이터 누수 특성
leakage_features = ['Close', 'High', 'Low', 'Open', 'bc_market_price', 'bc_market_cap']

# 제외할 특성 전체
exclude_features = list(set(technical_indicators + leakage_features))

# 사용 가능한 특성
all_columns = df_2021.columns.tolist()
available_features = [col for col in all_columns if col not in exclude_features]

print(f"\n전체 특성 수: {len(all_columns)}")
print(f"제거된 기술적 지표: {len(technical_indicators)}개")
print(f"제거된 데이터 누수 특성: {len(leakage_features)}개")
print(f"남은 특성: {len(available_features)}개")

# 카테고리별 분류
print("\n남은 특성 카테고리:")

# BTC 원본 (Volume만)
btc_features = [f for f in available_features if f == 'Volume']
print(f"  BTC 원본: {len(btc_features)}개")
if btc_features:
    for f in btc_features:
        print(f"    - {f}")

# 전통 시장
traditional_features = [f for f in available_features if f in ['QQQ', 'SPX', 'UUP', 'EURUSD', 'GOLD', 'SILVER', 'OIL', 'BSV']]
print(f"  전통 시장: {len(traditional_features)}개")
for f in traditional_features:
    print(f"    - {f}")

# 거시경제
macro_features = [f for f in available_features if f in ['DGS10', 'DFF', 'CPIAUCSL', 'UNRATE', 'M2SL', 'GDP', 'DEXUSEU', 'DTWEXBGS', 'T10Y2Y', 'VIXCLS']]
print(f"  거시경제: {len(macro_features)}개")
for f in macro_features:
    print(f"    - {f}")

# 감정/관심
sentiment_features = [f for f in available_features if 'fear' in f.lower() or 'google' in f.lower()]
print(f"  감정/관심: {len(sentiment_features)}개")
for f in sentiment_features:
    print(f"    - {f}")

# 온체인
onchain_features = [f for f in available_features if f.startswith('bc_')]
print(f"  온체인: {len(onchain_features)}개")
for f in onchain_features:
    print(f"    - {f}")

# ===== 3. 타겟 생성 =====
print("\n3. 타겟 생성")
print("-" * 70)

# 타겟: 다음 날 종가
df_2021['target'] = df_2021['Close'].shift(-1)
df_2021 = df_2021.dropna(subset=['target'])

exclude_features.append('target')
final_features = [f for f in available_features if f in df_2021.columns]

X = df_2021[final_features].copy()
y = df_2021['target'].copy()

print(f"최종 특성 수: {len(final_features)}개")
print(f"샘플 수: {len(X)}개")
print(f"기간: {X.index[0].date()} ~ {X.index[-1].date()}")

# ===== 4. 데이터 분할 =====
print("\n4. 데이터 분할 (7:3)")
print("-" * 70)

split_idx = int(len(X) * 0.7)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"훈련: {X_train.shape[0]}개 ({X_train.index[0].date()} ~ {X_train.index[-1].date()})")
print(f"테스트: {X_test.shape[0]}개 ({X_test.index[0].date()} ~ {X_test.index[-1].date()})")

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== 5. 모델 훈련 =====
print("\n" + "=" * 70)
print("5. 모델 훈련 (기술적 지표 없이)")
print("=" * 70)

results = []

# Random Forest
print("\n[Random Forest]")
print("-" * 70)
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
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
rf_train_mae = mean_absolute_error(y_train, y_train_pred_rf)
rf_test_mae = mean_absolute_error(y_test, y_test_pred_rf)
rf_train_mape = np.mean(np.abs((y_train - y_train_pred_rf) / y_train)) * 100
rf_test_mape = np.mean(np.abs((y_test - y_test_pred_rf) / y_test)) * 100

print(f"Train R²: {rf_train_r2:.4f} | RMSE: ${rf_train_rmse:,.2f} | MAPE: {rf_train_mape:.2f}%")
print(f"Test R²:  {rf_test_r2:.4f} | RMSE: ${rf_test_rmse:,.2f} | MAPE: {rf_test_mape:.2f}%")
print(f"과적합 지표 (R² Gap): {rf_train_r2 - rf_test_r2:.4f}")

results.append({
    'model': 'Random Forest',
    'train_r2': rf_train_r2,
    'test_r2': rf_test_r2,
    'train_rmse': rf_train_rmse,
    'test_rmse': rf_test_rmse,
    'train_mape': rf_train_mape,
    'test_mape': rf_test_mape
})

# Feature Importance (RF)
rf_importance = pd.DataFrame({
    'feature': final_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n상위 10개 중요 특성:")
for i, row in rf_importance.head(10).iterrows():
    print(f"  {i+1:2d}. {row['feature']:30s} : {row['importance']:.4f}")

# XGBoost
print("\n[XGBoost]")
print("-" * 70)
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
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
xgb_train_mae = mean_absolute_error(y_train, y_train_pred_xgb)
xgb_test_mae = mean_absolute_error(y_test, y_test_pred_xgb)
xgb_train_mape = np.mean(np.abs((y_train - y_train_pred_xgb) / y_train)) * 100
xgb_test_mape = np.mean(np.abs((y_test - y_test_pred_xgb) / y_test)) * 100

print(f"Train R²: {xgb_train_r2:.4f} | RMSE: ${xgb_train_rmse:,.2f} | MAPE: {xgb_train_mape:.2f}%")
print(f"Test R²:  {xgb_test_r2:.4f} | RMSE: ${xgb_test_rmse:,.2f} | MAPE: {xgb_test_mape:.2f}%")
print(f"과적합 지표 (R² Gap): {xgb_train_r2 - xgb_test_r2:.4f}")

results.append({
    'model': 'XGBoost',
    'train_r2': xgb_train_r2,
    'test_r2': xgb_test_r2,
    'train_rmse': xgb_train_rmse,
    'test_rmse': xgb_test_rmse,
    'train_mape': xgb_train_mape,
    'test_mape': xgb_test_mape
})

# Feature Importance (XGB)
xgb_importance = pd.DataFrame({
    'feature': final_features,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n상위 10개 중요 특성:")
for i, row in xgb_importance.head(10).iterrows():
    print(f"  {i+1:2d}. {row['feature']:30s} : {row['importance']:.4f}")

# ===== 6. 비교: 기술적 지표 포함 vs 제외 =====
print("\n" + "=" * 70)
print("6. 비교: 기술적 지표 포함 vs 제외")
print("=" * 70)

print("\n기술적 지표 제외 (현재 모델):")
print(f"  Random Forest: Test R² = {rf_test_r2:.4f}, RMSE = ${rf_test_rmse:,.2f}")
print(f"  XGBoost:       Test R² = {xgb_test_r2:.4f}, RMSE = ${xgb_test_rmse:,.2f}")

print("\n기술적 지표 포함 (step9 결과 - Top 10):")
print(f"  XGBoost:       Test R² = 0.7583, RMSE = $3,416.71")

print("\n성능 비교:")
if xgb_test_r2 > 0:
    performance_drop = ((0.7583 - xgb_test_r2) / 0.7583) * 100
    print(f"  R² 하락: {performance_drop:.1f}%")
else:
    print(f"  R² 하락: 무한대 (음수로 전환)")

# ===== 7. 시각화 =====
print("\n7. 시각화")
print("-" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Train vs Test R² 비교
ax1 = axes[0, 0]
models = ['RF', 'XGB']
train_r2 = [rf_train_r2, xgb_train_r2]
test_r2 = [rf_test_r2, xgb_test_r2]
x = np.arange(len(models))
width = 0.35

ax1.bar(x - width/2, train_r2, width, label='Train R²', alpha=0.8)
ax1.bar(x + width/2, test_r2, width, label='Test R²', alpha=0.8)
ax1.set_xlabel('Model', fontsize=11)
ax1.set_ylabel('R² Score', fontsize=11)
ax1.set_title('Train vs Test R² (No Technical Indicators)', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)

# 2. Feature Importance (RF)
ax2 = axes[0, 1]
top_rf = rf_importance.head(15)
ax2.barh(range(len(top_rf)), top_rf['importance'], alpha=0.7)
ax2.set_yticks(range(len(top_rf)))
ax2.set_yticklabels(top_rf['feature'], fontsize=9)
ax2.set_xlabel('Importance', fontsize=11)
ax2.set_title('Feature Importance (Random Forest)', fontsize=12, fontweight='bold')
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3, axis='x')

# 3. Feature Importance (XGB)
ax3 = axes[0, 2]
top_xgb = xgb_importance.head(15)
ax3.barh(range(len(top_xgb)), top_xgb['importance'], alpha=0.7, color='orange')
ax3.set_yticks(range(len(top_xgb)))
ax3.set_yticklabels(top_xgb['feature'], fontsize=9)
ax3.set_xlabel('Importance', fontsize=11)
ax3.set_title('Feature Importance (XGBoost)', fontsize=12, fontweight='bold')
ax3.invert_yaxis()
ax3.grid(True, alpha=0.3, axis='x')

# 4. Actual vs Predicted (RF)
ax4 = axes[1, 0]
ax4.scatter(y_test, y_test_pred_rf, alpha=0.5, s=30)
ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        'r--', linewidth=2, label='Perfect Prediction')
ax4.set_xlabel('Actual Price ($)', fontsize=11)
ax4.set_ylabel('Predicted Price ($)', fontsize=11)
ax4.set_title(f'Random Forest (R²={rf_test_r2:.4f})', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Actual vs Predicted (XGB)
ax5 = axes[1, 1]
ax5.scatter(y_test, y_test_pred_xgb, alpha=0.5, s=30, color='orange')
ax5.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        'r--', linewidth=2, label='Perfect Prediction')
ax5.set_xlabel('Actual Price ($)', fontsize=11)
ax5.set_ylabel('Predicted Price ($)', fontsize=11)
ax5.set_title(f'XGBoost (R²={xgb_test_r2:.4f})', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Time Series Prediction
ax6 = axes[1, 2]
test_dates = y_test.index
ax6.plot(test_dates, y_test.values, label='Actual', linewidth=2, color='blue')
ax6.plot(test_dates, y_test_pred_xgb, label='Predicted (XGB)',
        linewidth=2, color='red', alpha=0.7, linestyle='--')
ax6.set_xlabel('Date', fontsize=11)
ax6.set_ylabel('BTC Price ($)', fontsize=11)
ax6.set_title(f'Time Series (MAPE={xgb_test_mape:.2f}%)', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('no_technical_indicators_prediction.png', dpi=300, bbox_inches='tight')
print("✓ no_technical_indicators_prediction.png")

plt.close()

# ===== 8. 결과 저장 =====
print("\n8. 결과 저장")
print("-" * 70)

results_df = pd.DataFrame(results)
results_df.to_csv('no_technical_indicators_results.csv', index=False)
print("✓ no_technical_indicators_results.csv")

rf_importance.to_csv('feature_importance_no_technical_rf.csv', index=False)
xgb_importance.to_csv('feature_importance_no_technical_xgb.csv', index=False)
print("✓ feature_importance_no_technical_rf.csv")
print("✓ feature_importance_no_technical_xgb.csv")

# 특성 목록 저장
with open('features_no_technical.txt', 'w') as f:
    f.write("기술적 지표 제외 특성 목록\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"총 {len(final_features)}개\n\n")
    for i, feat in enumerate(final_features, 1):
        f.write(f"{i}. {feat}\n")
print("✓ features_no_technical.txt")

print("\n" + "=" * 70)
print("기술적 지표 제외 가격 예측 완료!")
print("=" * 70)

print("\n📊 최종 요약:")
print("-" * 70)
print(f"사용된 특성: {len(final_features)}개")
print(f"  - BTC 원본: {len(btc_features)}개")
print(f"  - 전통 시장: {len(traditional_features)}개")
print(f"  - 거시경제: {len(macro_features)}개")
print(f"  - 감정/관심: {len(sentiment_features)}개")
print(f"  - 온체인: {len(onchain_features)}개")

print(f"\n최고 성능 모델: {'Random Forest' if rf_test_r2 > xgb_test_r2 else 'XGBoost'}")
best_r2 = max(rf_test_r2, xgb_test_r2)
best_rmse = rf_test_rmse if rf_test_r2 > xgb_test_r2 else xgb_test_rmse
best_mape = rf_test_mape if rf_test_r2 > xgb_test_r2 else xgb_test_mape

print(f"  Test R²: {best_r2:.4f}")
print(f"  Test RMSE: ${best_rmse:,.2f}")
print(f"  Test MAPE: {best_mape:.2f}%")

print("\n💡 결론:")
if best_r2 > 0.5:
    print("✅ 기술적 지표 없이도 좋은 성능!")
elif best_r2 > 0:
    print("⚠️ 기술적 지표 없이는 성능 제한적")
else:
    print("❌ 기술적 지표 필수 - 거시경제/온체인 데이터만으로는 1일 예측 불가능")

print("=" * 70)
