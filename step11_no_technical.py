import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("모델 훈련: 기술적 지표 제거 (2024년~최신, 7:3)")
print("=" * 70)

# ===== 1. 데이터 로드 =====
print("\n1. 데이터 로드 및 특성 선택")
print("-" * 70)

df = pd.read_csv('integrated_data_full.csv', index_col=0, parse_dates=True)
print(f"전체 데이터: {df.shape}")

# 2024년 이후 데이터
df_2024 = df[df.index >= '2024-01-01'].copy()
print(f"2024년 이후: {df_2024.shape} ({df_2024.index[0].date()} ~ {df_2024.index[-1].date()})")

# 타겟 설정
df_2024['target'] = df_2024['Close'].shift(-1)
df_2024 = df_2024.dropna(subset=['target'])

# ===== 2. 제거할 특성 정의 =====
print("\n2. 제거할 특성 정의")
print("-" * 70)

# Close와 거의 같은 값들
exclude_similar_to_close = [
    'Close', 'target',
    'High', 'Low', 'Open',  # 같은 날 가격
    'cumulative_return',  # Close로부터 직접 계산
    'bc_market_price',  # Close와 거의 동일
    'bc_market_cap',  # Close * Supply
]

# 기술적 지표 (모두 제거)
exclude_technical = [
    # 이동평균
    col for col in df_2024.columns if 'EMA' in col or 'SMA' in col
] + [
    # 모멘텀 지표
    'RSI', 'Stoch_K', 'Stoch_D', 'Williams_R', 'ROC', 'MFI',
    # 트렌드 지표
    'MACD', 'MACD_signal', 'MACD_diff', 'ADX', 'CCI',
    # 변동성 지표
    'BB_high', 'BB_low', 'BB_mid', 'BB_width', 'ATR', 'volatility_20d',
    # 거래량 지표
    'OBV', 'volume_change',
    # 수익률
    'daily_return',
    # 시가총액 (Close 기반)
    'market_cap_approx',
]

# 전체 제외 목록
exclude_cols = list(set(exclude_similar_to_close + exclude_technical))

print(f"제거할 특성 수: {len(exclude_cols)}개")
print("\n제거 카테고리:")
print(f"  - Close 유사 특성: {len(exclude_similar_to_close)}개")
print(f"  - 기술적 지표: {len([c for c in exclude_technical if c in df_2024.columns])}개")

# 남은 특성
all_features = [col for col in df_2024.columns if col not in exclude_cols]

print(f"\n남은 특성: {len(all_features)}개")

# 카테고리별 특성 확인
categories = {
    '원본 데이터': [c for c in all_features if c in ['Volume', 'Dividends', 'Stock Splits']],
    '전통 시장': [c for c in all_features if c in ['QQQ', 'SPX', 'UUP', 'EURUSD', 'GOLD', 'SILVER', 'OIL', 'BSV']],
    '거시경제': [c for c in all_features if c in ['DGS10', 'DFF', 'CPIAUCSL', 'UNRATE', 'M2SL', 'GDP', 'DEXUSEU', 'DTWEXBGS', 'T10Y2Y', 'VIXCLS']],
    '감정/관심': [c for c in all_features if 'fear_greed' in c or 'google_trends' in c],
    '온체인': [c for c in all_features if c.startswith('bc_') or c.startswith('cm_') or c.startswith('gn_')],
}

print("\n남은 특성 카테고리:")
for cat, feats in categories.items():
    if feats:
        print(f"  {cat}: {len(feats)}개")
        for feat in feats:
            print(f"    - {feat}")

X = df_2024[all_features].copy()
y = df_2024['target'].copy()

print(f"\n최종 데이터: {X.shape}")
print(f"샘플: {len(X)}개")

# ===== 3. 7:3 분할 =====
print("\n3. 데이터 분할 (7:3)")
print("-" * 70)

split_idx = int(len(X) * 0.7)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"훈련: {X_train.shape[0]}개 ({X_train.index[0].date()} ~ {X_train.index[-1].date()})")
print(f"테스트: {X_test.shape[0]}개 ({X_test.index[0].date()} ~ {X_test.index[-1].date()})")

# ===== 4. 모델 훈련 =====
print("\n" + "=" * 70)
print("4. 모델 훈련 및 평가")
print("=" * 70)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

all_results = []
predictions = {}

# ===== Random Forest =====
print("\n[Random Forest]")
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

rf_results = {
    'model': 'Random Forest',
    'train_r2': r2_score(y_train, y_train_pred_rf),
    'test_r2': r2_score(y_test, y_test_pred_rf),
    'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred_rf)),
    'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred_rf)),
    'train_mae': mean_absolute_error(y_train, y_train_pred_rf),
    'test_mae': mean_absolute_error(y_test, y_test_pred_rf),
    'train_mape': np.mean(np.abs((y_train - y_train_pred_rf) / y_train)) * 100,
    'test_mape': np.mean(np.abs((y_test - y_test_pred_rf) / y_test)) * 100,
}

all_results.append(rf_results)
predictions['RF'] = y_test_pred_rf

print(f"  Train R²: {rf_results['train_r2']:.4f} | Test R²: {rf_results['test_r2']:.4f}")
print(f"  Train RMSE: ${rf_results['train_rmse']:,.2f} | Test RMSE: ${rf_results['test_rmse']:,.2f}")
print(f"  Train MAE: ${rf_results['train_mae']:,.2f} | Test MAE: ${rf_results['test_mae']:,.2f}")
print(f"  Train MAPE: {rf_results['train_mape']:.2f}% | Test MAPE: {rf_results['test_mape']:.2f}%")

# Feature Importance (Random Forest)
rf_importance = pd.DataFrame({
    'feature': all_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n  상위 10개 중요 특성:")
for i, row in rf_importance.head(10).iterrows():
    print(f"    {i+1:2}. {row['feature']:30} : {row['importance']:.4f}")

# ===== XGBoost =====
print("\n[XGBoost]")
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

xgb_results = {
    'model': 'XGBoost',
    'train_r2': r2_score(y_train, y_train_pred_xgb),
    'test_r2': r2_score(y_test, y_test_pred_xgb),
    'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred_xgb)),
    'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred_xgb)),
    'train_mae': mean_absolute_error(y_train, y_train_pred_xgb),
    'test_mae': mean_absolute_error(y_test, y_test_pred_xgb),
    'train_mape': np.mean(np.abs((y_train - y_train_pred_xgb) / y_train)) * 100,
    'test_mape': np.mean(np.abs((y_test - y_test_pred_xgb) / y_test)) * 100,
}

all_results.append(xgb_results)
predictions['XGB'] = y_test_pred_xgb

print(f"  Train R²: {xgb_results['train_r2']:.4f} | Test R²: {xgb_results['test_r2']:.4f}")
print(f"  Train RMSE: ${xgb_results['train_rmse']:,.2f} | Test RMSE: ${xgb_results['test_rmse']:,.2f}")
print(f"  Train MAE: ${xgb_results['train_mae']:,.2f} | Test MAE: ${xgb_results['test_mae']:,.2f}")
print(f"  Train MAPE: {xgb_results['train_mape']:.2f}% | Test MAPE: {xgb_results['test_mape']:.2f}%")

# Feature Importance (XGBoost)
xgb_importance = pd.DataFrame({
    'feature': all_features,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n  상위 10개 중요 특성:")
for i, row in xgb_importance.head(10).iterrows():
    print(f"    {i+1:2}. {row['feature']:30} : {row['importance']:.4f}")

# ===== 5. 결과 저장 =====
print("\n" + "=" * 70)
print("5. 결과 저장")
print("=" * 70)

results_df = pd.DataFrame(all_results)
results_df.to_csv('model_results_no_technical.csv', index=False)
print("✓ model_results_no_technical.csv")

# 최고 성능 모델
best_model = results_df.loc[results_df['test_r2'].idxmax()]
print(f"\n🏆 최고 성능: {best_model['model']}")
print(f"   Test R²: {best_model['test_r2']:.4f}")
print(f"   Test RMSE: ${best_model['test_rmse']:,.2f}")
print(f"   Test MAPE: {best_model['test_mape']:.2f}%")

# ===== 6. 시각화 =====
print("\n6. 결과 시각화")
print("-" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 모델 비교 (R²)
models = ['Random Forest', 'XGBoost']
train_r2 = [rf_results['train_r2'], xgb_results['train_r2']]
test_r2 = [rf_results['test_r2'], xgb_results['test_r2']]

x_pos = np.arange(len(models))
width = 0.35

axes[0, 0].bar(x_pos - width/2, train_r2, width, label='Train R²', alpha=0.8)
axes[0, 0].bar(x_pos + width/2, test_r2, width, label='Test R²', alpha=0.8)
axes[0, 0].set_xlabel('Model')
axes[0, 0].set_ylabel('R² Score')
axes[0, 0].set_title('Model Comparison (R²)', fontweight='bold')
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(models)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis='y')
axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)

# 2. 모델 비교 (RMSE & MAPE)
ax2 = axes[0, 1]
ax2_twin = ax2.twinx()

test_rmse = [rf_results['test_rmse'], xgb_results['test_rmse']]
test_mape = [rf_results['test_mape'], xgb_results['test_mape']]

color1 = 'tab:blue'
ax2.bar(x_pos - width/2, test_rmse, width, label='Test RMSE', color=color1, alpha=0.8)
ax2.set_xlabel('Model')
ax2.set_ylabel('RMSE ($)', color=color1)
ax2.tick_params(axis='y', labelcolor=color1)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(models)

color2 = 'tab:orange'
ax2_twin.bar(x_pos + width/2, test_mape, width, label='Test MAPE', color=color2, alpha=0.8)
ax2_twin.set_ylabel('MAPE (%)', color=color2)
ax2_twin.tick_params(axis='y', labelcolor=color2)

axes[0, 1].set_title('Model Comparison (RMSE & MAPE)', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# 3. Feature Importance 비교 (상위 15개)
top_rf = rf_importance.head(15)
top_xgb = xgb_importance.head(15)

axes[1, 0].barh(range(len(top_rf)), top_rf['importance'], alpha=0.7)
axes[1, 0].set_yticks(range(len(top_rf)))
axes[1, 0].set_yticklabels(top_rf['feature'], fontsize=8)
axes[1, 0].set_xlabel('Importance')
axes[1, 0].set_title('Random Forest: Top 15 Features', fontweight='bold')
axes[1, 0].invert_yaxis()
axes[1, 0].grid(True, alpha=0.3, axis='x')

axes[1, 1].barh(range(len(top_xgb)), top_xgb['importance'], alpha=0.7, color='orange')
axes[1, 1].set_yticks(range(len(top_xgb)))
axes[1, 1].set_yticklabels(top_xgb['feature'], fontsize=8)
axes[1, 1].set_xlabel('Importance')
axes[1, 1].set_title('XGBoost: Top 15 Features', fontweight='bold')
axes[1, 1].invert_yaxis()
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('model_no_technical_comparison.png', dpi=300, bbox_inches='tight')
print("✓ model_no_technical_comparison.png")

# 시계열 예측 플롯
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

test_dates = y_test.index

# Random Forest
axes[0].plot(test_dates, y_test.values, label='Actual', linewidth=2, color='blue')
axes[0].plot(test_dates, y_test_pred_rf, label='Predicted', linewidth=2,
            color='red', alpha=0.7, linestyle='--')
axes[0].set_xlabel('Date', fontsize=11)
axes[0].set_ylabel('BTC Price ($)', fontsize=11)
axes[0].set_title(f'Random Forest: Time Series Prediction (R²={rf_results["test_r2"]:.4f})',
                 fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)

# XGBoost
axes[1].plot(test_dates, y_test.values, label='Actual', linewidth=2, color='blue')
axes[1].plot(test_dates, y_test_pred_xgb, label='Predicted', linewidth=2,
            color='red', alpha=0.7, linestyle='--')
axes[1].set_xlabel('Date', fontsize=11)
axes[1].set_ylabel('BTC Price ($)', fontsize=11)
axes[1].set_title(f'XGBoost: Time Series Prediction (R²={xgb_results["test_r2"]:.4f})',
                 fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('model_no_technical_timeseries.png', dpi=300, bbox_inches='tight')
print("✓ model_no_technical_timeseries.png")

plt.close('all')

# ===== 7. 2021년 결과와 비교 =====
print("\n" + "=" * 70)
print("7. 기술적 지표 포함 vs 제외 비교")
print("=" * 70)

print("\n2021년 데이터 (기술적 지표 포함):")
results_2021 = pd.read_csv('model_results_2021.csv')
best_2021 = results_2021.iloc[0]
print(f"  모델: {best_2021['model']}")
print(f"  Test R²: {best_2021['test_r2']:.4f}")
print(f"  Test RMSE: ${best_2021['test_rmse']:,.2f}")
print(f"  Test MAPE: {best_2021['test_mape']:.2f}%")

print(f"\n2024년 데이터 (기술적 지표 제외):")
print(f"  모델: {best_model['model']}")
print(f"  Test R²: {best_model['test_r2']:.4f}")
print(f"  Test RMSE: ${best_model['test_rmse']:,.2f}")
print(f"  Test MAPE: {best_model['test_mape']:.2f}%")

print("\n" + "=" * 70)
print("기술적 지표 제거 모델 훈련 완료!")
print("=" * 70)
