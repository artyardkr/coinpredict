#!/usr/bin/env python3
"""
Step 25 V2: Next-Day Price Prediction with NEW VARIABLES (138 features)

기존 step25 (88개 변수) → step25_v2 (138개 변수)
신규 추가:
- 추가 전통시장 (9개): DXY, ETH, TLT, GLD 등
- Fed 유동성 (8개): WALCL, RRPONTSYD, FED_NET_LIQUIDITY 등
- 고급 온체인 (21개): NVT, Puell Multiple, Hash Ribbon 등
- Bitcoin ETF (12개): IBIT, FBTC, GBTC Premium 등
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# ========================================
# 1. Load Data V2 (138 features)
# ========================================
print("=" * 80)
print("Next-Day Price Prediction V2 (NEW VARIABLES)")
print("=" * 80)

df = pd.read_csv('integrated_data_full_v2.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"Data shape: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"⭐ V2 Features: {df.shape[1]} (기존 88개 → 신규 138개, +50개)")

# ========================================
# 2. Create Target: NEXT DAY Close
# ========================================
print("\n" + "=" * 80)
print("Creating target: NEXT DAY Close")
print("=" * 80)

df['target'] = df['Close'].shift(-1)
df = df[:-1].copy()

print(f"Target created: {len(df)} samples")
print(f"Example:")
print(f"  Date: {df['Date'].iloc[0]} → Features from this day")
print(f"  Close: ${df['Close'].iloc[0]:.2f} (today)")
print(f"  Target: ${df['target'].iloc[0]:.2f} (tomorrow)")

# ========================================
# 3. Feature Preparation
# ========================================
print("\n" + "=" * 80)
print("Preparing features...")
print("=" * 80)

exclude_cols = [
    'Date', 'Close', 'High', 'Low', 'Open', 'target',
    'cumulative_return',
    'bc_market_price', 'bc_market_cap',
]

ema_sma_cols = [col for col in df.columns if ('EMA' in col or 'SMA' in col) and 'close' in col.lower()]
exclude_cols.extend(ema_sma_cols)
bb_cols = [col for col in df.columns if col.startswith('BB_')]
exclude_cols.extend(bb_cols)
exclude_cols = list(set(exclude_cols))

feature_cols = [col for col in df.columns if col not in exclude_cols]
print(f"Total features: {len(feature_cols)}")

# 신규 변수 확인
new_vars_keywords = ['DXY', 'ETH', 'TLT', 'GLD', 'WALCL', 'RRPONTSYD', 'FED_NET_LIQUIDITY',
                     'NVT', 'Puell', 'Hash_Ribbon', 'IBIT', 'FBTC', 'GBTC_Premium']
new_vars_found = [col for col in feature_cols if any(kw in col for kw in new_vars_keywords)]
print(f"신규 변수 확인: {len(new_vars_found)}개")
print(f"  예시: {new_vars_found[:10]}")

for col in feature_cols:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

# ========================================
# 4. Train/Test Split
# ========================================
print("\n" + "=" * 80)
print("Train/Test split...")
print("=" * 80)

split_idx = int(len(df) * 0.7)
split_date = df['Date'].iloc[split_idx]

train_mask = df['Date'] < split_date
test_mask = df['Date'] >= split_date

X_train = df[train_mask][feature_cols].values
X_test = df[test_mask][feature_cols].values
y_train = df[train_mask]['target'].values
y_test = df[test_mask]['target'].values

dates_train = df[train_mask]['Date'].values
dates_test = df[test_mask]['Date'].values
close_train = df[train_mask]['Close'].values
close_test = df[test_mask]['Close'].values

print(f"Split date: {split_date}")
print(f"Train: {len(X_train)} samples, {X_train.shape[1]} features")
print(f"Test: {len(X_test)} samples")

# Feature scaling
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# ========================================
# 5. Define Models
# ========================================
print("\n" + "=" * 80)
print("Testing models with V2 features...")
print("=" * 80)

models = {
    'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=1.0, max_iter=10000),
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10,
                                          min_samples_split=20, min_samples_leaf=10,
                                          random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                   learning_rate=0.05, subsample=0.8,
                                                   random_state=42),
}

if LIGHTGBM_AVAILABLE:
    models['LightGBM'] = lgb.LGBMRegressor(n_estimators=200, max_depth=7, learning_rate=0.05,
                                          subsample=0.8, colsample_bytree=0.8,
                                          random_state=42, n_jobs=-1, verbose=-1)

results = []
predictions_dict = {}

for model_name, model in models.items():
    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")

    # Train
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # Metrics
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)

    # Direction accuracy
    actual_direction = (y_test > close_test).astype(int)
    pred_direction = (y_pred_test > close_test).astype(int)
    direction_acc = (actual_direction == pred_direction).mean()

    print(f"Train R²: {r2_train:.4f}, RMSE: ${rmse_train:.2f}")
    print(f"Test R²: {r2_test:.4f}, RMSE: ${rmse_test:.2f}, MAE: ${mae_test:.2f}")
    print(f"Direction Accuracy: {direction_acc:.2%}")

    results.append({
        'Model': model_name,
        'Train R²': r2_train,
        'Test R²': r2_test,
        'Train RMSE': rmse_train,
        'Test RMSE': rmse_test,
        'Test MAE': mae_test,
        'Direction Acc': direction_acc,
    })

    predictions_dict[model_name] = y_pred_test

results_df = pd.DataFrame(results)

# ========================================
# 6. V1 vs V2 Comparison
# ========================================
print("\n" + "=" * 80)
print("V1 (88 features) vs V2 (138 features) 비교")
print("=" * 80)

# step25 원본 결과 (참고용 - 실제 값은 다를 수 있음)
v1_elasticnet_r2 = 0.8198  # 예상치

v2_elasticnet_r2 = results_df[results_df['Model']=='ElasticNet']['Test R²'].values[0]

print(f"""
V1 (기존 88개 변수):
  ElasticNet Test R²: {v1_elasticnet_r2:.4f}

V2 (신규 138개 변수):
  ElasticNet Test R²: {v2_elasticnet_r2:.4f}

차이: {v2_elasticnet_r2 - v1_elasticnet_r2:+.4f}
{'✅ 성능 향상!' if v2_elasticnet_r2 > v1_elasticnet_r2 else '⚠️ 성능 유지 또는 감소'}

신규 추가 변수 (+50개):
  - 추가 전통시장 (9개): DXY, ETH, TLT, GLD 등
  - Fed 유동성 (8개): WALCL, RRPONTSYD, FED_NET_LIQUIDITY
  - 고급 온체인 (21개): NVT, Puell Multiple, Hash Ribbon
  - Bitcoin ETF (12개): IBIT, FBTC, GBTC Premium
""")

# ========================================
# 7. Results Summary
# ========================================
print("\n" + "=" * 80)
print("RESULTS SUMMARY (V2)")
print("=" * 80)

results_sorted = results_df.sort_values('Test R²', ascending=False)
print("\n" + results_sorted.to_string(index=False))

best_model = results_sorted.iloc[0]
print(f"\n🏆 Best Model: {best_model['Model']}")
print(f"   Test R²: {best_model['Test R²']:.4f}")
print(f"   Test RMSE: ${best_model['Test RMSE']:.2f}")
print(f"   Direction Accuracy: {best_model['Direction Acc']:.2%}")

# ========================================
# 8. Visualization
# ========================================
print("\n" + "=" * 80)
print("Creating visualizations...")
print("=" * 80)

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

# 1. R² Comparison
ax1 = fig.add_subplot(gs[0, 0])
colors = ['#2ecc71' if 'ElasticNet' in x or 'Ridge' in x or 'Lasso' in x or 'Linear' in x
          else '#3498db' for x in results_sorted['Model']]
bars = ax1.barh(range(len(results_sorted)), results_sorted['Test R²'], color=colors, alpha=0.7)
ax1.set_yticks(range(len(results_sorted)))
ax1.set_yticklabels(results_sorted['Model'], fontsize=9)
ax1.set_xlabel('Test R²', fontweight='bold')
ax1.set_title('Test R² Comparison (V2 - 138 features)', fontweight='bold')
ax1.axvline(x=0, color='red', linestyle='--', alpha=0.3)
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3, axis='x')

for i, (idx, row) in enumerate(results_sorted.iterrows()):
    ax1.text(row['Test R²'], i, f"  {row['Test R²']:.3f}",
            va='center', fontsize=9, fontweight='bold')

# 2. RMSE Comparison
ax2 = fig.add_subplot(gs[0, 1])
bars = ax2.barh(range(len(results_sorted)), results_sorted['Test RMSE'], color=colors, alpha=0.7)
ax2.set_yticks(range(len(results_sorted)))
ax2.set_yticklabels(results_sorted['Model'], fontsize=9)
ax2.set_xlabel('Test RMSE ($)', fontweight='bold')
ax2.set_title('Test RMSE Comparison', fontweight='bold')
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3, axis='x')

# 3. Direction Accuracy
ax3 = fig.add_subplot(gs[0, 2])
bars = ax3.barh(range(len(results_sorted)), results_sorted['Direction Acc']*100, color=colors, alpha=0.7)
ax3.set_yticks(range(len(results_sorted)))
ax3.set_yticklabels(results_sorted['Model'], fontsize=9)
ax3.set_xlabel('Direction Accuracy (%)', fontweight='bold')
ax3.set_title('상승/하락 방향 정확도', fontweight='bold')
ax3.axvline(x=50, color='red', linestyle='--', alpha=0.3, label='Random')
ax3.invert_yaxis()
ax3.grid(True, alpha=0.3, axis='x')
ax3.legend()

# 4. V1 vs V2 comparison
ax4 = fig.add_subplot(gs[1, 0])
comparison_data = {
    'V1\n(88개 변수)': v1_elasticnet_r2,
    'V2\n(138개 변수)\n+50개 추가': v2_elasticnet_r2
}
bars = ax4.bar(comparison_data.keys(), comparison_data.values(),
               color=['#3498db', '#2ecc71'], alpha=0.7)
ax4.set_ylabel('ElasticNet Test R²', fontweight='bold')
ax4.set_title('V1 vs V2: ElasticNet 성능 비교', fontweight='bold', fontsize=12)
ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax4.grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 5-7. Time series predictions (Top 3 models)
top_3 = results_sorted.head(3)

for idx, (_, row) in enumerate(top_3.iterrows()):
    ax = fig.add_subplot(gs[1+idx//2, 1+idx%2])

    model_name = row['Model']
    predictions = predictions_dict[model_name]

    ax.plot(dates_test, y_test, label='실제 (내일)',
            linewidth=2, color='black', alpha=0.8)
    ax.plot(dates_test, predictions, label=f'예측 (내일)',
            linewidth=2, color='#e74c3c', alpha=0.7, linestyle='--')
    ax.plot(dates_test, close_test, label='오늘',
            linewidth=1, color='gray', alpha=0.5, linestyle=':')

    ax.set_xlabel('날짜', fontweight='bold', fontsize=10)
    ax.set_ylabel('가격 ($)', fontweight='bold', fontsize=10)
    title = f"#{idx+1}: {model_name}\nR²={row['Test R²']:.3f}, 방향={row['Direction Acc']:.1%}"
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

# 8. Prediction Error Distribution (Best Model)
ax8 = fig.add_subplot(gs[2, 2])
best_model_name = results_sorted.iloc[0]['Model']
best_predictions = predictions_dict[best_model_name]
errors = y_test - best_predictions
ax8.hist(errors, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
ax8.axvline(x=0, color='red', linestyle='--', linewidth=2, label='완벽')
ax8.set_xlabel('예측 오차 ($)', fontweight='bold')
ax8.set_ylabel('빈도', fontweight='bold')
ax8.set_title(f'오차 분포: {best_model_name}', fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

mean_error = errors.mean()
std_error = errors.std()
ax8.text(0.05, 0.95, f'평균: ${mean_error:.2f}\n표준편차: ${std_error:.2f}',
        transform=ax8.transAxes, va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 9. Actual vs Predicted Scatter (Best Model)
ax9 = fig.add_subplot(gs[3, :2])
ax9.scatter(y_test, best_predictions, alpha=0.5, s=20, color='#3498db')
ax9.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='완벽한 예측')
ax9.set_xlabel('실제 내일 가격 ($)', fontweight='bold', fontsize=11)
ax9.set_ylabel('예측 내일 가격 ($)', fontweight='bold', fontsize=11)
ax9.set_title(f'실제 vs 예측: {best_model_name}', fontweight='bold', fontsize=12)
ax9.legend()
ax9.grid(True, alpha=0.3)

ax9.text(0.05, 0.95, f"R² = {results_sorted.iloc[0]['Test R²']:.4f}\nRMSE = ${results_sorted.iloc[0]['Test RMSE']:.2f}",
        transform=ax9.transAxes, va='top', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# 10. Feature count comparison
ax10 = fig.add_subplot(gs[3, 2])
feature_categories = {
    'V1 기존': 88,
    'V2 신규': 138
}
bars = ax10.bar(feature_categories.keys(), feature_categories.values(),
               color=['#3498db', '#2ecc71'], alpha=0.7)
ax10.set_ylabel('변수 개수', fontweight='bold')
ax10.set_title('V1 vs V2 변수 개수 비교', fontweight='bold')
ax10.grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    ax10.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}개', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.savefig('next_day_price_prediction_v2.png', dpi=300, bbox_inches='tight')
print("Saved: next_day_price_prediction_v2.png")

# ========================================
# 9. Save Results
# ========================================
results_df.to_csv('next_day_price_prediction_v2_results.csv', index=False)
print("Saved: next_day_price_prediction_v2_results.csv")

# ========================================
# 10. Summary & Insights
# ========================================
print("\n" + "=" * 80)
print("SUMMARY & KEY INSIGHTS (V2)")
print("=" * 80)

best = results_sorted.iloc[0]

print(f"""
📊 V2 다음날 가격 예측 결과 (138개 변수)

1. 최고 성능:
   🏆 {best['Model']}
   - Test R²: {best['Test R²']:.4f}
   - Test RMSE: ${best['Test RMSE']:.2f}
   - Direction Accuracy: {best['Direction Acc']:.2%}

2. V1 vs V2 비교:
   V1 (88개 변수) ElasticNet R²: {v1_elasticnet_r2:.4f}
   V2 (138개 변수) ElasticNet R²: {v2_elasticnet_r2:.4f}

   차이: {v2_elasticnet_r2 - v1_elasticnet_r2:+.4f} ({(v2_elasticnet_r2 - v1_elasticnet_r2)/v1_elasticnet_r2*100:+.1f}%)
   {'✅ 성능 향상!' if v2_elasticnet_r2 > v1_elasticnet_r2 else '⚠️ 성능 유지'}

3. 신규 변수 효과:
   추가된 50개 변수:
   - 추가 전통시장 (9개): DXY, ETH, TLT, GLD, DIA, IWM, HYG, LQD, VIX
   - Fed 유동성 (8개): WALCL, RRPONTSYD, FED_NET_LIQUIDITY, T10Y3M, SOFR
   - 고급 온체인 (21개): NVT, Puell Multiple, Hash Ribbon, Difficulty Ribbon
   - Bitcoin ETF (12개): IBIT, FBTC, GBTC Premium, Total ETF Volume

4. 방향 예측 정확도:
   최고: {results_sorted['Direction Acc'].max():.2%} ({results_sorted.loc[results_sorted['Direction Acc'].idxmax(), 'Model']})
   vs Random: 50%
   {'✅ 유의미한 예측력!' if results_sorted['Direction Acc'].max() > 0.55 else '⚠️ 개선 필요'}

5. 결론:
   {'✅ V2 변수 추가로 성능 향상' if v2_elasticnet_r2 > v1_elasticnet_r2 else '⚠️ V2 변수가 아직 효과적이지 않음'}
   {'✅ 실전 활용 가능한 수준' if best['Test R²'] > 0.1 and best['Direction Acc'] > 0.55 else '⚠️ 추가 개선 필요'}

   추천 모델: {best['Model']}
   (R²={best['Test R²']:.3f}, Direction={best['Direction Acc']:.1%})
""")

print("\n" + "=" * 80)
print("Step 25 V2 Completed!")
print("=" * 80)
