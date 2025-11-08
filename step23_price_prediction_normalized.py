#!/usr/bin/env python3
"""
Step 23: Price Prediction with Target Normalization

기존 가격 예측의 Extrapolation 문제를 해결하기 위해:
1. Z-score 표준화
2. Log 변환
3. Min-Max 정규화
4. Robust Scaler

각 방법의 성능을 비교하고 Extrapolation 문제 개선 확인
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ========================================
# 1. Load Data
# ========================================
print("=" * 80)
print("Price Prediction with Target Normalization")
print("=" * 80)

df = pd.read_csv('integrated_data_full.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"Data shape: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Price range: ${df['Close'].min():.0f} - ${df['Close'].max():.0f}")

ETF_DATE = pd.to_datetime('2024-01-10')

# ========================================
# 2. Feature Preparation
# ========================================
print("\n" + "=" * 80)
print("Preparing features...")
print("=" * 80)

exclude_cols = [
    'Date', 'Close', 'High', 'Low', 'Open',
    'cumulative_return',
    'bc_market_price', 'bc_market_cap',
]

# EMA/SMA close features (data leakage)
ema_sma_cols = [col for col in df.columns if ('EMA' in col or 'SMA' in col) and 'close' in col.lower()]
exclude_cols.extend(ema_sma_cols)

# Bollinger Bands
bb_cols = [col for col in df.columns if col.startswith('BB_')]
exclude_cols.extend(bb_cols)

exclude_cols = list(set(exclude_cols))

feature_cols = [col for col in df.columns if col not in exclude_cols]
print(f"Total features: {len(feature_cols)}")

# Handle missing/inf values
for col in feature_cols:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

# ========================================
# 3. Train/Test Split by Date (70:30)
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
y_train = df[train_mask]['Close'].values
y_test = df[test_mask]['Close'].values

dates_train = df[train_mask]['Date'].values
dates_test = df[test_mask]['Date'].values

print(f"Split date: {split_date}")
print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Train price range: ${y_train.min():.0f} - ${y_train.max():.0f} (mean: ${y_train.mean():.0f})")
print(f"Test price range: ${y_test.min():.0f} - ${y_test.max():.0f} (mean: ${y_test.mean():.0f})")
print(f"Extrapolation ratio: {y_test.mean() / y_train.mean():.2f}x")

# Feature scaling (always do this)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# ========================================
# 4. Baseline: No Target Normalization
# ========================================
print("\n" + "=" * 80)
print("METHOD 1: Baseline (No Target Normalization)")
print("=" * 80)

rf_baseline = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

rf_baseline.fit(X_train_scaled, y_train)

y_pred_train_baseline = rf_baseline.predict(X_train_scaled)
y_pred_test_baseline = rf_baseline.predict(X_test_scaled)

r2_train_baseline = r2_score(y_train, y_pred_train_baseline)
r2_test_baseline = r2_score(y_test, y_pred_test_baseline)
rmse_test_baseline = np.sqrt(mean_squared_error(y_test, y_pred_test_baseline))
mae_test_baseline = mean_absolute_error(y_test, y_pred_test_baseline)

print(f"Train R²: {r2_train_baseline:.4f}")
print(f"Test R²: {r2_test_baseline:.4f}")
print(f"Test RMSE: ${rmse_test_baseline:.2f}")
print(f"Test MAE: ${mae_test_baseline:.2f}")
print(f"Mean prediction: ${y_pred_test_baseline.mean():.2f}")
print(f"Prediction range: ${y_pred_test_baseline.min():.2f} - ${y_pred_test_baseline.max():.2f}")

# ========================================
# 5. Method 2: Z-score Standardization
# ========================================
print("\n" + "=" * 80)
print("METHOD 2: Z-score Standardization")
print("=" * 80)

scaler_y_zscore = StandardScaler()
y_train_zscore = scaler_y_zscore.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_zscore = scaler_y_zscore.transform(y_test.reshape(-1, 1)).flatten()

print(f"Original train: mean=${y_train.mean():.2f}, std=${y_train.std():.2f}")
print(f"Standardized train: mean={y_train_zscore.mean():.4f}, std={y_train_zscore.std():.4f}")

rf_zscore = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

rf_zscore.fit(X_train_scaled, y_train_zscore)

y_pred_train_zscore_norm = rf_zscore.predict(X_train_scaled)
y_pred_test_zscore_norm = rf_zscore.predict(X_test_scaled)

# Inverse transform
y_pred_train_zscore = scaler_y_zscore.inverse_transform(y_pred_train_zscore_norm.reshape(-1, 1)).flatten()
y_pred_test_zscore = scaler_y_zscore.inverse_transform(y_pred_test_zscore_norm.reshape(-1, 1)).flatten()

r2_train_zscore = r2_score(y_train, y_pred_train_zscore)
r2_test_zscore = r2_score(y_test, y_pred_test_zscore)
rmse_test_zscore = np.sqrt(mean_squared_error(y_test, y_pred_test_zscore))
mae_test_zscore = mean_absolute_error(y_test, y_pred_test_zscore)

print(f"Train R²: {r2_train_zscore:.4f}")
print(f"Test R²: {r2_test_zscore:.4f}")
print(f"Test RMSE: ${rmse_test_zscore:.2f}")
print(f"Test MAE: ${mae_test_zscore:.2f}")
print(f"Mean prediction: ${y_pred_test_zscore.mean():.2f}")
print(f"Prediction range: ${y_pred_test_zscore.min():.2f} - ${y_pred_test_zscore.max():.2f}")

# ========================================
# 6. Method 3: Log Transformation
# ========================================
print("\n" + "=" * 80)
print("METHOD 3: Log Transformation")
print("=" * 80)

y_train_log = np.log(y_train)
y_test_log = np.log(y_test)

print(f"Original train: ${y_train.min():.2f} - ${y_train.max():.2f}")
print(f"Log train: {y_train_log.min():.4f} - {y_train_log.max():.4f}")

rf_log = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

rf_log.fit(X_train_scaled, y_train_log)

y_pred_train_log_norm = rf_log.predict(X_train_scaled)
y_pred_test_log_norm = rf_log.predict(X_test_scaled)

# Inverse transform (exp)
y_pred_train_log = np.exp(y_pred_train_log_norm)
y_pred_test_log = np.exp(y_pred_test_log_norm)

r2_train_log = r2_score(y_train, y_pred_train_log)
r2_test_log = r2_score(y_test, y_pred_test_log)
rmse_test_log = np.sqrt(mean_squared_error(y_test, y_pred_test_log))
mae_test_log = mean_absolute_error(y_test, y_pred_test_log)

print(f"Train R²: {r2_train_log:.4f}")
print(f"Test R²: {r2_test_log:.4f}")
print(f"Test RMSE: ${rmse_test_log:.2f}")
print(f"Test MAE: ${mae_test_log:.2f}")
print(f"Mean prediction: ${y_pred_test_log.mean():.2f}")
print(f"Prediction range: ${y_pred_test_log.min():.2f} - ${y_pred_test_log.max():.2f}")

# ========================================
# 7. Method 4: Min-Max Normalization
# ========================================
print("\n" + "=" * 80)
print("METHOD 4: Min-Max Normalization (0-1)")
print("=" * 80)

scaler_y_minmax = MinMaxScaler()
y_train_minmax = scaler_y_minmax.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_minmax = scaler_y_minmax.transform(y_test.reshape(-1, 1)).flatten()

print(f"Original train: ${y_train.min():.2f} - ${y_train.max():.2f}")
print(f"MinMax train: {y_train_minmax.min():.4f} - {y_train_minmax.max():.4f}")
print(f"MinMax test: {y_test_minmax.min():.4f} - {y_test_minmax.max():.4f}")

rf_minmax = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

rf_minmax.fit(X_train_scaled, y_train_minmax)

y_pred_train_minmax_norm = rf_minmax.predict(X_train_scaled)
y_pred_test_minmax_norm = rf_minmax.predict(X_test_scaled)

# Inverse transform
y_pred_train_minmax = scaler_y_minmax.inverse_transform(y_pred_train_minmax_norm.reshape(-1, 1)).flatten()
y_pred_test_minmax = scaler_y_minmax.inverse_transform(y_pred_test_minmax_norm.reshape(-1, 1)).flatten()

r2_train_minmax = r2_score(y_train, y_pred_train_minmax)
r2_test_minmax = r2_score(y_test, y_pred_test_minmax)
rmse_test_minmax = np.sqrt(mean_squared_error(y_test, y_pred_test_minmax))
mae_test_minmax = mean_absolute_error(y_test, y_pred_test_minmax)

print(f"Train R²: {r2_train_minmax:.4f}")
print(f"Test R²: {r2_test_minmax:.4f}")
print(f"Test RMSE: ${rmse_test_minmax:.2f}")
print(f"Test MAE: ${mae_test_minmax:.2f}")
print(f"Mean prediction: ${y_pred_test_minmax.mean():.2f}")
print(f"Prediction range: ${y_pred_test_minmax.min():.2f} - ${y_pred_test_minmax.max():.2f}")

# ========================================
# 8. Method 5: Robust Scaler
# ========================================
print("\n" + "=" * 80)
print("METHOD 5: Robust Scaler (Median + IQR)")
print("=" * 80)

scaler_y_robust = RobustScaler()
y_train_robust = scaler_y_robust.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_robust = scaler_y_robust.transform(y_test.reshape(-1, 1)).flatten()

print(f"Original train median: ${np.median(y_train):.2f}, IQR: ${np.percentile(y_train, 75) - np.percentile(y_train, 25):.2f}")
print(f"Robust train: {y_train_robust.min():.4f} - {y_train_robust.max():.4f}")

rf_robust = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

rf_robust.fit(X_train_scaled, y_train_robust)

y_pred_train_robust_norm = rf_robust.predict(X_train_scaled)
y_pred_test_robust_norm = rf_robust.predict(X_test_scaled)

# Inverse transform
y_pred_train_robust = scaler_y_robust.inverse_transform(y_pred_train_robust_norm.reshape(-1, 1)).flatten()
y_pred_test_robust = scaler_y_robust.inverse_transform(y_pred_test_robust_norm.reshape(-1, 1)).flatten()

r2_train_robust = r2_score(y_train, y_pred_train_robust)
r2_test_robust = r2_score(y_test, y_pred_test_robust)
rmse_test_robust = np.sqrt(mean_squared_error(y_test, y_pred_test_robust))
mae_test_robust = mean_absolute_error(y_test, y_pred_test_robust)

print(f"Train R²: {r2_train_robust:.4f}")
print(f"Test R²: {r2_test_robust:.4f}")
print(f"Test RMSE: ${rmse_test_robust:.2f}")
print(f"Test MAE: ${mae_test_robust:.2f}")
print(f"Mean prediction: ${y_pred_test_robust.mean():.2f}")
print(f"Prediction range: ${y_pred_test_robust.min():.2f} - ${y_pred_test_robust.max():.2f}")

# ========================================
# 9. Comparison Summary
# ========================================
print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)

results = pd.DataFrame({
    'Method': ['Baseline (No Norm)', 'Z-score', 'Log Transform', 'Min-Max', 'Robust Scaler'],
    'Train R²': [r2_train_baseline, r2_train_zscore, r2_train_log, r2_train_minmax, r2_train_robust],
    'Test R²': [r2_test_baseline, r2_test_zscore, r2_test_log, r2_test_minmax, r2_test_robust],
    'Test RMSE': [rmse_test_baseline, rmse_test_zscore, rmse_test_log, rmse_test_minmax, rmse_test_robust],
    'Test MAE': [mae_test_baseline, mae_test_zscore, mae_test_log, mae_test_minmax, mae_test_robust],
    'Mean Pred': [y_pred_test_baseline.mean(), y_pred_test_zscore.mean(), y_pred_test_log.mean(),
                  y_pred_test_minmax.mean(), y_pred_test_robust.mean()]
})

print("\n" + results.to_string(index=False))

# Find best method
best_idx = results['Test R²'].idxmax()
best_method = results.loc[best_idx, 'Method']
best_r2 = results.loc[best_idx, 'Test R²']

print(f"\n🏆 Best Method: {best_method} (Test R² = {best_r2:.4f})")

# ========================================
# 10. Visualization
# ========================================
print("\n" + "=" * 80)
print("Creating visualizations...")
print("=" * 80)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. R² Comparison
ax1 = fig.add_subplot(gs[0, :])
x = np.arange(len(results))
width = 0.35
bars1 = ax1.bar(x - width/2, results['Train R²'], width, label='Train R²', color='#3498db', alpha=0.7)
bars2 = ax1.bar(x + width/2, results['Test R²'], width, label='Test R²', color='#e74c3c', alpha=0.7)
ax1.set_xlabel('Method', fontweight='bold', fontsize=12)
ax1.set_ylabel('R² Score', fontweight='bold', fontsize=12)
ax1.set_title('R² Comparison: Effect of Target Normalization', fontweight='bold', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(results['Method'], rotation=15, ha='right')
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9, fontweight='bold')

# 2. RMSE Comparison
ax2 = fig.add_subplot(gs[1, 0])
bars = ax2.bar(results['Method'], results['Test RMSE'], color='#9b59b6', alpha=0.7)
ax2.set_ylabel('RMSE ($)', fontweight='bold')
ax2.set_title('Test RMSE Comparison', fontweight='bold')
ax2.set_xticklabels(results['Method'], rotation=45, ha='right', fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'${height:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 3. MAE Comparison
ax3 = fig.add_subplot(gs[1, 1])
bars = ax3.bar(results['Method'], results['Test MAE'], color='#f39c12', alpha=0.7)
ax3.set_ylabel('MAE ($)', fontweight='bold')
ax3.set_title('Test MAE Comparison', fontweight='bold')
ax3.set_xticklabels(results['Method'], rotation=45, ha='right', fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'${height:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 4. Mean Prediction vs Actual
ax4 = fig.add_subplot(gs[1, 2])
actual_mean = y_test.mean()
x_pos = np.arange(len(results) + 1)
means = list(results['Mean Pred']) + [actual_mean]
colors_pred = ['#3498db'] * len(results) + ['#e74c3c']
bars = ax4.bar(x_pos, means, color=colors_pred, alpha=0.7)
ax4.set_ylabel('Mean Price ($)', fontweight='bold')
ax4.set_title('Mean Prediction vs Actual', fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(list(results['Method']) + ['Actual'], rotation=45, ha='right', fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'${height:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 5-7. Time Series Predictions (Best 3 methods + Baseline)
predictions_to_plot = [
    ('Baseline', y_pred_test_baseline, '#95a5a6'),
    ('Log Transform', y_pred_test_log, '#2ecc71'),
    (best_method, [y_pred_test_baseline, y_pred_test_zscore, y_pred_test_log,
                   y_pred_test_minmax, y_pred_test_robust][best_idx], '#e74c3c')
]

for idx, (method_name, predictions, color) in enumerate(predictions_to_plot):
    ax = fig.add_subplot(gs[2, idx])
    ax.plot(dates_test, y_test, label='Actual', linewidth=2, color='black', alpha=0.7)
    ax.plot(dates_test, predictions, label=f'Predicted ({method_name})',
            linewidth=2, color=color, alpha=0.7, linestyle='--')
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Price ($)', fontweight='bold')
    ax.set_title(f'{method_name} Predictions', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

plt.savefig('price_prediction_normalized.png', dpi=300, bbox_inches='tight')
print("Saved: price_prediction_normalized.png")

# ========================================
# 11. Detailed Analysis
# ========================================
print("\n" + "=" * 80)
print("DETAILED ANALYSIS")
print("=" * 80)

print(f"""
📊 Target Normalization 효과 분석

1. 성능 순위 (Test R²):
   1) {results.loc[results['Test R²'].idxmax(), 'Method']}: {results['Test R²'].max():.4f} 🏆
   2) {results.loc[results['Test R²'].nlargest(2).index[1], 'Method']}: {results['Test R²'].nlargest(2).iloc[1]:.4f}
   3) {results.loc[results['Test R²'].nlargest(3).index[2], 'Method']}: {results['Test R²'].nlargest(3).iloc[2]:.4f}

2. Baseline 대비 개선:
   Z-score: {(r2_test_zscore - r2_test_baseline):.4f} ({(r2_test_zscore - r2_test_baseline)/abs(r2_test_baseline)*100:+.1f}%)
   Log Transform: {(r2_test_log - r2_test_baseline):.4f} ({(r2_test_log - r2_test_baseline)/abs(r2_test_baseline)*100:+.1f}%)
   Min-Max: {(r2_test_minmax - r2_test_baseline):.4f} ({(r2_test_minmax - r2_test_baseline)/abs(r2_test_baseline)*100:+.1f}%)
   Robust: {(r2_test_robust - r2_test_baseline):.4f} ({(r2_test_robust - r2_test_baseline)/abs(r2_test_baseline)*100:+.1f}%)

3. 예측 평균 vs 실제 평균:
   실제: ${y_test.mean():.2f}
   Baseline: ${y_pred_test_baseline.mean():.2f} (차이: ${abs(y_test.mean() - y_pred_test_baseline.mean()):.2f})
   Z-score: ${y_pred_test_zscore.mean():.2f} (차이: ${abs(y_test.mean() - y_pred_test_zscore.mean()):.2f})
   Log: ${y_pred_test_log.mean():.2f} (차이: ${abs(y_test.mean() - y_pred_test_log.mean()):.2f})
   Min-Max: ${y_pred_test_minmax.mean():.2f} (차이: ${abs(y_test.mean() - y_pred_test_minmax.mean()):.2f})
   Robust: ${y_pred_test_robust.mean():.2f} (차이: ${abs(y_test.mean() - y_pred_test_robust.mean()):.2f})

4. RMSE 개선:
   Baseline: ${rmse_test_baseline:.2f}
   Best ({best_method}): ${results.loc[best_idx, 'Test RMSE']:.2f} (개선: ${rmse_test_baseline - results.loc[best_idx, 'Test RMSE']:.2f})

5. 결론:
   {'✅ Target 표준화로 성능 개선!' if best_r2 > r2_test_baseline else '❌ Target 표준화 효과 없음'}
   {'✅ Extrapolation 문제 완화' if best_r2 > 0 else '⚠️ 여전히 Extrapolation 문제 존재'}
   {'✅ 예측 평균이 실제에 근접' if abs(y_test.mean() - results.loc[best_idx, 'Mean Pred']) < abs(y_test.mean() - y_pred_test_baseline.mean()) else ''}
""")

# Save results
results.to_csv('price_prediction_normalized_results.csv', index=False)
print("\nSaved: price_prediction_normalized_results.csv")

print("\n" + "=" * 80)
print("Step 23 Completed!")
print("=" * 80)
