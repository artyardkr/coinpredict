#!/usr/bin/env python3
"""
Step 25 (4시간 버전): 4시간 후 가격 예측
integrated_data_4hour.csv 사용

step25와 동일:
- 오늘 features → 4시간 후 Close 예측
- Data Leakage 제거
- 10개 모델 비교
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
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# ========================================
# 1. Load Data
# ========================================
print("=" * 80)
print("4시간 후 가격 예측 (Step25 - 4Hour Version)")
print("=" * 80)

df = pd.read_csv('integrated_data_4hour.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"Data shape: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Period: 약 {(df['Date'].max() - df['Date'].min()).days}일 (4시간 단위)")

# ========================================
# 2. Create Target: 4시간 후 Close
# ========================================
print("\n" + "=" * 80)
print("Target 생성: 4시간 후 종가")
print("=" * 80)

# Close_x 사용 (진짜 4시간 데이터)
if 'Close_x' in df.columns:
    df['Close'] = df['Close_x']
    df['Open'] = df['Open_x']
    df['High'] = df['High_x']
    df['Low'] = df['Low_x']
    df['Volume'] = df['Volume_x']
    print("  ✅ Close_x를 Close로 변환")
elif 'Close' not in df.columns:
    print("  ❌ Close 컬럼이 없습니다!")
    raise ValueError("Close 컬럼 필요")

# 4시간 후 가격
df['target'] = df['Close'].shift(-1)

# 마지막 행 제거 (target NaN)
df = df[:-1].copy()

print(f"Target 생성 완료: {len(df)} samples")
print(f"Example:")
print(f"  Date: {df['Date'].iloc[0]} → 이 시점의 features")
print(f"  Close: ${df['Close'].iloc[0]:.2f} (현재)")
print(f"  Target: ${df['target'].iloc[0]:.2f} (4시간 후)")

# ========================================
# 3. Feature Preparation
# ========================================
print("\n" + "=" * 80)
print("Feature 준비...")
print("=" * 80)

# 제외할 컬럼
exclude_cols = [
    'Date', 'Close', 'High', 'Low', 'Open', 'target',
    'Close_x', 'High_x', 'Low_x', 'Open_x', 'Volume_x',  # _x 버전
    'Close_y', 'High_y', 'Low_y', 'Open_y', 'Volume_y',  # _y 버전
    'cumulative_return',
    'bc_market_price', 'bc_market_cap',
]

# EMA/SMA close 제외 (있다면)
ema_sma_cols = [col for col in df.columns
                if ('EMA' in col or 'SMA' in col) and 'close' in col.lower()]
exclude_cols.extend(ema_sma_cols)

# BB 제외
bb_cols = [col for col in df.columns if col.startswith('BB_')]
exclude_cols.extend(bb_cols)

exclude_cols = list(set(exclude_cols))

feature_cols = [col for col in df.columns if col not in exclude_cols]
print(f"Total features: {len(feature_cols)}")

# NaN/Inf 처리
for col in feature_cols:
    if col not in df.columns:
        continue
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

print(f"사용 가능한 features: {len([c for c in feature_cols if c in df.columns])}개")

# 실제 존재하는 컬럼만 필터링
feature_cols = [col for col in feature_cols if col in df.columns]

# ========================================
# 4. Train/Test Split (70/30)
# ========================================
print("\n" + "=" * 80)
print("Train/Test split (70/30)...")
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
print(f"Train: {len(X_train)} samples ({len(X_train)//6:.0f}일)")
print(f"  Current price: ${close_train.min():.0f}-${close_train.max():.0f}")
print(f"  4h later price: ${y_train.min():.0f}-${y_train.max():.0f}")
print(f"\nTest: {len(X_test)} samples ({len(X_test)//6:.0f}일)")
print(f"  Current price: ${close_test.min():.0f}-${close_test.max():.0f}")
print(f"  4h later price: ${y_test.min():.0f}-${y_test.max():.0f}")
print(f"\nExtrapolation ratio: {y_test.max() / y_train.max():.2f}x")

# Feature scaling
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# ========================================
# 5. Define Models
# ========================================
print("\n" + "=" * 80)
print("모델 테스트 (10개)...")
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
    'SVR': SVR(kernel='rbf', C=100, gamma='scale'),
    'KNN': KNeighborsRegressor(n_neighbors=10, weights='distance'),
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

    # Prediction stats
    pred_mean = y_pred_test.mean()
    pred_min = y_pred_test.min()
    pred_max = y_pred_test.max()

    # Direction accuracy
    actual_direction = (y_test > close_test).astype(int)
    pred_direction = (y_pred_test > close_test).astype(int)
    direction_acc = (actual_direction == pred_direction).mean()

    print(f"Train R²: {r2_train:.4f}, RMSE: ${rmse_train:.2f}")
    print(f"Test R²: {r2_test:.4f}, RMSE: ${rmse_test:.2f}, MAE: ${mae_test:.2f}")
    print(f"Predictions - Mean: ${pred_mean:.2f}, Range: ${pred_min:.2f}-${pred_max:.2f}")
    print(f"Direction Accuracy: {direction_acc:.2%}")

    # Extrapolation check
    extrapolates = pred_max > y_train.max()
    print(f"Extrapolates? {extrapolates} (Max pred: ${pred_max:.0f} vs Train max: ${y_train.max():.0f})")

    results.append({
        'Model': model_name,
        'Train R²': r2_train,
        'Test R²': r2_test,
        'Train RMSE': rmse_train,
        'Test RMSE': rmse_test,
        'Test MAE': mae_test,
        'Direction Acc': direction_acc,
        'Extrapolates': extrapolates
    })

    predictions_dict[model_name] = y_pred_test

results_df = pd.DataFrame(results)

# ========================================
# 6. Results Summary
# ========================================
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

results_sorted = results_df.sort_values('Test R²', ascending=False)
print("\n" + results_sorted.to_string(index=False))

best_model = results_sorted.iloc[0]
print(f"\n🏆 Best Model: {best_model['Model']}")
print(f"   Test R²: {best_model['Test R²']:.4f}")
print(f"   Test RMSE: ${best_model['Test RMSE']:.2f}")
print(f"   Direction Accuracy: {best_model['Direction Acc']:.2%}")

# ========================================
# 7. Visualization
# ========================================
print("\n" + "=" * 80)
print("시각화 생성 중...")
print("=" * 80)

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

# 1. R² Comparison
ax1 = fig.add_subplot(gs[0, 0])
colors = ['#2ecc71' if 'ElasticNet' in x or 'Ridge' in x or 'Lasso' in x or 'Linear' in x
          else '#3498db' for x in results_sorted['Model']]
ax1.barh(range(len(results_sorted)), results_sorted['Test R²'], color=colors, alpha=0.7)
ax1.set_yticks(range(len(results_sorted)))
ax1.set_yticklabels(results_sorted['Model'], fontsize=9)
ax1.set_xlabel('Test R²', fontweight='bold')
ax1.set_title('Test R² Comparison (4-Hour Prediction)', fontweight='bold')
ax1.axvline(x=0, color='red', linestyle='--', alpha=0.3)
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3, axis='x')

for i, (idx, row) in enumerate(results_sorted.iterrows()):
    ax1.text(row['Test R²'], i, f"  {row['Test R²']:.3f}",
            va='center', fontsize=9, fontweight='bold')

# 2. RMSE Comparison
ax2 = fig.add_subplot(gs[0, 1])
ax2.barh(range(len(results_sorted)), results_sorted['Test RMSE'], color=colors, alpha=0.7)
ax2.set_yticks(range(len(results_sorted)))
ax2.set_yticklabels(results_sorted['Model'], fontsize=9)
ax2.set_xlabel('Test RMSE ($)', fontweight='bold')
ax2.set_title('Test RMSE Comparison', fontweight='bold')
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3, axis='x')

# 3. Direction Accuracy
ax3 = fig.add_subplot(gs[0, 2])
ax3.barh(range(len(results_sorted)), results_sorted['Direction Acc']*100, color=colors, alpha=0.7)
ax3.set_yticks(range(len(results_sorted)))
ax3.set_yticklabels(results_sorted['Model'], fontsize=9)
ax3.set_xlabel('Direction Accuracy (%)', fontweight='bold')
ax3.set_title('Up/Down Direction Accuracy', fontweight='bold')
ax3.axvline(x=50, color='red', linestyle='--', alpha=0.3, label='Random')
ax3.invert_yaxis()
ax3.grid(True, alpha=0.3, axis='x')
ax3.legend()

# 4-6. Time series (Top 3)
top_3 = results_sorted.head(3)

for idx, (_, row) in enumerate(top_3.iterrows()):
    ax = fig.add_subplot(gs[1+idx//2, idx%2 if idx < 2 else 2])

    model_name = row['Model']
    predictions = predictions_dict[model_name]

    ax.plot(dates_test, y_test, label='Actual (4h later)',
            linewidth=2, color='black', alpha=0.8)
    ax.plot(dates_test, predictions, label='Predicted (4h later)',
            linewidth=2, color='#e74c3c', alpha=0.7, linestyle='--')
    ax.plot(dates_test, close_test, label='Current',
            linewidth=1, color='gray', alpha=0.5, linestyle=':')

    ax.set_xlabel('Date', fontweight='bold', fontsize=10)
    ax.set_ylabel('Price ($)', fontweight='bold', fontsize=10)
    title = f"#{idx+1}: {model_name}\nR²={row['Test R²']:.3f}, Dir={row['Direction Acc']:.1%}"
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

# 7. Error Distribution
ax7 = fig.add_subplot(gs[2, 2])
best_model_name = results_sorted.iloc[0]['Model']
best_predictions = predictions_dict[best_model_name]
errors = y_test - best_predictions
ax7.hist(errors, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
ax7.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax7.set_xlabel('Prediction Error ($)', fontweight='bold')
ax7.set_ylabel('Frequency', fontweight='bold')
ax7.set_title(f'Error Distribution: {best_model_name}', fontweight='bold')
ax7.grid(True, alpha=0.3)

mean_error = errors.mean()
std_error = errors.std()
ax7.text(0.05, 0.95, f'Mean: ${mean_error:.2f}\nStd: ${std_error:.2f}',
        transform=ax7.transAxes, va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 8. Actual vs Predicted Scatter
ax8 = fig.add_subplot(gs[3, :2])
ax8.scatter(y_test, best_predictions, alpha=0.5, s=20, color='#3498db')
ax8.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect Prediction')
ax8.set_xlabel('Actual 4h Later Price ($)', fontweight='bold', fontsize=11)
ax8.set_ylabel('Predicted 4h Later Price ($)', fontweight='bold', fontsize=11)
ax8.set_title(f'Actual vs Predicted: {best_model_name}', fontweight='bold', fontsize=12)
ax8.legend()
ax8.grid(True, alpha=0.3)

ax8.text(0.05, 0.95, f"R² = {results_sorted.iloc[0]['Test R²']:.4f}\nRMSE = ${results_sorted.iloc[0]['Test RMSE']:.2f}",
        transform=ax8.transAxes, va='top', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# 9. Linear vs Tree
ax9 = fig.add_subplot(gs[3, 2])
linear_models = ['Linear Regression', 'Ridge', 'Lasso', 'ElasticNet']
linear_r2 = [results_df[results_df['Model']==m]['Test R²'].values[0]
             for m in linear_models if m in results_df['Model'].values]
tree_models = ['Random Forest', 'XGBoost', 'Gradient Boosting', 'LightGBM']
tree_r2 = [results_df[results_df['Model']==m]['Test R²'].values[0]
           for m in tree_models if m in results_df['Model'].values]

x_pos = np.arange(max(len(linear_r2), len(tree_r2)))
width = 0.35

bars1 = ax9.bar(x_pos[:len(linear_r2)], linear_r2, width,
                label='Linear', color='#2ecc71', alpha=0.7)
bars2 = ax9.bar(x_pos[:len(tree_r2)] + width, tree_r2, width,
                label='Tree-based', color='#3498db', alpha=0.7)

ax9.set_ylabel('Test R²', fontweight='bold')
ax9.set_title('Linear vs Tree Models', fontweight='bold')
ax9.set_xticks(x_pos + width/2)
ax9.set_xticklabels(range(1, len(x_pos)+1))
ax9.axhline(y=0, color='red', linestyle='--', alpha=0.3)
ax9.legend()
ax9.grid(True, alpha=0.3, axis='y')

plt.savefig('4hour_price_prediction.png', dpi=300, bbox_inches='tight')
print("Saved: 4hour_price_prediction.png")

# ========================================
# 8. Save Results
# ========================================
results_df.to_csv('4hour_price_prediction_results.csv', index=False)
print("Saved: 4hour_price_prediction_results.csv")

# ========================================
# 9. Summary
# ========================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

best = results_sorted.iloc[0]

print(f"""
📊 4시간 후 가격 예측 결과

1. 최고 성능:
   🏆 {best['Model']}
   - Test R²: {best['Test R²']:.4f} {'✅' if best['Test R²'] > 0 else '❌'}
   - Test RMSE: ${best['Test RMSE']:.2f}
   - Direction Accuracy: {best['Direction Acc']:.2%}
   - Extrapolates: {best['Extrapolates']}

2. Linear vs Tree:
   Linear 최고: {results_df[results_df['Model'].isin(linear_models)]['Test R²'].max():.4f}
   Tree 최고: {results_df[results_df['Model'].isin(tree_models)]['Test R²'].max():.4f}

3. 데이터:
   샘플: {len(df)} (약 {len(df)//6}일)
   Features: {len(feature_cols)}개
   Train: {len(X_train)} ({len(X_train)//6}일)
   Test: {len(X_test)} ({len(X_test)//6}일)

4. 결론:
   {'✅ 4시간 후 가격 예측 가능! (R² > 0)' if best['Test R²'] > 0 else '❌ 4시간 후 가격 예측 어려움 (R² < 0)'}
   {'✅ 실전 활용 가능' if best['Test R²'] > 0.1 and best['Direction Acc'] > 0.55 else '⚠️ 실전 활용 제한적'}

   일별 step25와 비교:
   - 일별: 24시간 후 예측
   - 4시간: 4시간 후 예측 (더 단기)
   - {'더 예측하기 쉬움' if best['Test R²'] > 0.82 else '더 예측하기 어려움' if best['Test R²'] < 0.82 else '비슷한 난이도'}
""")

print("=" * 80)
print("Step 25 (4Hour Version) Completed!")
print("=" * 80)
