#!/usr/bin/env python3
"""
Step 25: CORRECT Next-Day Price Prediction

step24의 문제점 수정:
- 같은 날 가격 예측 (Data Leakage) ❌
- 다음날 가격 예측 (올바름) ✅

오늘의 features → 내일의 Close 예측
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
plt.rcParams['font.family'] = 'AppleGothic'  # macOS
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# ========================================
# 1. Load Data
# ========================================
print("=" * 80)
print("CORRECT Next-Day Price Prediction")
print("=" * 80)

df = pd.read_csv('integrated_data_full.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"Data shape: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

# ========================================
# 2. Create Target: NEXT DAY Close
# ========================================
print("\n" + "=" * 80)
print("Creating target: NEXT DAY Close")
print("=" * 80)

# ✅ 올바른 방법: 다음날 가격
df['target'] = df['Close'].shift(-1)

# Remove last row (no target)
df = df[:-1].copy()

print(f"Target created: {len(df)} samples")
print(f"Example:")
print(f"  Date: {df['Date'].iloc[0]} → Features from this day")
print(f"  Close: ${df['Close'].iloc[0]:.2f} (today)")
print(f"  Target: ${df['target'].iloc[0]:.2f} (tomorrow)")
print(f"\n  Date: {df['Date'].iloc[1]} → Features from this day")
print(f"  Close: ${df['Close'].iloc[1]:.2f} (today)")
print(f"  Target: ${df['target'].iloc[1]:.2f} (tomorrow)")

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
print(f"Train: {len(X_train)} samples")
print(f"  Today's price: ${close_train.min():.0f}-${close_train.max():.0f} (mean: ${close_train.mean():.0f})")
print(f"  Tomorrow's price: ${y_train.min():.0f}-${y_train.max():.0f} (mean: ${y_train.mean():.0f})")
print(f"\nTest: {len(X_test)} samples")
print(f"  Today's price: ${close_test.min():.0f}-${close_test.max():.0f} (mean: ${close_test.mean():.0f})")
print(f"  Tomorrow's price: ${y_test.min():.0f}-${y_test.max():.0f} (mean: ${y_test.mean():.0f})")
print(f"\nExtrapolation ratio: {y_test.mean() / y_train.mean():.2f}x")

# Feature scaling
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# ========================================
# 5. Define Models
# ========================================
print("\n" + "=" * 80)
print("Testing models...")
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

    # Prediction stats
    pred_mean = y_pred_test.mean()
    pred_std = y_pred_test.std()
    pred_min = y_pred_test.min()
    pred_max = y_pred_test.max()

    # Direction accuracy (bonus metric)
    actual_direction = (y_test > close_test).astype(int)  # Tomorrow > Today?
    pred_direction = (y_pred_test > close_test).astype(int)
    direction_acc = (actual_direction == pred_direction).mean()

    print(f"Train R²: {r2_train:.4f}, RMSE: ${rmse_train:.2f}")
    print(f"Test R²: {r2_test:.4f}, RMSE: ${rmse_test:.2f}, MAE: ${mae_test:.2f}")
    print(f"Predictions - Mean: ${pred_mean:.2f}, Range: ${pred_min:.2f}-${pred_max:.2f}")
    print(f"Direction Accuracy: {direction_acc:.2%} (Up/Down from today)")

    # Check extrapolation
    extrapolates = pred_max > y_train.max()
    print(f"Extrapolates? {extrapolates} (Max pred: ${pred_max:.0f} vs Train max: ${y_train.max():.0f})")

    results.append({
        'Model': model_name,
        'Train R²': r2_train,
        'Test R²': r2_test,
        'Train RMSE': rmse_train,
        'Test RMSE': rmse_test,
        'Test MAE': mae_test,
        'Pred Mean': pred_mean,
        'Pred Range': f"${pred_min:.0f}-${pred_max:.0f}",
        'Direction Acc': direction_acc,
        'Extrapolates': extrapolates
    })

    predictions_dict[model_name] = y_pred_test

results_df = pd.DataFrame(results)

# ========================================
# 6. Comparison with step24 (Same-Day)
# ========================================
print("\n" + "=" * 80)
print("COMPARISON: Next-Day vs Same-Day Prediction")
print("=" * 80)

# step24 results (for reference)
step24_elasticnet_r2 = 0.8001

print(f"""
step24 (Same-Day Prediction - DATA LEAKAGE):
  ElasticNet R²: {step24_elasticnet_r2:.4f} ❌ (too good to be true)

step25 (Next-Day Prediction - CORRECT):
  ElasticNet R²: {results_df[results_df['Model']=='ElasticNet']['Test R²'].values[0]:.4f} ✅ (realistic)

차이: {step24_elasticnet_r2 - results_df[results_df['Model']=='ElasticNet']['Test R²'].values[0]:.4f}

Next-Day 예측이 더 어려운 이유:
  1. 미래는 불확실함 (당연!)
  2. Data Leakage 제거됨
  3. 실전에서 실제로 사용 가능한 성능
""")

# ========================================
# 7. Results Summary
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
ax1.set_title('Test R² Comparison (Next-Day Prediction)', fontweight='bold')
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
ax3.set_title('Up/Down Direction Accuracy', fontweight='bold')
ax3.axvline(x=50, color='red', linestyle='--', alpha=0.3, label='Random')
ax3.invert_yaxis()
ax3.grid(True, alpha=0.3, axis='x')
ax3.legend()

# 4. step24 vs step25 comparison
ax4 = fig.add_subplot(gs[1, 0])
comparison_data = {
    'step24\n(Same-Day)\nDATA LEAKAGE': step24_elasticnet_r2,
    'step25\n(Next-Day)\nCORRECT': results_df[results_df['Model']=='ElasticNet']['Test R²'].values[0]
}
bars = ax4.bar(comparison_data.keys(), comparison_data.values(),
               color=['#e74c3c', '#2ecc71'], alpha=0.7)
ax4.set_ylabel('ElasticNet Test R²', fontweight='bold')
ax4.set_title('step24 vs step25: ElasticNet', fontweight='bold', fontsize=12)
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

    # Plot actual tomorrow's price
    ax.plot(dates_test, y_test, label='Actual (Tomorrow)',
            linewidth=2, color='black', alpha=0.8)

    # Plot predicted tomorrow's price
    ax.plot(dates_test, predictions, label=f'Predicted (Tomorrow)',
            linewidth=2, color='#e74c3c', alpha=0.7, linestyle='--')

    # Plot today's price (reference)
    ax.plot(dates_test, close_test, label='Today',
            linewidth=1, color='gray', alpha=0.5, linestyle=':')

    ax.set_xlabel('Date', fontweight='bold', fontsize=10)
    ax.set_ylabel('Price ($)', fontweight='bold', fontsize=10)
    title = f"#{idx+1}: {model_name}\nR²={row['Test R²']:.3f}, Dir Acc={row['Direction Acc']:.1%}"
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
ax8.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect')
ax8.set_xlabel('Prediction Error ($)', fontweight='bold')
ax8.set_ylabel('Frequency', fontweight='bold')
ax8.set_title(f'Error Distribution: {best_model_name}', fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

# Add stats
mean_error = errors.mean()
std_error = errors.std()
ax8.text(0.05, 0.95, f'Mean: ${mean_error:.2f}\nStd: ${std_error:.2f}',
        transform=ax8.transAxes, va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 9. Actual vs Predicted Scatter (Best Model)
ax9 = fig.add_subplot(gs[3, :2])
ax9.scatter(y_test, best_predictions, alpha=0.5, s=20, color='#3498db')
ax9.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect Prediction')
ax9.set_xlabel('Actual Tomorrow Price ($)', fontweight='bold', fontsize=11)
ax9.set_ylabel('Predicted Tomorrow Price ($)', fontweight='bold', fontsize=11)
ax9.set_title(f'Actual vs Predicted: {best_model_name}', fontweight='bold', fontsize=12)
ax9.legend()
ax9.grid(True, alpha=0.3)

# Add R² text
ax9.text(0.05, 0.95, f"R² = {results_sorted.iloc[0]['Test R²']:.4f}\nRMSE = ${results_sorted.iloc[0]['Test RMSE']:.2f}",
        transform=ax9.transAxes, va='top', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# 10. Linear vs Tree models
ax10 = fig.add_subplot(gs[3, 2])
linear_models = ['Linear Regression', 'Ridge', 'Lasso', 'ElasticNet']
linear_r2 = [results_df[results_df['Model']==m]['Test R²'].values[0]
             for m in linear_models if m in results_df['Model'].values]
tree_models = ['Random Forest', 'XGBoost', 'Gradient Boosting', 'LightGBM']
tree_r2 = [results_df[results_df['Model']==m]['Test R²'].values[0]
           for m in tree_models if m in results_df['Model'].values]

x_pos = np.arange(max(len(linear_r2), len(tree_r2)))
width = 0.35

bars1 = ax10.bar(x_pos[:len(linear_r2)], linear_r2, width,
                 label='Linear', color='#2ecc71', alpha=0.7)
bars2 = ax10.bar(x_pos[:len(tree_r2)] + width, tree_r2, width,
                 label='Tree-based', color='#3498db', alpha=0.7)

ax10.set_ylabel('Test R²', fontweight='bold')
ax10.set_title('Linear vs Tree-based Models', fontweight='bold')
ax10.set_xticks(x_pos + width/2)
ax10.set_xticklabels(range(1, len(x_pos)+1))
ax10.axhline(y=0, color='red', linestyle='--', alpha=0.3)
ax10.legend()
ax10.grid(True, alpha=0.3, axis='y')

plt.savefig('next_day_price_prediction.png', dpi=300, bbox_inches='tight')
print("Saved: next_day_price_prediction.png")

# ========================================
# 9. Save Results
# ========================================
results_df.to_csv('next_day_price_prediction_results.csv', index=False)
print("Saved: next_day_price_prediction_results.csv")

# ========================================
# 10. Summary & Insights
# ========================================
print("\n" + "=" * 80)
print("SUMMARY & KEY INSIGHTS")
print("=" * 80)

best = results_sorted.iloc[0]

print(f"""
📊 다음날 가격 예측 결과

1. 최고 성능:
   🏆 {best['Model']}
   - Test R²: {best['Test R²']:.4f} {'✅' if best['Test R²'] > 0 else '❌'}
   - Test RMSE: ${best['Test RMSE']:.2f}
   - Direction Accuracy: {best['Direction Acc']:.2%}
   - Extrapolates: {best['Extrapolates']}

2. step24 (잘못됨) vs step25 (올바름):
   step24 ElasticNet R²: 0.8001 ❌ (Data Leakage)
   step25 ElasticNet R²: {results_df[results_df['Model']=='ElasticNet']['Test R²'].values[0]:.4f} ✅ (Correct)

   차이: {0.8001 - results_df[results_df['Model']=='ElasticNet']['Test R²'].values[0]:.4f}
   → Data Leakage가 {(0.8001 - results_df[results_df['Model']=='ElasticNet']['Test R²'].values[0])/0.8001*100:.1f}% 성능을 부풀렸음!

3. Linear vs Tree:
   Linear 최고: {results_df[results_df['Model'].isin(linear_models)]['Test R²'].max():.4f}
   Tree 최고: {results_df[results_df['Model'].isin(tree_models)]['Test R²'].max():.4f}
   {'✅ Linear 모델 우수!' if results_df[results_df['Model'].isin(linear_models)]['Test R²'].max() > results_df[results_df['Model'].isin(tree_models)]['Test R²'].max() else '✅ Tree 모델 우수!'}

4. 실제 vs 예측 (최고 모델):
   실제 평균: ${y_test.mean():.2f}
   예측 평균: ${best['Pred Mean']:.2f}
   차이: ${abs(y_test.mean() - best['Pred Mean']):.2f} ({abs(y_test.mean() - best['Pred Mean'])/y_test.mean()*100:.1f}%)

5. 방향 예측 정확도:
   최고: {results_sorted['Direction Acc'].max():.2%} ({results_sorted.loc[results_sorted['Direction Acc'].idxmax(), 'Model']})
   평균: {results_sorted['Direction Acc'].mean():.2%}
   vs Random: 50%

6. 결론:
   {'✅ 다음날 가격 예측 가능! (R² > 0)' if best['Test R²'] > 0 else '❌ 다음날 가격 예측 매우 어려움 (R² < 0)'}
   {'✅ 실전 활용 가능한 수준' if best['Test R²'] > 0.1 and best['Direction Acc'] > 0.55 else '⚠️ 실전 활용 제한적'}
   추천: {best['Model']} (R²={best['Test R²']:.3f}, Dir={best['Direction Acc']:.1%})

7. 개선 필요 사항:
   {f"⚠️ R² {best['Test R²']:.3f}은 낮은 편 (설명력 {best['Test R²']*100:.1f}%)" if best['Test R²'] < 0.3 else "✅ R²가 양호한 수준"}
   {f"⚠️ RMSE ${best['Test RMSE']:.0f}은 높은 편 (평균의 {best['Test RMSE']/y_test.mean()*100:.1f}%)" if best['Test RMSE']/y_test.mean() > 0.1 else "✅ RMSE가 낮음"}
   {f"⚠️ 방향 정확도 {best['Direction Acc']:.1%}은 낮음" if best['Direction Acc'] < 0.55 else f"✅ 방향 정확도 {best['Direction Acc']:.1%}은 양호"}
""")

print("\n" + "=" * 80)
print("Step 25 Completed!")
print("=" * 80)
