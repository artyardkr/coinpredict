#!/usr/bin/env python3
"""
Step 24: Comprehensive Regression Model Comparison

다양한 회귀 모델 비교:
1. Linear Models (Extrapolation 가능):
   - Linear Regression
   - Ridge Regression
   - Lasso Regression
   - ElasticNet
   - SVR (Support Vector Regression)

2. Tree-based Models (Extrapolation 불가):
   - Random Forest
   - XGBoost
   - LightGBM

각 모델을 Target 표준화와 함께 테스트
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

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("LightGBM not available")
    LIGHTGBM_AVAILABLE = False

# ========================================
# 1. Load Data
# ========================================
print("=" * 80)
print("Comprehensive Regression Model Comparison")
print("=" * 80)

df = pd.read_csv('integrated_data_full.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"Data shape: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

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
# 3. Train/Test Split
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
dates_test = df[test_mask]['Date'].values

print(f"Split date: {split_date}")
print(f"Train: {len(X_train)} samples, price ${y_train.min():.0f}-${y_train.max():.0f} (mean: ${y_train.mean():.0f})")
print(f"Test: {len(X_test)} samples, price ${y_test.min():.0f}-${y_test.max():.0f} (mean: ${y_test.mean():.0f})")
print(f"Extrapolation ratio: {y_test.mean() / y_train.mean():.2f}x")

# Feature scaling
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# ========================================
# 4. Define Models
# ========================================
print("\n" + "=" * 80)
print("Defining models...")
print("=" * 80)

models = {
    # Linear Models (Extrapolation 가능)
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=1.0, max_iter=10000),
    'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000),
    'SVR (RBF)': SVR(kernel='rbf', C=100, gamma='scale', cache_size=1000),
    'SVR (Linear)': SVR(kernel='linear', C=1.0, cache_size=1000),

    # Tree-based Models (Extrapolation 불가)
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

print(f"Total models: {len(models)}")
for name in models.keys():
    model_type = "Linear" if name in ['Linear Regression', 'Ridge', 'Lasso', 'ElasticNet', 'SVR (RBF)', 'SVR (Linear)'] else "Tree-based"
    print(f"  - {name} ({model_type})")

# ========================================
# 5. Test with Different Target Transformations
# ========================================

transformations = ['None', 'Z-score', 'Log']
results_all = []

for transform_name in transformations:
    print("\n" + "=" * 80)
    print(f"TARGET TRANSFORMATION: {transform_name}")
    print("=" * 80)

    # Prepare target
    if transform_name == 'None':
        y_train_trans = y_train.copy()
        y_test_trans = y_test.copy()
        scaler_y = None

    elif transform_name == 'Z-score':
        scaler_y = StandardScaler()
        y_train_trans = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_trans = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    elif transform_name == 'Log':
        y_train_trans = np.log(y_train)
        y_test_trans = np.log(y_test)
        scaler_y = 'log'

    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")

        try:
            # Train
            model.fit(X_train_scaled, y_train_trans)

            # Predict
            y_pred_train_trans = model.predict(X_train_scaled)
            y_pred_test_trans = model.predict(X_test_scaled)

            # Inverse transform if needed
            if transform_name == 'Z-score':
                y_pred_train = scaler_y.inverse_transform(y_pred_train_trans.reshape(-1, 1)).flatten()
                y_pred_test = scaler_y.inverse_transform(y_pred_test_trans.reshape(-1, 1)).flatten()
            elif transform_name == 'Log':
                y_pred_train = np.exp(y_pred_train_trans)
                y_pred_test = np.exp(y_pred_test_trans)
            else:
                y_pred_train = y_pred_train_trans
                y_pred_test = y_pred_test_trans

            # Metrics
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            mae_test = mean_absolute_error(y_test, y_pred_test)

            # Prediction stats
            pred_mean = y_pred_test.mean()
            pred_std = y_pred_test.std()
            pred_min = y_pred_test.min()
            pred_max = y_pred_test.max()

            print(f"Train R²: {r2_train:.4f}")
            print(f"Test R²: {r2_test:.4f}")
            print(f"Test RMSE: ${rmse_test:.2f}")
            print(f"Test MAE: ${mae_test:.2f}")
            print(f"Predictions - Mean: ${pred_mean:.2f}, Range: ${pred_min:.2f}-${pred_max:.2f}")

            # Check if extrapolation works
            extrapolates = pred_max > y_train.max()
            print(f"Extrapolates beyond training max? {extrapolates} (Max pred: ${pred_max:.0f} vs Train max: ${y_train.max():.0f})")

            results_all.append({
                'Model': model_name,
                'Transform': transform_name,
                'Train R²': r2_train,
                'Test R²': r2_test,
                'Test RMSE': rmse_test,
                'Test MAE': mae_test,
                'Pred Mean': pred_mean,
                'Pred Std': pred_std,
                'Pred Min': pred_min,
                'Pred Max': pred_max,
                'Extrapolates': extrapolates,
                'predictions': y_pred_test
            })

        except Exception as e:
            print(f"ERROR: {e}")
            results_all.append({
                'Model': model_name,
                'Transform': transform_name,
                'Train R²': np.nan,
                'Test R²': np.nan,
                'Test RMSE': np.nan,
                'Test MAE': np.nan,
                'Pred Mean': np.nan,
                'Pred Std': np.nan,
                'Pred Min': np.nan,
                'Pred Max': np.nan,
                'Extrapolates': False,
                'predictions': None
            })

# ========================================
# 6. Results Analysis
# ========================================
print("\n" + "=" * 80)
print("COMPREHENSIVE RESULTS")
print("=" * 80)

results_df = pd.DataFrame(results_all)

# Best models by R²
print("\n🏆 TOP 10 MODELS (by Test R²):")
top_10 = results_df.nlargest(10, 'Test R²')[['Model', 'Transform', 'Test R²', 'Test RMSE', 'Test MAE', 'Extrapolates']]
print(top_10.to_string(index=False))

# Best by model type
print("\n📊 BEST PER MODEL (any transformation):")
best_per_model = results_df.loc[results_df.groupby('Model')['Test R²'].idxmax()]
best_per_model_display = best_per_model[['Model', 'Transform', 'Test R²', 'Test RMSE', 'Extrapolates']].sort_values('Test R²', ascending=False)
print(best_per_model_display.to_string(index=False))

# Linear vs Tree comparison
print("\n🔍 LINEAR vs TREE-BASED MODELS:")
linear_models = ['Linear Regression', 'Ridge', 'Lasso', 'ElasticNet', 'SVR (RBF)', 'SVR (Linear)']
tree_models = ['Random Forest', 'XGBoost', 'Gradient Boosting', 'LightGBM']

linear_results = results_df[results_df['Model'].isin(linear_models)]
tree_results = results_df[results_df['Model'].isin(tree_models)]

print(f"Linear models - Best R²: {linear_results['Test R²'].max():.4f} (Mean: {linear_results['Test R²'].mean():.4f})")
print(f"Tree models - Best R²: {tree_results['Test R²'].max():.4f} (Mean: {tree_results['Test R²'].mean():.4f})")

# Extrapolation check
print("\n🚀 EXTRAPOLATION CAPABILITY:")
extrap_yes = results_df[results_df['Extrapolates'] == True]
extrap_no = results_df[results_df['Extrapolates'] == False]
print(f"Models that extrapolate: {len(extrap_yes)}/{len(results_df)}")
print(f"Average R² (Extrapolates): {extrap_yes['Test R²'].mean():.4f}")
print(f"Average R² (No Extrapolate): {extrap_no['Test R²'].mean():.4f}")

# ========================================
# 7. Visualization
# ========================================
print("\n" + "=" * 80)
print("Creating visualizations...")
print("=" * 80)

fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

# 1. Overall R² comparison (heatmap)
ax1 = fig.add_subplot(gs[0, :])
pivot_r2 = results_df.pivot(index='Model', columns='Transform', values='Test R²')
sns.heatmap(pivot_r2, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
            cbar_kws={'label': 'Test R²'}, ax=ax1, linewidths=0.5)
ax1.set_title('Test R² Heatmap: All Models × Transformations', fontweight='bold', fontsize=14)
ax1.set_xlabel('Target Transformation', fontweight='bold')
ax1.set_ylabel('Model', fontweight='bold')

# 2. Best R² per model
ax2 = fig.add_subplot(gs[1, 0])
best_r2 = results_df.loc[results_df.groupby('Model')['Test R²'].idxmax()]
colors = ['#2ecc71' if x in linear_models else '#3498db' for x in best_r2['Model']]
bars = ax2.barh(range(len(best_r2)), best_r2['Test R²'], color=colors, alpha=0.7)
ax2.set_yticks(range(len(best_r2)))
ax2.set_yticklabels(best_r2['Model'], fontsize=9)
ax2.set_xlabel('Best Test R²', fontweight='bold')
ax2.set_title('Best R² per Model', fontweight='bold')
ax2.axvline(x=0, color='red', linestyle='--', alpha=0.3)
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3, axis='x')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2ecc71', alpha=0.7, label='Linear Model'),
                   Patch(facecolor='#3498db', alpha=0.7, label='Tree Model')]
ax2.legend(handles=legend_elements, loc='lower right')

# 3. RMSE comparison
ax3 = fig.add_subplot(gs[1, 1])
best_rmse = results_df.loc[results_df.groupby('Model')['Test RMSE'].idxmin()]
colors = ['#2ecc71' if x in linear_models else '#3498db' for x in best_rmse['Model']]
bars = ax3.barh(range(len(best_rmse)), best_rmse['Test RMSE'], color=colors, alpha=0.7)
ax3.set_yticks(range(len(best_rmse)))
ax3.set_yticklabels(best_rmse['Model'], fontsize=9)
ax3.set_xlabel('Best Test RMSE ($)', fontweight='bold')
ax3.set_title('Best RMSE per Model', fontweight='bold')
ax3.invert_yaxis()
ax3.grid(True, alpha=0.3, axis='x')

# 4. Extrapolation capability
ax4 = fig.add_subplot(gs[1, 2])
extrap_counts = results_df.groupby('Model')['Extrapolates'].sum()
colors = ['#2ecc71' if x in linear_models else '#3498db' for x in extrap_counts.index]
bars = ax4.barh(range(len(extrap_counts)), extrap_counts.values, color=colors, alpha=0.7)
ax4.set_yticks(range(len(extrap_counts)))
ax4.set_yticklabels(extrap_counts.index, fontsize=9)
ax4.set_xlabel('Count (out of 3 transforms)', fontweight='bold')
ax4.set_title('Extrapolation Capability', fontweight='bold')
ax4.invert_yaxis()
ax4.grid(True, alpha=0.3, axis='x')

# 5. Linear vs Tree comparison
ax5 = fig.add_subplot(gs[2, 0])
linear_best = linear_results.groupby('Model')['Test R²'].max()
tree_best = tree_results.groupby('Model')['Test R²'].max()
x = np.arange(max(len(linear_best), len(tree_best)))
width = 0.35

ax5.bar(range(len(linear_best)), linear_best.values, width,
        label='Linear', color='#2ecc71', alpha=0.7)
ax5.bar(np.arange(len(tree_best)) + width, tree_best.values, width,
        label='Tree-based', color='#3498db', alpha=0.7)
ax5.set_ylabel('Best Test R²', fontweight='bold')
ax5.set_title('Linear vs Tree-based Models', fontweight='bold')
ax5.axhline(y=0, color='red', linestyle='--', alpha=0.3)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# 6-8. Time series predictions (Top 3 models)
top_3_models = results_df.nlargest(3, 'Test R²')

for idx, (_, row) in enumerate(top_3_models.iterrows()):
    ax = fig.add_subplot(gs[2+idx//2, 1+idx%2])

    ax.plot(dates_test, y_test, label='Actual', linewidth=2, color='black', alpha=0.8)
    if row['predictions'] is not None:
        ax.plot(dates_test, row['predictions'], label='Predicted',
                linewidth=2, color='#e74c3c', alpha=0.7, linestyle='--')

    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Price ($)', fontweight='bold')
    title = f"#{idx+1}: {row['Model']} ({row['Transform']})\nR²={row['Test R²']:.3f}"
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

# 9. Transformation effect
ax9 = fig.add_subplot(gs[3, :])
for model_name in ['Linear Regression', 'Ridge', 'Random Forest', 'XGBoost']:
    model_data = results_df[results_df['Model'] == model_name]
    ax9.plot(model_data['Transform'], model_data['Test R²'],
             marker='o', linewidth=2, markersize=8, label=model_name)

ax9.set_xlabel('Target Transformation', fontweight='bold', fontsize=11)
ax9.set_ylabel('Test R²', fontweight='bold', fontsize=11)
ax9.set_title('Effect of Target Transformation on Different Models', fontweight='bold', fontsize=13)
ax9.axhline(y=0, color='red', linestyle='--', alpha=0.3)
ax9.legend()
ax9.grid(True, alpha=0.3)

plt.savefig('all_regression_models_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: all_regression_models_comparison.png")

# ========================================
# 8. Save Results
# ========================================
results_save = results_df.drop(columns=['predictions'])
results_save.to_csv('all_regression_models_results.csv', index=False)
print("Saved: all_regression_models_results.csv")

# ========================================
# 9. Summary & Insights
# ========================================
print("\n" + "=" * 80)
print("SUMMARY & KEY INSIGHTS")
print("=" * 80)

best_overall = results_df.loc[results_df['Test R²'].idxmax()]

print(f"""
📊 회귀 모델 종합 분석 결과

1. 최고 성능:
   🏆 {best_overall['Model']} ({best_overall['Transform']})
   - Test R²: {best_overall['Test R²']:.4f}
   - Test RMSE: ${best_overall['Test RMSE']:.2f}
   - Extrapolates: {best_overall['Extrapolates']}
   - Prediction range: ${best_overall['Pred Min']:.0f} - ${best_overall['Pred Max']:.0f}

2. Linear vs Tree-based:
   Linear 최고:  {linear_results['Test R²'].max():.4f} ({linear_results.loc[linear_results['Test R²'].idxmax(), 'Model']})
   Tree 최고:    {tree_results['Test R²'].max():.4f} ({tree_results.loc[tree_results['Test R²'].idxmax(), 'Model']})
   {'✅ Linear 모델이 우수!' if linear_results['Test R²'].max() > tree_results['Test R²'].max() else '✅ Tree 모델이 우수!'}

3. Extrapolation:
   가능한 모델: {extrap_yes['Model'].unique().tolist()}
   불가능 모델: {extrap_no['Model'].unique().tolist()[:3]}...

4. Target 변환 효과:
   None:    평균 R² {results_df[results_df['Transform']=='None']['Test R²'].mean():.4f}
   Z-score: 평균 R² {results_df[results_df['Transform']=='Z-score']['Test R²'].mean():.4f}
   Log:     평균 R² {results_df[results_df['Transform']=='Log']['Test R²'].mean():.4f}

5. 실제 vs 예측:
   실제 평균: ${y_test.mean():.2f}
   최고 모델 예측 평균: ${best_overall['Pred Mean']:.2f}
   차이: ${abs(y_test.mean() - best_overall['Pred Mean']):.2f}

6. 결론:
   {'✅ Linear 모델로 Extrapolation 개선!' if best_overall['Test R²'] > -1 else '❌ 모든 모델 Extrapolation 실패'}
   {'✅ Target 변환 효과 있음!' if results_df.groupby('Model')['Test R²'].apply(lambda x: x.max() - x.min()).mean() > 0.1 else '⚠️ Target 변환 효과 미미'}
   추천 모델: {best_overall['Model']} + {best_overall['Transform']} 변환
""")

print("\n" + "=" * 80)
print("Step 24 Completed!")
print("=" * 80)
