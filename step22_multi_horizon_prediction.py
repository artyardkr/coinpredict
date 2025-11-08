#!/usr/bin/env python3
"""
Step 22: Multi-Horizon Prediction (1-day, 7-day, 30-day)

다양한 예측 기간에 따른 성능 비교:
1. 1일 후 (일일 예측)
2. 7일 후 (일주일 예측)
3. 30일 후 (한 달 예측)

각 기간별:
- 방향 예측 정확도
- 수익률 예측 R²
- 중요 변수 변화
- ETF 전후 비교
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ========================================
# 1. Load Data
# ========================================
print("=" * 80)
print("Multi-Horizon Prediction Analysis")
print("=" * 80)

df = pd.read_csv('integrated_data_full.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"Data shape: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

ETF_DATE = pd.to_datetime('2024-01-10')

# ========================================
# 2. Create Targets for Multiple Horizons
# ========================================
print("\n" + "=" * 80)
print("Creating targets for 1-day, 7-day, 30-day horizons...")
print("=" * 80)

# Calculate future returns
for horizon in [1, 7, 30]:
    df[f'return_{horizon}d'] = (df['Close'].shift(-horizon) / df['Close'] - 1) * 100

    # Direction classification (Binary: Up/Down with 0.5% threshold)
    THRESHOLD = 0.5
    df[f'direction_{horizon}d'] = (df[f'return_{horizon}d'] > THRESHOLD).astype(int)

# Remove rows with no target
df = df[:-30].copy()  # Remove last 30 rows (for 30-day horizon)

print("\nTarget statistics:")
for horizon in [1, 7, 30]:
    returns = df[f'return_{horizon}d']
    directions = df[f'direction_{horizon}d']
    print(f"\n{horizon}-day horizon:")
    print(f"  Return - Mean: {returns.mean():.3f}%, Std: {returns.std():.3f}%")
    print(f"  Return - Min: {returns.min():.2f}%, Max: {returns.max():.2f}%")
    print(f"  Direction - Up: {(directions==1).sum()} ({(directions==1).mean()*100:.1f}%)")
    print(f"  Direction - Down: {(directions==0).sum()} ({(directions==0).mean()*100:.1f}%)")

# ========================================
# 3. Feature Preparation
# ========================================
print("\n" + "=" * 80)
print("Preparing features...")
print("=" * 80)

exclude_cols = [
    'Date', 'Close', 'High', 'Low', 'Open',
    'cumulative_return',
    'bc_market_price', 'bc_market_cap',
]

# Exclude all target columns
target_cols = [col for col in df.columns if col.startswith('return_') or col.startswith('direction_')]
exclude_cols.extend(target_cols)

# EMA/SMA close features
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
# 4. Train/Test Split by Date (70:30)
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

print(f"Split date: {split_date}")
print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========================================
# 5. Multi-Horizon Prediction
# ========================================

results_all = []
feature_importance_all = {}

for horizon in [1, 7, 30]:
    print("\n" + "=" * 80)
    print(f"{horizon}-DAY HORIZON PREDICTION")
    print("=" * 80)

    # Get targets
    y_train_direction = df[train_mask][f'direction_{horizon}d'].values
    y_test_direction = df[test_mask][f'direction_{horizon}d'].values
    y_train_return = df[train_mask][f'return_{horizon}d'].values
    y_test_return = df[test_mask][f'return_{horizon}d'].values

    print(f"\nTrain class distribution: {np.bincount(y_train_direction)}")
    print(f"Test class distribution: {np.bincount(y_test_direction)}")

    # ----------------------------------------
    # Direction Classification (Random Forest)
    # ----------------------------------------
    print(f"\n--- Direction Classification (RF) ---")

    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )

    rf_clf.fit(X_train_scaled, y_train_direction)

    y_pred_train_dir = rf_clf.predict(X_train_scaled)
    y_pred_test_dir = rf_clf.predict(X_test_scaled)

    acc_train_dir = accuracy_score(y_train_direction, y_pred_train_dir)
    acc_test_dir = accuracy_score(y_test_direction, y_pred_test_dir)

    print(f"Train Accuracy: {acc_train_dir:.4f}")
    print(f"Test Accuracy: {acc_test_dir:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test_direction, y_pred_test_dir, target_names=['Down', 'Up']))

    # Feature importance
    feature_importance_dir = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_clf.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 Important Features (Direction):")
    print(feature_importance_dir.head(10).to_string(index=False))

    feature_importance_all[f'{horizon}d_direction'] = feature_importance_dir

    # ----------------------------------------
    # Return Regression (Random Forest)
    # ----------------------------------------
    print(f"\n--- Return Regression (RF) ---")

    rf_reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )

    rf_reg.fit(X_train_scaled, y_train_return)

    y_pred_train_ret = rf_reg.predict(X_train_scaled)
    y_pred_test_ret = rf_reg.predict(X_test_scaled)

    r2_train_ret = r2_score(y_train_return, y_pred_train_ret)
    r2_test_ret = r2_score(y_test_return, y_pred_test_ret)
    rmse_test_ret = np.sqrt(mean_squared_error(y_test_return, y_pred_test_ret))

    print(f"Train R²: {r2_train_ret:.4f}")
    print(f"Test R²: {r2_test_ret:.4f}")
    print(f"Test RMSE: {rmse_test_ret:.4f}%")

    # Feature importance
    feature_importance_ret = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_reg.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 Important Features (Return):")
    print(feature_importance_ret.head(10).to_string(index=False))

    feature_importance_all[f'{horizon}d_return'] = feature_importance_ret

    # ----------------------------------------
    # XGBoost Classification
    # ----------------------------------------
    print(f"\n--- XGBoost Classification ---")

    xgb_clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=len(y_train_direction[y_train_direction==0])/len(y_train_direction[y_train_direction==1])
    )

    xgb_clf.fit(X_train_scaled, y_train_direction)

    y_pred_test_xgb = xgb_clf.predict(X_test_scaled)
    acc_test_xgb = accuracy_score(y_test_direction, y_pred_test_xgb)

    print(f"Test Accuracy: {acc_test_xgb:.4f}")

    # Store results
    results_all.append({
        'horizon': f'{horizon}d',
        'horizon_days': horizon,
        'rf_dir_train_acc': acc_train_dir,
        'rf_dir_test_acc': acc_test_dir,
        'xgb_dir_test_acc': acc_test_xgb,
        'rf_ret_train_r2': r2_train_ret,
        'rf_ret_test_r2': r2_test_ret,
        'rf_ret_test_rmse': rmse_test_ret,
        'test_samples': len(y_test_direction),
        'return_mean': y_test_return.mean(),
        'return_std': y_test_return.std(),
        'up_ratio': (y_test_direction==1).mean()
    })

# Convert to DataFrame
results_df = pd.DataFrame(results_all)

# ========================================
# 6. ETF Pre/Post Comparison
# ========================================
print("\n" + "=" * 80)
print("ETF PRE/POST COMPARISON BY HORIZON")
print("=" * 80)

etf_comparison_results = []

for horizon in [1, 7, 30]:
    print(f"\n--- {horizon}-day Horizon ---")

    # Pre-ETF
    pre_mask = df['Date'] < ETF_DATE
    X_pre = df[pre_mask][feature_cols].values
    y_pre_dir = df[pre_mask][f'direction_{horizon}d'].values

    split_pre = int(len(X_pre) * 0.7)
    X_train_pre = X_pre[:split_pre]
    X_test_pre = X_pre[split_pre:]
    y_train_pre = y_pre_dir[:split_pre]
    y_test_pre = y_pre_dir[split_pre:]

    scaler_pre = StandardScaler()
    X_train_pre_scaled = scaler_pre.fit_transform(X_train_pre)
    X_test_pre_scaled = scaler_pre.transform(X_test_pre)

    rf_pre = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42,
                                    n_jobs=-1, class_weight='balanced')
    rf_pre.fit(X_train_pre_scaled, y_train_pre)
    acc_pre = accuracy_score(y_test_pre, rf_pre.predict(X_test_pre_scaled))

    # Post-ETF
    post_mask = df['Date'] >= ETF_DATE
    X_post = df[post_mask][feature_cols].values
    y_post_dir = df[post_mask][f'direction_{horizon}d'].values

    split_post = int(len(X_post) * 0.7)
    X_train_post = X_post[:split_post]
    X_test_post = X_post[split_post:]
    y_train_post = y_post_dir[:split_post]
    y_test_post = y_post_dir[split_post:]

    scaler_post = StandardScaler()
    X_train_post_scaled = scaler_post.fit_transform(X_train_post)
    X_test_post_scaled = scaler_post.transform(X_test_post)

    rf_post = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42,
                                     n_jobs=-1, class_weight='balanced')
    rf_post.fit(X_train_post_scaled, y_train_post)
    acc_post = accuracy_score(y_test_post, rf_post.predict(X_test_post_scaled))

    print(f"Pre-ETF Accuracy: {acc_pre:.4f}")
    print(f"Post-ETF Accuracy: {acc_post:.4f}")
    print(f"Improvement: {(acc_post - acc_pre)*100:+.2f}%p")

    etf_comparison_results.append({
        'horizon': f'{horizon}d',
        'horizon_days': horizon,
        'pre_etf_acc': acc_pre,
        'post_etf_acc': acc_post,
        'improvement': acc_post - acc_pre
    })

etf_comparison_df = pd.DataFrame(etf_comparison_results)

# ========================================
# 7. Visualization
# ========================================
print("\n" + "=" * 80)
print("Creating visualizations...")
print("=" * 80)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# 1. Direction Accuracy by Horizon
ax1 = fig.add_subplot(gs[0, 0:2])
x = np.arange(len(results_df))
width = 0.35
bars1 = ax1.bar(x - width/2, results_df['rf_dir_test_acc'], width, label='RF', color='#3498db', alpha=0.8)
bars2 = ax1.bar(x + width/2, results_df['xgb_dir_test_acc'], width, label='XGBoost', color='#2ecc71', alpha=0.8)
ax1.set_xlabel('Prediction Horizon', fontweight='bold', fontsize=11)
ax1.set_ylabel('Direction Accuracy', fontweight='bold', fontsize=11)
ax1.set_title('Direction Prediction Accuracy by Horizon', fontweight='bold', fontsize=13)
ax1.set_xticks(x)
ax1.set_xticklabels(results_df['horizon'])
ax1.set_ylim([0, 1])
ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Random')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 2. Return R² by Horizon
ax2 = fig.add_subplot(gs[0, 2:4])
bars = ax2.bar(results_df['horizon'], results_df['rf_ret_test_r2'], color='#9b59b6', alpha=0.8)
ax2.set_xlabel('Prediction Horizon', fontweight='bold', fontsize=11)
ax2.set_ylabel('R² Score', fontweight='bold', fontsize=11)
ax2.set_title('Return Prediction R² by Horizon', fontweight='bold', fontsize=13)
ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
ax2.grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom' if height > 0 else 'top',
            fontsize=10, fontweight='bold')

# 3. ETF Pre/Post Comparison
ax3 = fig.add_subplot(gs[1, 0:2])
x = np.arange(len(etf_comparison_df))
width = 0.35
bars1 = ax3.bar(x - width/2, etf_comparison_df['pre_etf_acc'], width,
                label='Pre-ETF', color='#3498db', alpha=0.7)
bars2 = ax3.bar(x + width/2, etf_comparison_df['post_etf_acc'], width,
                label='Post-ETF', color='#27ae60', alpha=0.7)
ax3.set_xlabel('Prediction Horizon', fontweight='bold', fontsize=11)
ax3.set_ylabel('Accuracy', fontweight='bold', fontsize=11)
ax3.set_title('ETF Pre/Post Comparison by Horizon', fontweight='bold', fontsize=13)
ax3.set_xticks(x)
ax3.set_xticklabels(etf_comparison_df['horizon'])
ax3.set_ylim([0, 1])
ax3.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# 4. RMSE by Horizon
ax4 = fig.add_subplot(gs[1, 2:4])
bars = ax4.bar(results_df['horizon'], results_df['rf_ret_test_rmse'], color='#e74c3c', alpha=0.8)
ax4.set_xlabel('Prediction Horizon', fontweight='bold', fontsize=11)
ax4.set_ylabel('RMSE (%)', fontweight='bold', fontsize=11)
ax4.set_title('Prediction Error (RMSE) by Horizon', fontweight='bold', fontsize=13)
ax4.grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 5. Top Features - 1 day
ax5 = fig.add_subplot(gs[2, 0])
top_1d = feature_importance_all['1d_direction'].head(10)
ax5.barh(range(10), top_1d['importance'].values, color='#3498db', alpha=0.7)
ax5.set_yticks(range(10))
ax5.set_yticklabels(top_1d['feature'].values, fontsize=7)
ax5.set_xlabel('Importance', fontweight='bold', fontsize=9)
ax5.set_title('Top Features (1-day)', fontweight='bold', fontsize=10)
ax5.invert_yaxis()
ax5.grid(True, alpha=0.3, axis='x')

# 6. Top Features - 7 day
ax6 = fig.add_subplot(gs[2, 1])
top_7d = feature_importance_all['7d_direction'].head(10)
ax6.barh(range(10), top_7d['importance'].values, color='#2ecc71', alpha=0.7)
ax6.set_yticks(range(10))
ax6.set_yticklabels(top_7d['feature'].values, fontsize=7)
ax6.set_xlabel('Importance', fontweight='bold', fontsize=9)
ax6.set_title('Top Features (7-day)', fontweight='bold', fontsize=10)
ax6.invert_yaxis()
ax6.grid(True, alpha=0.3, axis='x')

# 7. Top Features - 30 day
ax7 = fig.add_subplot(gs[2, 2])
top_30d = feature_importance_all['30d_direction'].head(10)
ax7.barh(range(10), top_30d['importance'].values, color='#9b59b6', alpha=0.7)
ax7.set_yticks(range(10))
ax7.set_yticklabels(top_30d['feature'].values, fontsize=7)
ax7.set_xlabel('Importance', fontweight='bold', fontsize=9)
ax7.set_title('Top Features (30-day)', fontweight='bold', fontsize=10)
ax7.invert_yaxis()
ax7.grid(True, alpha=0.3, axis='x')

# 8. Accuracy vs Horizon (line plot)
ax8 = fig.add_subplot(gs[2, 3])
ax8.plot(results_df['horizon_days'], results_df['rf_dir_test_acc'],
         marker='o', linewidth=2, markersize=8, label='RF', color='#3498db')
ax8.plot(results_df['horizon_days'], results_df['xgb_dir_test_acc'],
         marker='s', linewidth=2, markersize=8, label='XGBoost', color='#2ecc71')
ax8.set_xlabel('Horizon (days)', fontweight='bold', fontsize=11)
ax8.set_ylabel('Accuracy', fontweight='bold', fontsize=11)
ax8.set_title('Accuracy Trend', fontweight='bold', fontsize=11)
ax8.set_ylim([0.4, 0.8])
ax8.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Random')
ax8.legend()
ax8.grid(True, alpha=0.3)
ax8.set_xticks([1, 7, 30])

plt.savefig('multi_horizon_prediction.png', dpi=300, bbox_inches='tight')
print("Saved: multi_horizon_prediction.png")

# ========================================
# 8. Summary Table
# ========================================
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)

summary_table = pd.DataFrame({
    'Horizon': results_df['horizon'],
    'RF Accuracy': results_df['rf_dir_test_acc'].apply(lambda x: f"{x:.2%}"),
    'XGB Accuracy': results_df['xgb_dir_test_acc'].apply(lambda x: f"{x:.2%}"),
    'Return R²': results_df['rf_ret_test_r2'].apply(lambda x: f"{x:.3f}"),
    'RMSE (%)': results_df['rf_ret_test_rmse'].apply(lambda x: f"{x:.2f}"),
    'Test Samples': results_df['test_samples'],
    'Top Feature': [
        feature_importance_all['1d_direction'].iloc[0]['feature'],
        feature_importance_all['7d_direction'].iloc[0]['feature'],
        feature_importance_all['30d_direction'].iloc[0]['feature']
    ]
})

print("\n" + summary_table.to_string(index=False))

# ETF comparison table
print("\n" + "=" * 80)
print("ETF PRE/POST COMPARISON")
print("=" * 80)

etf_table = pd.DataFrame({
    'Horizon': etf_comparison_df['horizon'],
    'Pre-ETF': etf_comparison_df['pre_etf_acc'].apply(lambda x: f"{x:.2%}"),
    'Post-ETF': etf_comparison_df['post_etf_acc'].apply(lambda x: f"{x:.2%}"),
    'Improvement': etf_comparison_df['improvement'].apply(lambda x: f"{x:+.2%}p")
})

print("\n" + etf_table.to_string(index=False))

# Top features comparison
print("\n" + "=" * 80)
print("TOP 5 FEATURES BY HORIZON")
print("=" * 80)

for horizon in [1, 7, 30]:
    print(f"\n{horizon}-day Horizon (Direction):")
    print(feature_importance_all[f'{horizon}d_direction'].head(5).to_string(index=False))

# Save results
results_df.to_csv('multi_horizon_results.csv', index=False)
etf_comparison_df.to_csv('multi_horizon_etf_comparison.csv', index=False)

# Save feature importance
for key, df_imp in feature_importance_all.items():
    df_imp.to_csv(f'feature_importance_{key}.csv', index=False)

print("\nSaved all results to CSV files.")

# ========================================
# 9. Key Insights
# ========================================
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

best_horizon_idx = results_df['rf_dir_test_acc'].idxmax()
best_horizon = results_df.loc[best_horizon_idx, 'horizon']
best_acc = results_df.loc[best_horizon_idx, 'rf_dir_test_acc']

print(f"""
📊 Multi-Horizon Prediction 분석 결과

1. 최고 성능 예측 기간:
   🏆 {best_horizon} 예측이 {best_acc:.2%} 정확도로 가장 우수

2. 기간별 성능:
   1일:  RF {results_df.loc[0, 'rf_dir_test_acc']:.2%}, XGB {results_df.loc[0, 'xgb_dir_test_acc']:.2%}, R² {results_df.loc[0, 'rf_ret_test_r2']:.3f}
   7일:  RF {results_df.loc[1, 'rf_dir_test_acc']:.2%}, XGB {results_df.loc[1, 'xgb_dir_test_acc']:.2%}, R² {results_df.loc[1, 'rf_ret_test_r2']:.3f}
   30일: RF {results_df.loc[2, 'rf_dir_test_acc']:.2%}, XGB {results_df.loc[2, 'xgb_dir_test_acc']:.2%}, R² {results_df.loc[2, 'rf_ret_test_r2']:.3f}

3. ETF 전후 비교 (정확도 개선폭):
   1일:  {etf_comparison_df.loc[0, 'improvement']:+.2%}p
   7일:  {etf_comparison_df.loc[1, 'improvement']:+.2%}p
   30일: {etf_comparison_df.loc[2, 'improvement']:+.2%}p

4. 중요 변수 변화:
   1일:  {feature_importance_all['1d_direction'].iloc[0]['feature']} ({feature_importance_all['1d_direction'].iloc[0]['importance']:.3f})
   7일:  {feature_importance_all['7d_direction'].iloc[0]['feature']} ({feature_importance_all['7d_direction'].iloc[0]['importance']:.3f})
   30일: {feature_importance_all['30d_direction'].iloc[0]['feature']} ({feature_importance_all['30d_direction'].iloc[0]['importance']:.3f})

5. 예측 오차 (RMSE):
   1일:  {results_df.loc[0, 'rf_ret_test_rmse']:.2f}%
   7일:  {results_df.loc[1, 'rf_ret_test_rmse']:.2f}%
   30일: {results_df.loc[2, 'rf_ret_test_rmse']:.2f}%

6. 결론:
   {'✅ 장기 예측일수록 정확도 향상 (노이즈 감소 효과)' if results_df.loc[2, 'rf_dir_test_acc'] > results_df.loc[0, 'rf_dir_test_acc'] else '⚠️ 단기 예측이 더 정확 (트렌드 변화 포착)'}
   {'✅ 모든 기간에서 ETF 이후 성능 개선' if all(etf_comparison_df['improvement'] > 0) else '⚠️ 일부 기간에서 ETF 이후 성능 하락'}
   {'✅ 장기 예측에서 R² 개선' if results_df.loc[2, 'rf_ret_test_r2'] > results_df.loc[0, 'rf_ret_test_r2'] else '⚠️ 수익률 예측은 여전히 어려움'}
""")

print("\n" + "=" * 80)
print("Step 22 Completed!")
print("=" * 80)
