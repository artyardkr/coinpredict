#!/usr/bin/env python3
"""
Step 21: ETF Approval Pre/Post Comparison

ETF 승인일(2024-01-10)을 기준으로 전후 기간을 분리하여:
1. 가격 범위 및 시장 특성 비교
2. 각 기간별 모델 학습 및 성능 비교
3. 특성 중요도 변화 분석
4. Extrapolation 문제 해결 여부 확인
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ========================================
# 1. Load Data
# ========================================
print("=" * 60)
print("Loading data...")
print("=" * 60)

df = pd.read_csv('integrated_data_full.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"Data shape: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

# ETF approval date
ETF_DATE = pd.to_datetime('2024-01-10')

# ========================================
# 2. Target: Direction (Binary)
# ========================================
print("\n" + "=" * 60)
print("Creating target...")
print("=" * 60)

df['next_return'] = (df['Close'].shift(-1) / df['Close'] - 1) * 100

THRESHOLD = 0.5

def classify_direction(return_pct, threshold=THRESHOLD):
    return 1 if return_pct > threshold else 0

df['target_direction'] = df['next_return'].apply(classify_direction)
df = df[:-1].copy()  # Remove last row

print(f"Threshold: {THRESHOLD}%")
print(f"Overall class distribution:\n{df['target_direction'].value_counts()}")

# ========================================
# 3. Split by ETF Date
# ========================================
print("\n" + "=" * 60)
print("Splitting by ETF approval date...")
print("=" * 60)

pre_etf = df[df['Date'] < ETF_DATE].copy()
post_etf = df[df['Date'] >= ETF_DATE].copy()

print(f"\nPre-ETF period: {pre_etf['Date'].min()} to {pre_etf['Date'].max()}")
print(f"  - Samples: {len(pre_etf)}")
print(f"  - Price range: ${pre_etf['Close'].min():.0f} - ${pre_etf['Close'].max():.0f}")
print(f"  - Mean price: ${pre_etf['Close'].mean():.0f}")
print(f"  - Volatility (std): ${pre_etf['Close'].std():.0f}")
print(f"  - Class distribution:\n{pre_etf['target_direction'].value_counts()}")

print(f"\nPost-ETF period: {post_etf['Date'].min()} to {post_etf['Date'].max()}")
print(f"  - Samples: {len(post_etf)}")
print(f"  - Price range: ${post_etf['Close'].min():.0f} - ${post_etf['Close'].max():.0f}")
print(f"  - Mean price: ${post_etf['Close'].mean():.0f}")
print(f"  - Volatility (std): ${post_etf['Close'].std():.0f}")
print(f"  - Class distribution:\n{post_etf['target_direction'].value_counts()}")

# ========================================
# 4. Feature Preparation
# ========================================
print("\n" + "=" * 60)
print("Preparing features...")
print("=" * 60)

# Exclude columns
exclude_cols = [
    'Date', 'target_direction', 'next_return',
    'Close', 'High', 'Low', 'Open',
    'cumulative_return',
    'bc_market_price', 'bc_market_cap',
]

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
# 5. Pre-ETF Analysis
# ========================================
print("\n" + "=" * 80)
print("PRE-ETF PERIOD ANALYSIS")
print("=" * 80)

pre_etf_clean = pre_etf.dropna(subset=feature_cols)
X_pre = pre_etf_clean[feature_cols].values
y_pre = pre_etf_clean['target_direction'].values

# Train/Test split (70:30)
split_idx_pre = int(len(X_pre) * 0.7)
X_train_pre = X_pre[:split_idx_pre]
X_test_pre = X_pre[split_idx_pre:]
y_train_pre = y_pre[:split_idx_pre]
y_test_pre = y_pre[split_idx_pre:]

print(f"Train samples: {len(X_train_pre)} ({len(X_train_pre)/len(X_pre)*100:.1f}%)")
print(f"Test samples: {len(X_test_pre)} ({len(X_test_pre)/len(X_pre)*100:.1f}%)")
print(f"Train class: {np.bincount(y_train_pre)}")
print(f"Test class: {np.bincount(y_test_pre)}")

# Scaling
scaler_pre = StandardScaler()
X_train_pre_scaled = scaler_pre.fit_transform(X_train_pre)
X_test_pre_scaled = scaler_pre.transform(X_test_pre)

# Random Forest
print("\nTraining Random Forest (Pre-ETF)...")
rf_pre = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf_pre.fit(X_train_pre_scaled, y_train_pre)

y_pred_rf_pre_train = rf_pre.predict(X_train_pre_scaled)
y_pred_rf_pre_test = rf_pre.predict(X_test_pre_scaled)

acc_rf_pre_train = accuracy_score(y_train_pre, y_pred_rf_pre_train)
acc_rf_pre_test = accuracy_score(y_test_pre, y_pred_rf_pre_test)

print(f"RF Train Accuracy: {acc_rf_pre_train:.4f}")
print(f"RF Test Accuracy: {acc_rf_pre_test:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_pre, y_pred_rf_pre_test, target_names=['Down', 'Up']))

# Feature importance
feature_importance_pre = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_pre.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features (Pre-ETF):")
print(feature_importance_pre.head(10).to_string(index=False))

# XGBoost
print("\nTraining XGBoost (Pre-ETF)...")
xgb_pre = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=len(y_train_pre[y_train_pre==0])/len(y_train_pre[y_train_pre==1])
)
xgb_pre.fit(X_train_pre_scaled, y_train_pre)

y_pred_xgb_pre_test = xgb_pre.predict(X_test_pre_scaled)
acc_xgb_pre_test = accuracy_score(y_test_pre, y_pred_xgb_pre_test)

print(f"XGBoost Test Accuracy: {acc_xgb_pre_test:.4f}")

# ========================================
# 6. Post-ETF Analysis
# ========================================
print("\n" + "=" * 80)
print("POST-ETF PERIOD ANALYSIS")
print("=" * 80)

post_etf_clean = post_etf.dropna(subset=feature_cols)
X_post = post_etf_clean[feature_cols].values
y_post = post_etf_clean['target_direction'].values

# Train/Test split (70:30)
split_idx_post = int(len(X_post) * 0.7)
X_train_post = X_post[:split_idx_post]
X_test_post = X_post[split_idx_post:]
y_train_post = y_post[:split_idx_post]
y_test_post = y_post[split_idx_post:]

print(f"Train samples: {len(X_train_post)} ({len(X_train_post)/len(X_post)*100:.1f}%)")
print(f"Test samples: {len(X_test_post)} ({len(X_test_post)/len(X_post)*100:.1f}%)")
print(f"Train class: {np.bincount(y_train_post)}")
print(f"Test class: {np.bincount(y_test_post)}")

# Scaling
scaler_post = StandardScaler()
X_train_post_scaled = scaler_post.fit_transform(X_train_post)
X_test_post_scaled = scaler_post.transform(X_test_post)

# Random Forest
print("\nTraining Random Forest (Post-ETF)...")
rf_post = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf_post.fit(X_train_post_scaled, y_train_post)

y_pred_rf_post_train = rf_post.predict(X_train_post_scaled)
y_pred_rf_post_test = rf_post.predict(X_test_post_scaled)

acc_rf_post_train = accuracy_score(y_train_post, y_pred_rf_post_train)
acc_rf_post_test = accuracy_score(y_test_post, y_pred_rf_post_test)

print(f"RF Train Accuracy: {acc_rf_post_train:.4f}")
print(f"RF Test Accuracy: {acc_rf_post_test:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_post, y_pred_rf_post_test, target_names=['Down', 'Up']))

# Feature importance
feature_importance_post = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_post.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features (Post-ETF):")
print(feature_importance_post.head(10).to_string(index=False))

# XGBoost
print("\nTraining XGBoost (Post-ETF)...")
xgb_post = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=len(y_train_post[y_train_post==0])/len(y_train_post[y_train_post==1])
)
xgb_post.fit(X_train_post_scaled, y_train_post)

y_pred_xgb_post_test = xgb_post.predict(X_test_post_scaled)
acc_xgb_post_test = accuracy_score(y_test_post, y_pred_xgb_post_test)

print(f"XGBoost Test Accuracy: {acc_xgb_post_test:.4f}")

# ========================================
# 7. Cross-Period Test (Extrapolation)
# ========================================
print("\n" + "=" * 80)
print("CROSS-PERIOD TEST (Extrapolation Problem Analysis)")
print("=" * 80)

# Test Pre-ETF model on Post-ETF data
print("\n1. Pre-ETF model → Post-ETF data:")
y_pred_pre_on_post = rf_pre.predict(scaler_pre.transform(X_post))
acc_pre_on_post = accuracy_score(y_post, y_pred_pre_on_post)
print(f"   Accuracy: {acc_pre_on_post:.4f}")
print(f"   Predictions: {np.bincount(y_pred_pre_on_post)} (Down/Up)")

# Test Post-ETF model on Pre-ETF data
print("\n2. Post-ETF model → Pre-ETF data:")
y_pred_post_on_pre = rf_post.predict(scaler_post.transform(X_pre))
acc_post_on_pre = accuracy_score(y_pre, y_pred_post_on_pre)
print(f"   Accuracy: {acc_post_on_pre:.4f}")
print(f"   Predictions: {np.bincount(y_pred_post_on_pre)} (Down/Up)")

# ========================================
# 8. Feature Importance Comparison
# ========================================
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE SHIFT ANALYSIS")
print("=" * 80)

# Compare top features
top_n = 15
top_pre = feature_importance_pre.head(top_n).copy()
top_post = feature_importance_post.head(top_n).copy()

# Create comparison dataframe
comparison = pd.merge(
    top_pre[['feature', 'importance']].rename(columns={'importance': 'pre_importance'}),
    top_post[['feature', 'importance']].rename(columns={'importance': 'post_importance'}),
    on='feature',
    how='outer'
).fillna(0)

comparison['change'] = comparison['post_importance'] - comparison['pre_importance']
comparison = comparison.sort_values('change', key=abs, ascending=False)

print("\nFeatures with Biggest Importance Shift:")
print(comparison.head(10).to_string(index=False))

# ========================================
# 9. Market Statistics Comparison
# ========================================
print("\n" + "=" * 80)
print("MARKET STATISTICS COMPARISON")
print("=" * 80)

stats_comparison = pd.DataFrame({
    'Metric': [
        'Samples',
        'Days',
        'Min Price',
        'Max Price',
        'Mean Price',
        'Median Price',
        'Std Dev',
        'Return (%)',
        'Daily Return Mean (%)',
        'Daily Return Std (%)',
        'Up Days (%)',
        'Down Days (%)'
    ],
    'Pre-ETF': [
        len(pre_etf),
        (pre_etf['Date'].max() - pre_etf['Date'].min()).days,
        f"${pre_etf['Close'].min():.0f}",
        f"${pre_etf['Close'].max():.0f}",
        f"${pre_etf['Close'].mean():.0f}",
        f"${pre_etf['Close'].median():.0f}",
        f"${pre_etf['Close'].std():.0f}",
        f"{(pre_etf['Close'].iloc[-1]/pre_etf['Close'].iloc[0]-1)*100:.1f}%",
        f"{pre_etf['next_return'].mean():.3f}%",
        f"{pre_etf['next_return'].std():.3f}%",
        f"{(pre_etf['target_direction']==1).sum()/len(pre_etf)*100:.1f}%",
        f"{(pre_etf['target_direction']==0).sum()/len(pre_etf)*100:.1f}%"
    ],
    'Post-ETF': [
        len(post_etf),
        (post_etf['Date'].max() - post_etf['Date'].min()).days,
        f"${post_etf['Close'].min():.0f}",
        f"${post_etf['Close'].max():.0f}",
        f"${post_etf['Close'].mean():.0f}",
        f"${post_etf['Close'].median():.0f}",
        f"${post_etf['Close'].std():.0f}",
        f"{(post_etf['Close'].iloc[-1]/post_etf['Close'].iloc[0]-1)*100:.1f}%",
        f"{post_etf['next_return'].mean():.3f}%",
        f"{post_etf['next_return'].std():.3f}%",
        f"{(post_etf['target_direction']==1).sum()/len(post_etf)*100:.1f}%",
        f"{(post_etf['target_direction']==0).sum()/len(post_etf)*100:.1f}%"
    ]
})

print("\n" + stats_comparison.to_string(index=False))

# ========================================
# 10. Visualization
# ========================================
print("\n" + "=" * 60)
print("Creating visualizations...")
print("=" * 60)

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Price over time with ETF line
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df['Date'], df['Close'], linewidth=2, color='#2c3e50')
ax1.axvline(ETF_DATE, color='red', linestyle='--', linewidth=2, label='ETF Approval', alpha=0.7)
ax1.fill_between(pre_etf['Date'], 0, pre_etf['Close'].max()*1.1, alpha=0.1, color='blue', label='Pre-ETF')
ax1.fill_between(post_etf['Date'], 0, post_etf['Close'].max()*1.1, alpha=0.1, color='green', label='Post-ETF')
ax1.set_ylabel('BTC Price (USD)', fontsize=12, fontweight='bold')
ax1.set_title('Bitcoin Price: Pre/Post ETF Approval', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# 2. Model Performance Comparison
ax2 = fig.add_subplot(gs[1, 0])
models = ['RF\nPre-ETF', 'RF\nPost-ETF', 'XGB\nPre-ETF', 'XGB\nPost-ETF']
accuracies = [acc_rf_pre_test, acc_rf_post_test, acc_xgb_pre_test, acc_xgb_post_test]
colors = ['#3498db', '#27ae60', '#9b59b6', '#16a085']
bars = ax2.bar(models, accuracies, color=colors, alpha=0.7)
ax2.set_ylabel('Test Accuracy', fontweight='bold')
ax2.set_title('Model Performance: Pre vs Post ETF', fontweight='bold')
ax2.set_ylim([0, 1])
ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Random')
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# 3. Extrapolation Test
ax3 = fig.add_subplot(gs[1, 1])
extrap_labels = ['Pre→Pre\n(Normal)', 'Pre→Post\n(Extrap)', 'Post→Post\n(Normal)', 'Post→Pre\n(Retro)']
extrap_accs = [acc_rf_pre_test, acc_pre_on_post, acc_rf_post_test, acc_post_on_pre]
colors_extrap = ['#3498db', '#e74c3c', '#27ae60', '#f39c12']
bars2 = ax3.bar(extrap_labels, extrap_accs, color=colors_extrap, alpha=0.7)
ax3.set_ylabel('Accuracy', fontweight='bold')
ax3.set_title('Cross-Period Test (Extrapolation)', fontweight='bold')
ax3.set_ylim([0, 1])
ax3.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
for bar in bars2:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# 4. Daily Returns Distribution
ax4 = fig.add_subplot(gs[1, 2])
ax4.hist(pre_etf['next_return'].dropna(), bins=50, alpha=0.5, label='Pre-ETF', color='blue', density=True)
ax4.hist(post_etf['next_return'].dropna(), bins=50, alpha=0.5, label='Post-ETF', color='green', density=True)
ax4.set_xlabel('Daily Return (%)', fontweight='bold')
ax4.set_ylabel('Density', fontweight='bold')
ax4.set_title('Daily Returns Distribution', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Top Features Pre-ETF
ax5 = fig.add_subplot(gs[2, 0])
top_10_pre = feature_importance_pre.head(10)
ax5.barh(range(10), top_10_pre['importance'].values, color='#3498db', alpha=0.7)
ax5.set_yticks(range(10))
ax5.set_yticklabels(top_10_pre['feature'].values, fontsize=8)
ax5.set_xlabel('Importance', fontweight='bold')
ax5.set_title('Top 10 Features (Pre-ETF)', fontweight='bold')
ax5.invert_yaxis()
ax5.grid(True, alpha=0.3, axis='x')

# 6. Top Features Post-ETF
ax6 = fig.add_subplot(gs[2, 1])
top_10_post = feature_importance_post.head(10)
ax6.barh(range(10), top_10_post['importance'].values, color='#27ae60', alpha=0.7)
ax6.set_yticks(range(10))
ax6.set_yticklabels(top_10_post['feature'].values, fontsize=8)
ax6.set_xlabel('Importance', fontweight='bold')
ax6.set_title('Top 10 Features (Post-ETF)', fontweight='bold')
ax6.invert_yaxis()
ax6.grid(True, alpha=0.3, axis='x')

# 7. Feature Importance Change
ax7 = fig.add_subplot(gs[2, 2])
top_changes = comparison.head(10)
colors_change = ['red' if x < 0 else 'green' for x in top_changes['change']]
ax7.barh(range(10), top_changes['change'].values, color=colors_change, alpha=0.7)
ax7.set_yticks(range(10))
ax7.set_yticklabels(top_changes['feature'].values, fontsize=8)
ax7.set_xlabel('Importance Change (Post - Pre)', fontweight='bold')
ax7.set_title('Biggest Feature Importance Shifts', fontweight='bold')
ax7.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax7.invert_yaxis()
ax7.grid(True, alpha=0.3, axis='x')

plt.savefig('etf_comparison_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: etf_comparison_analysis.png")

# ========================================
# 11. Summary & Conclusions
# ========================================
print("\n" + "=" * 80)
print("SUMMARY & CONCLUSIONS")
print("=" * 80)

print(f"""
📊 ETF 전후 비교 분석 결과

1. 모델 성능:
   Pre-ETF:  RF {acc_rf_pre_test:.2%}, XGBoost {acc_xgb_pre_test:.2%}
   Post-ETF: RF {acc_rf_post_test:.2%}, XGBoost {acc_xgb_post_test:.2%}

   {'✅ Post-ETF 성능 개선!' if acc_rf_post_test > acc_rf_pre_test else '⚠️ Post-ETF 성능 하락'}

2. Extrapolation 문제:
   Pre-ETF model → Post-ETF data: {acc_pre_on_post:.2%}
   {'❌ 심각한 Extrapolation 실패' if acc_pre_on_post < 0.45 else '✅ Extrapolation 문제 없음'}

   Post-ETF model → Pre-ETF data: {acc_post_on_pre:.2%}
   {'✅ 역방향 예측 성공' if acc_post_on_pre > 0.50 else '⚠️ 역방향 예측 어려움'}

3. 가격 범위:
   Pre-ETF:  ${pre_etf['Close'].min():.0f} - ${pre_etf['Close'].max():.0f} (평균 ${pre_etf['Close'].mean():.0f})
   Post-ETF: ${post_etf['Close'].min():.0f} - ${post_etf['Close'].max():.0f} (평균 ${post_etf['Close'].mean():.0f})

   평균 가격 {post_etf['Close'].mean()/pre_etf['Close'].mean():.2f}배 증가

4. 변동성:
   Pre-ETF:  일평균 변화율 {pre_etf['next_return'].std():.2f}%
   Post-ETF: 일평균 변화율 {post_etf['next_return'].std():.2f}%

   {'✅ 변동성 감소 (안정화)' if post_etf['next_return'].std() < pre_etf['next_return'].std() else '⚠️ 변동성 증가'}

5. 중요 변수 변화:
   Pre-ETF Top 3:
     1) {feature_importance_pre.iloc[0]['feature']} ({feature_importance_pre.iloc[0]['importance']:.3f})
     2) {feature_importance_pre.iloc[1]['feature']} ({feature_importance_pre.iloc[1]['importance']:.3f})
     3) {feature_importance_pre.iloc[2]['feature']} ({feature_importance_pre.iloc[2]['importance']:.3f})

   Post-ETF Top 3:
     1) {feature_importance_post.iloc[0]['feature']} ({feature_importance_post.iloc[0]['importance']:.3f})
     2) {feature_importance_post.iloc[1]['feature']} ({feature_importance_post.iloc[1]['importance']:.3f})
     3) {feature_importance_post.iloc[2]['feature']} ({feature_importance_post.iloc[2]['importance']:.3f})

6. 결론:
   {'✅ ETF 승인 후 예측 모델 성능이 개선되었습니다.' if acc_rf_post_test > acc_rf_pre_test else '⚠️ ETF 승인 후에도 예측이 어렵습니다.'}
   {'✅ 기간별 분리 학습으로 Extrapolation 문제를 완화할 수 있습니다.' if acc_pre_on_post > 0.45 else '❌ Pre-ETF 모델은 Post-ETF 데이터에 적용할 수 없습니다 (Extrapolation 실패).'}
   {'✅ 시장 안정화로 변동성이 감소했습니다.' if post_etf['next_return'].std() < pre_etf['next_return'].std() else '⚠️ 변동성이 증가하여 예측이 더 어려워졌습니다.'}
""")

# Save results
results = {
    'period': ['Pre-ETF', 'Post-ETF', 'Pre→Post (Extrap)', 'Post→Pre (Retro)'],
    'rf_accuracy': [acc_rf_pre_test, acc_rf_post_test, acc_pre_on_post, acc_post_on_pre],
    'xgb_accuracy': [acc_xgb_pre_test, acc_xgb_post_test, np.nan, np.nan],
    'samples': [len(pre_etf), len(post_etf), len(post_etf), len(pre_etf)],
    'price_mean': [pre_etf['Close'].mean(), post_etf['Close'].mean(), post_etf['Close'].mean(), pre_etf['Close'].mean()],
    'volatility': [pre_etf['next_return'].std(), post_etf['next_return'].std(), post_etf['next_return'].std(), pre_etf['next_return'].std()]
}

results_df = pd.DataFrame(results)
results_df.to_csv('etf_comparison_results.csv', index=False)
print("\nSaved: etf_comparison_results.csv")

print("\n" + "=" * 80)
print("Step 21 Completed!")
print("=" * 80)
