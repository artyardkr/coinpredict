#!/usr/bin/env python3
"""
ElasticNet XAI (설명가능한 AI) 분석

1. 계수 (Coefficients) 분석
2. Feature Importance
3. SHAP Values
4. 개별 예측 설명
5. 변수 간 상호작용
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import shap
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("ElasticNet XAI 분석")
print("="*80)

# ========================================
# 1. 데이터 준비 및 모델 학습
# ========================================
print("\n[1/6] 데이터 로드 및 모델 학습...")

df = pd.read_csv('integrated_data_full.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Target
df['target'] = df['Close'].shift(-1)
df = df[:-1].copy()

# Feature 선택 (step25와 동일)
exclude_cols = [
    'Date', 'Close', 'High', 'Low', 'Open', 'target',
    'cumulative_return', 'bc_market_price', 'bc_market_cap',
]

ema_sma_cols = [col for col in df.columns if ('EMA' in col or 'SMA' in col) and 'close' in col.lower()]
exclude_cols.extend(ema_sma_cols)
bb_cols = [col for col in df.columns if col.startswith('BB_')]
exclude_cols.extend(bb_cols)
exclude_cols = list(set(exclude_cols))

feature_cols = [col for col in df.columns if col not in exclude_cols]

for col in feature_cols:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

print(f"Features: {len(feature_cols)}개")

# Train/Test Split (70/30)
split_idx = int(len(df) * 0.7)
split_date = df['Date'].iloc[split_idx]

X_train = df.iloc[:split_idx][feature_cols].values
X_test = df.iloc[split_idx:][feature_cols].values
y_train = df.iloc[:split_idx]['target'].values
y_test = df.iloc[split_idx:]['target'].values

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ElasticNet 학습
print("ElasticNet 학습 중...")
model = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000, random_state=42)
model.fit(X_train_scaled, y_train)

# 예측
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# 성능
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\n성능:")
print(f"  Train R²: {r2_train:.4f}")
print(f"  Test R²:  {r2_test:.4f}")
print(f"  Test RMSE: ${rmse_test:,.2f}")

# ========================================
# 2. 계수 (Coefficients) 분석
# ========================================
print("\n" + "="*60)
print("[2/6] 계수 분석")
print("="*60)

# 계수 추출
coefficients = model.coef_
intercept = model.intercept_

print(f"\nIntercept (절편): ${intercept:,.2f}")
print(f"Non-zero 계수: {np.sum(coefficients != 0)}/{len(coefficients)}")

# 계수 DataFrame
coef_df = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

# 0이 아닌 계수만
coef_df_nonzero = coef_df[coef_df['Coefficient'] != 0].copy()

print(f"\nTop 10 중요 변수 (절댓값 기준):")
for i, row in coef_df_nonzero.head(10).iterrows():
    sign = '📈' if row['Coefficient'] > 0 else '📉'
    print(f"  {sign} {row['Feature']:<30s}: {row['Coefficient']:+.4f}")

print(f"\n양수 계수 (가격 상승 요인): {np.sum(coefficients > 0)}개")
print(f"음수 계수 (가격 하락 요인): {np.sum(coefficients < 0)}개")
print(f"0 계수 (제거된 변수): {np.sum(coefficients == 0)}개")

# ========================================
# 3. Feature Importance
# ========================================
print("\n" + "="*60)
print("[3/6] Feature Importance")
print("="*60)

# Standardized 계수 (중요도)
# 표준화된 데이터에서 학습했으므로, 계수의 절댓값이 곧 중요도
feature_importance = np.abs(coefficients)
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

# 누적 중요도
importance_df['Cumulative'] = importance_df['Importance'].cumsum() / importance_df['Importance'].sum()

# 80% 설명력을 위한 변수 개수
n_features_80 = (importance_df['Cumulative'] <= 0.8).sum() + 1

print(f"\n상위 10개 변수 중요도:")
for i, row in importance_df.head(10).iterrows():
    print(f"  {row['Feature']:<30s}: {row['Importance']:.4f} (누적: {row['Cumulative']*100:.1f}%)")

print(f"\n80% 설명력을 위한 변수 개수: {n_features_80}개 (총 {len(feature_cols)}개 중)")

# ========================================
# 4. SHAP Values 분석
# ========================================
print("\n" + "="*60)
print("[4/6] SHAP Values 분석")
print("="*60)

print("SHAP Explainer 생성 중...")
# Linear explainer (빠름)
explainer = shap.LinearExplainer(model, X_train_scaled)

# Test set SHAP values
print("SHAP values 계산 중 (Test set)...")
shap_values_test = explainer.shap_values(X_test_scaled)

# SHAP 평균 절댓값 (전역 중요도)
shap_importance = np.abs(shap_values_test).mean(axis=0)
shap_df = pd.DataFrame({
    'Feature': feature_cols,
    'SHAP_Importance': shap_importance
}).sort_values('SHAP_Importance', ascending=False)

print(f"\nTop 10 SHAP 중요도:")
for i, row in shap_df.head(10).iterrows():
    print(f"  {row['Feature']:<30s}: {row['SHAP_Importance']:.4f}")

# ========================================
# 5. 개별 예측 설명 (샘플 3개)
# ========================================
print("\n" + "="*60)
print("[5/6] 개별 예측 설명")
print("="*60)

# 3가지 케이스: 정확, 과대평가, 과소평가
errors = y_pred_test - y_test
abs_errors = np.abs(errors)

# 가장 정확한 예측
best_idx = np.argmin(abs_errors)
# 가장 과대평가 (예측 >> 실제)
overpredict_idx = np.argmax(errors)
# 가장 과소평가 (예측 << 실제)
underpredict_idx = np.argmin(errors)

sample_indices = [best_idx, overpredict_idx, underpredict_idx]
sample_names = ['가장 정확', '과대평가', '과소평가']

for idx, name in zip(sample_indices, sample_names):
    print(f"\n[{name}]")
    print(f"  실제: ${y_test[idx]:,.2f}")
    print(f"  예측: ${y_pred_test[idx]:,.2f}")
    print(f"  오차: ${errors[idx]:+,.2f}")

    # Top 5 기여 변수
    sample_shap = shap_values_test[idx]
    sample_shap_df = pd.DataFrame({
        'Feature': feature_cols,
        'SHAP': sample_shap
    }).sort_values('SHAP', key=lambda x: abs(x), ascending=False)

    print(f"  Top 5 기여 변수:")
    for i, row in sample_shap_df.head(5).iterrows():
        direction = '⬆️' if row['SHAP'] > 0 else '⬇️'
        print(f"    {direction} {row['Feature']:<25s}: {row['SHAP']:+.2f}")

# ========================================
# 6. 시각화
# ========================================
print("\n" + "="*60)
print("[6/6] 시각화 생성 중...")
print("="*60)

fig = plt.figure(figsize=(22, 16))

# (1) Top 20 계수
ax1 = plt.subplot(4, 4, 1)
top_coef = coef_df_nonzero.head(20).sort_values('Coefficient')
colors = ['red' if x < 0 else 'green' for x in top_coef['Coefficient']]
ax1.barh(range(len(top_coef)), top_coef['Coefficient'], color=colors, alpha=0.7)
ax1.set_yticks(range(len(top_coef)))
ax1.set_yticklabels(top_coef['Feature'], fontsize=8)
ax1.set_xlabel('계수 (Coefficient)', fontweight='bold')
ax1.set_title('Top 20 변수 계수', fontsize=12, fontweight='bold')
ax1.axvline(0, color='black', linewidth=1)
ax1.grid(True, alpha=0.3, axis='x')

# (2) Feature Importance (절댓값)
ax2 = plt.subplot(4, 4, 2)
top_importance = importance_df.head(20)
ax2.barh(range(len(top_importance)), top_importance['Importance'], color='steelblue', alpha=0.7)
ax2.set_yticks(range(len(top_importance)))
ax2.set_yticklabels(top_importance['Feature'], fontsize=8)
ax2.set_xlabel('중요도 (|계수|)', fontweight='bold')
ax2.set_title('Top 20 Feature Importance', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# (3) 누적 중요도
ax3 = plt.subplot(4, 4, 3)
ax3.plot(range(1, len(importance_df)+1), importance_df['Cumulative'], linewidth=2, color='purple')
ax3.axhline(0.8, color='red', linestyle='--', linewidth=1, label='80%')
ax3.axvline(n_features_80, color='red', linestyle='--', linewidth=1)
ax3.set_xlabel('변수 개수', fontweight='bold')
ax3.set_ylabel('누적 중요도', fontweight='bold')
ax3.set_title('누적 Feature Importance', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.text(n_features_80, 0.85, f'{n_features_80}개', ha='center', fontweight='bold')

# (4) 계수 분포
ax4 = plt.subplot(4, 4, 4)
ax4.hist(coefficients[coefficients != 0], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
ax4.axvline(0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('계수 값', fontweight='bold')
ax4.set_ylabel('빈도', fontweight='bold')
ax4.set_title('계수 분포 (Non-zero)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# (5) SHAP Summary Plot
ax5 = plt.subplot(4, 4, (5, 6))
shap.summary_plot(shap_values_test, X_test_scaled, feature_names=feature_cols,
                  show=False, max_display=15, plot_size=(8, 5))
plt.title('SHAP Summary Plot (Top 15)', fontsize=12, fontweight='bold')

# (6) SHAP Bar Plot
ax6 = plt.subplot(4, 4, (7, 8))
top_shap = shap_df.head(20)
ax6.barh(range(len(top_shap)), top_shap['SHAP_Importance'], color='coral', alpha=0.7)
ax6.set_yticks(range(len(top_shap)))
ax6.set_yticklabels(top_shap['Feature'], fontsize=8)
ax6.set_xlabel('평균 |SHAP Value|', fontweight='bold')
ax6.set_title('Top 20 SHAP Importance', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='x')

# (7) 개별 예측 1: 정확
ax7 = plt.subplot(4, 4, 9)
idx = sample_indices[0]
sample_shap = shap_values_test[idx]
sample_shap_df = pd.DataFrame({
    'Feature': feature_cols,
    'SHAP': sample_shap
}).sort_values('SHAP', key=lambda x: abs(x), ascending=False).head(10)
colors_sample = ['red' if x < 0 else 'green' for x in sample_shap_df['SHAP']]
ax7.barh(range(len(sample_shap_df)), sample_shap_df['SHAP'], color=colors_sample, alpha=0.7)
ax7.set_yticks(range(len(sample_shap_df)))
ax7.set_yticklabels(sample_shap_df['Feature'], fontsize=7)
ax7.set_xlabel('SHAP Value', fontweight='bold')
ax7.set_title(f'가장 정확한 예측\n실제: ${y_test[idx]:,.0f}, 예측: ${y_pred_test[idx]:,.0f}',
             fontsize=10, fontweight='bold')
ax7.axvline(0, color='black', linewidth=1)
ax7.grid(True, alpha=0.3, axis='x')

# (8) 개별 예측 2: 과대평가
ax8 = plt.subplot(4, 4, 10)
idx = sample_indices[1]
sample_shap = shap_values_test[idx]
sample_shap_df = pd.DataFrame({
    'Feature': feature_cols,
    'SHAP': sample_shap
}).sort_values('SHAP', key=lambda x: abs(x), ascending=False).head(10)
colors_sample = ['red' if x < 0 else 'green' for x in sample_shap_df['SHAP']]
ax8.barh(range(len(sample_shap_df)), sample_shap_df['SHAP'], color=colors_sample, alpha=0.7)
ax8.set_yticks(range(len(sample_shap_df)))
ax8.set_yticklabels(sample_shap_df['Feature'], fontsize=7)
ax8.set_xlabel('SHAP Value', fontweight='bold')
ax8.set_title(f'과대평가 (예측 > 실제)\n실제: ${y_test[idx]:,.0f}, 예측: ${y_pred_test[idx]:,.0f}',
             fontsize=10, fontweight='bold')
ax8.axvline(0, color='black', linewidth=1)
ax8.grid(True, alpha=0.3, axis='x')

# (9) 개별 예측 3: 과소평가
ax9 = plt.subplot(4, 4, 11)
idx = sample_indices[2]
sample_shap = shap_values_test[idx]
sample_shap_df = pd.DataFrame({
    'Feature': feature_cols,
    'SHAP': sample_shap
}).sort_values('SHAP', key=lambda x: abs(x), ascending=False).head(10)
colors_sample = ['red' if x < 0 else 'green' for x in sample_shap_df['SHAP']]
ax9.barh(range(len(sample_shap_df)), sample_shap_df['SHAP'], color=colors_sample, alpha=0.7)
ax9.set_yticks(range(len(sample_shap_df)))
ax9.set_yticklabels(sample_shap_df['Feature'], fontsize=7)
ax9.set_xlabel('SHAP Value', fontweight='bold')
ax9.set_title(f'과소평가 (예측 < 실제)\n실제: ${y_test[idx]:,.0f}, 예측: ${y_pred_test[idx]:,.0f}',
             fontsize=10, fontweight='bold')
ax9.axvline(0, color='black', linewidth=1)
ax9.grid(True, alpha=0.3, axis='x')

# (10) 계수 vs SHAP 비교
ax10 = plt.subplot(4, 4, 12)
comparison_df = coef_df.merge(shap_df, on='Feature')
comparison_df = comparison_df[comparison_df['Abs_Coefficient'] > 0]
ax10.scatter(comparison_df['Abs_Coefficient'], comparison_df['SHAP_Importance'],
            alpha=0.5, s=50, color='purple')
ax10.set_xlabel('계수 절댓값', fontweight='bold')
ax10.set_ylabel('SHAP 중요도', fontweight='bold')
ax10.set_title('계수 vs SHAP 중요도', fontsize=12, fontweight='bold')
ax10.grid(True, alpha=0.3)

# 상관계수
corr = comparison_df['Abs_Coefficient'].corr(comparison_df['SHAP_Importance'])
ax10.text(0.05, 0.95, f'상관계수: {corr:.3f}', transform=ax10.transAxes,
         fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# (11) 양수/음수 계수 비율
ax11 = plt.subplot(4, 4, 13)
coef_stats = [
    np.sum(coefficients > 0),
    np.sum(coefficients < 0),
    np.sum(coefficients == 0)
]
labels_coef = ['양수\n(상승요인)', '음수\n(하락요인)', '0\n(제거)']
colors_coef = ['green', 'red', 'gray']
bars = ax11.bar(labels_coef, coef_stats, color=colors_coef, alpha=0.7)
for bar, val in zip(bars, coef_stats):
    ax11.text(bar.get_x() + bar.get_width()/2, val, f'{val}개',
            ha='center', va='bottom', fontweight='bold')
ax11.set_ylabel('변수 개수', fontweight='bold')
ax11.set_title('계수 부호 분포', fontsize=12, fontweight='bold')
ax11.grid(True, alpha=0.3, axis='y')

# (12) 요약
ax12 = plt.subplot(4, 4, (14, 16))
ax12.axis('off')
summary = f"""
【ElasticNet XAI 분석 요약】

1. 모델 성능:
   Train R²: {r2_train:.4f}
   Test R²:  {r2_test:.4f}
   Test RMSE: ${rmse_test:,.2f}

2. 변수 선택:
   총 변수: {len(feature_cols)}개
   사용 변수: {np.sum(coefficients != 0)}개
   제거 변수: {np.sum(coefficients == 0)}개
   제거율: {np.sum(coefficients == 0)/len(coefficients)*100:.1f}%

3. Top 5 중요 변수:
   {coef_df_nonzero.iloc[0]['Feature'][:25]}: {coef_df_nonzero.iloc[0]['Coefficient']:+.4f}
   {coef_df_nonzero.iloc[1]['Feature'][:25]}: {coef_df_nonzero.iloc[1]['Coefficient']:+.4f}
   {coef_df_nonzero.iloc[2]['Feature'][:25]}: {coef_df_nonzero.iloc[2]['Coefficient']:+.4f}
   {coef_df_nonzero.iloc[3]['Feature'][:25]}: {coef_df_nonzero.iloc[3]['Coefficient']:+.4f}
   {coef_df_nonzero.iloc[4]['Feature'][:25]}: {coef_df_nonzero.iloc[4]['Coefficient']:+.4f}

4. 80% 설명력:
   필요 변수: {n_features_80}개
   (전체의 {n_features_80/len(feature_cols)*100:.1f}%)

5. 계수 분포:
   양수 계수: {np.sum(coefficients > 0)}개 (가격 상승)
   음수 계수: {np.sum(coefficients < 0)}개 (가격 하락)

6. SHAP vs 계수:
   상관계수: {corr:.3f}
   {'→ 일치도 높음' if corr > 0.8 else '→ 일치도 보통' if corr > 0.5 else '→ 일치도 낮음'}
"""
ax12.text(0.05, 0.95, summary, fontsize=9, verticalalignment='top',
         family='monospace', transform=ax12.transAxes)
ax12.set_title('종합 요약', fontsize=14, fontweight='bold', loc='left')

plt.tight_layout()
plt.savefig('elasticnet_xai_analysis.png', dpi=300, bbox_inches='tight')
print("✅ 저장: elasticnet_xai_analysis.png")

# ========================================
# 7. 결과 저장
# ========================================
# 계수
coef_df_nonzero.to_csv('elasticnet_coefficients.csv', index=False)
print("✅ 저장: elasticnet_coefficients.csv")

# SHAP
shap_df.to_csv('elasticnet_shap_importance.csv', index=False)
print("✅ 저장: elasticnet_shap_importance.csv")

# 비교
comparison_df.to_csv('elasticnet_coef_shap_comparison.csv', index=False)
print("✅ 저장: elasticnet_coef_shap_comparison.csv")

print("\n" + "="*80)
print("XAI 분석 완료!")
print("="*80)
