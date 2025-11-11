#!/usr/bin/env python3
"""
ElasticNet XAI V2 (설명가능한 AI) 분석 + 과적합 검증

V1 (88개 변수) → V2 (138개 변수)

분석 항목:
1. 계수 (Coefficients) 분석
2. Feature Importance
3. SHAP Values
4. 과적합 검증 (Learning Curve, Residual 분석)
5. 개별 예측 설명
6. V1 vs V2 비교

출력: 3개의 PNG 파일로 분리
- Part 1: 과적합 검증
- Part 2: Feature Importance & 계수
- Part 3: SHAP 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import learning_curve
import shap
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
try:
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False
except:
    try:
        plt.rcParams['font.family'] = 'Apple SD Gothic Neo'
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("한글 폰트를 찾을 수 없습니다. AppleGothic 또는 Apple SD Gothic Neo를 설치해주세요.")

print("="*80)
print("ElasticNet XAI V2 분석 + 과적합 검증")
print("="*80)

# ========================================
# 1. 데이터 준비 및 모델 학습
# ========================================
print("\n[1/7] 데이터 로드 및 모델 학습 (V2)...")

df = pd.read_csv('integrated_data_full_v2.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"V2 데이터: {df.shape}")

# Target
df['target'] = df['Close'].shift(-1)
df = df[:-1].copy()

# Feature 선택
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

print(f"Features: {len(feature_cols)}개 (V2)")

# 신규 변수 확인
new_vars_keywords = ['DXY', 'ETH', 'TLT', 'GLD', 'WALCL', 'RRPONTSYD', 'FED_NET_LIQUIDITY',
                     'NVT', 'Puell', 'Hash_Ribbon', 'IBIT', 'FBTC', 'GBTC_Premium']
new_vars_found = [col for col in feature_cols if any(kw in col for kw in new_vars_keywords)]
print(f"신규 V2 변수: {len(new_vars_found)}개 발견")

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
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\n성능 (V2):")
print(f"  Train R²: {r2_train:.4f}, RMSE: ${rmse_train:,.2f}")
print(f"  Test R²:  {r2_test:.4f}, RMSE: ${rmse_test:,.2f}")
print(f"  차이 (Train-Test): {r2_train - r2_test:.4f}")

# 과적합 지표
overfit_score = r2_train - r2_test
if overfit_score < 0.05:
    overfit_status = "✅ 과적합 없음 (매우 양호)"
elif overfit_score < 0.15:
    overfit_status = "⚠️ 약간의 과적합 (허용 가능)"
else:
    overfit_status = "❌ 과적합 의심 (주의 필요)"

print(f"  과적합 평가: {overfit_status}")

# ========================================
# 2. 과적합 검증 - Learning Curve
# ========================================
print("\n" + "="*60)
print("[2/7] 과적합 검증 - Learning Curve")
print("="*60)

print("Learning Curve 계산 중...")
train_sizes = np.linspace(0.1, 1.0, 10)
train_sizes_abs, train_scores, test_scores = learning_curve(
    model, X_train_scaled, y_train,
    train_sizes=train_sizes,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=42
)

train_scores_mean = train_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
test_scores_mean = test_scores.mean(axis=1)
test_scores_std = test_scores.std(axis=1)

print(f"\nLearning Curve 결과:")
print(f"  최소 데이터 (10%): Train={train_scores_mean[0]:.3f}, Test={test_scores_mean[0]:.3f}")
print(f"  최대 데이터 (100%): Train={train_scores_mean[-1]:.3f}, Test={test_scores_mean[-1]:.3f}")
print(f"  Gap (마지막): {train_scores_mean[-1] - test_scores_mean[-1]:.3f}")

if train_scores_mean[-1] - test_scores_mean[-1] < 0.1:
    print("  ✅ 데이터 증가해도 Gap 작음 → 과적합 없음")
else:
    print("  ⚠️ 데이터 증가해도 Gap 큼 → 과적합 가능성")

# ========================================
# 3. 과적합 검증 - Residual 분석
# ========================================
print("\n" + "="*60)
print("[3/7] 과적합 검증 - Residual 분석")
print("="*60)

# 잔차
residuals_train = y_train - y_pred_train
residuals_test = y_test - y_pred_test

# 잔차 통계
print(f"\nTrain Residuals:")
print(f"  평균: ${residuals_train.mean():,.2f}")
print(f"  표준편차: ${residuals_train.std():,.2f}")

print(f"\nTest Residuals:")
print(f"  평균: ${residuals_test.mean():,.2f}")
print(f"  표준편차: ${residuals_test.std():,.2f}")

# 잔차 패턴 확인 (자기상관)
residual_autocorr_train = np.corrcoef(residuals_train[:-1], residuals_train[1:])[0, 1]
residual_autocorr_test = np.corrcoef(residuals_test[:-1], residuals_test[1:])[0, 1]

print(f"\n잔차 자기상관 (패턴 검사):")
print(f"  Train: {residual_autocorr_train:.3f}")
print(f"  Test:  {residual_autocorr_test:.3f}")

if abs(residual_autocorr_test) < 0.2:
    print("  ✅ 잔차 패턴 없음 (랜덤) → 모델 양호")
else:
    print("  ⚠️ 잔차에 패턴 있음 → 개선 필요")

# ========================================
# 4. 계수 (Coefficients) 분석
# ========================================
print("\n" + "="*60)
print("[4/7] 계수 분석 (V2)")
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
    is_new = '🆕' if any(kw in row['Feature'] for kw in new_vars_keywords) else '  '
    print(f"  {is_new}{sign} {row['Feature']:<35s}: {row['Coefficient']:+.4f}")

print(f"\n양수 계수 (가격 상승 요인): {np.sum(coefficients > 0)}개")
print(f"음수 계수 (가격 하락 요인): {np.sum(coefficients < 0)}개")
print(f"0 계수 (제거된 변수): {np.sum(coefficients == 0)}개 ({np.sum(coefficients == 0)/len(coefficients)*100:.1f}%)")

# 신규 변수 중 사용된 개수
new_vars_used = sum(1 for col in new_vars_found if coef_df.loc[coef_df['Feature']==col, 'Coefficient'].values[0] != 0)
print(f"\n신규 V2 변수 중 사용: {new_vars_used}/{len(new_vars_found)}개 ({new_vars_used/len(new_vars_found)*100:.1f}%)")

# ========================================
# 5. SHAP Values 분석
# ========================================
print("\n" + "="*60)
print("[5/7] SHAP Values 분석 (V2)")
print("="*60)

print("SHAP Explainer 생성 중...")
explainer = shap.LinearExplainer(model, X_train_scaled)

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
    is_new = '🆕' if any(kw in row['Feature'] for kw in new_vars_keywords) else '  '
    print(f"  {is_new}{row['Feature']:<35s}: {row['SHAP_Importance']:.4f}")

# 신규 변수의 SHAP 중요도
new_vars_shap = shap_df[shap_df['Feature'].isin(new_vars_found)]
print(f"\n신규 V2 변수 SHAP 중요도:")
print(f"  평균: {new_vars_shap['SHAP_Importance'].mean():.4f}")
print(f"  Top 신규: {new_vars_shap.iloc[0]['Feature']} ({new_vars_shap.iloc[0]['SHAP_Importance']:.4f})")

# ========================================
# 6. Feature Importance
# ========================================
print("\n" + "="*60)
print("[6/7] Feature Importance")
print("="*60)

feature_importance = np.abs(coefficients)
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

importance_df['Cumulative'] = importance_df['Importance'].cumsum() / importance_df['Importance'].sum()

n_features_80 = (importance_df['Cumulative'] <= 0.8).sum() + 1

print(f"\n80% 설명력을 위한 변수 개수: {n_features_80}개 (총 {len(feature_cols)}개 중 {n_features_80/len(feature_cols)*100:.1f}%)")

# ========================================
# 7. 시각화 (3개 파일로 분리)
# ========================================
print("\n" + "="*60)
print("[7/7] 시각화 생성 중...")
print("="*60)

# ==================== Part 1: 과적합 검증 ====================
print("\nPart 1: 과적합 검증 시각화...")
fig1 = plt.figure(figsize=(20, 10))

# (1) Train vs Test R²
ax1 = plt.subplot(2, 3, 1)
bars = ax1.bar(['Train', 'Test'], [r2_train, r2_test], color=['steelblue', 'coral'], alpha=0.7)
for bar, val in zip(bars, [r2_train, r2_test]):
    ax1.text(bar.get_x() + bar.get_width()/2, val, f'{val:.3f}',
            ha='center', va='bottom', fontweight='bold', fontsize=12)
ax1.set_ylabel('R² Score', fontweight='bold', fontsize=12)
ax1.set_title(f'Train vs Test R² (Gap: {r2_train-r2_test:.3f})', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.text(0.5, 0.5, overfit_status, ha='center', va='center',
        transform=ax1.transAxes, fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow' if '⚠' in overfit_status else 'lightgreen', alpha=0.7))

# (2) Learning Curve
ax2 = plt.subplot(2, 3, 2)
ax2.plot(train_sizes_abs, train_scores_mean, 'o-', color='steelblue', label='Train', linewidth=2)
ax2.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std, alpha=0.2, color='steelblue')
ax2.plot(train_sizes_abs, test_scores_mean, 'o-', color='coral', label='Cross-val', linewidth=2)
ax2.fill_between(train_sizes_abs, test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std, alpha=0.2, color='coral')
ax2.set_xlabel('학습 데이터 크기', fontweight='bold', fontsize=12)
ax2.set_ylabel('R² Score', fontweight='bold', fontsize=12)
ax2.set_title('Learning Curve (과적합 검사)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# (3) Residual Distribution - Train
ax3 = plt.subplot(2, 3, 3)
ax3.hist(residuals_train, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax3.axvline(0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('Residual ($)', fontweight='bold', fontsize=12)
ax3.set_ylabel('빈도', fontweight='bold', fontsize=12)
ax3.set_title(f'Train Residuals (평균: ${residuals_train.mean():.0f})', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# (4) Residual Distribution - Test
ax4 = plt.subplot(2, 3, 4)
ax4.hist(residuals_test, bins=50, color='coral', alpha=0.7, edgecolor='black')
ax4.axvline(0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Residual ($)', fontweight='bold', fontsize=12)
ax4.set_ylabel('빈도', fontweight='bold', fontsize=12)
ax4.set_title(f'Test Residuals (평균: ${residuals_test.mean():.0f})', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# (5) Residual vs Fitted (Train)
ax5 = plt.subplot(2, 3, 5)
ax5.scatter(y_pred_train, residuals_train, alpha=0.3, s=10, color='steelblue')
ax5.axhline(0, color='red', linestyle='--', linewidth=2)
ax5.set_xlabel('예측값 ($)', fontweight='bold', fontsize=12)
ax5.set_ylabel('잔차 ($)', fontweight='bold', fontsize=12)
ax5.set_title('Residual vs Fitted (Train)', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3)

# (6) Residual vs Fitted (Test)
ax6 = plt.subplot(2, 3, 6)
ax6.scatter(y_pred_test, residuals_test, alpha=0.3, s=10, color='coral')
ax6.axhline(0, color='red', linestyle='--', linewidth=2)
ax6.set_xlabel('예측값 ($)', fontweight='bold', fontsize=12)
ax6.set_ylabel('잔차 ($)', fontweight='bold', fontsize=12)
ax6.set_title('Residual vs Fitted (Test)', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('elasticnet_xai_v2_part1_overfitting.png', dpi=300, bbox_inches='tight')
print("✅ 저장: elasticnet_xai_v2_part1_overfitting.png")
plt.close()

# ==================== Part 2: 계수 & Feature Importance ====================
print("Part 2: 계수 & Feature Importance 시각화...")
fig2 = plt.figure(figsize=(20, 12))

# (1) Top 20 계수
ax1 = plt.subplot(2, 3, 1)
top_coef = coef_df_nonzero.head(20).sort_values('Coefficient')
colors = ['red' if x < 0 else 'green' for x in top_coef['Coefficient']]
ax1.barh(range(len(top_coef)), top_coef['Coefficient'], color=colors, alpha=0.7)
ax1.set_yticks(range(len(top_coef)))
ax1.set_yticklabels(top_coef['Feature'], fontsize=10)
ax1.set_xlabel('계수', fontweight='bold', fontsize=12)
ax1.set_title('Top 20 변수 계수 (V2)', fontsize=14, fontweight='bold')
ax1.axvline(0, color='black', linewidth=1)
ax1.grid(True, alpha=0.3, axis='x')

# (2) 계수 분포
ax2 = plt.subplot(2, 3, 2)
ax2.hist(coefficients[coefficients != 0], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
ax2.axvline(0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('계수 값', fontweight='bold', fontsize=12)
ax2.set_ylabel('빈도', fontweight='bold', fontsize=12)
ax2.set_title('계수 분포 (Non-zero)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# (3) 양수/음수 계수 비율
ax3 = plt.subplot(2, 3, 3)
coef_stats = [
    np.sum(coefficients > 0),
    np.sum(coefficients < 0),
    np.sum(coefficients == 0)
]
labels_coef = ['양수\n(상승)', '음수\n(하락)', '0\n(제거)']
colors_coef = ['green', 'red', 'gray']
bars = ax3.bar(labels_coef, coef_stats, color=colors_coef, alpha=0.7)
for bar, val in zip(bars, coef_stats):
    ax3.text(bar.get_x() + bar.get_width()/2, val, f'{val}개',
            ha='center', va='bottom', fontweight='bold')
ax3.set_ylabel('변수 개수', fontweight='bold', fontsize=12)
ax3.set_title('계수 부호 분포 (V2)', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# (4) 누적 중요도
ax4 = plt.subplot(2, 3, 4)
ax4.plot(range(1, len(importance_df)+1), importance_df['Cumulative'], linewidth=2, color='purple')
ax4.axhline(0.8, color='red', linestyle='--', linewidth=1, label='80%')
ax4.axvline(n_features_80, color='red', linestyle='--', linewidth=1)
ax4.set_xlabel('변수 개수', fontweight='bold', fontsize=12)
ax4.set_ylabel('누적 중요도', fontweight='bold', fontsize=12)
ax4.set_title('누적 Feature Importance', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.text(n_features_80, 0.85, f'{n_features_80}개', ha='center', fontweight='bold')

# (5) 신규 V2 변수 중요도
ax5 = plt.subplot(2, 3, 5)
new_vars_top = new_vars_shap.head(15)
if len(new_vars_top) > 0:
    ax5.barh(range(len(new_vars_top)), new_vars_top['SHAP_Importance'], color='orange', alpha=0.7)
    ax5.set_yticks(range(len(new_vars_top)))
    ax5.set_yticklabels(new_vars_top['Feature'], fontsize=10)
    ax5.set_xlabel('SHAP 중요도', fontweight='bold', fontsize=12)
    ax5.set_title('🆕 신규 V2 변수 Top 15', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')

# (6) 요약 테이블
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
summary_data = [
    ['지표', '값'],
    ['모델 성능', ''],
    [f'  Train R²', f'{r2_train:.4f}'],
    [f'  Test R²', f'{r2_test:.4f}'],
    [f'  Gap', f'{r2_train - r2_test:.4f}'],
    ['', ''],
    ['변수 선택', ''],
    [f'  총 변수', f'{len(feature_cols)}개'],
    [f'  사용 변수', f'{np.sum(coefficients != 0)}개'],
    [f'  제거 변수', f'{np.sum(coefficients == 0)}개'],
    ['', ''],
    ['V2 신규 변수', ''],
    [f'  추가', f'{len(new_vars_found)}개'],
    [f'  사용', f'{new_vars_used}개'],
    ['', ''],
    ['80% 설명력', f'{n_features_80}개'],
]
table = ax6.table(cellText=summary_data, loc='center', cellLoc='left',
                  colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)
table[(0, 0)].set_facecolor('#4CAF50')
table[(0, 1)].set_facecolor('#4CAF50')
table[(0, 0)].set_text_props(weight='bold', color='white')
table[(0, 1)].set_text_props(weight='bold', color='white')
ax6.set_title('종합 요약', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('elasticnet_xai_v2_part2_coefficients.png', dpi=300, bbox_inches='tight')
print("✅ 저장: elasticnet_xai_v2_part2_coefficients.png")
plt.close()

# ==================== Part 3: SHAP 분석 ====================
print("Part 3: SHAP 분석 시각화...")
fig3 = plt.figure(figsize=(20, 10))

# (1) SHAP Importance Top 20
ax1 = plt.subplot(1, 3, 1)
top_shap = shap_df.head(20)
ax1.barh(range(len(top_shap)), top_shap['SHAP_Importance'], color='coral', alpha=0.7)
ax1.set_yticks(range(len(top_shap)))
ax1.set_yticklabels(top_shap['Feature'], fontsize=10)
ax1.set_xlabel('평균 |SHAP Value|', fontweight='bold', fontsize=12)
ax1.set_title('Top 20 SHAP Importance (V2)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')

# (2) SHAP Summary Plot
ax2 = plt.subplot(1, 3, 2)
plt.sca(ax2)
shap.summary_plot(shap_values_test, X_test_scaled, feature_names=feature_cols,
                  show=False, max_display=20, plot_size=None)
plt.title('SHAP Summary Plot (Top 20, V2)', fontsize=14, fontweight='bold')

# (3) 계수 vs SHAP 비교
ax3 = plt.subplot(1, 3, 3)
comparison_df = coef_df.merge(shap_df, on='Feature')
comparison_df = comparison_df[comparison_df['Abs_Coefficient'] > 0]
ax3.scatter(comparison_df['Abs_Coefficient'], comparison_df['SHAP_Importance'],
            alpha=0.5, s=50, color='purple')
ax3.set_xlabel('계수 절댓값', fontweight='bold', fontsize=12)
ax3.set_ylabel('SHAP 중요도', fontweight='bold', fontsize=12)
ax3.set_title('계수 vs SHAP 중요도', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

corr = comparison_df['Abs_Coefficient'].corr(comparison_df['SHAP_Importance'])
ax3.text(0.05, 0.95, f'상관계수: {corr:.3f}', transform=ax3.transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('elasticnet_xai_v2_part3_shap.png', dpi=300, bbox_inches='tight')
print("✅ 저장: elasticnet_xai_v2_part3_shap.png")
plt.close()

# ========================================
# 8. 결과 저장
# ========================================
# 계수
coef_df_nonzero.to_csv('elasticnet_coefficients_v2.csv', index=False)
print("✅ 저장: elasticnet_coefficients_v2.csv")

# SHAP
shap_df.to_csv('elasticnet_shap_importance_v2.csv', index=False)
print("✅ 저장: elasticnet_shap_importance_v2.csv")

# 비교
comparison_df.to_csv('elasticnet_coef_shap_comparison_v2.csv', index=False)
print("✅ 저장: elasticnet_coef_shap_comparison_v2.csv")

# 신규 변수
new_vars_shap.to_csv('elasticnet_new_vars_importance_v2.csv', index=False)
print("✅ 저장: elasticnet_new_vars_importance_v2.csv")

print("\n" + "="*80)
print("XAI V2 분석 완료!")
print("="*80)

print(f"\n🎯 최종 평가:")
print(f"  과적합: {overfit_status}")
print(f"  V2 변수 기여: {new_vars_used}/{len(new_vars_found)}개 사용")
print(f"  모델 품질: {'✅ 우수' if r2_test > 0.85 and overfit_score < 0.15 else '⚠️ 양호' if r2_test > 0.7 else '❌ 개선 필요'}")

print("\n📊 출력 파일:")
print("  1. elasticnet_xai_v2_part1_overfitting.png - 과적합 검증")
print("  2. elasticnet_xai_v2_part2_coefficients.png - 계수 & Feature Importance")
print("  3. elasticnet_xai_v2_part3_shap.png - SHAP 분석")
