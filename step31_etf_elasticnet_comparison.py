"""
Step 31: ETF 도입 전후 ElasticNet 비교

Pre-ETF (2021-2023) vs Post-ETF (2024-2025)
- 모델 성능 비교
- 중요 변수 분석 (Coefficient)
- 예측 vs 실제값 비교
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("ETF 도입 전후 ElasticNet 비교")
print("="*80)

# ========================================
# 1. 데이터 로드
# ========================================
df = pd.read_csv('integrated_data_full.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"\n전체 데이터: {df.shape}")
print(f"기간: {df['Date'].min().date()} ~ {df['Date'].max().date()}")

# ========================================
# 2. 타겟 생성
# ========================================
df['target'] = df['Close'].shift(-1)
df = df[:-1].copy()

# ========================================
# 3. ETF 도입 기준으로 분리
# ========================================
ETF_DATE = pd.to_datetime('2024-01-10')

pre_etf = df[df['Date'] < ETF_DATE].copy()
post_etf = df[df['Date'] >= ETF_DATE].copy()

print(f"\n{'='*60}")
print(f"ETF 도입일: {ETF_DATE.date()}")
print(f"{'='*60}")
print(f"Pre-ETF:  {len(pre_etf)} samples ({pre_etf['Date'].min().date()} ~ {pre_etf['Date'].max().date()})")
print(f"Post-ETF: {len(post_etf)} samples ({post_etf['Date'].min().date()} ~ {post_etf['Date'].max().date()})")

# ========================================
# 4. 특성 선택 (step25/step30 방식)
# ========================================
exclude_cols = [
    'Date', 'Close', 'High', 'Low', 'Open', 'target',
    'cumulative_return',
    'bc_market_price', 'bc_market_cap',
]

# EMA/SMA close 관련 제외
ema_sma_cols = [col for col in df.columns if ('EMA' in col or 'SMA' in col) and 'close' in col.lower()]
exclude_cols.extend(ema_sma_cols)

# BB 제외
bb_cols = [col for col in df.columns if col.startswith('BB_')]
exclude_cols.extend(bb_cols)

exclude_cols = list(set(exclude_cols))
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"\n특성 수: {len(feature_cols)}")

# NaN/Inf 처리
for col in feature_cols:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    pre_etf[col] = pre_etf[col].replace([np.inf, -np.inf], np.nan)
    pre_etf[col] = pre_etf[col].fillna(method='ffill').fillna(method='bfill')
    post_etf[col] = post_etf[col].replace([np.inf, -np.inf], np.nan)
    post_etf[col] = post_etf[col].fillna(method='ffill').fillna(method='bfill')

# ========================================
# 5. Pre-ETF 모델 (70/30 split)
# ========================================
print(f"\n{'='*60}")
print("Pre-ETF 모델 (2021-2023)")
print(f"{'='*60}")

split_idx_pre = int(len(pre_etf) * 0.7)
split_date_pre = pre_etf['Date'].iloc[split_idx_pre]

X_train_pre = pre_etf.iloc[:split_idx_pre][feature_cols].values
y_train_pre = pre_etf.iloc[:split_idx_pre]['target'].values
X_test_pre = pre_etf.iloc[split_idx_pre:][feature_cols].values
y_test_pre = pre_etf.iloc[split_idx_pre:]['target'].values
dates_test_pre = pre_etf.iloc[split_idx_pre:]['Date'].values
close_test_pre = pre_etf.iloc[split_idx_pre:]['Close'].values

print(f"\nTrain: {len(X_train_pre)} samples ({pre_etf['Date'].iloc[0].date()} ~ {split_date_pre.date()})")
print(f"Test:  {len(X_test_pre)} samples ({split_date_pre.date()} ~ {pre_etf['Date'].iloc[-1].date()})")

# 표준화
scaler_pre = StandardScaler()
X_train_pre_scaled = scaler_pre.fit_transform(X_train_pre)
X_test_pre_scaled = scaler_pre.transform(X_test_pre)

# ElasticNet
model_pre = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=10000)
model_pre.fit(X_train_pre_scaled, y_train_pre)

# 예측
y_train_pred_pre = model_pre.predict(X_train_pre_scaled)
y_test_pred_pre = model_pre.predict(X_test_pre_scaled)

# 성능
train_r2_pre = r2_score(y_train_pre, y_train_pred_pre)
test_r2_pre = r2_score(y_test_pre, y_test_pred_pre)
train_rmse_pre = np.sqrt(mean_squared_error(y_train_pre, y_train_pred_pre))
test_rmse_pre = np.sqrt(mean_squared_error(y_test_pre, y_test_pred_pre))
test_mae_pre = mean_absolute_error(y_test_pre, y_test_pred_pre)

# 방향 정확도
actual_dir_pre = (y_test_pre > close_test_pre).astype(int)
pred_dir_pre = (y_test_pred_pre > close_test_pre).astype(int)
dir_acc_pre = (actual_dir_pre == pred_dir_pre).mean()

print(f"\nPre-ETF 성능:")
print(f"  Train R²: {train_r2_pre:.4f}, RMSE: ${train_rmse_pre:,.2f}")
print(f"  Test R²:  {test_r2_pre:.4f}, RMSE: ${test_rmse_pre:,.2f}, MAE: ${test_mae_pre:,.2f}")
print(f"  방향 정확도: {dir_acc_pre:.2%}")

# ========================================
# 6. Post-ETF 모델 (70/30 split)
# ========================================
print(f"\n{'='*60}")
print("Post-ETF 모델 (2024-2025)")
print(f"{'='*60}")

split_idx_post = int(len(post_etf) * 0.7)
split_date_post = post_etf['Date'].iloc[split_idx_post]

X_train_post = post_etf.iloc[:split_idx_post][feature_cols].values
y_train_post = post_etf.iloc[:split_idx_post]['target'].values
X_test_post = post_etf.iloc[split_idx_post:][feature_cols].values
y_test_post = post_etf.iloc[split_idx_post:]['target'].values
dates_test_post = post_etf.iloc[split_idx_post:]['Date'].values
close_test_post = post_etf.iloc[split_idx_post:]['Close'].values

print(f"\nTrain: {len(X_train_post)} samples ({post_etf['Date'].iloc[0].date()} ~ {split_date_post.date()})")
print(f"Test:  {len(X_test_post)} samples ({split_date_post.date()} ~ {post_etf['Date'].iloc[-1].date()})")

# 표준화
scaler_post = StandardScaler()
X_train_post_scaled = scaler_post.fit_transform(X_train_post)
X_test_post_scaled = scaler_post.transform(X_test_post)

# ElasticNet
model_post = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=10000)
model_post.fit(X_train_post_scaled, y_train_post)

# 예측
y_train_pred_post = model_post.predict(X_train_post_scaled)
y_test_pred_post = model_post.predict(X_test_post_scaled)

# 성능
train_r2_post = r2_score(y_train_post, y_train_pred_post)
test_r2_post = r2_score(y_test_post, y_test_pred_post)
train_rmse_post = np.sqrt(mean_squared_error(y_train_post, y_train_pred_post))
test_rmse_post = np.sqrt(mean_squared_error(y_test_post, y_test_pred_post))
test_mae_post = mean_absolute_error(y_test_post, y_test_pred_post)

# 방향 정확도
actual_dir_post = (y_test_post > close_test_post).astype(int)
pred_dir_post = (y_test_pred_post > close_test_post).astype(int)
dir_acc_post = (actual_dir_post == pred_dir_post).mean()

print(f"\nPost-ETF 성능:")
print(f"  Train R²: {train_r2_post:.4f}, RMSE: ${train_rmse_post:,.2f}")
print(f"  Test R²:  {test_r2_post:.4f}, RMSE: ${test_rmse_post:,.2f}, MAE: ${test_mae_post:,.2f}")
print(f"  방향 정확도: {dir_acc_post:.2%}")

# ========================================
# 7. 중요 변수 분석 (Coefficients)
# ========================================
print(f"\n{'='*60}")
print("중요 변수 분석 (ElasticNet Coefficients)")
print(f"{'='*60}")

# Pre-ETF 중요 변수
coef_pre = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model_pre.coef_
})
coef_pre['Abs_Coef'] = np.abs(coef_pre['Coefficient'])
coef_pre = coef_pre[coef_pre['Abs_Coef'] > 0].sort_values('Abs_Coef', ascending=False)

print(f"\nPre-ETF Top 15 중요 변수:")
print(coef_pre.head(15).to_string(index=False))

# Post-ETF 중요 변수
coef_post = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model_post.coef_
})
coef_post['Abs_Coef'] = np.abs(coef_post['Coefficient'])
coef_post = coef_post[coef_post['Abs_Coef'] > 0].sort_values('Abs_Coef', ascending=False)

print(f"\nPost-ETF Top 15 중요 변수:")
print(coef_post.head(15).to_string(index=False))

# 비영 계수 개수
print(f"\n{'='*60}")
print(f"Pre-ETF:  {len(coef_pre)} / {len(feature_cols)} 변수 사용 ({len(coef_pre)/len(feature_cols)*100:.1f}%)")
print(f"Post-ETF: {len(coef_post)} / {len(feature_cols)} 변수 사용 ({len(coef_post)/len(feature_cols)*100:.1f}%)")

# ========================================
# 8. 시각화
# ========================================
fig = plt.figure(figsize=(20, 16))

# (1) 성능 비교 - R²
ax1 = plt.subplot(4, 4, 1)
periods = ['Pre-ETF\n(2021-2023)', 'Post-ETF\n(2024-2025)']
train_r2s = [train_r2_pre, train_r2_post]
bars = ax1.bar(periods, train_r2s, color=['steelblue', 'coral'], alpha=0.7)
for bar, val in zip(bars, train_r2s):
    ax1.text(bar.get_x() + bar.get_width()/2, val, f'{val:.4f}',
            ha='center', va='bottom', fontweight='bold')
ax1.set_ylabel('R²')
ax1.set_title('Train R² 비교', fontsize=12, fontweight='bold')
ax1.set_ylim([0, 1.0])
ax1.grid(True, alpha=0.3, axis='y')

# (2) Test R²
ax2 = plt.subplot(4, 4, 2)
test_r2s = [test_r2_pre, test_r2_post]
colors = ['green' if r > 0 else 'red' for r in test_r2s]
bars = ax2.bar(periods, test_r2s, color=colors, alpha=0.7)
for bar, val in zip(bars, test_r2s):
    ax2.text(bar.get_x() + bar.get_width()/2, val, f'{val:.4f}',
            ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')
ax2.set_ylabel('R²')
ax2.set_title('Test R² 비교', fontsize=12, fontweight='bold')
ax2.axhline(0, color='black', linewidth=0.8)
ax2.grid(True, alpha=0.3, axis='y')

# (3) RMSE 비교
ax3 = plt.subplot(4, 4, 3)
test_rmses = [test_rmse_pre, test_rmse_post]
bars = ax3.barh(periods, test_rmses, color=['steelblue', 'coral'], alpha=0.7)
for bar, val in zip(bars, test_rmses):
    ax3.text(val, bar.get_y() + bar.get_height()/2, f'${val:,.0f}',
            va='center', ha='left', fontweight='bold')
ax3.set_xlabel('RMSE ($)')
ax3.set_title('Test RMSE 비교', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

# (4) 방향 정확도
ax4 = plt.subplot(4, 4, 4)
dir_accs = [dir_acc_pre * 100, dir_acc_post * 100]
bars = ax4.barh(periods, dir_accs, color=['steelblue', 'coral'], alpha=0.7)
for bar, val in zip(bars, dir_accs):
    ax4.text(val, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
            va='center', ha='left', fontweight='bold')
ax4.set_xlabel('방향 정확도 (%)')
ax4.set_title('방향 예측 정확도', fontsize=12, fontweight='bold')
ax4.axvline(50, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax4.grid(True, alpha=0.3, axis='x')

# (5) Pre-ETF 예측 vs 실제
ax5 = plt.subplot(4, 4, 5)
ax5.plot(dates_test_pre, y_test_pre, label='실제', linewidth=2, color='black', alpha=0.8)
ax5.plot(dates_test_pre, y_test_pred_pre, label='예측', linewidth=2, color='red', alpha=0.6, linestyle='--')
ax5.plot(dates_test_pre, close_test_pre, label='오늘 가격', linewidth=1, color='gray', alpha=0.5, linestyle=':')
ax5.set_ylabel('가격 ($)')
ax5.set_title(f'Pre-ETF: 예측 vs 실제 (R²={test_r2_pre:.4f})', fontsize=11, fontweight='bold')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)
ax5.tick_params(axis='x', rotation=45, labelsize=8)

# (6) Post-ETF 예측 vs 실제
ax6 = plt.subplot(4, 4, 6)
ax6.plot(dates_test_post, y_test_post, label='실제', linewidth=2, color='black', alpha=0.8)
ax6.plot(dates_test_post, y_test_pred_post, label='예측', linewidth=2, color='red', alpha=0.6, linestyle='--')
ax6.plot(dates_test_post, close_test_post, label='오늘 가격', linewidth=1, color='gray', alpha=0.5, linestyle=':')
ax6.set_ylabel('가격 ($)')
ax6.set_title(f'Post-ETF: 예측 vs 실제 (R²={test_r2_post:.4f})', fontsize=11, fontweight='bold')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)
ax6.tick_params(axis='x', rotation=45, labelsize=8)

# (7) Pre-ETF Scatter
ax7 = plt.subplot(4, 4, 7)
ax7.scatter(y_test_pre, y_test_pred_pre, alpha=0.5, s=30, color='steelblue')
min_val = min(y_test_pre.min(), y_test_pred_pre.min())
max_val = max(y_test_pre.max(), y_test_pred_pre.max())
ax7.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
ax7.set_xlabel('실제 ($)')
ax7.set_ylabel('예측 ($)')
ax7.set_title('Pre-ETF: Actual vs Predicted', fontsize=11, fontweight='bold')
ax7.grid(True, alpha=0.3)

# (8) Post-ETF Scatter
ax8 = plt.subplot(4, 4, 8)
ax8.scatter(y_test_post, y_test_pred_post, alpha=0.5, s=30, color='coral')
min_val = min(y_test_post.min(), y_test_pred_post.min())
max_val = max(y_test_post.max(), y_test_pred_post.max())
ax8.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
ax8.set_xlabel('실제 ($)')
ax8.set_ylabel('예측 ($)')
ax8.set_title('Post-ETF: Actual vs Predicted', fontsize=11, fontweight='bold')
ax8.grid(True, alpha=0.3)

# (9) Pre-ETF Top 10 중요 변수
ax9 = plt.subplot(4, 4, 9)
top10_pre = coef_pre.head(10)
colors_pre = ['green' if c > 0 else 'red' for c in top10_pre['Coefficient']]
bars = ax9.barh(range(len(top10_pre)), top10_pre['Coefficient'], color=colors_pre, alpha=0.7)
ax9.set_yticks(range(len(top10_pre)))
ax9.set_yticklabels(top10_pre['Feature'], fontsize=8)
ax9.set_xlabel('Coefficient')
ax9.set_title('Pre-ETF: Top 10 중요 변수', fontsize=11, fontweight='bold')
ax9.axvline(0, color='black', linewidth=0.8)
ax9.grid(True, alpha=0.3, axis='x')
ax9.invert_yaxis()

# (10) Post-ETF Top 10 중요 변수
ax10 = plt.subplot(4, 4, 10)
top10_post = coef_post.head(10)
colors_post = ['green' if c > 0 else 'red' for c in top10_post['Coefficient']]
bars = ax10.barh(range(len(top10_post)), top10_post['Coefficient'], color=colors_post, alpha=0.7)
ax10.set_yticks(range(len(top10_post)))
ax10.set_yticklabels(top10_post['Feature'], fontsize=8)
ax10.set_xlabel('Coefficient')
ax10.set_title('Post-ETF: Top 10 중요 변수', fontsize=11, fontweight='bold')
ax10.axvline(0, color='black', linewidth=0.8)
ax10.grid(True, alpha=0.3, axis='x')
ax10.invert_yaxis()

# (11) 예측 오차 분포 - Pre-ETF
ax11 = plt.subplot(4, 4, 11)
errors_pre = y_test_pre - y_test_pred_pre
ax11.hist(errors_pre, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
ax11.axvline(0, color='red', linestyle='--', linewidth=2)
ax11.set_xlabel('예측 오차 ($)')
ax11.set_ylabel('빈도')
ax11.set_title(f'Pre-ETF: 오차 분포 (MAE=${test_mae_pre:,.0f})', fontsize=10, fontweight='bold')
ax11.grid(True, alpha=0.3)
mean_err = errors_pre.mean()
std_err = errors_pre.std()
ax11.text(0.05, 0.95, f'Mean: ${mean_err:.0f}\nStd: ${std_err:.0f}',
         transform=ax11.transAxes, va='top', fontsize=8,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# (12) 예측 오차 분포 - Post-ETF
ax12 = plt.subplot(4, 4, 12)
errors_post = y_test_post - y_test_pred_post
ax12.hist(errors_post, bins=30, color='coral', alpha=0.7, edgecolor='black')
ax12.axvline(0, color='red', linestyle='--', linewidth=2)
ax12.set_xlabel('예측 오차 ($)')
ax12.set_ylabel('빈도')
ax12.set_title(f'Post-ETF: 오차 분포 (MAE=${test_mae_post:,.0f})', fontsize=10, fontweight='bold')
ax12.grid(True, alpha=0.3)
mean_err = errors_post.mean()
std_err = errors_post.std()
ax12.text(0.05, 0.95, f'Mean: ${mean_err:.0f}\nStd: ${std_err:.0f}',
         transform=ax12.transAxes, va='top', fontsize=8,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# (13) 변수 사용 비교
ax13 = plt.subplot(4, 4, 13)
used_vars = [len(coef_pre), len(coef_post)]
bars = ax13.bar(periods, used_vars, color=['steelblue', 'coral'], alpha=0.7)
for bar, val in zip(bars, used_vars):
    percentage = val / len(feature_cols) * 100
    ax13.text(bar.get_x() + bar.get_width()/2, val, f'{val}\n({percentage:.1f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=9)
ax13.set_ylabel('사용된 변수 개수')
ax13.set_title(f'ElasticNet 변수 선택 (전체 {len(feature_cols)}개)', fontsize=11, fontweight='bold')
ax13.grid(True, alpha=0.3, axis='y')

# (14) 공통 Top 변수
ax14 = plt.subplot(4, 4, 14)
ax14.axis('off')
top5_pre_names = set(coef_pre.head(10)['Feature'])
top5_post_names = set(coef_post.head(10)['Feature'])
common = top5_pre_names & top5_post_names
only_pre = top5_pre_names - top5_post_names
only_post = top5_post_names - top5_pre_names

comparison_text = f"""
【공통 Top 10 변수】
{chr(10).join('- ' + f for f in sorted(common)[:5])}
{'...' if len(common) > 5 else ''}

【Pre-ETF만】
{chr(10).join('- ' + f for f in sorted(only_pre)[:3])}
{'...' if len(only_pre) > 3 else ''}

【Post-ETF만】
{chr(10).join('- ' + f for f in sorted(only_post)[:3])}
{'...' if len(only_post) > 3 else ''}
"""
ax14.text(0.1, 0.5, comparison_text, fontsize=8, family='monospace',
         verticalalignment='center')
ax14.set_title('중요 변수 비교', fontsize=11, fontweight='bold')

# (15) 요약
ax15 = plt.subplot(4, 4, 15)
ax15.axis('off')
summary_text = f"""
【Pre-ETF (2021-2023)】
Test R²: {test_r2_pre:.4f}
RMSE: ${test_rmse_pre:,.0f}
방향 정확도: {dir_acc_pre:.2%}
사용 변수: {len(coef_pre)}개

【Post-ETF (2024-2025)】
Test R²: {test_r2_post:.4f}
RMSE: ${test_rmse_post:,.0f}
방향 정확도: {dir_acc_post:.2%}
사용 변수: {len(coef_post)}개

【비교】
R² 차이: {test_r2_post - test_r2_pre:+.4f}
{'✅ Post-ETF가 더 좋음' if test_r2_post > test_r2_pre else '❌ Pre-ETF가 더 좋음'}

RMSE 차이: ${test_rmse_post - test_rmse_pre:+,.0f}
{'✅ Post-ETF 오차 작음' if test_rmse_post < test_rmse_pre else '❌ Post-ETF 오차 큼'}
"""
ax15.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
         verticalalignment='center')
ax15.set_title('종합 요약', fontsize=11, fontweight='bold')

# (16) 전체 기간 비교
ax16 = plt.subplot(4, 4, 16)
# Pre-ETF와 Post-ETF 전체 가격 흐름
all_dates = df['Date'].values
all_close = df['Close'].values
ax16.plot(all_dates, all_close, linewidth=2, color='gray', alpha=0.7)
ax16.axvline(ETF_DATE, color='red', linestyle='--', linewidth=2, label='ETF 도입')
ax16.fill_betweenx([all_close.min(), all_close.max()],
                    all_dates[0], ETF_DATE, alpha=0.1, color='blue', label='Pre-ETF')
ax16.fill_betweenx([all_close.min(), all_close.max()],
                    ETF_DATE, all_dates[-1], alpha=0.1, color='orange', label='Post-ETF')
ax16.set_ylabel('BTC 가격 ($)')
ax16.set_title('전체 기간 BTC 가격', fontsize=11, fontweight='bold')
ax16.legend(fontsize=8)
ax16.grid(True, alpha=0.3)
ax16.tick_params(axis='x', rotation=45, labelsize=8)

plt.tight_layout()
plt.savefig('etf_elasticnet_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n시각화 저장: etf_elasticnet_comparison.png")

# ========================================
# 9. 결과 저장
# ========================================
# 성능 비교
performance = pd.DataFrame({
    'Period': ['Pre-ETF', 'Post-ETF'],
    'Train R²': [train_r2_pre, train_r2_post],
    'Test R²': [test_r2_pre, test_r2_post],
    'Test RMSE': [test_rmse_pre, test_rmse_post],
    'Test MAE': [test_mae_pre, test_mae_post],
    'Direction Accuracy': [dir_acc_pre, dir_acc_post],
    'Vars Used': [len(coef_pre), len(coef_post)]
})
performance.to_csv('etf_elasticnet_performance.csv', index=False)

# 중요 변수
coef_pre.to_csv('etf_pre_important_features.csv', index=False)
coef_post.to_csv('etf_post_important_features.csv', index=False)

print(f"\n결과 저장:")
print(f"  - etf_elasticnet_performance.csv")
print(f"  - etf_pre_important_features.csv")
print(f"  - etf_post_important_features.csv")

# ========================================
# 10. 최종 결론
# ========================================
print(f"\n{'='*80}")
print("최종 결론")
print(f"{'='*80}")

print(f"""
1. 성능 비교:
   Pre-ETF  (2021-2023): Test R² {test_r2_pre:.4f}, RMSE ${test_rmse_pre:,.0f}
   Post-ETF (2024-2025): Test R² {test_r2_post:.4f}, RMSE ${test_rmse_post:,.0f}

   {'✅ Post-ETF 성능이 더 좋음!' if test_r2_post > test_r2_pre else '❌ Pre-ETF 성능이 더 좋음'}
   R² 개선: {(test_r2_post - test_r2_pre):+.4f}

2. 방향 예측:
   Pre-ETF:  {dir_acc_pre:.2%}
   Post-ETF: {dir_acc_post:.2%}
   {'✅ Post-ETF가 방향 예측도 더 정확' if dir_acc_post > dir_acc_pre else '❌ Pre-ETF가 방향 예측 더 정확'}

3. 변수 사용:
   Pre-ETF:  {len(coef_pre)} / {len(feature_cols)} 변수 ({len(coef_pre)/len(feature_cols)*100:.1f}%)
   Post-ETF: {len(coef_post)} / {len(feature_cols)} 변수 ({len(coef_post)/len(feature_cols)*100:.1f}%)

4. 중요 변수 (Top 5):
   Pre-ETF:
{chr(10).join('     ' + str(i+1) + '. ' + row['Feature'] + f' ({row["Coefficient"]:.4f})' for i, (_, row) in enumerate(coef_pre.head(5).iterrows()))}

   Post-ETF:
{chr(10).join('     ' + str(i+1) + '. ' + row['Feature'] + f' ({row["Coefficient"]:.4f})' for i, (_, row) in enumerate(coef_post.head(5).iterrows()))}

5. 핵심 발견:
   - ETF 도입 이후 {'모델 예측력 향상' if test_r2_post > test_r2_pre else '모델 예측력 감소'}
   - {'시장 효율성 증가 → 더 예측 가능' if test_r2_post > test_r2_pre else '시장 변동성 증가 → 예측 어려움'}
   - 중요 변수 변화: {len(common)} 개 공통, {len(only_pre)} 개 Pre만, {len(only_post)} 개 Post만

6. 실전 활용:
   {'✅ Post-ETF 모델이 더 신뢰할 수 있음' if test_r2_post > test_r2_pre and dir_acc_post > dir_acc_pre else '⚠️ 기간별로 모델 재학습 필요'}
   {'✅ 방향 예측 정확도 높음 (백테스팅 권장)' if min(dir_acc_pre, dir_acc_post) > 0.55 else '⚠️ 방향 예측 정확도 낮음 (신중히 사용)'}
""")

print("="*80)
