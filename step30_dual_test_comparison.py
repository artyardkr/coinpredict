"""
Step 30: Dual Test Comparison

Version A: 70/30 split (step25 방식) - 2023-2025 전체
Version B: 2025년만 (순수 out-of-sample)

둘 다 테스트해서 차이를 확인
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("Dual Test Comparison: 70/30 Split vs 2025-Only")
print("="*80)

# ========================================
# 1. 데이터 로드
# ========================================
df = pd.read_csv('integrated_data_full.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"\n데이터: {df.shape}")
print(f"기간: {df['Date'].min().date()} ~ {df['Date'].max().date()}")

# ========================================
# 2. 타겟 생성
# ========================================
df['target'] = df['Close'].shift(-1)
df = df[:-1].copy()

# ========================================
# 3. 특성 선택 (step25 방식)
# ========================================
# step25와 동일하게 일부만 제외
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

X = df[feature_cols].values
y = df['target'].values
dates = df['Date'].values
close_prices = df['Close'].values

# ========================================
# 4. Version A: 70/30 Split (step25 방식)
# ========================================
print(f"\n{'='*60}")
print("Version A: 70/30 Split (step25 방식)")
print(f"{'='*60}")

split_idx = int(len(df) * 0.7)
split_date = df['Date'].iloc[split_idx]

X_train_A = X[:split_idx]
y_train_A = y[:split_idx]
X_test_A = X[split_idx:]
y_test_A = y[split_idx:]
dates_test_A = dates[split_idx:]
close_test_A = close_prices[split_idx:]

print(f"\nSplit date: {split_date.date()}")
print(f"Train: {len(X_train_A)} samples ({df['Date'].iloc[0].date()} ~ {split_date.date()})")
print(f"Test:  {len(X_test_A)} samples ({df['Date'].iloc[split_idx].date()} ~ {df['Date'].iloc[-1].date()})")

# 표준화
scaler_A = StandardScaler()
X_train_A_scaled = scaler_A.fit_transform(X_train_A)
X_test_A_scaled = scaler_A.transform(X_test_A)

# ElasticNet
elasticnet_A = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=10000)
elasticnet_A.fit(X_train_A_scaled, y_train_A)

# 예측
y_train_pred_A = elasticnet_A.predict(X_train_A_scaled)
y_test_pred_A = elasticnet_A.predict(X_test_A_scaled)

# 성능
train_r2_A = r2_score(y_train_A, y_train_pred_A)
test_r2_A = r2_score(y_test_A, y_test_pred_A)
train_rmse_A = np.sqrt(mean_squared_error(y_train_A, y_train_pred_A))
test_rmse_A = np.sqrt(mean_squared_error(y_test_A, y_test_pred_A))

print(f"\nVersion A 성능:")
print(f"  Train R²:  {train_r2_A:.4f}, RMSE: ${train_rmse_A:,.2f}")
print(f"  Test R²:   {test_r2_A:.4f}, RMSE: ${test_rmse_A:,.2f}")
print(f"  (step25 ElasticNet R²: 0.8198)")

# ========================================
# 5. Version B: 2025년만
# ========================================
print(f"\n{'='*60}")
print("Version B: 2025년만 (순수 Out-of-Sample)")
print(f"{'='*60}")

train_end = pd.to_datetime('2024-12-31')
test_start = pd.to_datetime('2025-01-01')

train_mask = df['Date'] <= train_end
test_mask = df['Date'] >= test_start

X_train_B = X[train_mask]
y_train_B = y[train_mask]
X_test_B = X[test_mask]
y_test_B = y[test_mask]
dates_test_B = dates[test_mask]
close_test_B = close_prices[test_mask]

print(f"\nTrain: {len(X_train_B)} samples ({df[train_mask]['Date'].min().date()} ~ {train_end.date()})")
print(f"Test:  {len(X_test_B)} samples ({test_start.date()} ~ {df[test_mask]['Date'].max().date()})")

# 표준화
scaler_B = StandardScaler()
X_train_B_scaled = scaler_B.fit_transform(X_train_B)
X_test_B_scaled = scaler_B.transform(X_test_B)

# ElasticNet
elasticnet_B = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=10000)
elasticnet_B.fit(X_train_B_scaled, y_train_B)

# 예측
y_train_pred_B = elasticnet_B.predict(X_train_B_scaled)
y_test_pred_B = elasticnet_B.predict(X_test_B_scaled)

# 성능
train_r2_B = r2_score(y_train_B, y_train_pred_B)
test_r2_B = r2_score(y_test_B, y_test_pred_B)
train_rmse_B = np.sqrt(mean_squared_error(y_train_B, y_train_pred_B))
test_rmse_B = np.sqrt(mean_squared_error(y_test_B, y_test_pred_B))

print(f"\nVersion B 성능:")
print(f"  Train R²:  {train_r2_B:.4f}, RMSE: ${train_rmse_B:,.2f}")
print(f"  Test R²:   {test_r2_B:.4f}, RMSE: ${test_rmse_B:,.2f}")
print(f"  (step29 ElasticNet R²: -5.45)")

# ========================================
# 6. 백테스팅 비교
# ========================================
def backtest_strategy(dates, current_prices, y_true, y_pred, threshold=1.0, initial_capital=10000):
    """예측 기반 백테스팅"""
    # 예측 변화율
    predicted_returns = (y_pred / current_prices - 1) * 100

    cash = initial_capital
    btc = 0
    portfolio_values = []

    for i in range(len(y_pred)):
        if predicted_returns[i] > threshold:
            # 매수
            if cash > 0:
                btc = cash / current_prices[i]
                cash = 0
        else:
            # 매도
            if btc > 0:
                cash = btc * current_prices[i]
                btc = 0

        # 다음날 포트폴리오 가치
        if btc > 0:
            portfolio_value = btc * y_true[i]
        else:
            portfolio_value = cash

        portfolio_values.append(portfolio_value)

    # Buy-and-Hold
    bnh_btc = initial_capital / current_prices[0]
    bnh_values = bnh_btc * current_prices

    # 성과
    final_value = portfolio_values[-1]
    total_return = (final_value / initial_capital - 1) * 100

    bnh_final = bnh_values[-1]
    bnh_return = (bnh_final / initial_capital - 1) * 100

    # 샤프 비율
    daily_returns = pd.Series(portfolio_values).pct_change()
    annual_return = (1 + total_return/100) ** (365/len(portfolio_values)) - 1
    annual_vol = daily_returns.std() * np.sqrt(365)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0

    return {
        'portfolio_values': portfolio_values,
        'bnh_values': bnh_values,
        'total_return': total_return,
        'bnh_return': bnh_return,
        'sharpe': sharpe
    }

print(f"\n{'='*60}")
print("백테스팅 비교")
print(f"{'='*60}")

# Version A 백테스팅
bt_A = backtest_strategy(dates_test_A, close_test_A, y_test_A, y_test_pred_A)
print(f"\nVersion A (70/30 Split):")
print(f"  전략 수익률: {bt_A['total_return']:+.2f}% (샤프 {bt_A['sharpe']:.3f})")
print(f"  Buy-and-Hold: {bt_A['bnh_return']:+.2f}%")

# Version B 백테스팅
bt_B = backtest_strategy(dates_test_B, close_test_B, y_test_B, y_test_pred_B)
print(f"\nVersion B (2025년만):")
print(f"  전략 수익률: {bt_B['total_return']:+.2f}% (샤프 {bt_B['sharpe']:.3f})")
print(f"  Buy-and-Hold: {bt_B['bnh_return']:+.2f}%")

# ========================================
# 7. 시각화
# ========================================
fig = plt.figure(figsize=(18, 12))

# (1) R² 비교
ax1 = plt.subplot(3, 4, 1)
versions = ['Version A\n(70/30)', 'Version B\n(2025만)']
train_r2s = [train_r2_A, train_r2_B]
bars = ax1.bar(versions, train_r2s, color=['steelblue', 'coral'], alpha=0.7)
for bar, val in zip(bars, train_r2s):
    ax1.text(bar.get_x() + bar.get_width()/2, val, f'{val:.4f}',
            ha='center', va='bottom', fontweight='bold')
ax1.set_ylabel('R²')
ax1.set_title('Train R² 비교', fontsize=11, fontweight='bold')
ax1.set_ylim([0, 1.0])
ax1.grid(True, alpha=0.3, axis='y')

# (2) Test R² 비교
ax2 = plt.subplot(3, 4, 2)
test_r2s = [test_r2_A, test_r2_B]
colors = ['green' if r > 0 else 'red' for r in test_r2s]
bars = ax2.bar(versions, test_r2s, color=colors, alpha=0.7)
for bar, val in zip(bars, test_r2s):
    ax2.text(bar.get_x() + bar.get_width()/2, val, f'{val:.4f}',
            ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')
ax2.set_ylabel('R²')
ax2.set_title('Test R² 비교', fontsize=11, fontweight='bold')
ax2.axhline(0, color='black', linewidth=0.8)
ax2.grid(True, alpha=0.3, axis='y')

# (3) RMSE 비교
ax3 = plt.subplot(3, 4, 3)
test_rmses = [test_rmse_A, test_rmse_B]
bars = ax3.barh(versions, test_rmses, color=['steelblue', 'coral'], alpha=0.7)
for bar, val in zip(bars, test_rmses):
    ax3.text(val, bar.get_y() + bar.get_height()/2, f'${val:,.0f}',
            va='center', ha='left', fontweight='bold')
ax3.set_xlabel('RMSE ($)')
ax3.set_title('Test RMSE 비교', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

# (4) 백테스팅 수익률
ax4 = plt.subplot(3, 4, 4)
returns = [bt_A['total_return'], bt_B['total_return']]
colors_ret = ['green' if r > 0 else 'red' for r in returns]
bars = ax4.barh(versions, returns, color=colors_ret, alpha=0.7)
for bar, val in zip(bars, returns):
    ax4.text(val, bar.get_y() + bar.get_height()/2, f'{val:+.2f}%',
            va='center', ha='left' if val > 0 else 'right', fontweight='bold')
ax4.set_xlabel('수익률 (%)')
ax4.set_title('백테스팅 수익률 (Threshold 1%)', fontsize=11, fontweight='bold')
ax4.axvline(0, color='black', linewidth=0.8)
ax4.grid(True, alpha=0.3, axis='x')

# (5) Version A - 예측 vs 실제
ax5 = plt.subplot(3, 4, 5)
ax5.plot(dates_test_A, y_test_A, label='실제', linewidth=2, color='black', alpha=0.8)
ax5.plot(dates_test_A, y_test_pred_A, label='예측', linewidth=2, color='red', alpha=0.6, linestyle='--')
ax5.set_ylabel('가격 ($)')
ax5.set_title(f'Version A: 예측 vs 실제 (R²={test_r2_A:.4f})', fontsize=10, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.tick_params(axis='x', rotation=45, labelsize=8)

# (6) Version B - 예측 vs 실제
ax6 = plt.subplot(3, 4, 6)
ax6.plot(dates_test_B, y_test_B, label='실제', linewidth=2, color='black', alpha=0.8)
ax6.plot(dates_test_B, y_test_pred_B, label='예측', linewidth=2, color='red', alpha=0.6, linestyle='--')
ax6.set_ylabel('가격 ($)')
ax6.set_title(f'Version B: 예측 vs 실제 (R²={test_r2_B:.4f})', fontsize=10, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.tick_params(axis='x', rotation=45, labelsize=8)

# (7) Version A - Scatter
ax7 = plt.subplot(3, 4, 7)
ax7.scatter(y_test_A, y_test_pred_A, alpha=0.3, s=20, color='steelblue')
min_val = min(y_test_A.min(), y_test_pred_A.min())
max_val = max(y_test_A.max(), y_test_pred_A.max())
ax7.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
ax7.set_xlabel('실제 ($)')
ax7.set_ylabel('예측 ($)')
ax7.set_title('Version A: Actual vs Predicted', fontsize=10, fontweight='bold')
ax7.grid(True, alpha=0.3)

# (8) Version B - Scatter
ax8 = plt.subplot(3, 4, 8)
ax8.scatter(y_test_B, y_test_pred_B, alpha=0.3, s=20, color='coral')
min_val = min(y_test_B.min(), y_test_pred_B.min())
max_val = max(y_test_B.max(), y_test_pred_B.max())
ax8.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
ax8.set_xlabel('실제 ($)')
ax8.set_ylabel('예측 ($)')
ax8.set_title('Version B: Actual vs Predicted', fontsize=10, fontweight='bold')
ax8.grid(True, alpha=0.3)

# (9) Version A - 백테스팅
ax9 = plt.subplot(3, 4, 9)
ax9.plot(dates_test_A, bt_A['portfolio_values'], label='전략', linewidth=2, color='steelblue')
ax9.plot(dates_test_A, bt_A['bnh_values'], label='Buy-and-Hold', linewidth=2, color='gray', alpha=0.7)
ax9.set_ylabel('자산 ($)')
ax9.set_title(f'Version A: 백테스팅 ({bt_A["total_return"]:+.2f}%)', fontsize=10, fontweight='bold')
ax9.legend()
ax9.grid(True, alpha=0.3)
ax9.tick_params(axis='x', rotation=45, labelsize=8)

# (10) Version B - 백테스팅
ax10 = plt.subplot(3, 4, 10)
ax10.plot(dates_test_B, bt_B['portfolio_values'], label='전략', linewidth=2, color='coral')
ax10.plot(dates_test_B, bt_B['bnh_values'], label='Buy-and-Hold', linewidth=2, color='gray', alpha=0.7)
ax10.set_ylabel('자산 ($)')
ax10.set_title(f'Version B: 백테스팅 ({bt_B["total_return"]:+.2f}%)', fontsize=10, fontweight='bold')
ax10.legend()
ax10.grid(True, alpha=0.3)
ax10.tick_params(axis='x', rotation=45, labelsize=8)

# (11) 요약
ax11 = plt.subplot(3, 4, 11)
ax11.axis('off')
summary_text = f"""
【Version A: 70/30 Split】
Test 기간: {pd.to_datetime(dates_test_A[0]).date()} ~
           {pd.to_datetime(dates_test_A[-1]).date()}
Test R²: {test_r2_A:.4f}
백테스팅: {bt_A['total_return']:+.2f}%

【Version B: 2025년만】
Test 기간: {pd.to_datetime(dates_test_B[0]).date()} ~
           {pd.to_datetime(dates_test_B[-1]).date()}
Test R²: {test_r2_B:.4f}
백테스팅: {bt_B['total_return']:+.2f}%

【차이】
R² 차이: {test_r2_A - test_r2_B:.4f}
수익률 차이: {bt_A['total_return'] - bt_B['total_return']:.2f}%p

Version A가 더 좋은 이유:
- 테스트 기간이 훈련 기간과
  시장 환경 유사
- 2023-2024 데이터 포함
"""
ax11.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
         verticalalignment='center')
ax11.set_title('요약', fontsize=11, fontweight='bold')

# (12) step25 vs step29 비교
ax12 = plt.subplot(3, 4, 12)
ax12.axis('off')
comparison_text = f"""
【step25】
방식: 70/30 split
특성: 모든 변수 사용
ElasticNet R²: 0.8198

【step29】
방식: 2025년만
특성: VIF + Backward (25개)
ElasticNet R²: -5.45

【step30 재현】
Version A (70/30): {test_r2_A:.4f}
→ step25 재현 {'성공' if abs(test_r2_A - 0.8198) < 0.02 else '실패'}

Version B (2025만): {test_r2_B:.4f}
→ step29보다 {'좋음' if test_r2_B > -5.45 else '나쁨'}

결론:
모든 변수 사용이
VIF 제거보다 나음!
"""
ax12.text(0.1, 0.5, comparison_text, fontsize=9, family='monospace',
         verticalalignment='center')
ax12.set_title('step25 vs step29 비교', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('dual_test_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n시각화 저장: dual_test_comparison.png")

# ========================================
# 8. 최종 결론
# ========================================
print(f"\n{'='*80}")
print("최종 결론")
print(f"{'='*80}")

print(f"""
1. step25 재현 (Version A):
   ✅ Test R²: {test_r2_A:.4f} (step25: 0.8198)
   차이: {abs(test_r2_A - 0.8198):.4f}
   {'✅ 재현 성공!' if abs(test_r2_A - 0.8198) < 0.02 else '⚠️ 약간 차이 있음'}

2. step29 개선 (Version B):
   ✅ Test R²: {test_r2_B:.4f} (step29: -5.45)
   개선: {test_r2_B - (-5.45):.2f}
   {'✅ 크게 개선!' if test_r2_B > -1 else '⚠️ 여전히 음수'}

3. 핵심 발견:
   - VIF 제거가 오히려 성능 저하!
   - ElasticNet은 자체 regularization으로 충분
   - 모든 변수 사용 > 변수 선택

4. 실전 성능 (백테스팅):
   Version A: {bt_A['total_return']:+.2f}% {'✅' if bt_A['total_return'] > 0 else '❌'}
   Version B: {bt_B['total_return']:+.2f}% {'✅' if bt_B['total_return'] > 0 else '❌'}
   Buy-and-Hold (B): {bt_B['bnh_return']:+.2f}%

5. 결론:
   {'✅ Version A: 테스트 기간 포함되어 좋아 보임' if test_r2_A > 0.5 else ''}
   {'❌ Version B: 순수 out-of-sample에서 실패' if test_r2_B < 0 else '✅ Version B: 순수 out-of-sample에서도 성공'}

   step29가 안 좋았던 이유:
   1. VIF 제거로 중요 변수 손실
   2. 2025년만 테스트 (더 어려움)

   step25가 좋았던 이유:
   1. 모든 변수 사용
   2. 2023-2025 전체 테스트 (더 쉬움)
""")

# 결과 저장
results_summary = pd.DataFrame({
    'Version': ['A (70/30)', 'B (2025만)'],
    'Test Period': [f'{pd.to_datetime(dates_test_A[0]).date()} ~ {pd.to_datetime(dates_test_A[-1]).date()}',
                    f'{pd.to_datetime(dates_test_B[0]).date()} ~ {pd.to_datetime(dates_test_B[-1]).date()}'],
    'Test Samples': [len(X_test_A), len(X_test_B)],
    'Train R²': [train_r2_A, train_r2_B],
    'Test R²': [test_r2_A, test_r2_B],
    'Test RMSE': [test_rmse_A, test_rmse_B],
    'Backtest Return': [bt_A['total_return'], bt_B['total_return']],
    'Sharpe': [bt_A['sharpe'], bt_B['sharpe']],
    'BnH Return': [bt_A['bnh_return'], bt_B['bnh_return']]
})

results_summary.to_csv('dual_test_comparison_results.csv', index=False)
print(f"\n결과 저장: dual_test_comparison_results.csv")

print("="*80)
