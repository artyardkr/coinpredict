"""
Step 30 V2: Comprehensive Strategy Comparison with V2 Features (138 variables)

70/30 split 기반으로 다양한 전략 비교:
1. Buy-and-Hold
2. Long-Only (예측 상승 시 매수)
3. Long-Short (상승 시 롱, 하락 시 숏)
4. Threshold 1% (1% 이상 예측 시만 거래)
5. Threshold 2% (2% 이상 예측 시만 거래)
6. Direction Prediction (방향만 맞추기)

V2 변경사항:
- integrated_data_full_v2.csv 사용 (138 features)
- 신규 변수 포함: Fed liquidity, ETF flows, advanced on-chain
- 거래비용 없음
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
print("Comprehensive Strategy Comparison V2 (138 Features)")
print("="*80)

# ========================================
# 1. 데이터 로드 (V2)
# ========================================
df = pd.read_csv('integrated_data_full_v2.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"\n데이터: {df.shape}")
print(f"기간: {df['Date'].min().date()} ~ {df['Date'].max().date()}")
print(f"⭐ V2 Features: 138개 (V1 88개 → V2 138개, +50개 신규)")

# ========================================
# 2. 타겟 생성
# ========================================
df['target'] = df['Close'].shift(-1)
df = df[:-1].copy()

# ========================================
# 3. 특성 선택 (step25_v2 방식)
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

print(f"\n특성 수: {len(feature_cols)} (V2: 138 features)")

# 신규 변수 확인
new_vars_keywords = ['DXY', 'ETH', 'TLT', 'GLD', 'WALCL', 'RRPONTSYD', 'FED_NET_LIQUIDITY',
                     'NVT', 'Puell', 'Hash_Ribbon', 'IBIT', 'FBTC', 'GBTC_Premium']
new_vars_found = [col for col in feature_cols if any(kw in col for kw in new_vars_keywords)]
print(f"신규 변수: {len(new_vars_found)}개 감지")

# NaN/Inf 처리
for col in feature_cols:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

X = df[feature_cols].values
y = df['target'].values
dates = df['Date'].values
close_prices = df['Close'].values

# ========================================
# 4. 70/30 Split
# ========================================
print(f"\n{'='*60}")
print("70/30 Split (step25_v2 방식)")
print(f"{'='*60}")

split_idx = int(len(df) * 0.7)
split_date = df['Date'].iloc[split_idx]

X_train = X[:split_idx]
y_train = y[:split_idx]
X_test = X[split_idx:]
y_test = y[split_idx:]
dates_test = dates[split_idx:]
close_test = close_prices[split_idx:]

print(f"\nSplit date: {split_date.date()}")
print(f"Train: {len(X_train)} samples ({df['Date'].iloc[0].date()} ~ {split_date.date()})")
print(f"Test:  {len(X_test)} samples ({df['Date'].iloc[split_idx].date()} ~ {df['Date'].iloc[-1].date()})")

# 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ElasticNet 학습
elasticnet = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=10000)
elasticnet.fit(X_train_scaled, y_train)

# 예측
y_train_pred = elasticnet.predict(X_train_scaled)
y_test_pred = elasticnet.predict(X_test_scaled)

# 성능
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"\nElasticNet 성능 (V2 - 138 features):")
print(f"  Train R²:  {train_r2:.4f}, RMSE: ${train_rmse:,.2f}")
print(f"  Test R²:   {test_r2:.4f}, RMSE: ${test_rmse:,.2f}")

# 방향 정확도
actual_direction = (y_test > close_test).astype(int)
pred_direction = (y_test_pred > close_test).astype(int)
direction_accuracy = (actual_direction == pred_direction).mean()
print(f"  Direction Accuracy: {direction_accuracy:.2%}")

# ========================================
# 5. 백테스팅 전략들
# ========================================

def backtest_strategy(df_slice, strategy_type='long_only', threshold=0.0, initial_capital=10000):
    """
    백테스팅 함수 (거래비용 없음)

    Parameters:
    - df_slice: DataFrame with Date, Close, predicted_price, actual_price
    - strategy_type: 'long_only', 'long_short', 'threshold', 'direction'
    - threshold: 예측 변화율 임계값 (%)
    - initial_capital: 초기 자본
    """

    capital = initial_capital
    position = None  # None, 'long', 'short'
    entry_price = 0

    portfolio_values = []
    returns = []
    trades = []

    for i in range(len(df_slice)):
        current_price = df_slice['Close'].iloc[i]
        predicted_next = df_slice['predicted_price'].iloc[i]

        # 예측 변화율
        predicted_change_pct = (predicted_next - current_price) / current_price * 100

        # 다음날 실제 가격
        if i < len(df_slice) - 1:
            next_actual_price = df_slice['Close'].iloc[i + 1]
        else:
            next_actual_price = current_price

        # Strategy logic
        if strategy_type == 'long_only':
            # 상승 예측 시 매수
            if predicted_change_pct > 0:
                if position != 'long':
                    entry_price = current_price
                    position = 'long'
                    trades.append({'date': df_slice['Date'].iloc[i], 'action': 'buy', 'price': current_price})

                # 보유 중 - 다음날 가격으로 평가
                if i < len(df_slice) - 1:
                    capital = capital * (next_actual_price / entry_price)
                    entry_price = next_actual_price
            else:
                # 하락 예측 시 현금 보유
                if position == 'long':
                    position = None
                    trades.append({'date': df_slice['Date'].iloc[i], 'action': 'sell', 'price': current_price})

        elif strategy_type == 'long_short':
            # 상승 예측 시 롱, 하락 예측 시 숏
            if predicted_change_pct > 0:
                if position != 'long':
                    if position == 'short':
                        # 숏 청산
                        capital = capital * (entry_price / current_price)
                    # 롱 진입
                    entry_price = current_price
                    position = 'long'
                    trades.append({'date': df_slice['Date'].iloc[i], 'action': 'long', 'price': current_price})

                # 롱 보유
                if i < len(df_slice) - 1:
                    capital = capital * (next_actual_price / entry_price)
                    entry_price = next_actual_price
            else:
                if position != 'short':
                    if position == 'long':
                        # 롱 청산 (이미 반영됨)
                        pass
                    # 숏 진입
                    entry_price = current_price
                    position = 'short'
                    trades.append({'date': df_slice['Date'].iloc[i], 'action': 'short', 'price': current_price})

                # 숏 보유
                if i < len(df_slice) - 1:
                    capital = capital * (entry_price / next_actual_price)
                    entry_price = next_actual_price

        elif strategy_type == 'threshold':
            # threshold 이상 변화 예측 시만 거래
            if predicted_change_pct > threshold:
                if position != 'long':
                    entry_price = current_price
                    position = 'long'
                    trades.append({'date': df_slice['Date'].iloc[i], 'action': 'buy', 'price': current_price})

                if i < len(df_slice) - 1:
                    capital = capital * (next_actual_price / entry_price)
                    entry_price = next_actual_price
            elif predicted_change_pct < -threshold:
                if position == 'long':
                    position = None
                    trades.append({'date': df_slice['Date'].iloc[i], 'action': 'sell', 'price': current_price})

        elif strategy_type == 'direction':
            # 방향만 예측 (상승/하락)
            if predicted_change_pct > 0:
                if position != 'long':
                    entry_price = current_price
                    position = 'long'
                    trades.append({'date': df_slice['Date'].iloc[i], 'action': 'buy', 'price': current_price})

                if i < len(df_slice) - 1:
                    capital = capital * (next_actual_price / entry_price)
                    entry_price = next_actual_price
            else:
                if position == 'long':
                    position = None
                    trades.append({'date': df_slice['Date'].iloc[i], 'action': 'sell', 'price': current_price})

        portfolio_values.append(capital)

        # Daily return
        if i > 0:
            daily_return = (capital / portfolio_values[i-1] - 1) * 100
            returns.append(daily_return)
        else:
            returns.append(0)

    # Calculate metrics
    final_value = portfolio_values[-1]
    total_return = (final_value / initial_capital - 1) * 100

    # Annualized return
    days = len(df_slice)
    years = days / 365
    annual_return = (np.power(final_value / initial_capital, 1/years) - 1) * 100 if years > 0 else 0

    # Volatility (annualized)
    returns_array = np.array(returns)
    daily_volatility = np.std(returns_array)
    annual_volatility = daily_volatility * np.sqrt(252)

    # Sharpe ratio (assuming 0% risk-free rate)
    sharpe_ratio = (annual_return / annual_volatility) if annual_volatility > 0 else 0

    # Max drawdown
    cummax = np.maximum.accumulate(portfolio_values)
    drawdowns = (np.array(portfolio_values) - cummax) / cummax * 100
    max_drawdown = np.min(drawdowns)

    # Win rate
    winning_days = np.sum(returns_array > 0)
    win_rate = winning_days / len(returns_array) * 100 if len(returns_array) > 0 else 0

    results = {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'final_value': final_value,
        'win_rate': win_rate,
        'num_trades': len(trades),
        'portfolio_values': portfolio_values,
        'returns': returns,
        'trades': trades
    }

    return results

print(f"\n{'='*60}")
print("백테스팅: 다양한 전략 비교 (거래비용 없음)")
print(f"{'='*60}")

# Test 데이터프레임 준비
test_df = pd.DataFrame({
    'Date': dates_test,
    'Close': close_test,
    'predicted_price': y_test_pred,
    'actual_price': y_test
})

# 전략 정의
strategies = [
    ('Buy-and-Hold', None, None),
    ('Long-Only', 'long_only', 0.0),
    ('Long-Short', 'long_short', 0.0),
    ('Threshold 1%', 'threshold', 1.0),
    ('Threshold 2%', 'threshold', 2.0),
    ('Direction Only', 'direction', 0.0),
]

all_results = []

# Buy-and-Hold 기준선
initial_price = close_test[0]
final_price = close_test[-1]
bnh_return = (final_price / initial_price - 1) * 100
days = len(test_df)
years = days / 365
bnh_annual_return = (np.power(final_price / initial_price, 1/years) - 1) * 100
bnh_returns = pd.Series(close_test).pct_change().fillna(0) * 100
bnh_volatility = np.std(bnh_returns) * np.sqrt(252)
bnh_sharpe = (bnh_annual_return / bnh_volatility) if bnh_volatility > 0 else 0

# Max drawdown for buy-and-hold
bnh_values = (close_test / initial_price * 10000)
bnh_cummax = np.maximum.accumulate(bnh_values)
bnh_drawdowns = (bnh_values - bnh_cummax) / bnh_cummax * 100
bnh_max_dd = np.min(bnh_drawdowns)

all_results.append({
    'strategy': 'Buy-and-Hold',
    'total_return': bnh_return,
    'annual_return': bnh_annual_return,
    'annual_volatility': bnh_volatility,
    'sharpe_ratio': bnh_sharpe,
    'max_drawdown': bnh_max_dd,
    'final_value': final_price / initial_price * 10000,
    'win_rate': (bnh_returns > 0).sum() / len(bnh_returns) * 100,
    'num_trades': 0,
    'portfolio_values': bnh_values.tolist()
})

print(f"\n{'='*60}")
print(f"Buy-and-Hold")
print(f"{'='*60}")
print(f"Total Return: {bnh_return:.2f}%")
print(f"Annual Return: {bnh_annual_return:.2f}%")
print(f"Annual Volatility: {bnh_volatility:.2f}%")
print(f"Sharpe Ratio: {bnh_sharpe:.4f}")
print(f"Max Drawdown: {bnh_max_dd:.2f}%")

# ElasticNet 전략들
for strategy_name, strategy_type, threshold in strategies[1:]:
    print(f"\n{'='*60}")
    print(f"{strategy_name}")
    print(f"{'='*60}")

    results = backtest_strategy(test_df, strategy_type=strategy_type, threshold=threshold)

    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Annual Return: {results['annual_return']:.2f}%")
    print(f"Annual Volatility: {results['annual_volatility']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Number of Trades: {results['num_trades']}")

    all_results.append({
        'strategy': strategy_name,
        'total_return': results['total_return'],
        'annual_return': results['annual_return'],
        'annual_volatility': results['annual_volatility'],
        'sharpe_ratio': results['sharpe_ratio'],
        'max_drawdown': results['max_drawdown'],
        'final_value': results['final_value'],
        'win_rate': results['win_rate'],
        'num_trades': results['num_trades'],
        'portfolio_values': results['portfolio_values']
    })

results_df = pd.DataFrame(all_results)

# ========================================
# 6. 시각화
# ========================================
print(f"\n{'='*60}")
print("시각화 생성 중...")
print(f"{'='*60}")

fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

# (1) 총 수익률 비교
ax1 = fig.add_subplot(gs[0, 0])
colors = ['#95a5a6'] + ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
bars = ax1.barh(range(len(results_df)), results_df['total_return'], color=colors, alpha=0.7)
ax1.set_yticks(range(len(results_df)))
ax1.set_yticklabels(results_df['strategy'], fontsize=9)
ax1.set_xlabel('총 수익률 (%)', fontweight='bold')
ax1.set_title('총 수익률 비교 (V2 - 138 features)', fontweight='bold', fontsize=11)
ax1.axvline(x=0, color='black', linestyle='--', alpha=0.3)
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3, axis='x')

for i, (idx, row) in enumerate(results_df.iterrows()):
    ax1.text(row['total_return'], i, f"  {row['total_return']:.1f}%",
            va='center', fontsize=9, fontweight='bold')

# (2) 샤프 비율 비교
ax2 = fig.add_subplot(gs[0, 1])
bars = ax2.barh(range(len(results_df)), results_df['sharpe_ratio'], color=colors, alpha=0.7)
ax2.set_yticks(range(len(results_df)))
ax2.set_yticklabels(results_df['strategy'], fontsize=9)
ax2.set_xlabel('샤프 비율', fontweight='bold')
ax2.set_title('위험 조정 수익 (Sharpe Ratio)', fontweight='bold', fontsize=11)
ax2.axvline(x=0, color='black', linestyle='--', alpha=0.3)
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3, axis='x')

for i, (idx, row) in enumerate(results_df.iterrows()):
    ax2.text(row['sharpe_ratio'], i, f"  {row['sharpe_ratio']:.3f}",
            va='center', fontsize=9, fontweight='bold')

# (3) Max Drawdown
ax3 = fig.add_subplot(gs[0, 2])
bars = ax3.barh(range(len(results_df)), results_df['max_drawdown'], color=colors, alpha=0.7)
ax3.set_yticks(range(len(results_df)))
ax3.set_yticklabels(results_df['strategy'], fontsize=9)
ax3.set_xlabel('최대 낙폭 (%)', fontweight='bold')
ax3.set_title('최대 낙폭 (Max Drawdown)', fontweight='bold', fontsize=11)
ax3.invert_yaxis()
ax3.grid(True, alpha=0.3, axis='x')

# (4) 포트폴리오 가치 변화
ax4 = fig.add_subplot(gs[1, :])
for i, result in enumerate(all_results):
    ax4.plot(dates_test, result['portfolio_values'],
            label=result['strategy'], linewidth=2, alpha=0.8, color=colors[i])
ax4.set_xlabel('날짜', fontweight='bold', fontsize=11)
ax4.set_ylabel('포트폴리오 가치 ($)', fontweight='bold', fontsize=11)
ax4.set_title('포트폴리오 가치 추이 (V2 - 138 Features)', fontweight='bold', fontsize=13)
ax4.legend(loc='best', fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='x', rotation=45)

# (5) 연간 수익률 vs 변동성
ax5 = fig.add_subplot(gs[2, 0])
for i, row in results_df.iterrows():
    ax5.scatter(row['annual_volatility'], row['annual_return'],
               s=200, alpha=0.7, color=colors[i], label=row['strategy'])
ax5.set_xlabel('연간 변동성 (%)', fontweight='bold')
ax5.set_ylabel('연간 수익률 (%)', fontweight='bold')
ax5.set_title('수익률 vs 위험', fontweight='bold')
ax5.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=8)

# (6) 승률 비교
ax6 = fig.add_subplot(gs[2, 1])
bars = ax6.barh(range(len(results_df)), results_df['win_rate'], color=colors, alpha=0.7)
ax6.set_yticks(range(len(results_df)))
ax6.set_yticklabels(results_df['strategy'], fontsize=9)
ax6.set_xlabel('승률 (%)', fontweight='bold')
ax6.set_title('승률 (수익일 비율)', fontweight='bold')
ax6.axvline(x=50, color='red', linestyle='--', alpha=0.3, label='50%')
ax6.invert_yaxis()
ax6.grid(True, alpha=0.3, axis='x')
ax6.legend()

# (7) 거래 횟수
ax7 = fig.add_subplot(gs[2, 2])
trade_results = results_df[results_df['num_trades'] > 0]
if len(trade_results) > 0:
    bars = ax7.barh(range(len(trade_results)), trade_results['num_trades'],
                   color=colors[1:len(trade_results)+1], alpha=0.7)
    ax7.set_yticks(range(len(trade_results)))
    ax7.set_yticklabels(trade_results['strategy'], fontsize=9)
    ax7.set_xlabel('거래 횟수', fontweight='bold')
    ax7.set_title('거래 빈도', fontweight='bold')
    ax7.invert_yaxis()
    ax7.grid(True, alpha=0.3, axis='x')

# (8) 예측 vs 실제
ax8 = fig.add_subplot(gs[3, 0])
ax8.plot(dates_test, y_test, label='실제 (내일)', linewidth=2, color='black', alpha=0.8)
ax8.plot(dates_test, y_test_pred, label='예측 (내일)', linewidth=2, color='red', alpha=0.6, linestyle='--')
ax8.plot(dates_test, close_test, label='오늘', linewidth=1, color='gray', alpha=0.5, linestyle=':')
ax8.set_xlabel('날짜', fontweight='bold')
ax8.set_ylabel('가격 ($)', fontweight='bold')
ax8.set_title(f'ElasticNet 예측 (R²={test_r2:.4f})', fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)
ax8.tick_params(axis='x', rotation=45)

# (9) 요약 통계
ax9 = fig.add_subplot(gs[3, 1])
ax9.axis('off')
best_sharpe_idx = results_df['sharpe_ratio'].idxmax()
best_return_idx = results_df['total_return'].idxmax()
best_sharpe = results_df.loc[best_sharpe_idx]
best_return = results_df.loc[best_return_idx]

summary_text = f"""
【최고 샤프 비율】
전략: {best_sharpe['strategy']}
샤프: {best_sharpe['sharpe_ratio']:.3f}
수익률: {best_sharpe['total_return']:.2f}%
낙폭: {best_sharpe['max_drawdown']:.2f}%

【최고 수익률】
전략: {best_return['strategy']}
수익률: {best_return['total_return']:.2f}%
샤프: {best_return['sharpe_ratio']:.3f}
낙폭: {best_return['max_drawdown']:.2f}%

【ElasticNet 성능】
Test R²: {test_r2:.4f}
방향 정확도: {direction_accuracy:.1%}
RMSE: ${test_rmse:,.0f}
"""
ax9.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
         verticalalignment='center')
ax9.set_title('요약 통계', fontsize=11, fontweight='bold')

# (10) V2 신규 변수 효과
ax10 = fig.add_subplot(gs[3, 2])
ax10.axis('off')

v2_text = f"""
【V2 신규 변수 (50개)】

• Fed 유동성 (8개)
  - WALCL, RRPONTSYD
  - FED_NET_LIQUIDITY

• Bitcoin ETF (12개)
  - IBIT, FBTC
  - GBTC Premium

• 고급 온체인 (21개)
  - NVT, Puell Multiple
  - Hash Ribbon

• 추가 전통시장 (9개)
  - DXY, ETH, TLT, GLD

총 138개 변수
(V1 88개 → +50개)
"""
ax10.text(0.1, 0.5, v2_text, fontsize=9, family='monospace',
         verticalalignment='center')
ax10.set_title('V2 신규 변수', fontsize=11, fontweight='bold')

plt.savefig('dual_test_comparison_v2.png', dpi=300, bbox_inches='tight')
print(f"\n시각화 저장: dual_test_comparison_v2.png")

# ========================================
# 7. 결과 저장
# ========================================
results_save = results_df.drop(columns=['portfolio_values'])
results_save.to_csv('dual_test_comparison_v2_results.csv', index=False)
print(f"결과 저장: dual_test_comparison_v2_results.csv")

# ========================================
# 8. 최종 결론
# ========================================
print(f"\n{'='*80}")
print("최종 결론 (V2 - 138 Features)")
print(f"{'='*80}")

print(f"""
📊 전략 비교 결과 분석

1. ElasticNet 예측 성능:
   ✅ Test R²: {test_r2:.4f}
   ✅ 방향 정확도: {direction_accuracy:.2%}
   ✅ RMSE: ${test_rmse:,.2f}

2. 최고 전략 (샤프 비율):
   🏆 {best_sharpe['strategy']}
   - 수익률: {best_sharpe['total_return']:.2f}%
   - 샤프: {best_sharpe['sharpe_ratio']:.3f}
   - 최대낙폭: {best_sharpe['max_drawdown']:.2f}%
   - 거래횟수: {best_sharpe['num_trades']}회

3. 최고 전략 (총 수익률):
   🏆 {best_return['strategy']}
   - 수익률: {best_return['total_return']:.2f}%
   - 샤프: {best_return['sharpe_ratio']:.3f}
   - 최대낙폭: {best_return['max_drawdown']:.2f}%
   - 거래횟수: {best_return['num_trades']}회

4. Buy-and-Hold 대비:
   BnH 수익률: {bnh_return:.2f}%
   {'✅ 전략이 BnH 초과!' if best_return['total_return'] > bnh_return else '❌ BnH가 더 나음'}

5. 전략별 비교:
""")

for _, row in results_df.iterrows():
    print(f"   {row['strategy']:20s}: {row['total_return']:+7.2f}% (샤프 {row['sharpe_ratio']:.3f})")

print(f"""
6. V2 신규 변수 효과:
   - 138개 변수 (V1 88개 → +50개)
   - Fed 유동성, ETF 플로우, 고급 온체인 지표
   {'✅ 예측 성능 향상!' if test_r2 > 0.8 else '⚠️ 추가 개선 필요'}

7. 실전 권장:
   {'✅ ' + best_sharpe['strategy'] + ' (위험 조정 수익 최고)' if best_sharpe['sharpe_ratio'] > 0.5 else '⚠️ 추가 개선 필요'}
   {'✅ 거래비용 고려 시 재평가 필요' if best_sharpe['num_trades'] > 100 else ''}
""")

print("="*80)
print("Step 30 V2 Completed!")
print("="*80)
