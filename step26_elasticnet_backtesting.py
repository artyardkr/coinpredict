#!/usr/bin/env python3
"""
Step 26: ElasticNet Backtesting with Performance Metrics

ElasticNet 가격 예측 기반 트레이딩 전략:
1. Long-Only: 예측 상승 시 매수
2. Long-Short: 예측 상승 시 롱, 예측 하락 시 숏
3. Threshold 전략: 일정 % 이상 예측 시만 거래

성과 지표:
- 총 수익률, 연간 수익률
- 샤프 비율 (Sharpe Ratio)
- 변동성 (Volatility)
- 최대 낙폭 (Max Drawdown)
- 승률 (Win Rate)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ========================================
# 1. Load Data & Train ElasticNet
# ========================================
print("=" * 80)
print("ElasticNet Backtesting")
print("=" * 80)

df = pd.read_csv('integrated_data_full.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Create target: next day Close
df['target'] = df['Close'].shift(-1)
df = df[:-1].copy()

print(f"Data: {len(df)} samples, {df['Date'].min()} to {df['Date'].max()}")

# Feature preparation
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

# Train/Test split
split_idx = int(len(df) * 0.7)
split_date = df['Date'].iloc[split_idx]

train_mask = df['Date'] < split_date
test_mask = df['Date'] >= split_date

X_train = df[train_mask][feature_cols].values
X_test = df[test_mask][feature_cols].values
y_train = df[train_mask]['target'].values
y_test = df[test_mask]['target'].values

# Scale features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Train ElasticNet
print("\nTraining ElasticNet...")
elasticnet = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000)
elasticnet.fit(X_train_scaled, y_train)

y_pred_test = elasticnet.predict(X_test_scaled)

print(f"ElasticNet trained successfully")
print(f"Test samples: {len(y_test)}")

# Get test data
test_df = df[test_mask].copy()
test_df['predicted_price'] = y_pred_test
test_df['actual_price'] = y_test

print(f"Test period: {test_df['Date'].min()} to {test_df['Date'].max()}")
print(f"Test days: {len(test_df)}")

# ========================================
# 2. Backtesting Function
# ========================================

def backtest_strategy(df, strategy_type='long_only', threshold=0.0, initial_capital=10000):
    """
    백테스팅 함수

    Parameters:
    - df: 데이터프레임 (Date, Close, predicted_price, actual_price 포함)
    - strategy_type: 'long_only', 'long_short', 'threshold'
    - threshold: 예측 변화율 임계값 (%)
    - initial_capital: 초기 자본

    Returns:
    - results: 성과 지표 딕셔너리
    - portfolio_values: 일별 포트폴리오 가치
    """

    TRANSACTION_COST = 0.0  # 거래 비용 없음
    SHORT_COST = 0.0       # 숏 비용 없음

    capital = initial_capital
    position = None  # None, 'long', 'short'
    entry_price = 0

    portfolio_values = []
    returns = []
    trades = []

    for i in range(len(df)):
        current_price = df['Close'].iloc[i]
        predicted_next = df['predicted_price'].iloc[i]

        # 예측 변화율
        predicted_change_pct = (predicted_next - current_price) / current_price * 100

        # 전날 종가 (실제 진입 가격)
        if i < len(df) - 1:
            next_actual_price = df['Close'].iloc[i + 1]
        else:
            next_actual_price = current_price

        # Strategy logic
        if strategy_type == 'long_only':
            # 상승 예측 시 매수
            if predicted_change_pct > 0:
                if position != 'long':
                    # 매수
                    capital -= capital * TRANSACTION_COST
                    entry_price = current_price
                    position = 'long'
                    trades.append({'date': df['Date'].iloc[i], 'action': 'buy', 'price': current_price})

                # 보유 중 - 다음날 가격으로 평가
                if i < len(df) - 1:
                    capital = capital * (next_actual_price / entry_price)
                    entry_price = next_actual_price
            else:
                # 하락 예측 시 현금 보유
                if position == 'long':
                    # 매도
                    capital -= capital * TRANSACTION_COST
                    position = None
                    trades.append({'date': df['Date'].iloc[i], 'action': 'sell', 'price': current_price})

        elif strategy_type == 'long_short':
            # 상승 예측 시 롱, 하락 예측 시 숏
            if predicted_change_pct > 0:
                if position != 'long':
                    if position == 'short':
                        # 숏 청산
                        capital = capital * (entry_price / current_price)
                        capital -= capital * (TRANSACTION_COST + SHORT_COST)
                    # 롱 진입
                    capital -= capital * TRANSACTION_COST
                    entry_price = current_price
                    position = 'long'
                    trades.append({'date': df['Date'].iloc[i], 'action': 'long', 'price': current_price})

                # 롱 보유
                if i < len(df) - 1:
                    capital = capital * (next_actual_price / entry_price)
                    entry_price = next_actual_price
            else:
                if position != 'short':
                    if position == 'long':
                        # 롱 청산
                        capital -= capital * TRANSACTION_COST
                    # 숏 진입
                    capital -= capital * (TRANSACTION_COST + SHORT_COST)
                    entry_price = current_price
                    position = 'short'
                    trades.append({'date': df['Date'].iloc[i], 'action': 'short', 'price': current_price})

                # 숏 보유
                if i < len(df) - 1:
                    capital = capital * (entry_price / next_actual_price)
                    entry_price = next_actual_price

        elif strategy_type == 'threshold':
            # threshold 이상 변화 예측 시만 거래
            if predicted_change_pct > threshold:
                if position != 'long':
                    capital -= capital * TRANSACTION_COST
                    entry_price = current_price
                    position = 'long'
                    trades.append({'date': df['Date'].iloc[i], 'action': 'buy', 'price': current_price})

                if i < len(df) - 1:
                    capital = capital * (next_actual_price / entry_price)
                    entry_price = next_actual_price
            elif predicted_change_pct < -threshold:
                if position == 'long':
                    capital -= capital * TRANSACTION_COST
                    position = None
                    trades.append({'date': df['Date'].iloc[i], 'action': 'sell', 'price': current_price})

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
    days = len(df)
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

# ========================================
# 3. Run Backtests
# ========================================
print("\n" + "=" * 80)
print("Running Backtests...")
print("=" * 80)

strategies = [
    ('Buy-and-Hold', None, None),
    ('Long-Only (ElasticNet)', 'long_only', 0.0),
    ('Long-Short (ElasticNet)', 'long_short', 0.0),
    ('Threshold 1% (ElasticNet)', 'threshold', 1.0),
    ('Threshold 2% (ElasticNet)', 'threshold', 2.0),
]

all_results = []

# Buy-and-Hold baseline
initial_price = test_df['Close'].iloc[0]
final_price = test_df['Close'].iloc[-1]
bnh_return = (final_price / initial_price - 1) * 100
days = len(test_df)
years = days / 365
bnh_annual_return = (np.power(final_price / initial_price, 1/years) - 1) * 100
bnh_returns = test_df['Close'].pct_change().fillna(0) * 100
bnh_volatility = np.std(bnh_returns) * np.sqrt(252)
bnh_sharpe = (bnh_annual_return / bnh_volatility) if bnh_volatility > 0 else 0

# Max drawdown for buy-and-hold
bnh_values = (test_df['Close'] / initial_price * 10000).values
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

# ElasticNet strategies
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

# ========================================
# 4. Results Summary
# ========================================
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

results_df = pd.DataFrame(all_results)
print("\n" + results_df[['strategy', 'total_return', 'annual_return', 'sharpe_ratio',
                         'max_drawdown', 'win_rate']].to_string(index=False))

# Best strategy
best_sharpe_idx = results_df['sharpe_ratio'].idxmax()
best_return_idx = results_df['total_return'].idxmax()

print(f"\n🏆 Best Sharpe Ratio: {results_df.loc[best_sharpe_idx, 'strategy']} ({results_df.loc[best_sharpe_idx, 'sharpe_ratio']:.4f})")
print(f"🏆 Best Total Return: {results_df.loc[best_return_idx, 'strategy']} ({results_df.loc[best_return_idx, 'total_return']:.2f}%)")

# ========================================
# 5. Visualization
# ========================================
print("\n" + "=" * 80)
print("Creating visualizations...")
print("=" * 80)

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Portfolio Value Over Time
ax1 = fig.add_subplot(gs[0, :])
for result in all_results:
    ax1.plot(test_df['Date'], result['portfolio_values'],
            label=result['strategy'], linewidth=2, alpha=0.8)
ax1.set_xlabel('Date', fontweight='bold', fontsize=11)
ax1.set_ylabel('Portfolio Value ($)', fontweight='bold', fontsize=11)
ax1.set_title('Portfolio Value Over Time', fontweight='bold', fontsize=13)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# 2. Total Return Comparison
ax2 = fig.add_subplot(gs[1, 0])
colors = ['#95a5a6'] + ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
bars = ax2.barh(range(len(results_df)), results_df['total_return'], color=colors, alpha=0.7)
ax2.set_yticks(range(len(results_df)))
ax2.set_yticklabels(results_df['strategy'], fontsize=9)
ax2.set_xlabel('Total Return (%)', fontweight='bold')
ax2.set_title('Total Return Comparison', fontweight='bold')
ax2.axvline(x=0, color='black', linestyle='--', alpha=0.3)
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3, axis='x')

for i, (idx, row) in enumerate(results_df.iterrows()):
    ax2.text(row['total_return'], i, f"  {row['total_return']:.1f}%",
            va='center', fontsize=9, fontweight='bold')

# 3. Sharpe Ratio Comparison
ax3 = fig.add_subplot(gs[1, 1])
bars = ax3.barh(range(len(results_df)), results_df['sharpe_ratio'], color=colors, alpha=0.7)
ax3.set_yticks(range(len(results_df)))
ax3.set_yticklabels(results_df['strategy'], fontsize=9)
ax3.set_xlabel('Sharpe Ratio', fontweight='bold')
ax3.set_title('Risk-Adjusted Return (Sharpe Ratio)', fontweight='bold')
ax3.axvline(x=0, color='black', linestyle='--', alpha=0.3)
ax3.invert_yaxis()
ax3.grid(True, alpha=0.3, axis='x')

for i, (idx, row) in enumerate(results_df.iterrows()):
    ax3.text(row['sharpe_ratio'], i, f"  {row['sharpe_ratio']:.3f}",
            va='center', fontsize=9, fontweight='bold')

# 4. Max Drawdown
ax4 = fig.add_subplot(gs[1, 2])
bars = ax4.barh(range(len(results_df)), results_df['max_drawdown'], color=colors, alpha=0.7)
ax4.set_yticks(range(len(results_df)))
ax4.set_yticklabels(results_df['strategy'], fontsize=9)
ax4.set_xlabel('Max Drawdown (%)', fontweight='bold')
ax4.set_title('Maximum Drawdown', fontweight='bold')
ax4.invert_yaxis()
ax4.grid(True, alpha=0.3, axis='x')

# 5. Annual Return vs Volatility
ax5 = fig.add_subplot(gs[2, 0])
for i, row in results_df.iterrows():
    ax5.scatter(row['annual_volatility'], row['annual_return'],
               s=200, alpha=0.7, color=colors[i], label=row['strategy'])
    ax5.text(row['annual_volatility'], row['annual_return'],
            f"  {row['strategy'].split('(')[0].strip()}", fontsize=8)
ax5.set_xlabel('Annual Volatility (%)', fontweight='bold')
ax5.set_ylabel('Annual Return (%)', fontweight='bold')
ax5.set_title('Return vs Risk', fontweight='bold')
ax5.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax5.grid(True, alpha=0.3)

# 6. Win Rate
ax6 = fig.add_subplot(gs[2, 1])
bars = ax6.barh(range(len(results_df)), results_df['win_rate'], color=colors, alpha=0.7)
ax6.set_yticks(range(len(results_df)))
ax6.set_yticklabels(results_df['strategy'], fontsize=9)
ax6.set_xlabel('Win Rate (%)', fontweight='bold')
ax6.set_title('Win Rate (% of Profitable Days)', fontweight='bold')
ax6.axvline(x=50, color='red', linestyle='--', alpha=0.3, label='50%')
ax6.invert_yaxis()
ax6.grid(True, alpha=0.3, axis='x')
ax6.legend()

# 7. Number of Trades
ax7 = fig.add_subplot(gs[2, 2])
trade_results = results_df[results_df['num_trades'] > 0]
if len(trade_results) > 0:
    bars = ax7.barh(range(len(trade_results)), trade_results['num_trades'],
                   color=colors[1:len(trade_results)+1], alpha=0.7)
    ax7.set_yticks(range(len(trade_results)))
    ax7.set_yticklabels(trade_results['strategy'], fontsize=9)
    ax7.set_xlabel('Number of Trades', fontweight='bold')
    ax7.set_title('Trading Frequency', fontweight='bold')
    ax7.invert_yaxis()
    ax7.grid(True, alpha=0.3, axis='x')

plt.savefig('elasticnet_backtesting_results.png', dpi=300, bbox_inches='tight')
print("Saved: elasticnet_backtesting_results.png")

# ========================================
# 6. Save Results
# ========================================
results_save = results_df.drop(columns=['portfolio_values'])
results_save.to_csv('elasticnet_backtesting_results.csv', index=False)
print("Saved: elasticnet_backtesting_results.csv")

# ========================================
# 7. Final Analysis
# ========================================
print("\n" + "=" * 80)
print("FINAL ANALYSIS")
print("=" * 80)

bnh = results_df[results_df['strategy'] == 'Buy-and-Hold'].iloc[0]
best = results_df.loc[best_sharpe_idx]

print(f"""
📊 ElasticNet 백테스팅 결과 분석

1. Buy-and-Hold (기준선):
   - Total Return: {bnh['total_return']:.2f}%
   - Sharpe Ratio: {bnh['sharpe_ratio']:.4f}
   - Max Drawdown: {bnh['max_drawdown']:.2f}%
   - Volatility: {bnh['annual_volatility']:.2f}%

2. 최고 전략 (Sharpe 기준):
   - Strategy: {best['strategy']}
   - Total Return: {best['total_return']:.2f}% {'✅' if best['total_return'] > bnh['total_return'] else '❌'}
   - Sharpe Ratio: {best['sharpe_ratio']:.4f} {'✅' if best['sharpe_ratio'] > bnh['sharpe_ratio'] else '❌'}
   - Max Drawdown: {best['max_drawdown']:.2f}% {'✅' if best['max_drawdown'] > bnh['max_drawdown'] else '❌'}
   - Volatility: {best['annual_volatility']:.2f}%
   - Win Rate: {best['win_rate']:.2f}%

3. 전략별 비교:
   Long-Only: {results_df[results_df['strategy'].str.contains('Long-Only')]['total_return'].values[0]:.2f}% return
   Long-Short: {results_df[results_df['strategy'].str.contains('Long-Short')]['total_return'].values[0]:.2f}% return
   Threshold 1%: {results_df[results_df['strategy'].str.contains('Threshold 1%')]['total_return'].values[0]:.2f}% return
   Threshold 2%: {results_df[results_df['strategy'].str.contains('Threshold 2%')]['total_return'].values[0]:.2f}% return

4. 결론:
   {'✅ ElasticNet 전략이 Buy-and-Hold보다 우수!' if best['sharpe_ratio'] > bnh['sharpe_ratio'] else '❌ Buy-and-Hold가 더 나음'}
   {'✅ 위험 대비 수익 개선 (Sharpe ↑)' if best['sharpe_ratio'] > bnh['sharpe_ratio'] else ''}
   {'✅ 변동성 감소 (Volatility ↓)' if best['annual_volatility'] < bnh['annual_volatility'] else '⚠️ 변동성 증가'}
   {'✅ 낙폭 감소 (Max DD ↑)' if best['max_drawdown'] > bnh['max_drawdown'] else '⚠️ 낙폭 증가'}

5. 실전 적용 가능성:
   {f"✅ 연간 수익률 {best['annual_return']:.1f}%는 실용적" if best['annual_return'] > 10 else f"⚠️ 연간 수익률 {best['annual_return']:.1f}%는 낮음"}
   {f"✅ Sharpe {best['sharpe_ratio']:.2f}는 우수" if best['sharpe_ratio'] > 1.0 else f"⚠️ Sharpe {best['sharpe_ratio']:.2f}는 보통"}
   거래 횟수: {best['num_trades']}회 (평균 {len(test_df)/best['num_trades']:.1f}일마다 1회)
""")

print("\n" + "=" * 80)
print("Step 26 Completed!")
print("=" * 80)
