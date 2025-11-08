#!/usr/bin/env python3
"""
Step 27: ElasticNet Backtesting - 2025 Only

2025년도 데이터만으로 백테스팅:
- 기간: 2025-01-01 ~ 2025-10-13
- ETF 이후 최신 시장 성과 측정
- 같은 ElasticNet 모델 사용 (2021-2024 학습)
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
print("ElasticNet Backtesting - 2025 Only")
print("=" * 80)

df = pd.read_csv('integrated_data_full.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Create target: next day Close
df['target'] = df['Close'].shift(-1)
df = df[:-1].copy()

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

# Train on 2021-2024 data, Test on 2025 data
train_end = pd.to_datetime('2024-12-31')
test_start = pd.to_datetime('2025-01-01')

train_mask = df['Date'] <= train_end
test_mask = df['Date'] >= test_start

X_train = df[train_mask][feature_cols].values
X_test = df[test_mask][feature_cols].values
y_train = df[train_mask]['target'].values
y_test = df[test_mask]['target'].values

print(f"\nTrain period: {df[train_mask]['Date'].min()} to {df[train_mask]['Date'].max()}")
print(f"Train samples: {len(X_train)}")

print(f"\nTest period (2025 only): {df[test_mask]['Date'].min()} to {df[test_mask]['Date'].max()}")
print(f"Test samples: {len(X_test)}")
print(f"Test days: {len(X_test)}")

# Scale features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Train ElasticNet
print("\nTraining ElasticNet on 2021-2024 data...")
elasticnet = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000)
elasticnet.fit(X_train_scaled, y_train)

y_pred_test = elasticnet.predict(X_test_scaled)

# Get test data
test_df = df[test_mask].copy()
test_df['predicted_price'] = y_pred_test
test_df['actual_price'] = y_test

print(f"\n2025 Price range:")
print(f"  Start: ${test_df['Close'].iloc[0]:.2f} (2025-01-01)")
print(f"  End: ${test_df['Close'].iloc[-1]:.2f} ({test_df['Date'].iloc[-1].date()})")
print(f"  Min: ${test_df['Close'].min():.2f}")
print(f"  Max: ${test_df['Close'].max():.2f}")

# ========================================
# 2. Backtesting Function
# ========================================

def backtest_strategy(df, strategy_type='long_only', threshold=0.0, initial_capital=10000):
    """백테스팅 함수"""

    TRANSACTION_COST = 0.01  # 1%
    SHORT_COST = 0.005       # 0.5%

    capital = initial_capital
    position = None
    entry_price = 0

    portfolio_values = []
    returns = []
    trades = []

    for i in range(len(df)):
        current_price = df['Close'].iloc[i]
        predicted_next = df['predicted_price'].iloc[i]
        predicted_change_pct = (predicted_next - current_price) / current_price * 100

        if i < len(df) - 1:
            next_actual_price = df['Close'].iloc[i + 1]
        else:
            next_actual_price = current_price

        # Strategy logic
        if strategy_type == 'long_only':
            if predicted_change_pct > 0:
                if position != 'long':
                    capital -= capital * TRANSACTION_COST
                    entry_price = current_price
                    position = 'long'
                    trades.append({'date': df['Date'].iloc[i], 'action': 'buy', 'price': current_price})

                if i < len(df) - 1:
                    capital = capital * (next_actual_price / entry_price)
                    entry_price = next_actual_price
            else:
                if position == 'long':
                    capital -= capital * TRANSACTION_COST
                    position = None
                    trades.append({'date': df['Date'].iloc[i], 'action': 'sell', 'price': current_price})

        elif strategy_type == 'long_short':
            if predicted_change_pct > 0:
                if position != 'long':
                    if position == 'short':
                        capital = capital * (entry_price / current_price)
                        capital -= capital * (TRANSACTION_COST + SHORT_COST)
                    capital -= capital * TRANSACTION_COST
                    entry_price = current_price
                    position = 'long'
                    trades.append({'date': df['Date'].iloc[i], 'action': 'long', 'price': current_price})

                if i < len(df) - 1:
                    capital = capital * (next_actual_price / entry_price)
                    entry_price = next_actual_price
            else:
                if position != 'short':
                    if position == 'long':
                        capital -= capital * TRANSACTION_COST
                    capital -= capital * (TRANSACTION_COST + SHORT_COST)
                    entry_price = current_price
                    position = 'short'
                    trades.append({'date': df['Date'].iloc[i], 'action': 'short', 'price': current_price})

                if i < len(df) - 1:
                    capital = capital * (entry_price / next_actual_price)
                    entry_price = next_actual_price

        elif strategy_type == 'threshold':
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

        if i > 0:
            daily_return = (capital / portfolio_values[i-1] - 1) * 100
            returns.append(daily_return)
        else:
            returns.append(0)

    # Calculate metrics
    final_value = portfolio_values[-1]
    total_return = (final_value / initial_capital - 1) * 100

    days = len(df)
    years = days / 365
    annual_return = (np.power(final_value / initial_capital, 1/years) - 1) * 100 if years > 0 else 0

    returns_array = np.array(returns)
    daily_volatility = np.std(returns_array)
    annual_volatility = daily_volatility * np.sqrt(252)

    sharpe_ratio = (annual_return / annual_volatility) if annual_volatility > 0 else 0

    cummax = np.maximum.accumulate(portfolio_values)
    drawdowns = (np.array(portfolio_values) - cummax) / cummax * 100
    max_drawdown = np.min(drawdowns)

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
# 3. Run Backtests (2025 Only)
# ========================================
print("\n" + "=" * 80)
print("Running Backtests on 2025 Data...")
print("=" * 80)

strategies = [
    ('Buy-and-Hold', None, None),
    ('Long-Only (ElasticNet)', 'long_only', 0.0),
    ('Long-Short (ElasticNet)', 'long_short', 0.0),
    ('Threshold 0.5%', 'threshold', 0.5),
    ('Threshold 1%', 'threshold', 1.0),
    ('Threshold 2%', 'threshold', 2.0),
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
print(f"Buy-and-Hold (2025)")
print(f"{'='*60}")
print(f"Total Return: {bnh_return:.2f}%")
print(f"Annual Return: {bnh_annual_return:.2f}%")
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
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Trades: {results['num_trades']}")

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
print("2025 BACKTESTING RESULTS SUMMARY")
print("=" * 80)

results_df = pd.DataFrame(all_results)
print("\n" + results_df[['strategy', 'total_return', 'annual_return', 'sharpe_ratio',
                         'max_drawdown', 'win_rate', 'num_trades']].to_string(index=False))

best_sharpe_idx = results_df['sharpe_ratio'].idxmax()
best_return_idx = results_df['total_return'].idxmax()

print(f"\n🏆 Best Sharpe: {results_df.loc[best_sharpe_idx, 'strategy']} ({results_df.loc[best_sharpe_idx, 'sharpe_ratio']:.4f})")
print(f"🏆 Best Return: {results_df.loc[best_return_idx, 'strategy']} ({results_df.loc[best_return_idx, 'total_return']:.2f}%)")

# ========================================
# 5. Visualization
# ========================================
print("\n" + "=" * 80)
print("Creating visualizations...")
print("=" * 80)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# 1. Portfolio Value Over Time
ax1 = fig.add_subplot(gs[0, :])
for result in all_results:
    ax1.plot(test_df['Date'], result['portfolio_values'],
            label=result['strategy'], linewidth=2.5, alpha=0.8)
ax1.set_xlabel('Date (2025)', fontweight='bold', fontsize=12)
ax1.set_ylabel('Portfolio Value ($)', fontweight='bold', fontsize=12)
ax1.set_title('Portfolio Value Over Time - 2025 Only', fontweight='bold', fontsize=14)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# 2. Total Return Comparison
ax2 = fig.add_subplot(gs[1, 0])
colors = ['#95a5a6', '#3498db', '#e74c3c', '#9b59b6', '#2ecc71', '#f39c12']
bars = ax2.barh(range(len(results_df)), results_df['total_return'], color=colors, alpha=0.7)
ax2.set_yticks(range(len(results_df)))
ax2.set_yticklabels(results_df['strategy'], fontsize=9)
ax2.set_xlabel('Total Return (%)', fontweight='bold')
ax2.set_title('Total Return (2025)', fontweight='bold')
ax2.axvline(x=0, color='black', linestyle='--', alpha=0.3)
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3, axis='x')

for i, row in results_df.iterrows():
    ax2.text(row['total_return'], i, f"  {row['total_return']:.1f}%",
            va='center', fontsize=9, fontweight='bold')

# 3. Sharpe Ratio
ax3 = fig.add_subplot(gs[1, 1])
bars = ax3.barh(range(len(results_df)), results_df['sharpe_ratio'], color=colors, alpha=0.7)
ax3.set_yticks(range(len(results_df)))
ax3.set_yticklabels(results_df['strategy'], fontsize=9)
ax3.set_xlabel('Sharpe Ratio', fontweight='bold')
ax3.set_title('Sharpe Ratio (2025)', fontweight='bold')
ax3.axvline(x=0, color='black', linestyle='--', alpha=0.3)
ax3.invert_yaxis()
ax3.grid(True, alpha=0.3, axis='x')

for i, row in results_df.iterrows():
    ax3.text(row['sharpe_ratio'], i, f"  {row['sharpe_ratio']:.2f}",
            va='center', fontsize=9, fontweight='bold')

# 4. Max Drawdown
ax4 = fig.add_subplot(gs[1, 2])
bars = ax4.barh(range(len(results_df)), results_df['max_drawdown'], color=colors, alpha=0.7)
ax4.set_yticks(range(len(results_df)))
ax4.set_yticklabels(results_df['strategy'], fontsize=9)
ax4.set_xlabel('Max Drawdown (%)', fontweight='bold')
ax4.set_title('Max Drawdown (2025)', fontweight='bold')
ax4.invert_yaxis()
ax4.grid(True, alpha=0.3, axis='x')

# 5. Win Rate
ax5 = fig.add_subplot(gs[2, 0])
bars = ax5.barh(range(len(results_df)), results_df['win_rate'], color=colors, alpha=0.7)
ax5.set_yticks(range(len(results_df)))
ax5.set_yticklabels(results_df['strategy'], fontsize=9)
ax5.set_xlabel('Win Rate (%)', fontweight='bold')
ax5.set_title('Win Rate (2025)', fontweight='bold')
ax5.axvline(x=50, color='red', linestyle='--', alpha=0.3)
ax5.invert_yaxis()
ax5.grid(True, alpha=0.3, axis='x')

# 6. 2025 BTC Price
ax6 = fig.add_subplot(gs[2, 1:])
ax6.plot(test_df['Date'], test_df['Close'], linewidth=2, color='black', label='BTC Price')
ax6.fill_between(test_df['Date'], test_df['Close'].min(), test_df['Close'],
                 alpha=0.3, color='blue')
ax6.set_xlabel('Date', fontweight='bold', fontsize=12)
ax6.set_ylabel('BTC Price ($)', fontweight='bold', fontsize=12)
ax6.set_title('Bitcoin Price - 2025', fontweight='bold', fontsize=13)
ax6.grid(True, alpha=0.3)
ax6.tick_params(axis='x', rotation=45)

# Add annotations
max_price_idx = test_df['Close'].idxmax()
min_price_idx = test_df['Close'].idxmin()
ax6.scatter(test_df.loc[max_price_idx, 'Date'], test_df.loc[max_price_idx, 'Close'],
           color='red', s=100, zorder=5, label=f"High: ${test_df['Close'].max():.0f}")
ax6.scatter(test_df.loc[min_price_idx, 'Date'], test_df.loc[min_price_idx, 'Close'],
           color='green', s=100, zorder=5, label=f"Low: ${test_df['Close'].min():.0f}")
ax6.legend()

plt.savefig('elasticnet_2025_only_results.png', dpi=300, bbox_inches='tight')
print("Saved: elasticnet_2025_only_results.png")

# ========================================
# 6. Save Results
# ========================================
results_save = results_df.drop(columns=['portfolio_values'])
results_save.to_csv('elasticnet_2025_only_results.csv', index=False)
print("Saved: elasticnet_2025_only_results.csv")

# ========================================
# 7. Detailed Analysis
# ========================================
print("\n" + "=" * 80)
print("2025 DETAILED ANALYSIS")
print("=" * 80)

bnh = results_df[results_df['strategy'] == 'Buy-and-Hold'].iloc[0]
best = results_df.loc[best_sharpe_idx]

print(f"""
📊 2025년도 백테스팅 결과 분석

기간: 2025-01-01 ~ 2025-10-13 ({len(test_df)}일)

1. 시장 상황 (2025):
   시작가: ${initial_price:.2f}
   종가: ${final_price:.2f}
   최고가: ${test_df['Close'].max():.2f}
   최저가: ${test_df['Close'].min():.2f}
   변동폭: ${test_df['Close'].max() - test_df['Close'].min():.2f}

2. Buy-and-Hold (기준선):
   총 수익률: {bnh['total_return']:.2f}%
   연간 수익률: {bnh['annual_return']:.2f}%
   Sharpe: {bnh['sharpe_ratio']:.4f}
   Max DD: {bnh['max_drawdown']:.2f}%

3. 최고 전략 (Sharpe 기준):
   전략: {best['strategy']}
   총 수익률: {best['total_return']:.2f}% {'✅' if best['total_return'] > bnh['total_return'] else '❌'}
   연간 수익률: {best['annual_return']:.2f}%
   Sharpe: {best['sharpe_ratio']:.4f} {'✅' if best['sharpe_ratio'] > bnh['sharpe_ratio'] else '❌'}
   Max DD: {best['max_drawdown']:.2f}% {'✅' if best['max_drawdown'] > bnh['max_drawdown'] else '❌'}
   거래 횟수: {best['num_trades']}회

4. 전체 기간 vs 2025년 비교:
   (step26 결과와 비교)

5. 결론:
   {'✅ 2025년에도 ElasticNet 전략 유효!' if best['sharpe_ratio'] > bnh['sharpe_ratio'] else '❌ 2025년에는 Buy-and-Hold가 우수'}
   {'✅ 최신 시장에서도 작동!' if best['total_return'] > bnh['total_return'] else '⚠️ 최신 시장에서 성능 저하'}

   2025년 특징:
   - 학습 데이터: 2021-2024 (4년)
   - 테스트 데이터: 2025 ({len(test_df)}일)
   - Out-of-sample 테스트 (완전히 미래 데이터)
""")

print("\n" + "=" * 80)
print("Step 27 Completed!")
print("=" * 80)
