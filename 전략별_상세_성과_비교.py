#!/usr/bin/env python3
"""
전략별 상세 성과 비교: V1 vs V2
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read V1 and V2 results
v1_df = pd.read_csv('elasticnet_backtesting_results.csv')
v2_df = pd.read_csv('elasticnet_backtesting_results_v2.csv')

# Add version column
v1_df['version'] = 'V1 (88 features)'
v2_df['version'] = 'V2 (138 features)'

# Combine
combined_df = pd.concat([v1_df, v2_df], ignore_index=True)

print("=" * 100)
print("전략별 상세 성과 비교: V1 (88 Features) vs V2 (138 Features)")
print("=" * 100)

# For each strategy, compare V1 vs V2
strategies = v1_df['strategy'].unique()

for strategy in strategies:
    print(f"\n{'='*100}")
    print(f"전략: {strategy}")
    print(f"{'='*100}")

    v1_row = v1_df[v1_df['strategy'] == strategy].iloc[0]
    v2_row = v2_df[v2_df['strategy'] == strategy].iloc[0]

    print(f"\n{'지표':<25} {'V1 (88 Features)':<25} {'V2 (138 Features)':<25} {'변화':<20}")
    print("-" * 100)

    # Total Return
    v1_return = v1_row['total_return']
    v2_return = v2_row['total_return']
    diff_return = v2_return - v1_return
    arrow = "✅" if diff_return > 0 else "⚠️" if diff_return < -5 else "➖"
    print(f"{'Total Return (%)':<25} {v1_return:>20.2f}% {v2_return:>20.2f}% {diff_return:>+15.2f}%p {arrow}")

    # Annual Return
    v1_annual = v1_row['annual_return']
    v2_annual = v2_row['annual_return']
    diff_annual = v2_annual - v1_annual
    arrow = "✅" if diff_annual > 0 else "⚠️" if diff_annual < -3 else "➖"
    print(f"{'Annual Return (%)':<25} {v1_annual:>20.2f}% {v2_annual:>20.2f}% {diff_annual:>+15.2f}%p {arrow}")

    # Sharpe Ratio
    v1_sharpe = v1_row['sharpe_ratio']
    v2_sharpe = v2_row['sharpe_ratio']
    diff_sharpe = v2_sharpe - v1_sharpe
    arrow = "✅" if diff_sharpe > 0 else "⚠️" if diff_sharpe < -0.1 else "➖"
    print(f"{'Sharpe Ratio':<25} {v1_sharpe:>24.4f} {v2_sharpe:>24.4f} {diff_sharpe:>+16.4f} {arrow}")

    # Volatility
    v1_vol = v1_row['annual_volatility']
    v2_vol = v2_row['annual_volatility']
    diff_vol = v2_vol - v1_vol
    arrow = "✅" if diff_vol < 0 else "⚠️" if diff_vol > 1 else "➖"
    print(f"{'Volatility (%)':<25} {v1_vol:>20.2f}% {v2_vol:>20.2f}% {diff_vol:>+15.2f}%p {arrow}")

    # Max Drawdown
    v1_dd = v1_row['max_drawdown']
    v2_dd = v2_row['max_drawdown']
    diff_dd = v2_dd - v1_dd
    arrow = "✅" if diff_dd > 0 else "⚠️" if diff_dd < -2 else "➖"
    print(f"{'Max Drawdown (%)':<25} {v1_dd:>20.2f}% {v2_dd:>20.2f}% {diff_dd:>+15.2f}%p {arrow}")

    # Win Rate
    v1_win = v1_row['win_rate']
    v2_win = v2_row['win_rate']
    diff_win = v2_win - v1_win
    arrow = "✅" if diff_win > 0 else "⚠️" if diff_win < -3 else "➖"
    print(f"{'Win Rate (%)':<25} {v1_win:>20.2f}% {v2_win:>20.2f}% {diff_win:>+15.2f}%p {arrow}")

    # Number of Trades
    v1_trades = int(v1_row['num_trades'])
    v2_trades = int(v2_row['num_trades'])
    diff_trades = v2_trades - v1_trades
    arrow = "✅" if abs(diff_trades) <= 2 else "➖"
    print(f"{'거래 횟수':<25} {v1_trades:>24d} {v2_trades:>24d} {diff_trades:>+16d} {arrow}")

    # Final Value
    v1_final = v1_row['final_value']
    v2_final = v2_row['final_value']
    diff_final = v2_final - v1_final
    arrow = "✅" if diff_final > 0 else "⚠️"
    print(f"{'최종 자본 ($)':<25} {v1_final:>20,.2f} {v2_final:>20,.2f} {diff_final:>+15,.2f} {arrow}")

# Summary Table
print(f"\n\n{'='*100}")
print("종합 비교표")
print(f"{'='*100}\n")

summary_data = []
for strategy in strategies:
    v1_row = v1_df[v1_df['strategy'] == strategy].iloc[0]
    v2_row = v2_df[v2_df['strategy'] == strategy].iloc[0]

    summary_data.append({
        'Strategy': strategy,
        'Version': 'V1',
        'Return (%)': f"{v1_row['total_return']:.2f}",
        'Sharpe': f"{v1_row['sharpe_ratio']:.3f}",
        'Volatility (%)': f"{v1_row['annual_volatility']:.2f}",
        'Max DD (%)': f"{v1_row['max_drawdown']:.2f}",
        'Win Rate (%)': f"{v1_row['win_rate']:.2f}",
        'Trades': int(v1_row['num_trades'])
    })

    summary_data.append({
        'Strategy': strategy,
        'Version': 'V2',
        'Return (%)': f"{v2_row['total_return']:.2f}",
        'Sharpe': f"{v2_row['sharpe_ratio']:.3f}",
        'Volatility (%)': f"{v2_row['annual_volatility']:.2f}",
        'Max DD (%)': f"{v2_row['max_drawdown']:.2f}",
        'Win Rate (%)': f"{v2_row['win_rate']:.2f}",
        'Trades': int(v2_row['num_trades'])
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Best performers
print(f"\n\n{'='*100}")
print("🏆 최고 성과 전략")
print(f"{'='*100}\n")

# Best Return
best_return_idx = combined_df['total_return'].idxmax()
best_return = combined_df.loc[best_return_idx]
print(f"✅ 최고 수익률: {best_return['strategy']} ({best_return['version']})")
print(f"   Total Return: {best_return['total_return']:.2f}%")
print(f"   Annual Return: {best_return['annual_return']:.2f}%")
print(f"   Sharpe: {best_return['sharpe_ratio']:.3f}")

# Best Sharpe
best_sharpe_idx = combined_df['sharpe_ratio'].idxmax()
best_sharpe = combined_df.loc[best_sharpe_idx]
print(f"\n✅ 최고 Sharpe Ratio: {best_sharpe['strategy']} ({best_sharpe['version']})")
print(f"   Sharpe: {best_sharpe['sharpe_ratio']:.4f}")
print(f"   Total Return: {best_sharpe['total_return']:.2f}%")
print(f"   Max DD: {best_sharpe['max_drawdown']:.2f}%")

# Best Max Drawdown (least negative)
best_dd_idx = combined_df['max_drawdown'].idxmax()
best_dd = combined_df.loc[best_dd_idx]
print(f"\n✅ 최소 낙폭: {best_dd['strategy']} ({best_dd['version']})")
print(f"   Max Drawdown: {best_dd['max_drawdown']:.2f}%")
print(f"   Total Return: {best_dd['total_return']:.2f}%")
print(f"   Sharpe: {best_dd['sharpe_ratio']:.3f}")

# Best Risk-Adjusted (Sharpe > 1.5 and Return > 50%)
quality_strategies = combined_df[(combined_df['sharpe_ratio'] > 1.5) & (combined_df['total_return'] > 50)]
if len(quality_strategies) > 0:
    print(f"\n✅ 우수 전략 (Sharpe > 1.5 & Return > 50%):")
    for idx, row in quality_strategies.iterrows():
        print(f"   - {row['strategy']} ({row['version']}): Return {row['total_return']:.2f}%, Sharpe {row['sharpe_ratio']:.3f}")

# Visualization
print(f"\n\n{'='*100}")
print("시각화 생성 중...")
print(f"{'='*100}")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('V1 vs V2 전략별 상세 비교', fontsize=16, fontweight='bold')

# Remove Buy-and-Hold for clearer comparison of ElasticNet strategies
elasticnet_df = combined_df[combined_df['strategy'] != 'Buy-and-Hold'].copy()
elasticnet_df['strategy_short'] = elasticnet_df['strategy'].str.replace(' (ElasticNet)', '')

# 1. Total Return
ax1 = axes[0, 0]
x = np.arange(len(elasticnet_df['strategy_short'].unique()))
width = 0.35
v1_data = elasticnet_df[elasticnet_df['version'].str.contains('V1')]['total_return'].values
v2_data = elasticnet_df[elasticnet_df['version'].str.contains('V2')]['total_return'].values
ax1.bar(x - width/2, v1_data, width, label='V1', alpha=0.8, color='#3498db')
ax1.bar(x + width/2, v2_data, width, label='V2', alpha=0.8, color='#e74c3c')
ax1.set_xlabel('Strategy', fontweight='bold')
ax1.set_ylabel('Total Return (%)', fontweight='bold')
ax1.set_title('Total Return Comparison', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(elasticnet_df['strategy_short'].unique(), rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# 2. Sharpe Ratio
ax2 = axes[0, 1]
v1_data = elasticnet_df[elasticnet_df['version'].str.contains('V1')]['sharpe_ratio'].values
v2_data = elasticnet_df[elasticnet_df['version'].str.contains('V2')]['sharpe_ratio'].values
ax2.bar(x - width/2, v1_data, width, label='V1', alpha=0.8, color='#3498db')
ax2.bar(x + width/2, v2_data, width, label='V2', alpha=0.8, color='#e74c3c')
ax2.set_xlabel('Strategy', fontweight='bold')
ax2.set_ylabel('Sharpe Ratio', fontweight='bold')
ax2.set_title('Sharpe Ratio Comparison', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(elasticnet_df['strategy_short'].unique(), rotation=45, ha='right')
ax2.legend()
ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Good (>1.0)')
ax2.grid(True, alpha=0.3, axis='y')

# 3. Max Drawdown
ax3 = axes[0, 2]
v1_data = elasticnet_df[elasticnet_df['version'].str.contains('V1')]['max_drawdown'].values
v2_data = elasticnet_df[elasticnet_df['version'].str.contains('V2')]['max_drawdown'].values
ax3.bar(x - width/2, v1_data, width, label='V1', alpha=0.8, color='#3498db')
ax3.bar(x + width/2, v2_data, width, label='V2', alpha=0.8, color='#e74c3c')
ax3.set_xlabel('Strategy', fontweight='bold')
ax3.set_ylabel('Max Drawdown (%)', fontweight='bold')
ax3.set_title('Max Drawdown Comparison', fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(elasticnet_df['strategy_short'].unique(), rotation=45, ha='right')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 4. Volatility
ax4 = axes[1, 0]
v1_data = elasticnet_df[elasticnet_df['version'].str.contains('V1')]['annual_volatility'].values
v2_data = elasticnet_df[elasticnet_df['version'].str.contains('V2')]['annual_volatility'].values
ax4.bar(x - width/2, v1_data, width, label='V1', alpha=0.8, color='#3498db')
ax4.bar(x + width/2, v2_data, width, label='V2', alpha=0.8, color='#e74c3c')
ax4.set_xlabel('Strategy', fontweight='bold')
ax4.set_ylabel('Volatility (%)', fontweight='bold')
ax4.set_title('Volatility Comparison', fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(elasticnet_df['strategy_short'].unique(), rotation=45, ha='right')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# 5. Win Rate
ax5 = axes[1, 1]
v1_data = elasticnet_df[elasticnet_df['version'].str.contains('V1')]['win_rate'].values
v2_data = elasticnet_df[elasticnet_df['version'].str.contains('V2')]['win_rate'].values
ax5.bar(x - width/2, v1_data, width, label='V1', alpha=0.8, color='#3498db')
ax5.bar(x + width/2, v2_data, width, label='V2', alpha=0.8, color='#e74c3c')
ax5.set_xlabel('Strategy', fontweight='bold')
ax5.set_ylabel('Win Rate (%)', fontweight='bold')
ax5.set_title('Win Rate Comparison', fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(elasticnet_df['strategy_short'].unique(), rotation=45, ha='right')
ax5.axhline(y=50, color='green', linestyle='--', alpha=0.5, label='50%')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# 6. Number of Trades
ax6 = axes[1, 2]
v1_data = elasticnet_df[elasticnet_df['version'].str.contains('V1')]['num_trades'].values
v2_data = elasticnet_df[elasticnet_df['version'].str.contains('V2')]['num_trades'].values
ax6.bar(x - width/2, v1_data, width, label='V1', alpha=0.8, color='#3498db')
ax6.bar(x + width/2, v2_data, width, label='V2', alpha=0.8, color='#e74c3c')
ax6.set_xlabel('Strategy', fontweight='bold')
ax6.set_ylabel('Number of Trades', fontweight='bold')
ax6.set_title('Trading Frequency Comparison', fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(elasticnet_df['strategy_short'].unique(), rotation=45, ha='right')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('strategy_detailed_comparison_v1_vs_v2.png', dpi=300, bbox_inches='tight')
print("✅ 저장: strategy_detailed_comparison_v1_vs_v2.png")

# Save detailed comparison to CSV
summary_df.to_csv('strategy_detailed_comparison_table.csv', index=False)
print("✅ 저장: strategy_detailed_comparison_table.csv")

print(f"\n{'='*100}")
print("분석 완료!")
print(f"{'='*100}")
