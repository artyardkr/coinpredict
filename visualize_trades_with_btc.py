#!/usr/bin/env python3
"""
거래 내역을 비트코인 가격과 함께 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("거래 내역 시각화 (비트코인 가격과 함께)")
print("=" * 80)

# 거래 데이터 로드
trades_df = pd.read_csv('step30_all_trades.csv')
trades_df['date'] = pd.to_datetime(trades_df['date'])

print(f"\n거래 내역: {len(trades_df)} 건")
print(f"기간: {trades_df['date'].min()} ~ {trades_df['date'].max()}")
print(f"\n버전별 거래 수:")
print(trades_df['version'].value_counts())

# 비트코인 가격 데이터 로드
btc_df = pd.read_csv('integrated_data_full_v2.csv')
btc_df['Date'] = pd.to_datetime(btc_df['Date'])
btc_df = btc_df[['Date', 'Close']].copy()
btc_df = btc_df.sort_values('Date')

# 거래 기간에 맞춰 필터링
start_date = trades_df['date'].min()
end_date = trades_df['date'].max()
btc_df = btc_df[(btc_df['Date'] >= start_date) & (btc_df['Date'] <= end_date)]

print(f"\n비트코인 가격 데이터: {len(btc_df)} 일")

# 버전별로 분리
versions = trades_df['version'].unique()
print(f"\n버전: {versions}")

# 시각화
fig = plt.figure(figsize=(24, 14))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. 전체 거래 내역 (모든 버전)
ax1 = fig.add_subplot(gs[0, :])

# 비트코인 가격
ax1.plot(btc_df['Date'], btc_df['Close'],
         color='black', linewidth=2, alpha=0.7, label='BTC 가격')

# 거래 표시
buy_trades = trades_df[trades_df['action'] == 'BUY']
sell_trades = trades_df[trades_df['action'] == 'SELL']

ax1.scatter(buy_trades['date'], buy_trades['price'],
           color='green', s=200, marker='^', alpha=0.8,
           edgecolors='darkgreen', linewidth=2,
           label=f'매수 ({len(buy_trades)})', zorder=5)

ax1.scatter(sell_trades['date'], sell_trades['price'],
           color='red', s=200, marker='v', alpha=0.8,
           edgecolors='darkred', linewidth=2,
           label=f'매도 ({len(sell_trades)})', zorder=5)

ax1.set_xlabel('날짜', fontweight='bold', fontsize=12)
ax1.set_ylabel('가격 ($)', fontweight='bold', fontsize=12)
ax1.set_title('전체 거래 내역 (모든 버전)', fontweight='bold', fontsize=14)
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax1.tick_params(axis='x', rotation=45)

# 가격 범위 정보
price_min = btc_df['Close'].min()
price_max = btc_df['Close'].max()
price_range = price_max - price_min

ax1.text(0.02, 0.98,
        f'가격 범위: ${price_min:,.0f} ~ ${price_max:,.0f}\n변동폭: ${price_range:,.0f} ({price_range/price_min*100:.1f}%)',
        transform=ax1.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# 2. 버전별 거래 내역
for idx, version in enumerate(sorted(versions)):
    row = (idx // 2) + 1
    col = idx % 2
    ax = fig.add_subplot(gs[row, col])

    version_trades = trades_df[trades_df['version'] == version]
    buy_v = version_trades[version_trades['action'] == 'BUY']
    sell_v = version_trades[version_trades['action'] == 'SELL']

    # 비트코인 가격
    ax.plot(btc_df['Date'], btc_df['Close'],
           color='gray', linewidth=2, alpha=0.5, label='BTC 가격')

    # 거래 표시
    ax.scatter(buy_v['date'], buy_v['price'],
              color='green', s=150, marker='^', alpha=0.8,
              edgecolors='darkgreen', linewidth=2,
              label=f'매수 ({len(buy_v)})', zorder=5)

    ax.scatter(sell_v['date'], sell_v['price'],
              color='red', s=150, marker='v', alpha=0.8,
              edgecolors='darkred', linewidth=2,
              label=f'매도 ({len(sell_v)})', zorder=5)

    # 매수-매도 연결선
    for i in range(min(len(buy_v), len(sell_v))):
        buy_date = buy_v.iloc[i]['date']
        buy_price = buy_v.iloc[i]['price']
        sell_date = sell_v.iloc[i]['date']
        sell_price = sell_v.iloc[i]['price']

        profit = (sell_price - buy_price) / buy_price * 100
        color = 'green' if profit > 0 else 'red'

        ax.plot([buy_date, sell_date], [buy_price, sell_price],
               color=color, linewidth=1.5, alpha=0.4, linestyle='--')

    ax.set_xlabel('날짜', fontweight='bold', fontsize=11)
    ax.set_ylabel('가격 ($)', fontweight='bold', fontsize=11)
    ax.set_title(f'버전 {version} 거래 내역', fontweight='bold', fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.tick_params(axis='x', rotation=45)

    # 거래 통계
    total_trades = len(version_trades)
    buy_count = len(buy_v)
    sell_count = len(sell_v)

    # 수익률 계산 (매칭된 거래만)
    profits = []
    for i in range(min(len(buy_v), len(sell_v))):
        buy_price = buy_v.iloc[i]['price']
        sell_price = sell_v.iloc[i]['price']
        profit = (sell_price - buy_price) / buy_price * 100
        profits.append(profit)

    if profits:
        avg_profit = np.mean(profits)
        win_rate = (np.array(profits) > 0).sum() / len(profits) * 100

        ax.text(0.02, 0.98,
               f'거래: {total_trades}건\n평균 수익: {avg_profit:.2f}%\n승률: {win_rate:.1f}%',
               transform=ax.transAxes, va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

plt.savefig('trades_visualization_with_btc.png', dpi=300, bbox_inches='tight')
print("\n✅ 저장 완료: trades_visualization_with_btc.png")

# 거래 통계 출력
print("\n" + "=" * 80)
print("거래 통계")
print("=" * 80)

for version in sorted(versions):
    version_trades = trades_df[trades_df['version'] == version]
    buy_v = version_trades[version_trades['action'] == 'BUY']
    sell_v = version_trades[version_trades['action'] == 'SELL']

    print(f"\n【버전 {version}】")
    print(f"  총 거래: {len(version_trades)}건")
    print(f"  매수: {len(buy_v)}건")
    print(f"  매도: {len(sell_v)}건")

    # 매칭된 거래 수익률
    profits = []
    for i in range(min(len(buy_v), len(sell_v))):
        buy_price = buy_v.iloc[i]['price']
        sell_price = sell_v.iloc[i]['price']
        profit = (sell_price - buy_price) / buy_price * 100
        profits.append(profit)

    if profits:
        print(f"  완료된 거래: {len(profits)}건")
        print(f"  평균 수익: {np.mean(profits):.2f}%")
        print(f"  최고 수익: {np.max(profits):.2f}%")
        print(f"  최저 수익: {np.min(profits):.2f}%")
        print(f"  승률: {(np.array(profits) > 0).sum() / len(profits) * 100:.1f}%")
        print(f"  총 수익률: {np.sum(profits):.2f}%")

# 상세 거래 내역 출력
print("\n" + "=" * 80)
print("상세 거래 내역")
print("=" * 80)

for version in sorted(versions):
    version_trades = trades_df[trades_df['version'] == version]
    buy_v = version_trades[version_trades['action'] == 'BUY'].reset_index(drop=True)
    sell_v = version_trades[version_trades['action'] == 'SELL'].reset_index(drop=True)

    print(f"\n【버전 {version}】")
    print(f"{'번호':>4} {'매수일':>12} {'매수가':>12} {'매도일':>12} {'매도가':>12} {'수익률':>10} {'보유기간':>8}")
    print("-" * 80)

    for i in range(min(len(buy_v), len(sell_v))):
        buy_date = buy_v.iloc[i]['date'].strftime('%Y-%m-%d')
        buy_price = buy_v.iloc[i]['price']
        sell_date = sell_v.iloc[i]['date'].strftime('%Y-%m-%d')
        sell_price = sell_v.iloc[i]['price']

        profit = (sell_price - buy_price) / buy_price * 100
        hold_days = (sell_v.iloc[i]['date'] - buy_v.iloc[i]['date']).days

        print(f"{i+1:4d} {buy_date:>12} ${buy_price:>10,.2f} {sell_date:>12} ${sell_price:>10,.2f} {profit:>9.2f}% {hold_days:>7}일")

print("\n" + "=" * 80)
print("분석 완료!")
print("=" * 80)
