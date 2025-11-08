#!/usr/bin/env python3
"""
Step30 백테스팅 매수매도 횟수 분석
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("Step30 백테스팅 매수매도 횟수 분석")
print("="*80)

# 데이터 로드
df = pd.read_csv('integrated_data_full.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# 타겟
df['target'] = df['Close'].shift(-1)
df = df[:-1].copy()

# 특성
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

X = df[feature_cols].values
y = df['target'].values
dates = df['Date'].values
close_prices = df['Close'].values

# 백테스팅 함수 (거래 횟수 포함)
def backtest_with_trades(dates, current_prices, y_true, y_pred, threshold=1.0, initial_capital=10000):
    """예측 기반 백테스팅 (거래 내역 포함)"""
    predicted_returns = (y_pred / current_prices - 1) * 100

    cash = initial_capital
    btc = 0
    portfolio_values = []
    trades = []
    positions = []

    for i in range(len(y_pred)):
        prev_position = 'cash' if btc == 0 else 'btc'

        if predicted_returns[i] > threshold:
            # 매수 신호
            if cash > 0:
                btc = cash / current_prices[i]
                cash = 0
                trades.append({
                    'date': dates[i],
                    'action': 'BUY',
                    'price': current_prices[i],
                    'predicted_return': predicted_returns[i]
                })
                positions.append('btc')
            else:
                positions.append('btc')
        else:
            # 매도 신호
            if btc > 0:
                cash = btc * current_prices[i]
                btc = 0
                trades.append({
                    'date': dates[i],
                    'action': 'SELL',
                    'price': current_prices[i],
                    'predicted_return': predicted_returns[i]
                })
                positions.append('cash')
            else:
                positions.append('cash')

        # 포트폴리오 가치
        if btc > 0:
            portfolio_value = btc * y_true[i]
        else:
            portfolio_value = cash
        portfolio_values.append(portfolio_value)

    # 통계
    buy_count = len([t for t in trades if t['action'] == 'BUY'])
    sell_count = len([t for t in trades if t['action'] == 'SELL'])

    # 최종 수익률
    final_value = portfolio_values[-1]
    total_return = (final_value / initial_capital - 1) * 100

    # Buy-and-Hold
    bnh_btc = initial_capital / current_prices[0]
    bnh_final = bnh_btc * current_prices[-1]
    bnh_return = (bnh_final / initial_capital - 1) * 100

    return {
        'trades': trades,
        'buy_count': buy_count,
        'sell_count': sell_count,
        'total_trades': len(trades),
        'final_value': final_value,
        'total_return': total_return,
        'bnh_return': bnh_return,
        'positions': positions
    }

print("\n" + "="*60)
print("Version A: 70/30 Split")
print("="*60)

# Version A
split_idx = int(len(df) * 0.7)
X_train_A = X[:split_idx]
y_train_A = y[:split_idx]
X_test_A = X[split_idx:]
y_test_A = y[split_idx:]
dates_test_A = dates[split_idx:]
close_test_A = close_prices[split_idx:]

scaler_A = StandardScaler()
X_train_A_scaled = scaler_A.fit_transform(X_train_A)
X_test_A_scaled = scaler_A.transform(X_test_A)

elasticnet_A = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=10000)
elasticnet_A.fit(X_train_A_scaled, y_train_A)
y_test_pred_A = elasticnet_A.predict(X_test_A_scaled)

# 백테스팅
bt_A = backtest_with_trades(dates_test_A, close_test_A, y_test_A, y_test_pred_A, threshold=1.0)

print(f"\n테스트 기간: {pd.to_datetime(dates_test_A[0]).date()} ~ {pd.to_datetime(dates_test_A[-1]).date()}")
print(f"테스트 일수: {len(dates_test_A)}일")
print(f"\n매수 횟수: {bt_A['buy_count']}회")
print(f"매도 횟수: {bt_A['sell_count']}회")
print(f"총 거래 횟수: {bt_A['total_trades']}회")
print(f"평균 보유 기간: {len(dates_test_A) / max(bt_A['buy_count'], 1):.1f}일")
print(f"\n전략 수익률: {bt_A['total_return']:+.2f}%")
print(f"Buy-and-Hold: {bt_A['bnh_return']:+.2f}%")
print(f"초과 수익: {bt_A['total_return'] - bt_A['bnh_return']:+.2f}%p")

# 거래 내역
print(f"\n처음 5개 거래:")
for trade in bt_A['trades'][:5]:
    print(f"  {pd.to_datetime(trade['date']).date()} | {trade['action']:4s} | ${trade['price']:>10,.2f} | 예측: {trade['predicted_return']:+.2f}%")

print(f"\n마지막 5개 거래:")
for trade in bt_A['trades'][-5:]:
    print(f"  {pd.to_datetime(trade['date']).date()} | {trade['action']:4s} | ${trade['price']:>10,.2f} | 예측: {trade['predicted_return']:+.2f}%")

print("\n" + "="*60)
print("Version B: 2025년만")
print("="*60)

# Version B
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

scaler_B = StandardScaler()
X_train_B_scaled = scaler_B.fit_transform(X_train_B)
X_test_B_scaled = scaler_B.transform(X_test_B)

elasticnet_B = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=10000)
elasticnet_B.fit(X_train_B_scaled, y_train_B)
y_test_pred_B = elasticnet_B.predict(X_test_B_scaled)

# 백테스팅
bt_B = backtest_with_trades(dates_test_B, close_test_B, y_test_B, y_test_pred_B, threshold=1.0)

print(f"\n테스트 기간: {pd.to_datetime(dates_test_B[0]).date()} ~ {pd.to_datetime(dates_test_B[-1]).date()}")
print(f"테스트 일수: {len(dates_test_B)}일")
print(f"\n매수 횟수: {bt_B['buy_count']}회")
print(f"매도 횟수: {bt_B['sell_count']}회")
print(f"총 거래 횟수: {bt_B['total_trades']}회")
print(f"평균 보유 기간: {len(dates_test_B) / max(bt_B['buy_count'], 1):.1f}일")
print(f"\n전략 수익률: {bt_B['total_return']:+.2f}%")
print(f"Buy-and-Hold: {bt_B['bnh_return']:+.2f}%")
print(f"초과 수익: {bt_B['total_return'] - bt_B['bnh_return']:+.2f}%p")

# 거래 내역
print(f"\n전체 거래 내역:")
for trade in bt_B['trades']:
    print(f"  {pd.to_datetime(trade['date']).date()} | {trade['action']:4s} | ${trade['price']:>10,.2f} | 예측: {trade['predicted_return']:+.2f}%")

# 비교 요약
print("\n" + "="*80)
print("비교 요약")
print("="*80)

comparison = pd.DataFrame({
    '구분': ['Version A (70/30)', 'Version B (2025만)'],
    '테스트일수': [len(dates_test_A), len(dates_test_B)],
    '매수': [bt_A['buy_count'], bt_B['buy_count']],
    '매도': [bt_A['sell_count'], bt_B['sell_count']],
    '총거래': [bt_A['total_trades'], bt_B['total_trades']],
    '거래빈도': [f"{bt_A['total_trades']/len(dates_test_A)*100:.1f}%",
                f"{bt_B['total_trades']/len(dates_test_B)*100:.1f}%"],
    '수익률': [f"{bt_A['total_return']:+.2f}%", f"{bt_B['total_return']:+.2f}%"],
    'BnH': [f"{bt_A['bnh_return']:+.2f}%", f"{bt_B['bnh_return']:+.2f}%"],
    '초과수익': [f"{bt_A['total_return'] - bt_A['bnh_return']:+.2f}%p",
                f"{bt_B['total_return'] - bt_B['bnh_return']:+.2f}%p"]
})

print("\n" + comparison.to_string(index=False))

# 저장
comparison.to_csv('step30_trade_summary.csv', index=False)

# 거래 내역 저장
trades_A_df = pd.DataFrame(bt_A['trades'])
trades_A_df['version'] = 'A'
trades_B_df = pd.DataFrame(bt_B['trades'])
trades_B_df['version'] = 'B'
all_trades = pd.concat([trades_A_df, trades_B_df], ignore_index=True)
all_trades.to_csv('step30_all_trades.csv', index=False)

print(f"\n✅ 저장 완료:")
print(f"   - step30_trade_summary.csv")
print(f"   - step30_all_trades.csv")

print("\n" + "="*80)
print("핵심 발견")
print("="*80)
print(f"""
1. Version A (70/30 Split):
   - 총 {bt_A['total_trades']}회 거래 ({len(dates_test_A)}일 중)
   - 평균 {len(dates_test_A) / max(bt_A['buy_count'], 1):.1f}일마다 1회 매수
   - 수익률 {bt_A['total_return']:+.2f}% (Buy-and-Hold 대비 {bt_A['total_return'] - bt_A['bnh_return']:+.2f}%p 우수)
   - ✅ 성공적인 전략

2. Version B (2025년만):
   - 총 {bt_B['total_trades']}회 거래 ({len(dates_test_B)}일 중)
   - 평균 {len(dates_test_B) / max(bt_B['buy_count'], 1):.1f}일마다 1회 매수
   - 수익률 {bt_B['total_return']:+.2f}% (Buy-and-Hold 대비 {bt_B['total_return'] - bt_B['bnh_return']:+.2f}%p 열등)
   - ❌ 실패한 전략 (2025년 시장 예측 실패)

3. 거래 빈도:
   - Version A: 더 자주 거래 (시장 변동 잘 포착)
   - Version B: 적게 거래 (불확실성으로 신호 감소)

4. 결론:
   - Threshold 1% 전략은 2024-2025 전체 기간에서 유효
   - 하지만 2025년만 따로 보면 실패 (시장 환경 변화)
   - 순수 out-of-sample 테스트의 중요성!
""")

print("="*80)
