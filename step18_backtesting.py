import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("백테스팅 (Backtesting) - 실전 트레이딩 시뮬레이션")
print("=" * 70)

# ===== 1. 데이터 로드 및 전처리 =====
print("\n1. 데이터 로드 및 전처리")
print("-" * 70)

df = pd.read_csv('integrated_data_full.csv', index_col=0, parse_dates=True)
print(f"전체 데이터: {df.shape} ({df.index[0].date()} ~ {df.index[-1].date()})")

# 제거할 특성
exclude_features = [
    'Close', 'High', 'Low', 'Open', 'cumulative_return',
    'bc_market_price', 'bc_market_cap',
    'EMA5_close', 'EMA10_close', 'EMA14_close', 'EMA20_close', 'EMA30_close', 'EMA100_close',
    'SMA5_close', 'SMA10_close', 'SMA20_close', 'SMA30_close',
    'BB_high', 'BB_mid', 'BB_low',
]

# 타겟: 일별 수익률
df['target_return'] = (df['Close'].shift(-1) / df['Close'] - 1) * 100

# Feature Engineering
if 'bc_miners_revenue' in df.columns:
    df['miners_revenue_normalized'] = df['bc_miners_revenue'] / df['Close']

for col in ['RSI', 'MACD', 'ATR', 'OBV', 'ADX', 'CCI', 'MFI']:
    if col in df.columns:
        df[f'{col}_change'] = df[col].pct_change() * 100

for col in ['DGS10', 'CPIAUCSL', 'UNRATE', 'M2SL']:
    if col in df.columns and df[col].notna().sum() > 0:
        df[f'{col}_change'] = df[col].pct_change() * 100

df_clean = df.dropna(subset=['target_return']).copy()

# 특성 선택
all_features = [col for col in df_clean.columns
                if col not in exclude_features
                and col != 'target_return'
                and not col.endswith('_change')]

change_features = [col for col in df_clean.columns if col.endswith('_change')]
all_features.extend(change_features)

if 'miners_revenue_normalized' in df_clean.columns:
    all_features.append('miners_revenue_normalized')
    if 'bc_miners_revenue' in all_features:
        all_features.remove('bc_miners_revenue')

X = df_clean[all_features].copy()
y = df_clean['target_return'].copy()

mask = X.notna().all(axis=1) & y.notna()
X = X[mask]
y = y[mask]

print(f"특성 수: {len(all_features)}개")
print(f"샘플 수: {len(X)}개")

# ===== 2. 데이터 분할 =====
print("\n2. 데이터 분할")
print("-" * 70)

split_idx = int(len(X) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"Train: {len(X_train)}개 ({X_train.index[0].date()} ~ {X_train.index[-1].date()})")
print(f"Test:  {len(X_test)}개 ({X_test.index[0].date()} ~ {X_test.index[-1].date()})")

# ===== 3. 모델 훈련 =====
print("\n3. 모델 훈련")
print("-" * 70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest
print("Random Forest 훈련 중...")
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)

y_test_pred_rf = rf.predict(X_test_scaled)
rf_test_r2 = r2_score(y_test, y_test_pred_rf)
print(f"✓ Random Forest (Test R²: {rf_test_r2:.4f})")

# XGBoost
print("XGBoost 훈련 중...")
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train_scaled, y_train)

y_test_pred_xgb = xgb_model.predict(X_test_scaled)
xgb_test_r2 = r2_score(y_test, y_test_pred_xgb)
print(f"✓ XGBoost (Test R²: {xgb_test_r2:.4f})")

# ===== 4. 백테스팅 설정 =====
print("\n" + "=" * 70)
print("4. 백테스팅 설정")
print("=" * 70)

# Test 기간의 실제 가격 데이터
test_dates = y_test.index
test_prices = df.loc[test_dates, 'Close'].values

# 거래 비용
TRANSACTION_COST = 0.01  # 1%
SHORT_COST = 0.005  # 0.5% (숏 포지션 유지 비용)

print(f"\n거래 설정:")
print(f"  거래 비용: {TRANSACTION_COST*100:.1f}%")
print(f"  숏 포지션 비용: {SHORT_COST*100:.1f}%")
print(f"  초기 자본: $10,000")
print(f"  백테스팅 기간: {test_dates[0].date()} ~ {test_dates[-1].date()}")
print(f"  거래일수: {len(test_dates)}일")

# ===== 5. 트레이딩 전략 구현 =====
print("\n5. 트레이딩 전략 시뮬레이션")
print("-" * 70)

def backtest_strategy(predicted_returns, actual_returns, prices, strategy_type='long_only',
                     initial_capital=10000):
    """
    백테스팅 함수

    strategy_type:
        - 'buy_hold': 매수 후 보유
        - 'long_only': 상승 예측 시만 매수
        - 'long_short': 상승 시 매수, 하락 시 공매도
    """
    capital = initial_capital
    position = None  # 'long', 'short', None
    entry_price = None
    portfolio_values = [initial_capital]
    positions = []
    trades = []

    num_days = len(prices) - 1 if strategy_type == 'buy_hold' else len(predicted_returns) - 1

    for i in range(num_days):
        current_price = prices[i]
        next_price = prices[i + 1]

        pred_return = predicted_returns[i] if predicted_returns is not None else 0
        actual_return = actual_returns[i] if actual_returns is not None else 0

        if strategy_type == 'buy_hold':
            # 첫날 매수 후 계속 보유
            if i == 0:
                position = 'long'
                entry_price = current_price
                capital -= capital * TRANSACTION_COST

            # 현재 가치 계산
            current_value = capital * (next_price / entry_price)
            portfolio_values.append(current_value)
            positions.append(position)

        elif strategy_type == 'long_only':
            # 상승 예측 시 매수, 하락 예측 시 현금 보유
            if pred_return > 0:  # 상승 예측
                if position != 'long':
                    # 매수 진입
                    position = 'long'
                    entry_price = current_price
                    capital -= capital * TRANSACTION_COST
                    trades.append(('BUY', current_price, i))

                # 포지션 유지
                current_value = capital * (next_price / entry_price)
            else:  # 하락 예측 또는 중립
                if position == 'long':
                    # 매도
                    capital = capital * (current_price / entry_price)
                    capital -= capital * TRANSACTION_COST
                    position = None
                    trades.append(('SELL', current_price, i))

                # 현금 보유
                current_value = capital

            portfolio_values.append(current_value)
            positions.append(position)

        elif strategy_type == 'long_short':
            # 상승 예측 시 매수, 하락 예측 시 공매도
            if pred_return > 0.5:  # 상승 예측 (0.5% 이상)
                if position != 'long':
                    # 기존 포지션 정리
                    if position == 'short':
                        capital = capital * (2 - next_price / entry_price)
                        capital -= capital * (TRANSACTION_COST + SHORT_COST)

                    # 롱 진입
                    position = 'long'
                    entry_price = current_price
                    capital -= capital * TRANSACTION_COST
                    trades.append(('LONG', current_price, i))

                current_value = capital * (next_price / entry_price)

            elif pred_return < -0.5:  # 하락 예측 (0.5% 이상)
                if position != 'short':
                    # 기존 포지션 정리
                    if position == 'long':
                        capital = capital * (current_price / entry_price)
                        capital -= capital * TRANSACTION_COST

                    # 숏 진입
                    position = 'short'
                    entry_price = current_price
                    capital -= capital * TRANSACTION_COST
                    trades.append(('SHORT', current_price, i))

                # 숏 수익 = 2 - (현재가 / 진입가)
                current_value = capital * (2 - next_price / entry_price)
                current_value -= capital * SHORT_COST

            else:  # 중립
                if position == 'long':
                    capital = capital * (current_price / entry_price)
                    capital -= capital * TRANSACTION_COST
                    position = None
                elif position == 'short':
                    capital = capital * (2 - current_price / entry_price)
                    capital -= capital * (TRANSACTION_COST + SHORT_COST)
                    position = None

                current_value = capital

            portfolio_values.append(current_value)
            positions.append(position)

    return np.array(portfolio_values), positions, trades

# 전략별 백테스팅
print("\n[1] Buy-and-Hold 전략")
bh_values, _, _ = backtest_strategy(None, None, test_prices, 'buy_hold')

print("[2] Long-Only (Random Forest)")
long_rf_values, long_rf_pos, long_rf_trades = backtest_strategy(
    y_test_pred_rf, y_test.values, test_prices, 'long_only'
)

print("[3] Long-Only (XGBoost)")
long_xgb_values, long_xgb_pos, long_xgb_trades = backtest_strategy(
    y_test_pred_xgb, y_test.values, test_prices, 'long_only'
)

print("[4] Long-Short (Random Forest)")
ls_rf_values, ls_rf_pos, ls_rf_trades = backtest_strategy(
    y_test_pred_rf, y_test.values, test_prices, 'long_short'
)

print("[5] Long-Short (XGBoost)")
ls_xgb_values, ls_xgb_pos, ls_xgb_trades = backtest_strategy(
    y_test_pred_xgb, y_test.values, test_prices, 'long_short'
)

print("✓ 백테스팅 완료")

# ===== 6. 성과 지표 계산 =====
print("\n6. 성과 지표 계산")
print("-" * 70)

def calculate_metrics(portfolio_values, days):
    """성과 지표 계산"""
    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    # 총 수익률
    total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100

    # 연간 수익률
    years = days / 365
    annual_return = ((portfolio_values[-1] / portfolio_values[0]) ** (1 / years) - 1) * 100

    # 연간 변동성
    annual_volatility = returns.std() * np.sqrt(365) * 100

    # 샤프 비율 (무위험 이자율 0% 가정)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0

    # 최대 낙폭
    cummax = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - cummax) / cummax * 100
    max_drawdown = drawdown.min()

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'final_value': portfolio_values[-1]
    }

days = len(test_dates)

metrics = {
    'Buy-and-Hold': calculate_metrics(bh_values, days),
    'Long-Only (RF)': calculate_metrics(long_rf_values, days),
    'Long-Only (XGB)': calculate_metrics(long_xgb_values, days),
    'Long-Short (RF)': calculate_metrics(ls_rf_values, days),
    'Long-Short (XGB)': calculate_metrics(ls_xgb_values, days),
}

# ===== 7. 결과 출력 =====
print("\n" + "=" * 70)
print("7. 백테스팅 결과")
print("=" * 70)

results_df = pd.DataFrame(metrics).T

print("\n성과 지표:")
print("-" * 70)
print(f"{'전략':<20} {'총수익률':<12} {'연수익률':<12} {'샤프비율':<12} {'최대낙폭':<12} {'최종자산':<12}")
print("-" * 70)
for strategy, result in metrics.items():
    print(f"{strategy:<20} "
          f"{result['total_return']:>10.2f}% "
          f"{result['annual_return']:>10.2f}% "
          f"{result['sharpe_ratio']:>10.2f} "
          f"{result['max_drawdown']:>10.2f}% "
          f"${result['final_value']:>10,.0f}")

# 거래 횟수
print(f"\n거래 횟수:")
print(f"  Long-Only (RF):  {len(long_rf_trades)}회")
print(f"  Long-Only (XGB): {len(long_xgb_trades)}회")
print(f"  Long-Short (RF):  {len(ls_rf_trades)}회")
print(f"  Long-Short (XGB): {len(ls_xgb_trades)}회")

# ===== 8. 시각화 =====
print("\n8. 시각화")
print("-" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 포트폴리오 가치 변화
ax1 = axes[0, 0]
dates = test_dates[:len(bh_values)]
ax1.plot(dates, bh_values, label='Buy-and-Hold', linewidth=2, alpha=0.8)
ax1.plot(dates, long_rf_values, label='Long-Only (RF)', linewidth=2, alpha=0.8)
ax1.plot(dates, long_xgb_values, label='Long-Only (XGB)', linewidth=2, alpha=0.8)
ax1.set_xlabel('Date', fontsize=11)
ax1.set_ylabel('Portfolio Value ($)', fontsize=11)
ax1.set_title('Portfolio Value Over Time', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# 2. Long-Short 전략
ax2 = axes[0, 1]
ax2.plot(dates, bh_values, label='Buy-and-Hold', linewidth=2, alpha=0.8)
ax2.plot(dates, ls_rf_values, label='Long-Short (RF)', linewidth=2, alpha=0.8)
ax2.plot(dates, ls_xgb_values, label='Long-Short (XGB)', linewidth=2, alpha=0.8)
ax2.set_xlabel('Date', fontsize=11)
ax2.set_ylabel('Portfolio Value ($)', fontsize=11)
ax2.set_title('Long-Short Strategy', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# 3. 수익률 비교
ax3 = axes[1, 0]
strategies = list(metrics.keys())
returns = [metrics[s]['total_return'] for s in strategies]
colors = ['gray', 'blue', 'orange', 'green', 'red']
ax3.bar(strategies, returns, color=colors, alpha=0.7)
ax3.set_ylabel('Total Return (%)', fontsize=11)
ax3.set_title('Total Return Comparison', fontsize=12, fontweight='bold')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, alpha=0.3, axis='y')
ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)

# 4. 샤프 비율 비교
ax4 = axes[1, 1]
sharpe = [metrics[s]['sharpe_ratio'] for s in strategies]
ax4.bar(strategies, sharpe, color=colors, alpha=0.7)
ax4.set_ylabel('Sharpe Ratio', fontsize=11)
ax4.set_title('Risk-Adjusted Return (Sharpe Ratio)', fontsize=12, fontweight='bold')
ax4.tick_params(axis='x', rotation=45)
ax4.grid(True, alpha=0.3, axis='y')
ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('backtesting_results.png', dpi=300, bbox_inches='tight')
print("✓ backtesting_results.png")
plt.close()

# ===== 9. 결과 저장 =====
print("\n9. 결과 저장")
print("-" * 70)

results_df.to_csv('backtesting_results.csv')
print("✓ backtesting_results.csv")

# ===== 10. 최종 평가 =====
print("\n" + "=" * 70)
print("10. 최종 평가")
print("=" * 70)

best_strategy = max(metrics.items(), key=lambda x: x[1]['sharpe_ratio'])
best_return = max(metrics.items(), key=lambda x: x[1]['total_return'])

print(f"\n최고 샤프 비율: {best_strategy[0]}")
print(f"  샤프 비율: {best_strategy[1]['sharpe_ratio']:.2f}")
print(f"  연간 수익률: {best_strategy[1]['annual_return']:.2f}%")
print(f"  최대 낙폭: {best_strategy[1]['max_drawdown']:.2f}%")

print(f"\n최고 수익률: {best_return[0]}")
print(f"  총 수익률: {best_return[1]['total_return']:.2f}%")
print(f"  최종 자산: ${best_return[1]['final_value']:,.0f}")

# Buy-and-Hold 대비 성능
bh_return = metrics['Buy-and-Hold']['total_return']
print(f"\nBuy-and-Hold 대비:")
print("-" * 70)
for strategy, result in metrics.items():
    if strategy != 'Buy-and-Hold':
        diff = result['total_return'] - bh_return
        symbol = '✅' if diff > 0 else '❌'
        print(f"{symbol} {strategy:<20} {diff:+.2f}%p")

print("\n💡 결론:")
print("-" * 70)
if best_strategy[1]['total_return'] > bh_return:
    print(f"✅ {best_strategy[0]} 전략이 Buy-and-Hold보다 {best_strategy[1]['total_return'] - bh_return:.2f}%p 더 높음")
    print(f"   → 모델이 실전에서 유용할 수 있음")
else:
    print(f"❌ 모든 전략이 Buy-and-Hold보다 낮음")
    print(f"   → 모델 개선 필요 (방향 예측, 기간 변경 등)")

print("\n" + "=" * 70)
print("백테스팅 완료!")
print("=" * 70)
