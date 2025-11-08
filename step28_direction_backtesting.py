import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("방향 예측 모델 백테스팅")
print("="*80)

# 1. 데이터 로드
df = pd.read_csv('integrated_data_full.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# 2. 제외할 특성 (데이터 누수 가능성)
EXCLUDE_FEATURES = [
    'Close', 'High', 'Low', 'Volume', 'Date',
    'EMA_12', 'EMA_26', 'SMA_20', 'SMA_50', 'SMA_200',
    'BB_upper', 'BB_middle', 'BB_lower',
    'ATR', 'ADX', 'OBV', 'VWAP'
]

# 3. 타겟 생성: 다음날 방향 (상승=1, 하락=0)
THRESHOLD = 0.0  # 0%보다 크면 상승
df['next_day_return'] = (df['Close'].shift(-1) / df['Close'] - 1) * 100
df['direction'] = (df['next_day_return'] > THRESHOLD).astype(int)
df = df[:-1].copy()  # 마지막 행 제거

# 4. 특성 선택
all_features = [col for col in df.columns if col not in EXCLUDE_FEATURES + ['next_day_return', 'direction']]
X = df[all_features]
y = df['direction']

print(f"\n총 샘플 수: {len(df)}")
print(f"특성 수: {len(all_features)}")
print(f"상승 비율: {y.mean()*100:.2f}%")

# 5. 백테스팅 함수
def backtest_direction_model(df, X, y, train_end, test_start, test_end, test_name):
    """방향 예측 모델로 백테스팅"""

    # 훈련/테스트 분할
    train_mask = df['Date'] <= train_end
    test_mask = (df['Date'] >= test_start) & (df['Date'] <= test_end)

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    test_df = df[test_mask].copy()

    print(f"\n{'='*60}")
    print(f"{test_name}")
    print(f"{'='*60}")
    print(f"훈련 기간: {df[train_mask]['Date'].min().date()} ~ {df[train_mask]['Date'].max().date()}")
    print(f"테스트 기간: {test_df['Date'].min().date()} ~ {test_df['Date'].max().date()}")
    print(f"훈련 샘플: {len(X_train)}, 테스트 샘플: {len(X_test)}")

    # 모델 훈련
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # 예측
    y_pred = model.predict(X_test)

    # 정확도
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n방향 예측 정확도: {accuracy*100:.2f}%")
    print(f"상승 예측 비율: {y_pred.mean()*100:.2f}%")
    print(f"실제 상승 비율: {y_test.mean()*100:.2f}%")

    # 백테스팅
    test_df = test_df.copy()
    test_df['predicted_direction'] = y_pred
    test_df['actual_direction'] = y_test.values

    # 전략: 상승 예측시 매수, 하락 예측시 현금 보유
    initial_capital = 10000
    cash = initial_capital
    btc_holdings = 0
    portfolio_values = []

    for idx, row in test_df.iterrows():
        current_price = row['Close']
        next_price = current_price * (1 + row['next_day_return'] / 100)
        predicted_up = row['predicted_direction'] == 1

        if predicted_up:
            # 상승 예측: 전액 매수
            if cash > 0:
                btc_holdings = cash / current_price
                cash = 0
            # 다음날 가격으로 포트폴리오 가치 계산
            portfolio_value = btc_holdings * next_price
        else:
            # 하락 예측: 전액 현금 보유
            if btc_holdings > 0:
                cash = btc_holdings * current_price
                btc_holdings = 0
            portfolio_value = cash

        portfolio_values.append(portfolio_value)

    test_df['portfolio_value'] = portfolio_values

    # Buy-and-Hold 전략
    bnh_initial_btc = initial_capital / test_df.iloc[0]['Close']
    test_df['bnh_value'] = bnh_initial_btc * test_df['Close']

    # 성과 지표 계산
    final_value = test_df['portfolio_value'].iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100

    bnh_final = test_df['bnh_value'].iloc[-1]
    bnh_return = (bnh_final / initial_capital - 1) * 100

    # 일별 수익률
    test_df['daily_return'] = test_df['portfolio_value'].pct_change()
    test_df['bnh_daily_return'] = test_df['bnh_value'].pct_change()

    # 연율화된 수익률 및 변동성
    days = len(test_df)
    annual_return = (1 + total_return/100) ** (365/days) - 1
    annual_vol = test_df['daily_return'].std() * np.sqrt(365)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0

    bnh_annual_return = (1 + bnh_return/100) ** (365/days) - 1
    bnh_annual_vol = test_df['bnh_daily_return'].std() * np.sqrt(365)
    bnh_sharpe = bnh_annual_return / bnh_annual_vol if bnh_annual_vol > 0 else 0

    # 최대 낙폭
    cummax = test_df['portfolio_value'].cummax()
    drawdown = (test_df['portfolio_value'] / cummax - 1) * 100
    max_dd = drawdown.min()

    bnh_cummax = test_df['bnh_value'].cummax()
    bnh_drawdown = (test_df['bnh_value'] / bnh_cummax - 1) * 100
    bnh_max_dd = bnh_drawdown.min()

    # 승률
    correct_predictions = (test_df['predicted_direction'] == test_df['actual_direction']).sum()
    win_rate = correct_predictions / len(test_df) * 100

    # 거래 빈도
    trades = (test_df['predicted_direction'].diff() != 0).sum()

    print(f"\n{'='*60}")
    print("백테스팅 결과")
    print(f"{'='*60}")
    print(f"\n방향 예측 전략:")
    print(f"  최종 자산: ${final_value:,.2f}")
    print(f"  총 수익률: {total_return:+.2f}%")
    print(f"  연율화 수익률: {annual_return*100:+.2f}%")
    print(f"  샤프 비율: {sharpe:.3f}")
    print(f"  최대 낙폭: {max_dd:.2f}%")
    print(f"  변동성: {annual_vol*100:.2f}%")
    print(f"  승률: {win_rate:.2f}%")
    print(f"  거래 횟수: {trades}")

    print(f"\nBuy-and-Hold:")
    print(f"  최종 자산: ${bnh_final:,.2f}")
    print(f"  총 수익률: {bnh_return:+.2f}%")
    print(f"  연율화 수익률: {bnh_annual_return*100:+.2f}%")
    print(f"  샤프 비율: {bnh_sharpe:.3f}")
    print(f"  최대 낙폭: {bnh_max_dd:.2f}%")
    print(f"  변동성: {bnh_annual_vol*100:.2f}%")

    return {
        'test_df': test_df,
        'accuracy': accuracy,
        'total_return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'bnh_return': bnh_return,
        'bnh_sharpe': bnh_sharpe,
        'bnh_max_dd': bnh_max_dd,
        'trades': trades
    }

# 6. 두 가지 백테스팅 수행
# (1) 전체 기간 (step26과 동일)
result1 = backtest_direction_model(
    df, X, y,
    train_end=pd.to_datetime('2024-05-16'),
    test_start=pd.to_datetime('2024-05-17'),
    test_end=pd.to_datetime('2025-10-13'),
    test_name="전체 기간 백테스팅 (2024-05-17 ~ 2025-10-13)"
)

# (2) 2025년만 (step27과 동일)
result2 = backtest_direction_model(
    df, X, y,
    train_end=pd.to_datetime('2024-12-31'),
    test_start=pd.to_datetime('2025-01-01'),
    test_end=pd.to_datetime('2025-10-13'),
    test_name="2025년만 백테스팅 (2025-01-01 ~ 2025-10-13)"
)

# 7. 시각화
fig = plt.figure(figsize=(16, 12))

# 첫 번째 행: 전체 기간
# (1-1) 포트폴리오 가치
ax1 = plt.subplot(3, 3, 1)
ax1.plot(result1['test_df']['Date'], result1['test_df']['portfolio_value'],
         label='방향 예측', linewidth=2, color='blue')
ax1.plot(result1['test_df']['Date'], result1['test_df']['bnh_value'],
         label='Buy-and-Hold', linewidth=2, color='gray', alpha=0.7)
ax1.set_title('포트폴리오 가치 - 전체 기간 (2024-05 ~ 2025-10)', fontsize=11, fontweight='bold')
ax1.set_ylabel('자산 ($)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# (1-2) 성과 비교
ax2 = plt.subplot(3, 3, 2)
strategies = ['방향 예측', 'Buy-and-Hold']
returns = [result1['total_return'], result1['bnh_return']]
colors_bar = ['blue' if r > 0 else 'red' for r in returns]
bars = ax2.barh(strategies, returns, color=colors_bar, alpha=0.7)
for i, (bar, val) in enumerate(zip(bars, returns)):
    ax2.text(val, i, f'{val:+.2f}%', va='center', ha='left' if val > 0 else 'right', fontweight='bold')
ax2.set_title('총 수익률 비교 - 전체 기간', fontsize=11, fontweight='bold')
ax2.set_xlabel('수익률 (%)')
ax2.axvline(0, color='black', linewidth=0.8)
ax2.grid(True, alpha=0.3, axis='x')

# (1-3) 샤프 비율
ax3 = plt.subplot(3, 3, 3)
sharpe_vals = [result1['sharpe'], result1['bnh_sharpe']]
colors_sharpe = ['blue', 'gray']
bars = ax3.barh(strategies, sharpe_vals, color=colors_sharpe, alpha=0.7)
for i, (bar, val) in enumerate(zip(bars, sharpe_vals)):
    ax3.text(val, i, f'{val:.3f}', va='center', ha='left', fontweight='bold')
ax3.set_title('샤프 비율 - 전체 기간', fontsize=11, fontweight='bold')
ax3.set_xlabel('샤프 비율')
ax3.grid(True, alpha=0.3, axis='x')

# 두 번째 행: 2025년만
# (2-1) 포트폴리오 가치
ax4 = plt.subplot(3, 3, 4)
ax4.plot(result2['test_df']['Date'], result2['test_df']['portfolio_value'],
         label='방향 예측', linewidth=2, color='blue')
ax4.plot(result2['test_df']['Date'], result2['test_df']['bnh_value'],
         label='Buy-and-Hold', linewidth=2, color='gray', alpha=0.7)
ax4.set_title('포트폴리오 가치 - 2025년만', fontsize=11, fontweight='bold')
ax4.set_ylabel('자산 ($)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# (2-2) 성과 비교
ax5 = plt.subplot(3, 3, 5)
returns2 = [result2['total_return'], result2['bnh_return']]
colors_bar2 = ['blue' if r > 0 else 'red' for r in returns2]
bars = ax5.barh(strategies, returns2, color=colors_bar2, alpha=0.7)
for i, (bar, val) in enumerate(zip(bars, returns2)):
    ax5.text(val, i, f'{val:+.2f}%', va='center', ha='left' if val > 0 else 'right', fontweight='bold')
ax5.set_title('총 수익률 비교 - 2025년만', fontsize=11, fontweight='bold')
ax5.set_xlabel('수익률 (%)')
ax5.axvline(0, color='black', linewidth=0.8)
ax5.grid(True, alpha=0.3, axis='x')

# (2-3) 샤프 비율
ax6 = plt.subplot(3, 3, 6)
sharpe_vals2 = [result2['sharpe'], result2['bnh_sharpe']]
bars = ax6.barh(strategies, sharpe_vals2, color=colors_sharpe, alpha=0.7)
for i, (bar, val) in enumerate(zip(bars, sharpe_vals2)):
    ax6.text(val, i, f'{val:.3f}', va='center', ha='left' if val > 0 else 'right', fontweight='bold')
ax6.set_title('샤프 비율 - 2025년만', fontsize=11, fontweight='bold')
ax6.set_xlabel('샤프 비율')
ax6.grid(True, alpha=0.3, axis='x')

# 세 번째 행: 추가 분석
# (3-1) 최대 낙폭
ax7 = plt.subplot(3, 3, 7)
max_dds_1 = [result1['max_dd'], result1['bnh_max_dd']]
max_dds_2 = [result2['max_dd'], result2['bnh_max_dd']]
x = np.arange(len(strategies))
width = 0.35
bars1 = ax7.bar(x - width/2, max_dds_1, width, label='전체 기간', alpha=0.7, color='steelblue')
bars2 = ax7.bar(x + width/2, max_dds_2, width, label='2025년만', alpha=0.7, color='coral')
ax7.set_ylabel('최대 낙폭 (%)')
ax7.set_title('최대 낙폭 비교', fontsize=11, fontweight='bold')
ax7.set_xticks(x)
ax7.set_xticklabels(strategies)
ax7.legend()
ax7.grid(True, alpha=0.3, axis='y')
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# (3-2) 승률
ax8 = plt.subplot(3, 3, 8)
win_rates = [result1['win_rate'], result2['win_rate']]
colors_wr = ['green' if w > 50 else 'red' for w in win_rates]
bars = ax8.barh(['전체 기간', '2025년만'], win_rates, color=colors_wr, alpha=0.7)
for i, (bar, val) in enumerate(zip(bars, win_rates)):
    ax8.text(val, i, f'{val:.2f}%', va='center', ha='left', fontweight='bold')
ax8.set_title('방향 예측 승률', fontsize=11, fontweight='bold')
ax8.set_xlabel('승률 (%)')
ax8.axvline(50, color='black', linewidth=0.8, linestyle='--', label='50% 기준')
ax8.legend()
ax8.grid(True, alpha=0.3, axis='x')

# (3-3) 거래 빈도
ax9 = plt.subplot(3, 3, 9)
trade_counts = [result1['trades'], result2['trades']]
test_days = [len(result1['test_df']), len(result2['test_df'])]
trade_freq = [t/d*100 for t, d in zip(trade_counts, test_days)]
bars = ax9.barh(['전체 기간', '2025년만'], trade_freq, color='purple', alpha=0.7)
for i, (bar, val, count) in enumerate(zip(bars, trade_freq, trade_counts)):
    ax9.text(val, i, f'{val:.1f}% ({count}회)', va='center', ha='left', fontweight='bold')
ax9.set_title('거래 빈도', fontsize=11, fontweight='bold')
ax9.set_xlabel('거래 비율 (%)')
ax9.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('direction_backtesting_results.png', dpi=300, bbox_inches='tight')
print(f"\n시각화 저장 완료: direction_backtesting_results.png")

# 8. 최종 요약
print(f"\n{'='*80}")
print("최종 요약: 방향 예측 vs ElasticNet 가격 예측")
print(f"{'='*80}")

print(f"\n전체 기간 (2024-05-17 ~ 2025-10-13):")
print(f"  방향 예측: {result1['total_return']:+.2f}% (샤프 {result1['sharpe']:.3f})")
print(f"  Buy-and-Hold: {result1['bnh_return']:+.2f}% (샤프 {result1['bnh_sharpe']:.3f})")
print(f"  ElasticNet (step26 참고): +89.94% (샤프 1.97, Threshold 1%)")

print(f"\n2025년만 (2025-01-01 ~ 2025-10-13):")
print(f"  방향 예측: {result2['total_return']:+.2f}% (샤프 {result2['sharpe']:.3f})")
print(f"  Buy-and-Hold: {result2['bnh_return']:+.2f}% (샤프 {result2['bnh_sharpe']:.3f})")
print(f"  ElasticNet (step27 참고): -7.01% (샤프 -0.51, Threshold 1%)")

print(f"\n핵심 발견:")
if result2['total_return'] < 0:
    print(f"  ❌ 방향 예측 모델도 2025년에 실패 ({result2['total_return']:+.2f}%)")
else:
    print(f"  ✅ 방향 예측 모델은 2025년에 성공 ({result2['total_return']:+.2f}%)")

if result2['total_return'] < result2['bnh_return']:
    print(f"  ❌ Buy-and-Hold보다 낮은 수익률")
else:
    print(f"  ✅ Buy-and-Hold보다 높은 수익률")

print(f"\n  방향 예측 정확도: {result2['accuracy']*100:.2f}% (2025년)")
print(f"  거래 승률: {result2['win_rate']:.2f}% (2025년)")

print("="*80)
