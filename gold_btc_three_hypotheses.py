#!/usr/bin/env python3
"""
금-비트코인 "후행성" 3가지 가설 검증

가설 1: 거시경제적 선행 (90-180일 지연, "100-Day Rule")
가설 2: 단기 가격 추종 (40일 지연, 92% 상관)
가설 3: 역관계 및 자본 순환 (로테이션)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("금-비트코인 '후행성' 3가지 가설 검증")
print("="*80)

# 데이터 로드
df = pd.read_csv('integrated_data_full.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# GOLD & BTC
if 'GOLD' in df.columns:
    df['Gold_Price'] = df['GOLD']
df['BTC_Price'] = df['Close']
df = df[['Date', 'BTC_Price', 'Gold_Price']].dropna()

# 수익률
df['BTC_Return'] = df['BTC_Price'].pct_change() * 100
df['Gold_Return'] = df['Gold_Price'].pct_change() * 100

# 누적 수익률 (YTD 스타일)
df['BTC_Cumulative'] = (df['BTC_Price'] / df['BTC_Price'].iloc[0] - 1) * 100
df['Gold_Cumulative'] = (df['Gold_Price'] / df['Gold_Price'].iloc[0] - 1) * 100

print(f"\n데이터 기간: {df['Date'].min().date()} ~ {df['Date'].max().date()}")
print(f"총 일수: {len(df)}일")

# ========================================
# 가설 1: 거시경제적 선행 (90-180일)
# ========================================
print("\n" + "="*80)
print("가설 1: 거시경제적 선행 (3-6개월 지연)")
print("="*80)

print("\n[가설 1-1] 90일~180일 lag 상관관계")

# 장기 lag 상관관계 테스트
long_lags = range(0, 201, 10)  # 0~200일, 10일 간격
long_lag_corrs = []

for lag in long_lags:
    if lag == 0:
        corr = df['Gold_Return'].corr(df['BTC_Return'])
    else:
        gold_now = df['Gold_Return'][:-lag]
        btc_future = df['BTC_Return'][lag:]
        corr = gold_now.corr(btc_future)
    long_lag_corrs.append(corr)

# 90일, 100일, 180일 특별 체크
lag_90 = long_lag_corrs[9]  # index 9 = 90일
lag_100 = long_lag_corrs[10]  # index 10 = 100일
lag_180 = long_lag_corrs[18]  # index 18 = 180일

print(f"\n수익률 상관관계:")
print(f"  동시 (0일):   {long_lag_corrs[0]:+.4f}")
print(f"  90일 후:      {lag_90:+.4f}")
print(f"  100일 후:     {lag_100:+.4f} ← '100-Day Rule'")
print(f"  180일 후:     {lag_180:+.4f}")

# 최대 상관 지점
max_long_idx = np.argmax(np.abs(long_lag_corrs))
max_long_lag = list(long_lags)[max_long_idx]
max_long_corr = long_lag_corrs[max_long_idx]

print(f"\n최대 상관:")
print(f"  Lag: {max_long_lag}일")
print(f"  상관계수: {max_long_corr:+.4f}")

if 90 <= max_long_lag <= 180:
    print(f"  ✅ 가설 1 지지: {max_long_lag}일은 3-6개월 범위 내")
else:
    print(f"  ❌ 가설 1 미지지: {max_long_lag}일은 범위 밖")

# 가격 추세 분석 (누적 수익률)
print("\n[가설 1-2] 누적 수익률 추세 분석")

# 3개월(90일) 롤링 상관관계
df['Rolling_Corr_90d_Cumulative'] = df['Gold_Cumulative'].rolling(90).corr(df['BTC_Cumulative'])

# 2024-2025 비교
df_2024 = df[(df['Date'] >= '2024-01-01') & (df['Date'] < '2025-01-01')]
df_2025 = df[df['Date'] >= '2025-01-01']

if len(df_2024) > 0:
    gold_ytd_2024 = (df_2024['Gold_Price'].iloc[-1] / df_2024['Gold_Price'].iloc[0] - 1) * 100
    btc_ytd_2024 = (df_2024['BTC_Price'].iloc[-1] / df_2024['BTC_Price'].iloc[0] - 1) * 100
    print(f"\n2024년 YTD:")
    print(f"  GOLD: {gold_ytd_2024:+.1f}%")
    print(f"  BTC:  {btc_ytd_2024:+.1f}%")
    print(f"  차이: {btc_ytd_2024 - gold_ytd_2024:+.1f}%p")

if len(df_2025) > 0:
    gold_ytd_2025 = (df_2025['Gold_Price'].iloc[-1] / df_2025['Gold_Price'].iloc[0] - 1) * 100
    btc_ytd_2025 = (df_2025['BTC_Price'].iloc[-1] / df_2025['BTC_Price'].iloc[0] - 1) * 100
    print(f"\n2025년 YTD (5월 12일 기준 주장: GOLD +20%, BTC +10%):")
    print(f"  GOLD: {gold_ytd_2025:+.1f}%")
    print(f"  BTC:  {btc_ytd_2025:+.1f}%")
    print(f"  차이: {btc_ytd_2025 - gold_ytd_2025:+.1f}%p")

    if gold_ytd_2025 > btc_ytd_2025:
        print(f"  ✅ GOLD가 BTC보다 높음 (가설 1 상황과 일치)")
        print(f"  → BTC는 'coiled spring' 상태?")
    else:
        print(f"  ❌ BTC가 GOLD보다 높음 (가설과 다름)")

# ========================================
# 가설 2: 단기 가격 추종 (40일, 92% 상관)
# ========================================
print("\n" + "="*80)
print("가설 2: 단기 가격 추종 (40일 지연, 92% 상관)")
print("="*80)

print("\n[가설 2-1] 40일 lag 상관관계 검증")

# 정밀 lag 분석 (30~50일)
precise_lags = range(30, 51)
precise_corrs = []

for lag in precise_lags:
    gold_now = df['Gold_Return'][:-lag]
    btc_future = df['BTC_Return'][lag:]
    corr = gold_now.corr(btc_future)
    precise_corrs.append(corr)

lag_40_idx = 10  # 40 - 30 = 10
lag_40_corr = precise_corrs[lag_40_idx]

print(f"\n40일 lag 상관관계: {lag_40_corr:+.4f}")
print(f"주장: 92% (0.92)")

if lag_40_corr >= 0.92:
    print(f"✅ 가설 2 완전 지지: {lag_40_corr:.4f} >= 0.92")
elif lag_40_corr >= 0.50:
    print(f"⚠️ 가설 2 부분 지지: {lag_40_corr:.4f} < 0.92 하지만 양의 상관")
else:
    print(f"❌ 가설 2 미지지: {lag_40_corr:.4f} << 0.92")

# 30-50일 범위 최대값
max_precise_idx = np.argmax(precise_corrs)
max_precise_lag = 30 + max_precise_idx
max_precise_corr = precise_corrs[max_precise_idx]

print(f"\n30~50일 범위 최대 상관:")
print(f"  Lag: {max_precise_lag}일")
print(f"  상관계수: {max_precise_corr:+.4f}")

# 2025년만 분석
print("\n[가설 2-2] 2025년 40일 추종 패턴")

if len(df_2025) > 40:
    gold_2025 = df_2025['Gold_Return'].values
    btc_2025 = df_2025['BTC_Return'].values

    gold_now_2025 = gold_2025[:-40]
    btc_future_2025 = btc_2025[40:]

    corr_2025_40d = np.corrcoef(gold_now_2025, btc_future_2025)[0, 1]

    print(f"2025년 40일 lag 상관: {corr_2025_40d:+.4f}")
    print(f"주장: 2025년에 92% 이상")

    if corr_2025_40d >= 0.92:
        print(f"✅ 2025년 가설 지지!")
    else:
        print(f"❌ 2025년 가설 미지지")

# ========================================
# 가설 3: 역관계 및 자본 순환
# ========================================
print("\n" + "="*80)
print("가설 3: 역관계 및 자본 순환 (로테이션)")
print("="*80)

print("\n[가설 3-1] 금 조정 후 비트코인 가속화")

# 금 피크 찾기 (상대 고점)
gold_prices = df['Gold_Price'].values
gold_peaks, _ = find_peaks(gold_prices, distance=30, prominence=50)

print(f"\n금 피크 (고점) 발견: {len(gold_peaks)}개")

# 각 피크 후 30일간 BTC 수익률
btc_returns_after_gold_peak = []
for peak_idx in gold_peaks:
    if peak_idx + 30 < len(df):
        btc_30d = (df['BTC_Price'].iloc[peak_idx+30] / df['BTC_Price'].iloc[peak_idx] - 1) * 100
        btc_returns_after_gold_peak.append(btc_30d)

if len(btc_returns_after_gold_peak) > 0:
    avg_btc_after_gold_peak = np.mean(btc_returns_after_gold_peak)

    # 랜덤 비교
    random_30d = []
    for i in range(len(df) - 30):
        ret = (df['BTC_Price'].iloc[i+30] / df['BTC_Price'].iloc[i] - 1) * 100
        random_30d.append(ret)
    avg_random = np.mean(random_30d)

    print(f"금 피크 후 30일 BTC 평균: {avg_btc_after_gold_peak:+.2f}%")
    print(f"랜덤 30일 BTC 평균:      {avg_random:+.2f}%")
    print(f"차이:                    {avg_btc_after_gold_peak - avg_random:+.2f}%p")

    if avg_btc_after_gold_peak > avg_random + 1:
        print(f"✅ 가설 3 지지: 금 피크 후 BTC가 더 상승!")
    elif avg_btc_after_gold_peak > avg_random:
        print(f"⚠️ 가설 3 약한 지지")
    else:
        print(f"❌ 가설 3 미지지")

print("\n[가설 3-2] 금 하락 시 BTC 급등")

# 금 급락일 정의 (하위 5%)
gold_crash_threshold = df['Gold_Return'].quantile(0.05)
gold_crash_days = df[df['Gold_Return'] < gold_crash_threshold]

print(f"\n금 급락일 (<{gold_crash_threshold:.2f}%): {len(gold_crash_days)}일")
print(f"  평균 GOLD: {gold_crash_days['Gold_Return'].mean():.2f}%")
print(f"  평균 BTC:  {gold_crash_days['BTC_Return'].mean():+.2f}%")

# 다음날 BTC 반응
gold_crash_indices = gold_crash_days.index
btc_next_day_returns = []

for idx in gold_crash_indices:
    if idx + 1 < len(df):
        btc_next = df['BTC_Return'].iloc[idx + 1]
        btc_next_day_returns.append(btc_next)

if len(btc_next_day_returns) > 0:
    avg_btc_next = np.mean(btc_next_day_returns)
    print(f"  다음날 BTC: {avg_btc_next:+.2f}%")

    if avg_btc_next > 0:
        print(f"  ✅ 로테이션: 금 급락 → 다음날 BTC 상승")
    else:
        print(f"  ❌ 로테이션 없음")

print("\n[가설 3-3] 금 과열 후 자본 순환")

# 금 급등일 (상위 5%)
gold_surge_threshold = df['Gold_Return'].quantile(0.95)
gold_surge_days = df[df['Gold_Return'] > gold_surge_threshold]

print(f"\n금 급등일 (>{gold_surge_threshold:.2f}%): {len(gold_surge_days)}일")

# 급등 이후 7일간 금 vs BTC 수익률
gold_surge_indices = gold_surge_days.index
gold_7d_after_surge = []
btc_7d_after_surge = []

for idx in gold_surge_indices:
    if idx + 7 < len(df):
        gold_7d = (df['Gold_Price'].iloc[idx+7] / df['Gold_Price'].iloc[idx] - 1) * 100
        btc_7d = (df['BTC_Price'].iloc[idx+7] / df['BTC_Price'].iloc[idx] - 1) * 100
        gold_7d_after_surge.append(gold_7d)
        btc_7d_after_surge.append(btc_7d)

if len(gold_7d_after_surge) > 0:
    avg_gold_7d = np.mean(gold_7d_after_surge)
    avg_btc_7d = np.mean(btc_7d_after_surge)

    print(f"급등 후 7일:")
    print(f"  GOLD 평균: {avg_gold_7d:+.2f}%")
    print(f"  BTC 평균:  {avg_btc_7d:+.2f}%")

    if avg_btc_7d > avg_gold_7d:
        print(f"  ✅ 로테이션: 금 과열 후 BTC로 자본 이동 ({avg_btc_7d - avg_gold_7d:+.2f}%p)")
    else:
        print(f"  ❌ 로테이션 없음")

# ========================================
# 2025년 ETF 자금 흐름 (간접 검증)
# ========================================
print("\n[가설 3-4] 2025년 상대 성과")

if len(df_2025) > 0:
    # 월별 승자
    df_2025_copy = df_2025.copy()
    df_2025_copy['Month'] = df_2025_copy['Date'].dt.to_period('M')

    monthly_winners = []
    for month, group in df_2025_copy.groupby('Month'):
        if len(group) > 1:
            gold_ret = (group['Gold_Price'].iloc[-1] / group['Gold_Price'].iloc[0] - 1) * 100
            btc_ret = (group['BTC_Price'].iloc[-1] / group['BTC_Price'].iloc[0] - 1) * 100
            winner = 'GOLD' if gold_ret > btc_ret else 'BTC'
            monthly_winners.append({
                'Month': str(month),
                'GOLD': gold_ret,
                'BTC': btc_ret,
                'Winner': winner
            })

    winners_df = pd.DataFrame(monthly_winners)

    print(f"\n2025년 월별 승자:")
    for _, row in winners_df.iterrows():
        marker = '🥇' if row['Winner'] == 'GOLD' else '🥈'
        print(f"  {row['Month']}: GOLD {row['GOLD']:+5.1f}%, BTC {row['BTC']:+5.1f}% {marker} {row['Winner']}")

    gold_wins = (winners_df['Winner'] == 'GOLD').sum()
    btc_wins = (winners_df['Winner'] == 'BTC').sum()

    print(f"\n승수: GOLD {gold_wins}개월, BTC {btc_wins}개월")

    if btc_wins > gold_wins:
        print(f"✅ 2025년 BTC가 더 많이 승리 (로테이션 시사)")
    else:
        print(f"⚠️ 2025년 GOLD가 우세")

# ========================================
# 시각화
# ========================================
print("\n" + "="*80)
print("시각화 생성 중...")
print("="*80)

fig = plt.figure(figsize=(22, 16))

# (1) 가설 1: 장기 lag 상관관계
ax1 = plt.subplot(4, 4, 1)
ax1.plot(long_lags, long_lag_corrs, marker='o', linewidth=2, color='purple')
ax1.axhline(0, color='black', linestyle='--', linewidth=1)
ax1.axvline(90, color='red', linestyle='--', linewidth=1, alpha=0.5, label='90일')
ax1.axvline(100, color='red', linestyle='--', linewidth=1.5, label='100일')
ax1.axvline(180, color='red', linestyle='--', linewidth=1, alpha=0.5, label='180일')
ax1.set_xlabel('Lag (일)', fontweight='bold')
ax1.set_ylabel('상관계수', fontweight='bold')
ax1.set_title('가설 1: 거시경제적 선행 (90-180일)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# (2) 가설 2: 40일 부근 정밀 분석
ax2 = plt.subplot(4, 4, 2)
ax2.plot(precise_lags, precise_corrs, marker='o', linewidth=2, color='green')
ax2.axhline(0, color='black', linestyle='--', linewidth=1)
ax2.axhline(0.92, color='red', linestyle='--', linewidth=1.5, label='92% (주장)')
ax2.axvline(40, color='red', linestyle='--', linewidth=1.5, label='40일')
ax2.set_xlabel('Lag (일)', fontweight='bold')
ax2.set_ylabel('상관계수', fontweight='bold')
ax2.set_title(f'가설 2: 40일 추종 (실제: {lag_40_corr:.3f})', fontsize=12, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# (3) 2024 vs 2025 YTD 비교
ax3 = plt.subplot(4, 4, 3)
if len(df_2024) > 0 and len(df_2025) > 0:
    years = ['2024', '2025']
    gold_ytds = [gold_ytd_2024, gold_ytd_2025]
    btc_ytds = [btc_ytd_2024, btc_ytd_2025]

    x = np.arange(len(years))
    width = 0.35

    bars1 = ax3.bar(x - width/2, gold_ytds, width, label='GOLD', color='gold', alpha=0.7)
    bars2 = ax3.bar(x + width/2, btc_ytds, width, label='BTC', color='orange', alpha=0.7)

    ax3.set_ylabel('YTD 수익률 (%)', fontweight='bold')
    ax3.set_title('YTD 수익률 비교 (가설 1 검증)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(years)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(0, color='black', linewidth=0.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:+.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                    fontweight='bold', fontsize=9)

# (4) 금 피크 후 BTC 반응
ax4 = plt.subplot(4, 4, 4)
if len(btc_returns_after_gold_peak) > 0:
    ax4.hist(btc_returns_after_gold_peak, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
    ax4.axvline(avg_btc_after_gold_peak, color='red', linestyle='--', linewidth=2,
               label=f'평균: {avg_btc_after_gold_peak:+.1f}%')
    ax4.axvline(avg_random, color='blue', linestyle='--', linewidth=2,
               label=f'랜덤: {avg_random:+.1f}%')
    ax4.set_xlabel('30일 후 BTC 수익률 (%)', fontweight='bold')
    ax4.set_ylabel('빈도', fontweight='bold')
    ax4.set_title('가설 3: 금 피크 후 BTC 반응', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

# (5) 누적 수익률 추이
ax5 = plt.subplot(4, 4, (5, 6))
ax5.plot(df['Date'], df['Gold_Cumulative'], label='GOLD', linewidth=2, color='gold')
ax5.plot(df['Date'], df['BTC_Cumulative'], label='BTC', linewidth=2, color='orange')
ax5.axhline(0, color='black', linewidth=0.5)
ax5.set_ylabel('누적 수익률 (%)', fontweight='bold')
ax5.set_title('누적 수익률 추이 (전체 기간)', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.tick_params(axis='x', rotation=45)

# (6) 2025년 상세
ax6 = plt.subplot(4, 4, (7, 8))
if len(df_2025) > 0:
    df_2025_reset = df_2025.copy()
    df_2025_reset['Gold_2025_Cum'] = (df_2025_reset['Gold_Price'] / df_2025_reset['Gold_Price'].iloc[0] - 1) * 100
    df_2025_reset['BTC_2025_Cum'] = (df_2025_reset['BTC_Price'] / df_2025_reset['BTC_Price'].iloc[0] - 1) * 100

    ax6.plot(df_2025_reset['Date'], df_2025_reset['Gold_2025_Cum'], label='GOLD', linewidth=2, color='gold')
    ax6.plot(df_2025_reset['Date'], df_2025_reset['BTC_2025_Cum'], label='BTC', linewidth=2, color='orange')
    ax6.axhline(0, color='black', linewidth=0.5)
    ax6.axhline(20, color='gold', linestyle='--', linewidth=1, alpha=0.5, label='GOLD +20% (주장)')
    ax6.axhline(10, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='BTC +10% (주장)')
    ax6.set_ylabel('누적 수익률 (%)', fontweight='bold')
    ax6.set_title('2025년 누적 수익률 (가설 1 검증)', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.tick_params(axis='x', rotation=45)

# (7) 금 급락 vs BTC
ax7 = plt.subplot(4, 4, 9)
crash_data = [
    df['BTC_Return'].mean(),
    gold_crash_days['BTC_Return'].mean(),
    avg_btc_next if len(btc_next_day_returns) > 0 else 0
]
labels_crash = ['전체 평균', '금 급락일\n동시', '금 급락\n다음날']
colors_crash = ['gray', 'orange', 'green']
bars = ax7.bar(labels_crash, crash_data, color=colors_crash, alpha=0.7)
ax7.axhline(0, color='black', linewidth=1)
for bar, val in zip(bars, crash_data):
    ax7.text(bar.get_x() + bar.get_width()/2, val, f'{val:+.3f}%',
            ha='center', va='bottom' if val > 0 else 'top', fontweight='bold', fontsize=9)
ax7.set_ylabel('BTC 평균 수익률 (%)', fontweight='bold')
ax7.set_title('가설 3: 금 급락 시 BTC 반응', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')

# (8) 금 급등 후 7일 비교
ax8 = plt.subplot(4, 4, 10)
if len(gold_7d_after_surge) > 0:
    surge_data = [avg_gold_7d, avg_btc_7d]
    labels_surge = ['GOLD\n(7일 후)', 'BTC\n(7일 후)']
    colors_surge = ['gold', 'orange']
    bars = ax8.bar(labels_surge, surge_data, color=colors_surge, alpha=0.7)
    ax8.axhline(0, color='black', linewidth=1)
    for bar, val in zip(bars, surge_data):
        ax8.text(bar.get_x() + bar.get_width()/2, val, f'{val:+.2f}%',
                ha='center', va='bottom' if val > 0 else 'top', fontweight='bold', fontsize=10)
    ax8.set_ylabel('평균 수익률 (%)', fontweight='bold')
    ax8.set_title('가설 3: 금 급등 후 자본 순환', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')

# (9) 2025년 월별 승자
ax9 = plt.subplot(4, 4, 11)
if len(df_2025) > 0 and len(monthly_winners) > 0:
    months_display = [w['Month'][-2:] for w in monthly_winners]  # MM만
    gold_rets = [w['GOLD'] for w in monthly_winners]
    btc_rets = [w['BTC'] for w in monthly_winners]

    x = np.arange(len(months_display))
    width = 0.35

    bars1 = ax9.bar(x - width/2, gold_rets, width, label='GOLD', color='gold', alpha=0.7)
    bars2 = ax9.bar(x + width/2, btc_rets, width, label='BTC', color='orange', alpha=0.7)

    ax9.set_ylabel('월간 수익률 (%)', fontweight='bold')
    ax9.set_title('2025년 월별 수익률 (가설 3)', fontsize=12, fontweight='bold')
    ax9.set_xticks(x)
    ax9.set_xticklabels(months_display, fontsize=8)
    ax9.legend()
    ax9.grid(True, alpha=0.3, axis='y')
    ax9.axhline(0, color='black', linewidth=0.5)

# (10) 롤링 상관관계
ax10 = plt.subplot(4, 4, 12)
ax10.plot(df['Date'], df['Rolling_Corr_90d_Cumulative'], linewidth=2, color='purple')
ax10.axhline(0, color='black', linestyle='--', linewidth=1)
ax10.set_ylabel('90일 롤링 상관계수', fontweight='bold')
ax10.set_title('누적 수익률 롤링 상관 (90일)', fontsize=12, fontweight='bold')
ax10.grid(True, alpha=0.3)
ax10.tick_params(axis='x', rotation=45)

# (11-12) 종합 요약
ax11 = plt.subplot(4, 4, (13, 16))
ax11.axis('off')

summary = f"""
【3가지 가설 검증 결과】

1. 가설 1: 거시경제적 선행 (90-180일)

   100일 lag 상관: {lag_100:+.4f}
   최대 상관 lag: {max_long_lag}일 ({max_long_corr:+.4f})
   {'✅ 지지' if 90 <= max_long_lag <= 180 else '❌ 미지지'}

   2025 YTD:
   GOLD: {gold_ytd_2025:+.1f}% vs BTC: {btc_ytd_2025:+.1f}%
   {'✅ GOLD 우세 (가설 상황)' if gold_ytd_2025 > btc_ytd_2025 else '❌ BTC 우세'}

2. 가설 2: 단기 추종 (40일, 92% 상관)

   40일 lag 상관: {lag_40_corr:+.4f}
   주장: 0.92 (92%)
   {'✅ 지지' if lag_40_corr >= 0.92 else '⚠️ 부분 지지' if lag_40_corr >= 0.50 else '❌ 미지지'}

   30~50일 최대: {max_precise_lag}일 ({max_precise_corr:+.4f})

3. 가설 3: 역관계 & 로테이션

   금 피크 후 BTC: {avg_btc_after_gold_peak:+.2f}%
   랜덤 평균:      {avg_random:+.2f}%
   차이:          {avg_btc_after_gold_peak - avg_random:+.2f}%p
   {'✅ 로테이션 존재' if avg_btc_after_gold_peak > avg_random + 1 else '⚠️ 약함'}

   금 급락 → 다음날 BTC: {avg_btc_next if len(btc_next_day_returns) > 0 else 0:+.2f}%
   금 급등 후 7일: GOLD {avg_gold_7d:+.1f}% vs BTC {avg_btc_7d:+.1f}%
   {'✅ BTC로 순환' if avg_btc_7d > avg_gold_7d else '❌ 순환 없음'}

【종합 결론】

가설 1 (100-Day Rule):    {'✅ 지지' if 90 <= max_long_lag <= 180 else '❌ 미지지'}
가설 2 (40일, 92%):      {'✅ 지지' if lag_40_corr >= 0.80 else '⚠️ 부분' if lag_40_corr >= 0.40 else '❌ 미지지'}
가설 3 (로테이션):       {'✅ 지지' if avg_btc_after_gold_peak > avg_random + 1 else '⚠️ 약함'}

가장 강한 증거: {'가설 ' + str(np.argmax([abs(max_long_corr) if 90 <= max_long_lag <= 180 else 0, lag_40_corr, avg_btc_after_gold_peak - avg_random]) + 1)}
"""

ax11.text(0.05, 0.95, summary, fontsize=9, verticalalignment='top',
         family='monospace', transform=ax11.transAxes)
ax11.set_title('종합 요약', fontsize=14, fontweight='bold', loc='left')

plt.tight_layout()
plt.savefig('gold_btc_three_hypotheses.png', dpi=300, bbox_inches='tight')
print("✅ 저장: gold_btc_three_hypotheses.png")

# 결과 저장
results = {
    '가설': [
        '가설 1: 100일 lag',
        '가설 1: 최대 상관 lag',
        '가설 1: 2025 GOLD vs BTC',
        '가설 2: 40일 lag 상관',
        '가설 2: 30-50일 최대',
        '가설 3: 금 피크 후 BTC',
        '가설 3: 금 급락 후 BTC',
        '가설 3: 금 급등 후 순환',
    ],
    '결과': [
        f'{lag_100:+.4f}',
        f'{max_long_lag}일 ({max_long_corr:+.4f})',
        f'GOLD {gold_ytd_2025:+.1f}% vs BTC {btc_ytd_2025:+.1f}%',
        f'{lag_40_corr:+.4f} (주장: 0.92)',
        f'{max_precise_lag}일 ({max_precise_corr:+.4f})',
        f'{avg_btc_after_gold_peak - avg_random:+.2f}%p',
        f'{avg_btc_next if len(btc_next_day_returns) > 0 else 0:+.2f}%',
        f'GOLD {avg_gold_7d:+.1f}% vs BTC {avg_btc_7d:+.1f}%',
    ],
    '판정': [
        '✅' if abs(lag_100) > 0.10 else '⚠️',
        '✅' if 90 <= max_long_lag <= 180 else '❌',
        '✅' if gold_ytd_2025 > btc_ytd_2025 else '❌',
        '✅' if lag_40_corr >= 0.92 else '⚠️' if lag_40_corr >= 0.50 else '❌',
        '✅' if 35 <= max_precise_lag <= 45 else '⚠️',
        '✅' if avg_btc_after_gold_peak > avg_random + 1 else '⚠️',
        '✅' if avg_btc_next > 0 else '❌',
        '✅' if avg_btc_7d > avg_gold_7d else '❌',
    ]
}

results_df = pd.DataFrame(results)
results_df.to_csv('gold_btc_three_hypotheses_results.csv', index=False)
print("✅ 저장: gold_btc_three_hypotheses_results.csv")

print("\n" + "="*80)
print("3가지 가설 검증 완료!")
print("="*80)
