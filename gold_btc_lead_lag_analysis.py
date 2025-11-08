#!/usr/bin/env python3
"""
금-비트코인 선행/로테이션 패턴 검증

가설 1: 금 선행 상승 패턴 (Gold Leads Bitcoin)
- 금이 먼저 상승 → 70일 후 비트코인 상승
- 금 신고가 → 비트코인 모멘텀

가설 2: 금 하락 시 로테이션 (Rotation)
- 금 하락 → 비트코인 상승
- 자본 이동 현상
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("금-비트코인 선행/로테이션 패턴 검증")
print("="*80)

# 데이터 로드
df = pd.read_csv('integrated_data_full.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# GOLD 컬럼
if 'GOLD' in df.columns:
    df['Gold_Price'] = df['GOLD']
elif 'gold_close' in df.columns:
    df['Gold_Price'] = df['gold_close']

df['BTC_Price'] = df['Close']
df = df[['Date', 'BTC_Price', 'Gold_Price']].dropna()

# 수익률 계산
df['BTC_Return'] = df['BTC_Price'].pct_change() * 100
df['Gold_Return'] = df['Gold_Price'].pct_change() * 100

print(f"\n데이터 기간: {df['Date'].min().date()} ~ {df['Date'].max().date()}")
print(f"총 일수: {len(df)}일")

# ========================================
# 가설 1: 금 선행 패턴 (70일)
# ========================================
print("\n" + "="*60)
print("가설 1: 금 선행 상승 패턴 검증")
print("="*60)

# 70일 lag 상관관계 확인
print("\n[1-1] Long-term Lag 상관관계 (최대 100일)")

lags = range(0, 101, 5)  # 0, 5, 10, ..., 100일
lag_correlations = []

for lag in lags:
    if lag == 0:
        corr = df['Gold_Return'].corr(df['BTC_Return'])
    else:
        # GOLD(t) vs BTC(t+lag)
        gold_now = df['Gold_Return'][:-lag]
        btc_future = df['BTC_Return'][lag:]
        corr = gold_now.corr(btc_future)
    lag_correlations.append(corr)
    if lag <= 80 and lag % 10 == 0:
        print(f"  {lag}일 후: {corr:+.4f}")

# 70일 lag 특별 체크
if 70 not in lags:
    gold_now = df['Gold_Return'][:-70]
    btc_future = df['BTC_Return'][70:]
    corr_70 = gold_now.corr(btc_future)
else:
    corr_70 = lag_correlations[lags.index(70)]

print(f"\n⭐ 70일 lag 상관관계: {corr_70:+.4f}")
print(f"   (주장: 92% 상관성)")

if corr_70 > 0.5:
    print(f"   ✅ 높은 상관성 확인! (금 선행 패턴 지지)")
elif corr_70 > 0.2:
    print(f"   ⚠️ 약한 상관성 (금 선행 패턴 부분 지지)")
else:
    print(f"   ❌ 상관성 낮음 (금 선행 패턴 미지지)")

# 최대 상관 lag 찾기
max_corr_idx = np.argmax(np.abs(lag_correlations))
max_lag = list(lags)[max_corr_idx]
max_corr = lag_correlations[max_corr_idx]

print(f"\n최대 상관 발생:")
print(f"  Lag: {max_lag}일")
print(f"  상관계수: {max_corr:+.4f}")

# ========================================
# 가설 1-2: 금 신고가 후 BTC 모멘텀
# ========================================
print("\n[1-2] 금 신고가 이후 비트코인 반응")

# 금 신고가 날짜 찾기 (이전 90일 대비)
df['Gold_90d_High'] = df['Gold_Price'].rolling(90).max()
df['Is_Gold_ATH'] = df['Gold_Price'] >= df['Gold_90d_High']

# 금 신고가 이후 30일간 BTC 수익률
gold_ath_dates = df[df['Is_Gold_ATH']].index

btc_returns_after_gold_ath = []
for idx in gold_ath_dates:
    if idx + 30 < len(df):
        future_30d_return = (df['BTC_Price'].iloc[idx+30] / df['BTC_Price'].iloc[idx] - 1) * 100
        btc_returns_after_gold_ath.append(future_30d_return)

if len(btc_returns_after_gold_ath) > 0:
    avg_btc_return_after_gold_ath = np.mean(btc_returns_after_gold_ath)
    print(f"\n금 신고가 발생: {len(gold_ath_dates)}회")
    print(f"신고가 후 30일 BTC 평균 수익률: {avg_btc_return_after_gold_ath:+.2f}%")

    # 비교: 랜덤 날짜 30일 후 수익률
    random_30d_returns = []
    for i in range(len(df) - 30):
        ret = (df['BTC_Price'].iloc[i+30] / df['BTC_Price'].iloc[i] - 1) * 100
        random_30d_returns.append(ret)
    avg_random_return = np.mean(random_30d_returns)

    print(f"랜덤 날짜 후 30일 BTC 평균: {avg_random_return:+.2f}%")
    print(f"차이: {avg_btc_return_after_gold_ath - avg_random_return:+.2f}%p")

    if avg_btc_return_after_gold_ath > avg_random_return:
        print(f"✅ 금 신고가 후 BTC가 더 상승!")
    else:
        print(f"❌ 금 신고가 효과 없음")

# ========================================
# 가설 2: 로테이션 패턴
# ========================================
print("\n" + "="*60)
print("가설 2: 금 하락 시 비트코인 로테이션 검증")
print("="*60)

# 금 하락일 정의
gold_down_days = df[df['Gold_Return'] < 0].copy()
gold_up_days = df[df['Gold_Return'] > 0].copy()

# 금 급락일 정의 (하위 10%)
gold_crash_threshold = df['Gold_Return'].quantile(0.10)
gold_crash_days = df[df['Gold_Return'] < gold_crash_threshold].copy()

# 금 급등일 정의 (상위 10%)
gold_surge_threshold = df['Gold_Return'].quantile(0.90)
gold_surge_days = df[df['Gold_Return'] > gold_surge_threshold].copy()

print("\n[2-1] 금 하락일의 비트코인 반응")
print(f"\n금 하락일: {len(gold_down_days)}일")
print(f"  평균 GOLD: {gold_down_days['Gold_Return'].mean():.3f}%")
print(f"  평균 BTC:  {gold_down_days['BTC_Return'].mean():+.3f}%")

btc_up_when_gold_down = (gold_down_days['BTC_Return'] > 0).sum()
print(f"  BTC 상승: {btc_up_when_gold_down}일 ({btc_up_when_gold_down/len(gold_down_days)*100:.1f}%)")

if gold_down_days['BTC_Return'].mean() > 0:
    print(f"  ✅ 로테이션 패턴 존재! (금↓ → BTC↑)")
else:
    print(f"  ❌ 로테이션 패턴 미확인")

print(f"\n금 급락일 (하위 10%, <{gold_crash_threshold:.2f}%): {len(gold_crash_days)}일")
print(f"  평균 GOLD: {gold_crash_days['Gold_Return'].mean():.3f}%")
print(f"  평균 BTC:  {gold_crash_days['BTC_Return'].mean():+.3f}%")

if gold_crash_days['BTC_Return'].mean() > 0:
    print(f"  ✅ 강한 로테이션! (금 급락 → BTC 상승)")
else:
    print(f"  ⚠️ 로테이션 약함")

# ========================================
# 가설 2-2: 반대 패턴 (금 상승 시 BTC)
# ========================================
print("\n[2-2] 금 상승일의 비트코인 반응 (역로테이션)")

print(f"\n금 상승일: {len(gold_up_days)}일")
print(f"  평균 GOLD: {gold_up_days['Gold_Return'].mean():.3f}%")
print(f"  평균 BTC:  {gold_up_days['BTC_Return'].mean():+.3f}%")

btc_down_when_gold_up = (gold_up_days['BTC_Return'] < 0).sum()
print(f"  BTC 하락: {btc_down_when_gold_up}일 ({btc_down_when_gold_up/len(gold_up_days)*100:.1f}%)")

print(f"\n금 급등일 (상위 10%, >{gold_surge_threshold:.2f}%): {len(gold_surge_days)}일")
print(f"  평균 GOLD: {gold_surge_days['Gold_Return'].mean():.3f}%")
print(f"  평균 BTC:  {gold_surge_days['BTC_Return'].mean():+.3f}%")

if gold_surge_days['BTC_Return'].mean() < 0:
    print(f"  ✅ 역로테이션 존재! (금↑ → BTC↓)")
else:
    print(f"  ❌ 역로테이션 미확인")

# ========================================
# 가설 2-3: 로테이션 강도 비교
# ========================================
print("\n[2-3] 로테이션 패턴 강도")

# 4가지 경우
case1 = len(df[(df['Gold_Return'] < 0) & (df['BTC_Return'] > 0)])  # 금↓ BTC↑
case2 = len(df[(df['Gold_Return'] > 0) & (df['BTC_Return'] < 0)])  # 금↑ BTC↓
case3 = len(df[(df['Gold_Return'] > 0) & (df['BTC_Return'] > 0)])  # 둘 다 ↑
case4 = len(df[(df['Gold_Return'] < 0) & (df['BTC_Return'] < 0)])  # 둘 다 ↓

total = case1 + case2 + case3 + case4

print(f"\n금↓ BTC↑ (로테이션):     {case1}일 ({case1/total*100:.1f}%)")
print(f"금↑ BTC↓ (역로테이션):   {case2}일 ({case2/total*100:.1f}%)")
print(f"둘 다 상승:             {case3}일 ({case3/total*100:.1f}%)")
print(f"둘 다 하락:             {case4}일 ({case4/total*100:.1f}%)")

print(f"\n로테이션 총합: {case1 + case2}일 ({(case1 + case2)/total*100:.1f}%)")
print(f"동조화 총합:   {case3 + case4}일 ({(case3 + case4)/total*100:.1f}%)")

if case1 + case2 > case3 + case4:
    print(f"✅ 로테이션 패턴이 더 강함!")
else:
    print(f"⚠️ 동조화 패턴이 더 강함")

# ========================================
# 최근 사례 검증
# ========================================
print("\n[2-4] 최근 사례 검증 (2025년)")

df_2025 = df[df['Date'] >= '2025-01-01'].copy()

if len(df_2025) > 0:
    # 금 -6% 이상 하락일
    gold_big_drop = df_2025[df_2025['Gold_Return'] < -3].copy()

    print(f"\n2025년 금 급락일 (<-3%): {len(gold_big_drop)}일")
    if len(gold_big_drop) > 0:
        print(f"  평균 GOLD: {gold_big_drop['Gold_Return'].mean():.2f}%")
        print(f"  평균 BTC:  {gold_big_drop['BTC_Return'].mean():+.2f}%")

        btc_up_count = (gold_big_drop['BTC_Return'] > 0).sum()
        print(f"  BTC 상승: {btc_up_count}/{len(gold_big_drop)}일 ({btc_up_count/len(gold_big_drop)*100:.1f}%)")

        # 날짜별 상세
        print(f"\n  상세 내역:")
        for _, row in gold_big_drop.head(10).iterrows():
            print(f"    {row['Date'].date()}: GOLD {row['Gold_Return']:+.2f}%, BTC {row['BTC_Return']:+.2f}%")

# ========================================
# 통계적 유의성 검증
# ========================================
print("\n" + "="*60)
print("통계적 유의성 검증")
print("="*60)

# t-검정: 금 하락일 BTC 수익률 vs 전체 평균
btc_return_gold_down = gold_down_days['BTC_Return'].dropna()
btc_return_all = df['BTC_Return'].dropna()

t_stat, p_value = stats.ttest_ind(btc_return_gold_down, btc_return_all)

print(f"\n금 하락일 BTC 수익률 t-검정:")
print(f"  t-통계량: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")

if p_value < 0.05:
    print(f"  ✅ 통계적으로 유의함 (p < 0.05)")
else:
    print(f"  ❌ 통계적으로 유의하지 않음 (p >= 0.05)")

# ========================================
# 시각화
# ========================================
print("\n시각화 생성 중...")

fig = plt.figure(figsize=(20, 14))

# (1) Long-term Lag 상관관계
ax1 = plt.subplot(3, 3, 1)
ax1.plot(lags, lag_correlations, marker='o', linewidth=2, color='purple')
ax1.axhline(0, color='black', linestyle='--', linewidth=1)
ax1.axvline(70, color='red', linestyle='--', linewidth=1, label='70일')
ax1.set_xlabel('Lag (일)', fontweight='bold')
ax1.set_ylabel('상관계수', fontweight='bold')
ax1.set_title('금 선행 패턴: Long-term Lag 상관관계', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# (2) 금 신고가 후 BTC 반응
ax2 = plt.subplot(3, 3, 2)
if len(btc_returns_after_gold_ath) > 0:
    ax2.hist(btc_returns_after_gold_ath, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax2.axvline(avg_btc_return_after_gold_ath, color='red', linestyle='--', linewidth=2,
               label=f'평균: {avg_btc_return_after_gold_ath:+.1f}%')
    ax2.axvline(avg_random_return, color='blue', linestyle='--', linewidth=2,
               label=f'랜덤: {avg_random_return:+.1f}%')
ax2.set_xlabel('30일 후 BTC 수익률 (%)', fontweight='bold')
ax2.set_ylabel('빈도', fontweight='bold')
ax2.set_title('금 신고가 후 BTC 반응', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# (3) 로테이션 패턴
ax3 = plt.subplot(3, 3, 3)
rotation_data = [case1, case2, case3, case4]
labels_rot = ['금↓\nBTC↑', '금↑\nBTC↓', '둘 다\n상승', '둘 다\n하락']
colors_rot = ['green', 'red', 'blue', 'gray']
bars = ax3.bar(labels_rot, rotation_data, color=colors_rot, alpha=0.7)
for bar, val in zip(bars, rotation_data):
    ax3.text(bar.get_x() + bar.get_width()/2, val, f'{val}일\n({val/total*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold')
ax3.set_ylabel('일수', fontweight='bold')
ax3.set_title('로테이션 패턴 발생 빈도', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# (4) 금 하락 vs BTC 수익률
ax4 = plt.subplot(3, 3, 4)
conditions = ['전체', '금 하락', '금 급락']
btc_means = [
    df['BTC_Return'].mean(),
    gold_down_days['BTC_Return'].mean(),
    gold_crash_days['BTC_Return'].mean()
]
colors_cond = ['gray', 'orange', 'red']
bars = ax4.barh(conditions, btc_means, color=colors_cond, alpha=0.7)
ax4.axvline(0, color='black', linewidth=1)
for bar, val in zip(bars, btc_means):
    ax4.text(val, bar.get_y() + bar.get_height()/2, f'{val:+.3f}%',
            va='center', ha='left' if val > 0 else 'right', fontweight='bold')
ax4.set_xlabel('평균 BTC 수익률 (%)', fontweight='bold')
ax4.set_title('금 하락 시 BTC 평균 수익률', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

# (5) 금 상승 vs BTC 수익률
ax5 = plt.subplot(3, 3, 5)
conditions2 = ['전체', '금 상승', '금 급등']
btc_means2 = [
    df['BTC_Return'].mean(),
    gold_up_days['BTC_Return'].mean(),
    gold_surge_days['BTC_Return'].mean()
]
colors_cond2 = ['gray', 'gold', 'darkgoldenrod']
bars = ax5.barh(conditions2, btc_means2, color=colors_cond2, alpha=0.7)
ax5.axvline(0, color='black', linewidth=1)
for bar, val in zip(bars, btc_means2):
    ax5.text(val, bar.get_y() + bar.get_height()/2, f'{val:+.3f}%',
            va='center', ha='left' if val > 0 else 'right', fontweight='bold')
ax5.set_xlabel('평균 BTC 수익률 (%)', fontweight='bold')
ax5.set_title('금 상승 시 BTC 평균 수익률', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='x')

# (6) 시계열: 금과 BTC 수익률
ax6 = plt.subplot(3, 3, 6)
recent = df.tail(200)
ax6.plot(recent['Date'], recent['Gold_Return'], label='GOLD', linewidth=1.5, alpha=0.7, color='gold')
ax6.plot(recent['Date'], recent['BTC_Return'], label='BTC', linewidth=1.5, alpha=0.7, color='orange')
ax6.axhline(0, color='black', linewidth=0.5)
ax6.set_ylabel('수익률 (%)', fontweight='bold')
ax6.set_title('최근 200일 수익률 추이', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.tick_params(axis='x', rotation=45)

# (7) 로테이션 vs 동조화 비율
ax7 = plt.subplot(3, 3, 7)
pie_data = [case1 + case2, case3 + case4]
pie_labels = [f'로테이션\n{case1+case2}일\n({(case1+case2)/total*100:.1f}%)',
              f'동조화\n{case3+case4}일\n({(case3+case4)/total*100:.1f}%)']
colors_pie = ['green', 'blue']
ax7.pie(pie_data, labels=pie_labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
ax7.set_title('로테이션 vs 동조화', fontsize=12, fontweight='bold')

# (8) 2025년 사례
ax8 = plt.subplot(3, 3, 8)
if len(df_2025) > 0:
    ax8.scatter(df_2025['Gold_Return'], df_2025['BTC_Return'], alpha=0.5, s=30, color='purple')
    ax8.axhline(0, color='black', linewidth=0.5)
    ax8.axvline(0, color='black', linewidth=0.5)
    # 사분면 표시
    ax8.axhline(0, color='red', linewidth=1, linestyle='--', alpha=0.3)
    ax8.axvline(0, color='red', linewidth=1, linestyle='--', alpha=0.3)
    ax8.set_xlabel('GOLD 수익률 (%)', fontweight='bold')
    ax8.set_ylabel('BTC 수익률 (%)', fontweight='bold')
    ax8.set_title('2025년 로테이션 패턴', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)

# (9) 요약
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
summary = f"""
【가설 검증 결과】

1. 금 선행 패턴 (70일):
   70일 lag 상관: {corr_70:+.4f}
   {'✅ 지지' if corr_70 > 0.2 else '❌ 미지지'}
   (주장: 0.92)

   금 신고가 후 30일:
   BTC: {avg_btc_return_after_gold_ath:+.1f}%
   랜덤: {avg_random_return:+.1f}%
   {'✅ 효과 있음' if avg_btc_return_after_gold_ath > avg_random_return else '❌ 효과 없음'}

2. 로테이션 패턴:
   금↓ → BTC평균: {gold_down_days['BTC_Return'].mean():+.3f}%
   {'✅ 로테이션 존재' if gold_down_days['BTC_Return'].mean() > 0 else '❌ 없음'}

   금 급락 → BTC: {gold_crash_days['BTC_Return'].mean():+.3f}%
   {'✅ 강한 로테이션' if gold_crash_days['BTC_Return'].mean() > 0.5 else '⚠️ 약함'}

3. 패턴 빈도:
   로테이션: {(case1+case2)/total*100:.1f}%
   동조화: {(case3+case4)/total*100:.1f}%
   {'✅ 로테이션 우세' if case1+case2 > case3+case4 else '⚠️ 동조화 우세'}

4. 통계 유의성:
   p-value: {p_value:.4f}
   {'✅ 유의함' if p_value < 0.05 else '❌ 유의하지 않음'}
"""
ax9.text(0.05, 0.5, summary, fontsize=9, verticalalignment='center', family='monospace')
ax9.set_title('종합 요약', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('gold_btc_lead_lag_analysis.png', dpi=300, bbox_inches='tight')
print("✅ 저장: gold_btc_lead_lag_analysis.png")

# 결과 저장
results = {
    '가설': [
        '70일 선행 상관관계',
        '금 신고가 후 BTC 효과',
        '금 하락 → BTC 상승',
        '금 급락 → BTC 상승',
        '로테이션 비율',
        '통계 유의성'
    ],
    '결과': [
        f'{corr_70:+.4f}',
        f'{avg_btc_return_after_gold_ath - avg_random_return:+.2f}%p',
        f'{gold_down_days["BTC_Return"].mean():+.3f}%',
        f'{gold_crash_days["BTC_Return"].mean():+.3f}%',
        f'{(case1+case2)/total*100:.1f}%',
        f'p={p_value:.4f}'
    ],
    '판정': [
        '✅ 지지' if corr_70 > 0.2 else '❌ 미지지',
        '✅' if avg_btc_return_after_gold_ath > avg_random_return else '❌',
        '✅' if gold_down_days['BTC_Return'].mean() > 0 else '❌',
        '✅' if gold_crash_days['BTC_Return'].mean() > 0.5 else '⚠️',
        '✅ 로테이션' if case1+case2 > case3+case4 else '⚠️ 동조화',
        '✅' if p_value < 0.05 else '❌'
    ]
}

results_df = pd.DataFrame(results)
results_df.to_csv('gold_btc_hypothesis_test.csv', index=False)
print("✅ 저장: gold_btc_hypothesis_test.csv")

print("\n" + "="*80)
print("분석 완료!")
print("="*80)
