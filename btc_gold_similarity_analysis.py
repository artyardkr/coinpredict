#!/usr/bin/env python3
"""
BTC vs GOLD 유사도 분석
- 상관관계 분석
- 수익률 패턴 비교
- 변동성 비교
- 시계열 동조화 분석
- 위기 시 반응 비교
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import correlate
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("BTC vs GOLD 유사도 분석")
print("="*80)

# 데이터 로드
df = pd.read_csv('integrated_data_full.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# BTC와 GOLD 데이터 확인
print(f"\n데이터 기간: {df['Date'].min().date()} ~ {df['Date'].max().date()}")
print(f"총 일수: {len(df)}일")

# GOLD 컬럼 확인
gold_cols = [col for col in df.columns if 'gold' in col.lower() or 'GC=F' in col]
print(f"\nGOLD 관련 컬럼: {gold_cols}")

# GOLD 데이터 선택
if 'GOLD' in df.columns:
    df['Gold_Price'] = df['GOLD']
elif 'gold_close' in df.columns:
    df['Gold_Price'] = df['gold_close']
elif len(gold_cols) > 0:
    df['Gold_Price'] = df[gold_cols[0]]
else:
    print("❌ GOLD 데이터를 찾을 수 없습니다!")
    exit(1)

# BTC 가격
df['BTC_Price'] = df['Close']

# 결측치 제거
df = df[['Date', 'BTC_Price', 'Gold_Price']].dropna()

print(f"\n유효 데이터: {len(df)}일")
print(f"BTC 가격 범위: ${df['BTC_Price'].min():,.2f} ~ ${df['BTC_Price'].max():,.2f}")
print(f"GOLD 가격 범위: ${df['Gold_Price'].min():.2f} ~ ${df['Gold_Price'].max():.2f}")

# ========================================
# 1. 정규화 (0~100 스케일)
# ========================================
df['BTC_Normalized'] = (df['BTC_Price'] - df['BTC_Price'].min()) / (df['BTC_Price'].max() - df['BTC_Price'].min()) * 100
df['Gold_Normalized'] = (df['Gold_Price'] - df['Gold_Price'].min()) / (df['Gold_Price'].max() - df['Gold_Price'].min()) * 100

# ========================================
# 2. 일별 수익률 계산
# ========================================
df['BTC_Return'] = df['BTC_Price'].pct_change() * 100
df['Gold_Return'] = df['Gold_Price'].pct_change() * 100

# ========================================
# 3. 상관관계 분석
# ========================================
print("\n" + "="*60)
print("1. 상관관계 분석")
print("="*60)

# 가격 상관관계
price_corr = df['BTC_Price'].corr(df['Gold_Price'])
print(f"\n가격 상관관계: {price_corr:.4f}")

# 수익률 상관관계
return_corr = df['BTC_Return'].corr(df['Gold_Return'])
print(f"일별 수익률 상관관계: {return_corr:.4f}")

# 기간별 상관관계
periods = {
    '전체': (df['Date'].min(), df['Date'].max()),
    '2021': ('2021-01-01', '2021-12-31'),
    '2022': ('2022-01-01', '2022-12-31'),
    '2023': ('2023-01-01', '2023-12-31'),
    '2024': ('2024-01-01', '2024-12-31'),
    '2025': ('2025-01-01', '2025-12-31'),
}

print("\n기간별 수익률 상관관계:")
period_corrs = []
for period_name, (start, end) in periods.items():
    mask = (df['Date'] >= start) & (df['Date'] <= end)
    period_df = df[mask]
    if len(period_df) > 30:
        corr = period_df['BTC_Return'].corr(period_df['Gold_Return'])
        print(f"  {period_name}: {corr:+.4f} ({len(period_df)}일)")
        period_corrs.append((period_name, corr))

# 롤링 상관관계 (90일)
df['Rolling_Corr_90d'] = df['BTC_Return'].rolling(90).corr(df['Gold_Return'])

# ========================================
# 4. 변동성 비교
# ========================================
print("\n" + "="*60)
print("2. 변동성 비교")
print("="*60)

btc_vol = df['BTC_Return'].std()
gold_vol = df['Gold_Return'].std()
btc_vol_annual = btc_vol * np.sqrt(252)
gold_vol_annual = gold_vol * np.sqrt(252)

print(f"\n일일 변동성:")
print(f"  BTC:  {btc_vol:.3f}%")
print(f"  GOLD: {gold_vol:.3f}%")
print(f"  비율: BTC는 GOLD의 {btc_vol/gold_vol:.1f}배")

print(f"\n연율 변동성:")
print(f"  BTC:  {btc_vol_annual:.2f}%")
print(f"  GOLD: {gold_vol_annual:.2f}%")

# ========================================
# 5. 수익률 분포 비교
# ========================================
print("\n" + "="*60)
print("3. 수익률 분포")
print("="*60)

btc_mean = df['BTC_Return'].mean()
gold_mean = df['Gold_Return'].mean()

print(f"\n평균 일일 수익률:")
print(f"  BTC:  {btc_mean:+.3f}%")
print(f"  GOLD: {gold_mean:+.3f}%")

btc_positive = (df['BTC_Return'] > 0).sum() / len(df) * 100
gold_positive = (df['Gold_Return'] > 0).sum() / len(df) * 100

print(f"\n상승 비율:")
print(f"  BTC:  {btc_positive:.1f}%")
print(f"  GOLD: {gold_positive:.1f}%")

# 극단값 비교
btc_extreme_up = (df['BTC_Return'] > 5).sum()
btc_extreme_down = (df['BTC_Return'] < -5).sum()
gold_extreme_up = (df['Gold_Return'] > 2).sum()
gold_extreme_down = (df['Gold_Return'] < -2).sum()

print(f"\n극단값 (BTC: ±5%, GOLD: ±2%):")
print(f"  BTC 급등(>5%):  {btc_extreme_up}일")
print(f"  BTC 급락(<-5%): {btc_extreme_down}일")
print(f"  GOLD 상승(>2%): {gold_extreme_up}일")
print(f"  GOLD 하락(<-2%): {gold_extreme_down}일")

# ========================================
# 6. 동시 움직임 분석
# ========================================
print("\n" + "="*60)
print("4. 동시 움직임 분석")
print("="*60)

# 같은 방향 움직임
df['Same_Direction'] = (np.sign(df['BTC_Return']) == np.sign(df['Gold_Return']))
same_direction_pct = df['Same_Direction'].sum() / len(df) * 100

print(f"\n같은 방향 움직임: {same_direction_pct:.1f}%")

# 강한 동시 움직임 (둘 다 ±1% 이상)
df['Both_Up'] = (df['BTC_Return'] > 1) & (df['Gold_Return'] > 0.5)
df['Both_Down'] = (df['BTC_Return'] < -1) & (df['Gold_Return'] < -0.5)

both_up = df['Both_Up'].sum()
both_down = df['Both_Down'].sum()

print(f"\n강한 동시 움직임:")
print(f"  둘 다 상승: {both_up}일")
print(f"  둘 다 하락: {both_down}일")
print(f"  총: {both_up + both_down}일 ({(both_up + both_down)/len(df)*100:.1f}%)")

# ========================================
# 7. 위기 시 반응 비교
# ========================================
print("\n" + "="*60)
print("5. 위기/급락 시 반응")
print("="*60)

# BTC 급락일 (하위 5%)
btc_crash_threshold = df['BTC_Return'].quantile(0.05)
btc_crash_days = df[df['BTC_Return'] < btc_crash_threshold]

print(f"\nBTC 급락일 (하위 5%, <{btc_crash_threshold:.2f}%):")
print(f"  일수: {len(btc_crash_days)}일")
print(f"  평균 BTC 수익률: {btc_crash_days['BTC_Return'].mean():.2f}%")
print(f"  평균 GOLD 수익률: {btc_crash_days['Gold_Return'].mean():.2f}%")
if btc_crash_days['Gold_Return'].mean() > 0:
    print(f"  → ✅ GOLD는 상승 (안전자산 역할)")
else:
    print(f"  → ❌ GOLD도 하락 (위험자산화)")

# BTC 급등일 (상위 5%)
btc_surge_threshold = df['BTC_Return'].quantile(0.95)
btc_surge_days = df[df['BTC_Return'] > btc_surge_threshold]

print(f"\nBTC 급등일 (상위 5%, >{btc_surge_threshold:.2f}%):")
print(f"  일수: {len(btc_surge_days)}일")
print(f"  평균 BTC 수익률: {btc_surge_days['BTC_Return'].mean():.2f}%")
print(f"  평균 GOLD 수익률: {btc_surge_days['Gold_Return'].mean():.2f}%")

# ========================================
# 8. 시계열 동조화 분석 (Cross-correlation)
# ========================================
print("\n" + "="*60)
print("6. 시계열 선후행 관계")
print("="*60)

# 정규화된 수익률
btc_returns_norm = (df['BTC_Return'] - df['BTC_Return'].mean()) / df['BTC_Return'].std()
gold_returns_norm = (df['Gold_Return'] - df['Gold_Return'].mean()) / df['Gold_Return'].std()

# Cross-correlation (lag -10 ~ +10일)
lags = range(-10, 11)
cross_corr = []
for lag in lags:
    if lag < 0:
        # BTC가 선행
        corr = btc_returns_norm[:-abs(lag)].corr(gold_returns_norm[abs(lag):])
    elif lag > 0:
        # GOLD가 선행
        corr = btc_returns_norm[lag:].corr(gold_returns_norm[:-lag])
    else:
        # 동시
        corr = btc_returns_norm.corr(gold_returns_norm)
    cross_corr.append(corr)

max_corr_idx = np.argmax(np.abs(cross_corr))
max_lag = lags[max_corr_idx]
max_corr = cross_corr[max_corr_idx]

print(f"\n최대 상관관계:")
print(f"  Lag: {max_lag}일")
print(f"  상관계수: {max_corr:.4f}")
if max_lag < 0:
    print(f"  → BTC가 GOLD보다 {abs(max_lag)}일 선행")
elif max_lag > 0:
    print(f"  → GOLD가 BTC보다 {max_lag}일 선행")
else:
    print(f"  → 동시 움직임")

# ========================================
# 9. 시각화
# ========================================
print("\n" + "="*60)
print("7. 시각화 생성 중...")
print("="*60)

fig = plt.figure(figsize=(20, 16))

# (1) 가격 추이 (정규화)
ax1 = plt.subplot(4, 3, 1)
ax1.plot(df['Date'], df['BTC_Normalized'], label='BTC', linewidth=2, color='orange')
ax1.plot(df['Date'], df['Gold_Normalized'], label='GOLD', linewidth=2, color='gold')
ax1.set_ylabel('정규화 가격 (0-100)')
ax1.set_title('가격 추이 비교 (정규화)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# (2) 수익률 산점도
ax2 = plt.subplot(4, 3, 2)
ax2.scatter(df['Gold_Return'], df['BTC_Return'], alpha=0.3, s=10, color='steelblue')
ax2.axhline(0, color='black', linewidth=0.5)
ax2.axvline(0, color='black', linewidth=0.5)
ax2.set_xlabel('GOLD 수익률 (%)')
ax2.set_ylabel('BTC 수익률 (%)')
ax2.set_title(f'수익률 상관관계 (r={return_corr:.3f})', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# (3) 롤링 상관관계 (90일)
ax3 = plt.subplot(4, 3, 3)
ax3.plot(df['Date'], df['Rolling_Corr_90d'], linewidth=2, color='purple')
ax3.axhline(0, color='black', linestyle='--', linewidth=1)
ax3.axhline(return_corr, color='red', linestyle='--', linewidth=1, label=f'평균: {return_corr:.3f}')
ax3.set_ylabel('상관계수')
ax3.set_title('롤링 상관관계 (90일)', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

# (4) 수익률 분포 (BTC)
ax4 = plt.subplot(4, 3, 4)
ax4.hist(df['BTC_Return'].dropna(), bins=50, alpha=0.7, color='orange', edgecolor='black')
ax4.axvline(0, color='red', linestyle='--', linewidth=2)
ax4.axvline(btc_mean, color='blue', linestyle='--', linewidth=2, label=f'평균: {btc_mean:.3f}%')
ax4.set_xlabel('일일 수익률 (%)')
ax4.set_ylabel('빈도')
ax4.set_title(f'BTC 수익률 분포 (σ={btc_vol:.2f}%)', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# (5) 수익률 분포 (GOLD)
ax5 = plt.subplot(4, 3, 5)
ax5.hist(df['Gold_Return'].dropna(), bins=50, alpha=0.7, color='gold', edgecolor='black')
ax5.axvline(0, color='red', linestyle='--', linewidth=2)
ax5.axvline(gold_mean, color='blue', linestyle='--', linewidth=2, label=f'평균: {gold_mean:.3f}%')
ax5.set_xlabel('일일 수익률 (%)')
ax5.set_ylabel('빈도')
ax5.set_title(f'GOLD 수익률 분포 (σ={gold_vol:.2f}%)', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# (6) 변동성 비교
ax6 = plt.subplot(4, 3, 6)
volatilities = [btc_vol_annual, gold_vol_annual]
colors_vol = ['orange', 'gold']
bars = ax6.bar(['BTC', 'GOLD'], volatilities, color=colors_vol, alpha=0.7, edgecolor='black')
for bar, vol in zip(bars, volatilities):
    ax6.text(bar.get_x() + bar.get_width()/2, vol, f'{vol:.1f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=11)
ax6.set_ylabel('연율 변동성 (%)')
ax6.set_title('변동성 비교', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

# (7) 기간별 상관관계
ax7 = plt.subplot(4, 3, 7)
if len(period_corrs) > 0:
    periods_names = [p[0] for p in period_corrs]
    periods_values = [p[1] for p in period_corrs]
    colors_period = ['green' if v > 0 else 'red' for v in periods_values]
    bars = ax7.barh(periods_names, periods_values, color=colors_period, alpha=0.7)
    ax7.axvline(0, color='black', linewidth=1)
    ax7.set_xlabel('상관계수')
    ax7.set_title('기간별 수익률 상관관계', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars, periods_values):
        ax7.text(val, bar.get_y() + bar.get_height()/2, f'{val:+.3f}',
                va='center', ha='left' if val > 0 else 'right', fontweight='bold')

# (8) Cross-correlation
ax8 = plt.subplot(4, 3, 8)
ax8.plot(lags, cross_corr, marker='o', linewidth=2, color='purple')
ax8.axhline(0, color='black', linestyle='--', linewidth=1)
ax8.axvline(0, color='red', linestyle='--', linewidth=1)
ax8.set_xlabel('Lag (일)')
ax8.set_ylabel('상관계수')
ax8.set_title('시계열 선후행 관계 (Cross-correlation)', fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3)

# (9) 동시 움직임
ax9 = plt.subplot(4, 3, 9)
movement_data = [same_direction_pct, 100 - same_direction_pct]
colors_move = ['green', 'red']
bars = ax9.bar(['같은 방향', '반대 방향'], movement_data, color=colors_move, alpha=0.7)
for bar, val in zip(bars, movement_data):
    ax9.text(bar.get_x() + bar.get_width()/2, val, f'{val:.1f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=11)
ax9.set_ylabel('비율 (%)')
ax9.set_title('움직임 방향 일치도', fontsize=12, fontweight='bold')
ax9.grid(True, alpha=0.3, axis='y')

# (10) 위기 시 반응
ax10 = plt.subplot(4, 3, 10)
crisis_data = [
    btc_crash_days['BTC_Return'].mean(),
    btc_crash_days['Gold_Return'].mean()
]
colors_crisis = ['red', 'gold']
bars = ax10.bar(['BTC (급락일)', 'GOLD (급락일)'], crisis_data, color=colors_crisis, alpha=0.7)
ax10.axhline(0, color='black', linewidth=1)
for bar, val in zip(bars, crisis_data):
    ax10.text(bar.get_x() + bar.get_width()/2, val, f'{val:+.2f}%',
            ha='center', va='bottom' if val > 0 else 'top', fontweight='bold', fontsize=10)
ax10.set_ylabel('평균 수익률 (%)')
ax10.set_title('BTC 급락일의 GOLD 반응', fontsize=12, fontweight='bold')
ax10.grid(True, alpha=0.3, axis='y')

# (11) 극단값 비교
ax11 = plt.subplot(4, 3, 11)
extreme_data = [btc_extreme_up, btc_extreme_down, gold_extreme_up, gold_extreme_down]
labels_extreme = ['BTC\n급등', 'BTC\n급락', 'GOLD\n상승', 'GOLD\n하락']
colors_extreme = ['orange', 'red', 'yellowgreen', 'darkred']
bars = ax11.bar(labels_extreme, extreme_data, color=colors_extreme, alpha=0.7)
for bar, val in zip(bars, extreme_data):
    ax11.text(bar.get_x() + bar.get_width()/2, val, f'{val}일',
            ha='center', va='bottom', fontweight='bold', fontsize=10)
ax11.set_ylabel('일수')
ax11.set_title('극단값 발생 빈도', fontsize=12, fontweight='bold')
ax11.grid(True, alpha=0.3, axis='y')

# (12) 요약
ax12 = plt.subplot(4, 3, 12)
ax12.axis('off')
summary = f"""
【BTC vs GOLD 유사도 분석】

1. 상관관계:
   가격: {price_corr:+.4f}
   수익률: {return_corr:+.4f}
   {'→ 약한 양의 상관' if 0 < return_corr < 0.3 else '→ 음의 상관' if return_corr < 0 else '→ 강한 양의 상관'}

2. 변동성:
   BTC: {btc_vol_annual:.1f}% (연율)
   GOLD: {gold_vol_annual:.1f}% (연율)
   → BTC는 {btc_vol/gold_vol:.1f}배 더 변동성

3. 동시 움직임:
   같은 방향: {same_direction_pct:.1f}%
   {'→ 독립적 움직임' if same_direction_pct < 55 else '→ 유사한 움직임'}

4. 안전자산 역할:
   BTC 급락 시 GOLD:
   {btc_crash_days['Gold_Return'].mean():+.2f}%
   {'→ ✅ 안전자산 역할' if btc_crash_days['Gold_Return'].mean() > 0 else '→ ❌ 같이 하락'}

5. 선후행 관계:
   최대 상관 Lag: {max_lag}일
   {'→ BTC 선행' if max_lag < 0 else '→ GOLD 선행' if max_lag > 0 else '→ 동시'}

6. 결론:
   {'✅ 유사성 높음 (분산 효과 낮음)' if return_corr > 0.5 else '⚠️ 중간 유사성 (약간 분산)' if return_corr > 0.2 else '✅ 유사성 낮음 (분산 효과 높음)'}
"""
ax12.text(0.05, 0.5, summary, fontsize=9, verticalalignment='center', family='monospace')
ax12.set_title('종합 요약', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('btc_gold_similarity_analysis.png', dpi=300, bbox_inches='tight')
print("✅ 저장: btc_gold_similarity_analysis.png")

# ========================================
# 10. 결과 저장
# ========================================
results = {
    '지표': [
        '가격 상관관계',
        '수익률 상관관계',
        'BTC 일일 변동성',
        'GOLD 일일 변동성',
        'BTC 연율 변동성',
        'GOLD 연율 변동성',
        '같은 방향 움직임',
        'BTC 급락일 GOLD 수익률',
        '최대 상관 Lag',
    ],
    '값': [
        f'{price_corr:.4f}',
        f'{return_corr:.4f}',
        f'{btc_vol:.3f}%',
        f'{gold_vol:.3f}%',
        f'{btc_vol_annual:.2f}%',
        f'{gold_vol_annual:.2f}%',
        f'{same_direction_pct:.1f}%',
        f'{btc_crash_days["Gold_Return"].mean():+.2f}%',
        f'{max_lag}일',
    ]
}

results_df = pd.DataFrame(results)
results_df.to_csv('btc_gold_similarity_results.csv', index=False)
print("✅ 저장: btc_gold_similarity_results.csv")

print("\n" + "="*80)
print("분석 완료!")
print("="*80)
