#!/usr/bin/env python3
"""
Quandt-Andrews Test 결과 시기별 카테고리 변화 분석

목적:
- 변화점 날짜를 시기별로 그룹화
- 6개 카테고리별로 어느 시기에 변화가 집중되었는지 분석
- 시계열 히트맵 및 요약 통계 생성

작성일: 2025-11-12
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 1. 데이터 로드 및 카테고리 분류
# ============================================================================

print("=" * 80)
print("Quandt-Andrews 시기별 카테고리 변화 분석")
print("=" * 80)

# 데이터 로드
df = pd.read_csv('zscore_qa_test_results.csv')
df['breakpoint_date'] = pd.to_datetime(df['breakpoint_date'])

print(f"\n총 변수 수: {len(df)}")
print(f"변화점 날짜 범위: {df['breakpoint_date'].min()} ~ {df['breakpoint_date'].max()}")

# 카테고리 정의
def categorize_variable(var_name):
    """변수명을 6개 카테고리로 분류"""

    # 1. Bitcoin 가격 및 기술적 지표
    price_tech = [
        'Close', 'High', 'Low', 'Open', 'Volume', 'cumulative_return',
        'EMA', 'SMA', 'BB_', 'RSI', 'MACD', 'Stoch', 'Williams', 'ROC',
        'MFI', 'OBV', 'ATR', 'CCI', 'ADX', 'volatility', 'daily_return',
        'volume_change', 'market_cap', 'Price_MA', 'Price_to_MA'
    ]

    # 2. 전통 시장 지수
    traditional = [
        'SPX', 'QQQ', 'DIA', 'IWM',  # 주식
        'TLT', 'LQD', 'HYG',  # 채권
        'GLD', 'GOLD', 'SILVER', 'OIL',  # 원자재
        'DXY', 'EURUSD', 'DEXUSEU', 'DTWEXBGS', 'UUP',  # 환율
        'VIX', 'VIXCLS',  # 변동성
        'ETH', 'BSV'  # 다른 암호화폐
    ]

    # 3. 거시경제 지표
    macro = [
        'DFF', 'DGS10', 'SOFR',  # 금리
        'T10Y2Y', 'T10Y3M',  # 금리차
        'M2SL',  # 통화량
        'GDP', 'UNRATE', 'CPIAUCSL',  # 경제
        'WALCL', 'RRPONTSYD', 'WTREGEN', 'FED_NET_LIQUIDITY',  # Fed
        'BAMLC0A0CM', 'BAMLH0A0HYM2'  # 신용 스프레드
    ]

    # 4. 온체인 데이터
    onchain = [
        'bc_', 'cm_', 'Active_Addresses', 'Mempool',
        'Avg_Fee', 'transaction', 'n_unique_addresses',
        'n_transactions', 'transaction_fees'
    ]

    # 5. Bitcoin ETF 데이터
    etf = [
        'IBIT', 'FBTC', 'GBTC', 'ARKB', 'BITB',
        'ETF_Volume', 'Total_BTC_ETF'
    ]

    # 6. 채굴 & 고급 온체인
    mining = [
        'Hash', 'Difficulty', 'Puell', 'NVT', 'Miner',
        'miners_revenue', 'hash_rate', 'difficulty'
    ]

    var_upper = var_name.upper()

    # 채굴 & 고급 온체인 (먼저 체크)
    if any(keyword in var_upper for keyword in [k.upper() for k in mining]):
        return '채굴_고급온체인'

    # ETF
    if any(keyword in var_upper for keyword in [k.upper() for k in etf]):
        return 'Bitcoin_ETF'

    # 온체인
    if any(keyword in var_upper for keyword in [k.upper() for k in onchain]):
        return '온체인'

    # 거시경제
    if any(keyword in var_upper for keyword in [k.upper() for k in macro]):
        return '거시경제'

    # 전통 시장
    if any(keyword in var_upper for keyword in [k.upper() for k in traditional]):
        return '전통시장'

    # 가격 및 기술지표
    if any(keyword in var_upper for keyword in [k.upper() for k in price_tech]):
        return '가격_기술지표'

    return '기타'

# 카테고리 분류
df['Category'] = df['Variable'].apply(categorize_variable)

print("\n카테고리별 변수 수:")
print(df['Category'].value_counts().sort_index())

# ============================================================================
# 2. 시기 구분
# ============================================================================

def classify_period(date):
    """날짜를 시기로 분류"""

    if date < pd.Timestamp('2022-01-01'):
        return '2021년 (초기)'
    elif date < pd.Timestamp('2022-07-01'):
        return '2022 상반기 (하락장)'
    elif date < pd.Timestamp('2023-01-01'):
        return '2022 하반기 (바닥)'
    elif date < pd.Timestamp('2023-07-01'):
        return '2023 상반기 (회복)'
    elif date < pd.Timestamp('2024-01-01'):
        return '2023 하반기 (상승)'
    elif date < pd.Timestamp('2024-01-11'):
        return '2024-01 (ETF 전)'
    elif date < pd.Timestamp('2024-04-01'):
        return '2024 Q1 (ETF 직후)'
    elif date < pd.Timestamp('2024-07-01'):
        return '2024 Q2'
    elif date < pd.Timestamp('2024-10-01'):
        return '2024 Q3'
    elif date < pd.Timestamp('2024-11-01'):
        return '2024-10 (변화 시작)'
    elif date < pd.Timestamp('2024-12-01'):
        return '2024-11 (변화 폭발)'
    else:
        return '2024-12 이후'

df['Period'] = df['breakpoint_date'].apply(classify_period)

print("\n시기별 변화점 수:")
period_counts = df['Period'].value_counts().sort_index()
print(period_counts)

# ============================================================================
# 3. 시기별 카테고리 분석
# ============================================================================

# 피벗 테이블: 시기 x 카테고리
pivot = pd.crosstab(df['Period'], df['Category'])

print("\n시기별 카테고리 변화점 수 (피벗 테이블):")
print(pivot)

# 비율 계산
pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

print("\n시기별 카테고리 비율 (%):")
print(pivot_pct.round(1))

# ============================================================================
# 4. 시각화
# ============================================================================

fig = plt.figure(figsize=(24, 16))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. 히트맵: 변화점 수
ax1 = fig.add_subplot(gs[0, :])
sns.heatmap(pivot.T, annot=True, fmt='d', cmap='YlOrRd',
            cbar_kws={'label': '변화점 수'}, ax=ax1, linewidths=0.5)
ax1.set_title('시기별 카테고리 변화점 히트맵 (절대 수)',
              fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('시기', fontsize=12, fontweight='bold')
ax1.set_ylabel('카테고리', fontsize=12, fontweight='bold')
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

# 2. 히트맵: 비율
ax2 = fig.add_subplot(gs[1, :])
sns.heatmap(pivot_pct.T, annot=True, fmt='.1f', cmap='Blues',
            cbar_kws={'label': '비율 (%)'}, ax=ax2, linewidths=0.5)
ax2.set_title('시기별 카테고리 변화점 비율 (%)',
              fontsize=16, fontweight='bold', pad=20)
ax2.set_xlabel('시기', fontsize=12, fontweight='bold')
ax2.set_ylabel('카테고리', fontsize=12, fontweight='bold')
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

# 3. 시계열 라인 차트
ax3 = fig.add_subplot(gs[2, 0])
for category in pivot.columns:
    ax3.plot(pivot.index, pivot[category], marker='o', label=category, linewidth=2)
ax3.set_xlabel('시기', fontsize=12, fontweight='bold')
ax3.set_ylabel('변화점 수', fontsize=12, fontweight='bold')
ax3.set_title('시기별 카테고리 변화점 추이', fontsize=14, fontweight='bold')
ax3.legend(loc='best', fontsize=9)
ax3.grid(alpha=0.3)
plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

# 4. 카테고리별 총 변화점 수
ax4 = fig.add_subplot(gs[2, 1])
category_totals = df['Category'].value_counts().sort_values(ascending=True)
colors = plt.cm.Set3(np.linspace(0, 1, len(category_totals)))
ax4.barh(category_totals.index, category_totals.values, color=colors)
ax4.set_xlabel('변화점 수', fontsize=12, fontweight='bold')
ax4.set_title('카테고리별 총 변화점 수', fontsize=14, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

for i, v in enumerate(category_totals.values):
    ax4.text(v + 1, i, str(v), va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('qa_시기별_카테고리_변화_히트맵.png', dpi=300, bbox_inches='tight')
print("\n저장 완료: qa_시기별_카테고리_변화_히트맵.png")

# ============================================================================
# 5. 주요 시기별 대표 변수 분석
# ============================================================================

print("\n" + "=" * 80)
print("주요 시기별 TOP 변수 (sup_F 기준)")
print("=" * 80)

key_periods = [
    '2022 상반기 (하락장)',
    '2023 하반기 (상승)',
    '2024 Q1 (ETF 직후)',
    '2024-10 (변화 시작)',
    '2024-11 (변화 폭발)'
]

for period in key_periods:
    period_data = df[df['Period'] == period].nlargest(5, 'sup_F')

    if len(period_data) > 0:
        print(f"\n📅 {period}")
        print("-" * 80)
        for idx, row in period_data.iterrows():
            print(f"  {row['Variable']:30s} | {row['Category']:15s} | "
                  f"sup_F: {row['sup_F']:8.0f} | {row['breakpoint_date'].date()}")

# ============================================================================
# 6. 카테고리별 변화점 날짜 분포
# ============================================================================

categories = sorted(df['Category'].unique())
n_categories = len(categories)
n_cols = 2
n_rows = (n_categories + 1) // 2

fig2, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
axes = axes.flatten()

for i, category in enumerate(categories):
    ax = axes[i]
    cat_data = df[df['Category'] == category]

    # 날짜별 카운트
    date_counts = cat_data['breakpoint_date'].value_counts().sort_index()

    ax.bar(date_counts.index, date_counts.values, width=10, alpha=0.7)
    ax.set_xlabel('날짜', fontsize=10)
    ax.set_ylabel('변화점 수', fontsize=10)
    ax.set_title(f'{category} (총 {len(cat_data)}개)',
                fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # ETF 승인일 표시
    etf_date = pd.Timestamp('2024-01-10')
    ax.axvline(etf_date, color='red', linestyle='--', linewidth=2, alpha=0.7, label='ETF 승인')
    ax.legend(fontsize=8)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)

plt.tight_layout()
plt.savefig('qa_카테고리별_변화점_분포.png', dpi=300, bbox_inches='tight')
print("저장 완료: qa_카테고리별_변화점_분포.png")

# ============================================================================
# 7. 요약 통계 저장
# ============================================================================

# 시기별 카테고리 통계
summary_df = pd.DataFrame({
    '시기': pivot.index,
    **{f'{cat}_수': pivot[cat].values for cat in pivot.columns},
    **{f'{cat}_%': pivot_pct[cat].values for cat in pivot_pct.columns},
    '총계': pivot.sum(axis=1).values
})

summary_df.to_csv('qa_시기별_카테고리_요약.csv', index=False, encoding='utf-8-sig')
print("저장 완료: qa_시기별_카테고리_요약.csv")

# 카테고리별 변수 목록
category_vars = df.groupby('Category')['Variable'].apply(list).reset_index()
category_vars['변수_수'] = category_vars['Variable'].apply(len)
category_vars.to_csv('qa_카테고리별_변수목록.csv', index=False, encoding='utf-8-sig')
print("저장 완료: qa_카테고리별_변수목록.csv")

# ============================================================================
# 8. 최종 요약 리포트
# ============================================================================

print("\n" + "=" * 80)
print("📊 최종 요약 리포트")
print("=" * 80)

# 가장 많은 변화가 있었던 시기
max_period = period_counts.idxmax()
max_count = period_counts.max()

print(f"\n🔥 가장 많은 변화가 있었던 시기: {max_period} ({max_count}개 변화점)")

# 시기별 주요 카테고리
print("\n📌 시기별 주요 카테고리:")
for period in pivot.index:
    top_cat = pivot.loc[period].idxmax()
    top_count = pivot.loc[period].max()
    pct = pivot_pct.loc[period, top_cat]
    print(f"  {period:30s}: {top_cat:20s} ({top_count:2.0f}개, {pct:4.1f}%)")

# 카테고리별 주요 시기
print("\n🎯 카테고리별 가장 많이 변화한 시기:")
for category in pivot.columns:
    top_period = pivot[category].idxmax()
    top_count = pivot[category].max()
    print(f"  {category:20s}: {top_period:30s} ({top_count:2.0f}개)")

# 특이 사항
print("\n⚠️ 주요 발견:")

# 2024-11 폭발적 증가
nov_2024 = df[df['Period'] == '2024-11 (변화 폭발)']
if len(nov_2024) > 0:
    print(f"\n1. 2024년 11월 변화 폭발:")
    print(f"   - 총 {len(nov_2024)}개 변수에서 변화점 감지")
    print(f"   - 주요 카테고리: {nov_2024['Category'].value_counts().head(3).to_dict()}")

# ETF 전후 비교
etf_before = df[df['Period'] == '2024-01 (ETF 전)']
etf_after_q1 = df[df['Period'] == '2024 Q1 (ETF 직후)']
print(f"\n2. ETF 승인 전후 비교:")
print(f"   - ETF 전 (2024-01): {len(etf_before)}개")
print(f"   - ETF 직후 (Q1): {len(etf_after_q1)}개")

# 2022년 하락장
crash_2022 = df[(df['Period'] == '2022 상반기 (하락장)') |
                 (df['Period'] == '2022 하반기 (바닥)')]
if len(crash_2022) > 0:
    print(f"\n3. 2022년 하락장:")
    print(f"   - 총 {len(crash_2022)}개 변수에서 변화점")
    print(f"   - 주요 카테고리: {crash_2022['Category'].value_counts().head(3).to_dict()}")

print("\n" + "=" * 80)
print("분석 완료!")
print("=" * 80)
