#!/usr/bin/env python3
"""
ETF 이전/이후 주요 변수들의 롤링 상관계수 분석
- 비트코인 가격과의 롤링 상관계수 계산
- ETF 승인 시점(2024-01-10) 전후 비교
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
print("=" * 80)
print("롤링 상관계수 분석")
print("=" * 80)

df = pd.read_csv('integrated_data_full_v2.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"데이터 기간: {df['Date'].min()} ~ {df['Date'].max()}")
print(f"총 {len(df)} 샘플")

ETF_DATE = pd.Timestamp('2024-01-10')

# 주요 변수 선택 (카테고리별 대표 변수)
key_variables = {
    '금리': ['T10Y3M', 'T10Y2Y', 'DFF', 'SOFR'],
    'ETF': ['IBIT_Price', 'FBTC_Price', 'GBTC_Price', 'ARKB_Price', 'BITB_Price'],
    '온체인': ['NVT_Ratio', 'Hash_Price', 'bc_miners_revenue', 'Miner_Revenue_to_Cap_MA30',
              'Difficulty_MA60', 'Hash_Ribbon_MA30'],
    '전통시장': ['ETH', 'IWM', 'SPX', 'QQQ', 'GOLD'],
    'Fed유동성': ['RRPONTSYD', 'WTREGEN', 'WALCL', 'FED_NET_LIQUIDITY'],
    '기술적': ['RSI', 'Volume', 'fear_greed_index']
}

# 롤링 윈도우 설정
WINDOW = 90  # 90일 롤링

# 가격과의 롤링 상관계수 계산
print(f"\n롤링 윈도우: {WINDOW}일")
print("비트코인 가격과의 상관계수 계산 중...")

correlations = {}

for category, variables in key_variables.items():
    print(f"\n【{category}】")
    for var in variables:
        if var in df.columns:
            # 결측치 처리
            df[var] = df[var].replace([np.inf, -np.inf], np.nan)

            # 롤링 상관계수 계산
            rolling_corr = df['Close'].rolling(window=WINDOW).corr(df[var])
            correlations[var] = rolling_corr

            # ETF 이전/이후 평균 상관계수
            pre_etf_mask = df['Date'] < ETF_DATE
            post_etf_mask = df['Date'] >= ETF_DATE

            pre_corr = rolling_corr[pre_etf_mask].mean()
            post_corr = rolling_corr[post_etf_mask].mean()

            # ETF 이후는 데이터가 있을 때만
            if post_etf_mask.sum() > WINDOW:
                change = post_corr - pre_corr
                print(f"  {var:30s} | 이전: {pre_corr:6.3f} | 이후: {post_corr:6.3f} | 변화: {change:+6.3f}")
            else:
                print(f"  {var:30s} | 이전: {pre_corr:6.3f} | 이후: N/A")
        else:
            print(f"  {var:30s} | 데이터 없음")

# 시각화
fig = plt.figure(figsize=(24, 16))
gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

# 카테고리별 롤링 상관계수 플롯
plot_configs = [
    ('금리', 0, 0),
    ('ETF', 0, 1),
    ('온체인', 0, 2),
    ('전통시장', 1, 0),
    ('Fed유동성', 1, 1),
    ('기술적', 1, 2),
]

for category, row, col in plot_configs:
    ax = fig.add_subplot(gs[row, col])

    variables = key_variables[category]
    for var in variables:
        if var in correlations:
            corr = correlations[var]
            # NaN 제거
            valid_mask = ~corr.isna()
            ax.plot(df['Date'][valid_mask], corr[valid_mask],
                   label=var, linewidth=2, alpha=0.7)

    # ETF 승인일 표시
    ax.axvline(x=ETF_DATE, color='red', linestyle='--', linewidth=2,
              label='ETF 승인', alpha=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)

    ax.set_xlabel('날짜', fontweight='bold')
    ax.set_ylabel(f'{WINDOW}일 롤링 상관계수', fontweight='bold')
    ax.set_title(f'【{category}】 vs BTC 가격', fontweight='bold', fontsize=12)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

# ETF 이전/이후 평균 상관계수 비교 (주요 변수만)
ax_comparison = fig.add_subplot(gs[2, :])

# 주요 변수 선택
comparison_vars = [
    # 금리
    'T10Y3M', 'T10Y2Y',
    # ETF
    'IBIT_Price', 'FBTC_Price', 'GBTC_Price',
    # 온체인
    'NVT_Ratio', 'Hash_Price', 'Miner_Revenue_to_Cap_MA30',
    # 전통시장
    'ETH', 'IWM',
    # Fed
    'RRPONTSYD', 'WTREGEN',
]

pre_corrs = []
post_corrs = []
var_labels = []

for var in comparison_vars:
    if var in correlations:
        pre_etf_mask = df['Date'] < ETF_DATE
        post_etf_mask = df['Date'] >= ETF_DATE

        corr = correlations[var]
        pre_corr = corr[pre_etf_mask].mean()
        post_corr = corr[post_etf_mask].mean()

        if not np.isnan(post_corr):
            pre_corrs.append(pre_corr)
            post_corrs.append(post_corr)
            var_labels.append(var[:15])  # 변수명 15자로 제한

x = np.arange(len(var_labels))
width = 0.35

bars1 = ax_comparison.bar(x - width/2, pre_corrs, width, label='ETF 이전',
                          alpha=0.8, color='#3498db')
bars2 = ax_comparison.bar(x + width/2, post_corrs, width, label='ETF 이후',
                          alpha=0.8, color='#e74c3c')

ax_comparison.set_ylabel('평균 상관계수', fontweight='bold')
ax_comparison.set_xlabel('변수', fontweight='bold')
ax_comparison.set_title('ETF 이전 vs 이후: 평균 상관계수 비교', fontweight='bold', fontsize=14)
ax_comparison.set_xticks(x)
ax_comparison.set_xticklabels(var_labels, rotation=45, ha='right', fontsize=9)
ax_comparison.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax_comparison.legend(fontsize=11)
ax_comparison.grid(True, alpha=0.3, axis='y')

# 상관계수 변화량 순위
ax_change = fig.add_subplot(gs[3, :])

changes = np.array(post_corrs) - np.array(pre_corrs)
sorted_idx = np.argsort(np.abs(changes))[::-1]

top_n = min(15, len(sorted_idx))
top_vars = [var_labels[i] for i in sorted_idx[:top_n]]
top_changes = [changes[i] for i in sorted_idx[:top_n]]

colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in top_changes]
bars = ax_change.barh(range(top_n), top_changes, color=colors, alpha=0.7)

ax_change.set_yticks(range(top_n))
ax_change.set_yticklabels(top_vars, fontsize=10)
ax_change.set_xlabel('상관계수 변화량 (이후 - 이전)', fontweight='bold')
ax_change.set_title('ETF 승인 전후 상관계수 변화량 TOP 15', fontweight='bold', fontsize=14)
ax_change.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
ax_change.invert_yaxis()
ax_change.grid(True, alpha=0.3, axis='x')

for i, change in enumerate(top_changes):
    ax_change.text(change, i, f'  {change:+.3f}',
                  va='center', fontsize=9, fontweight='bold')

plt.savefig('rolling_correlation_analysis.png', dpi=300, bbox_inches='tight')
print("\n저장 완료: rolling_correlation_analysis.png")

# 요약 테이블 생성
print("\n" + "=" * 80)
print("ETF 이전/이후 상관계수 변화 요약")
print("=" * 80)

summary_data = []
for i, var in enumerate(var_labels):
    summary_data.append({
        '변수': var,
        'ETF 이전': f"{pre_corrs[i]:.3f}",
        'ETF 이후': f"{post_corrs[i]:.3f}",
        '변화량': f"{changes[i]:+.3f}",
        '변화율': f"{(changes[i]/abs(pre_corrs[i])*100 if pre_corrs[i] != 0 else 0):+.1f}%"
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('변화량', key=lambda x: abs(x.str.replace('+', '').astype(float)),
                                    ascending=False)

print("\n" + summary_df.to_string(index=False))

# CSV 저장
summary_df.to_csv('rolling_correlation_summary.csv', index=False)
print("\n저장 완료: rolling_correlation_summary.csv")

# 핵심 인사이트
print("\n" + "=" * 80)
print("🎯 핵심 인사이트")
print("=" * 80)

# 상관계수가 가장 많이 증가한 변수
max_increase_idx = np.argmax(changes)
max_increase_var = var_labels[max_increase_idx]
max_increase_val = changes[max_increase_idx]

# 상관계수가 가장 많이 감소한 변수
min_decrease_idx = np.argmin(changes)
min_decrease_var = var_labels[min_decrease_idx]
min_decrease_val = changes[min_decrease_idx]

print(f"""
1. 상관계수 가장 큰 증가:
   - {max_increase_var}: {pre_corrs[max_increase_idx]:.3f} → {post_corrs[max_increase_idx]:.3f} ({max_increase_val:+.3f})

2. 상관계수 가장 큰 감소:
   - {min_decrease_var}: {pre_corrs[min_decrease_idx]:.3f} → {post_corrs[min_decrease_idx]:.3f} ({min_decrease_val:+.3f})

3. ETF 변수들의 상관계수:
""")

etf_vars = ['IBIT_Price', 'FBTC_Price', 'GBTC_Price', 'ARKB_Price', 'BITB_Price']
for var in etf_vars:
    if var in var_labels:
        idx = var_labels.index(var)
        print(f"   - {var:15s}: {post_corrs[idx]:.3f} (ETF 이후)")

print("""
4. 금리 변수들의 변화:
""")
rate_vars = ['T10Y3M', 'T10Y2Y', 'DFF', 'SOFR']
for var in rate_vars:
    if var in var_labels:
        idx = var_labels.index(var)
        print(f"   - {var:10s}: {pre_corrs[idx]:.3f} → {post_corrs[idx]:.3f} ({changes[idx]:+.3f})")

print("\n" + "=" * 80)
print("분석 완료!")
print("=" * 80)
