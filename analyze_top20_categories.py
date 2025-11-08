#!/usr/bin/env python3
"""
ETF 이전/이후 TOP 20 변수를 카테고리별로 분류
"""

import pandas as pd

# ETF 이전 TOP 20
pre_etf_top20 = [
    'bc_miners_revenue',           # 온체인
    'Hash_Price',                  # 온체인
    'NVT_Ratio',                   # 온체인
    'EMA100_volume',               # 시장/기술적
    'Volume',                      # 시장/기술적
    'SMA20_marketcap',             # 시장/기술적
    'T10Y3M',                      # 거시경제
    'market_cap_approx',           # 시장/기술적
    'SMA30_marketcap',             # 시장/기술적
    'Active_Addresses_MA90',       # 온체인
    'fear_greed_index',            # 심리/기술적
    'WTREGEN',                     # 거시경제
    'Price_to_MA200',              # 시장/기술적
    'EMA30_marketcap',             # 시장/기술적
    'Price_MA200',                 # 시장/기술적
    'EMA200_volume',               # 시장/기술적
    'RSI',                         # 시장/기술적
    'EMA100_marketcap',            # 시장/기술적
    'T10Y2Y',                      # 거시경제
    'TLT',                         # 거시경제
]

# ETF 이후 TOP 20
post_etf_top20 = [
    'Miner_Revenue_to_Cap_MA30',   # 온체인
    'ETH',                         # 전통시장
    'Price_to_MA200',              # 시장/기술적
    'IWM',                         # 전통시장
    'NVT_Ratio_MA90',              # 온체인
    'SMA10_marketcap',             # 시장/기술적
    'RRPONTSYD',                   # 거시경제
    'SMA30_marketcap',             # 시장/기술적
    'Difficulty_MA60',             # 온체인
    'BITB_Price',                  # ETF
    'ARKB_Price',                  # ETF
    'Hash_Ribbon_MA30',            # 온체인
    'GBTC_Price',                  # ETF
    'CCI',                         # 시장/기술적
    'FBTC_Price',                  # ETF
    'IBIT_Price',                  # ETF
    'Difficulty_MA30',             # 온체인
    'Hash_Ribbon_MA60',            # 온체인
    'EMA100_volume',               # 시장/기술적
    'Difficulty_MA90',             # 온체인
]

def categorize_variable(var):
    """변수를 카테고리로 분류"""
    # ETF
    if any(x in var for x in ['IBIT', 'FBTC', 'GBTC', 'ARKB', 'BITB']):
        return 'ETF'

    # 온체인
    onchain_keywords = ['Hash', 'NVT', 'Puell', 'Miner', 'Active_Address',
                        'bc_', 'Difficulty', 'Hash_Ribbon', 'Revenue_to_Cap']
    if any(kw in var for kw in onchain_keywords):
        return '온체인'

    # 거시경제/Fed
    macro_keywords = ['T10Y', 'WTREGEN', 'RRPONTSYD', 'FED', 'WALCL', 'DFF',
                      'SOFR', 'GDP', 'UNRATE', 'M2SL', 'CPIAUCSL', 'TLT', 'DXY']
    if any(kw in var for kw in macro_keywords):
        return '거시경제'

    # 전통시장
    market_keywords = ['SPX', 'QQQ', 'ETH', 'IWM', 'DIA', 'GLD', 'GOLD', 'SILVER',
                       'OIL', 'VIX', 'EURUSD', 'UUP', 'HYG', 'LQD']
    if any(kw in var for kw in market_keywords):
        return '전통시장'

    # 심리지표
    if 'fear_greed' in var or 'google_trends' in var:
        return '심리지표'

    # 나머지는 시장/기술적지표
    return '시장/기술적지표'

# 카테고리별 분류
print("=" * 80)
print("ETF 이전 기간 TOP 20 변수 카테고리 분석")
print("=" * 80)

pre_categories = {}
for i, var in enumerate(pre_etf_top20, 1):
    cat = categorize_variable(var)
    if cat not in pre_categories:
        pre_categories[cat] = []
    pre_categories[cat].append((i, var))

print(f"\n총 {len(pre_etf_top20)}개 변수\n")
for cat in sorted(pre_categories.keys()):
    vars_list = pre_categories[cat]
    print(f"【{cat}】 {len(vars_list)}개")
    for rank, var in vars_list:
        print(f"  {rank:2d}. {var}")
    print()

print("\n" + "=" * 80)
print("ETF 이후 기간 TOP 20 변수 카테고리 분석")
print("=" * 80)

post_categories = {}
for i, var in enumerate(post_etf_top20, 1):
    cat = categorize_variable(var)
    if cat not in post_categories:
        post_categories[cat] = []
    post_categories[cat].append((i, var))

print(f"\n총 {len(post_etf_top20)}개 변수\n")
for cat in sorted(post_categories.keys()):
    vars_list = post_categories[cat]
    print(f"【{cat}】 {len(vars_list)}개")
    for rank, var in vars_list:
        etf_mark = "🔥" if cat == 'ETF' else "  "
        print(f"  {rank:2d}. {etf_mark} {var}")
    print()

# 비교 요약
print("\n" + "=" * 80)
print("카테고리별 비교 요약")
print("=" * 80)

all_categories = set(list(pre_categories.keys()) + list(post_categories.keys()))

summary = []
for cat in sorted(all_categories):
    pre_count = len(pre_categories.get(cat, []))
    post_count = len(post_categories.get(cat, []))
    change = post_count - pre_count
    change_str = f"+{change}" if change > 0 else str(change) if change < 0 else "0"

    summary.append({
        '카테고리': cat,
        'ETF 이전': pre_count,
        'ETF 이후': post_count,
        '변화': change_str
    })

summary_df = pd.DataFrame(summary)
print("\n" + summary_df.to_string(index=False))

# 핵심 인사이트
print("\n" + "=" * 80)
print("🎯 핵심 인사이트")
print("=" * 80)

print(f"""
1. ETF 변수:
   - ETF 이전: {len(pre_categories.get('ETF', []))}개
   - ETF 이후: {len(post_categories.get('ETF', []))}개 ({post_categories.get('ETF', [])[0][0]}~{post_categories.get('ETF', [])[-1][0]}위)
   → ETF 승인 후 BTC ETF가 TOP 20의 25% 차지!

2. 온체인 데이터:
   - ETF 이전: {len(pre_categories.get('온체인', []))}개
   - ETF 이후: {len(post_categories.get('온체인', []))}개
   → 온체인 중요도 {'증가' if len(post_categories.get('온체인', [])) > len(pre_categories.get('온체인', [])) else '유지' if len(post_categories.get('온체인', [])) == len(pre_categories.get('온체인', [])) else '감소'}

3. 거시경제:
   - ETF 이전: {len(pre_categories.get('거시경제', []))}개
   - ETF 이후: {len(post_categories.get('거시경제', []))}개
   → 거시경제 중요도 {'증가' if len(post_categories.get('거시경제', [])) > len(pre_categories.get('거시경제', [])) else '유지' if len(post_categories.get('거시경제', [])) == len(pre_categories.get('거시경제', [])) else '감소'}

4. 전통시장:
   - ETF 이전: {len(pre_categories.get('전통시장', []))}개
   - ETF 이후: {len(post_categories.get('전통시장', []))}개
   → ETH, IWM 등 전통시장 상관관계 {'증가' if len(post_categories.get('전통시장', [])) > len(pre_categories.get('전통시장', [])) else '유지'}

5. 시장/기술적지표:
   - ETF 이전: {len(pre_categories.get('시장/기술적지표', []))}개
   - ETF 이후: {len(post_categories.get('시장/기술적지표', []))}개
   → 순수 기술적 분석의 중요도 {'감소' if len(post_categories.get('시장/기술적지표', [])) < len(pre_categories.get('시장/기술적지표', [])) else '증가' if len(post_categories.get('시장/기술적지표', [])) > len(pre_categories.get('시장/기술적지표', [])) else '유지'}
""")

print("=" * 80)
print("분석 완료!")
print("=" * 80)
