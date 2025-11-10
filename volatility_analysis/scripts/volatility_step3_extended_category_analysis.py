"""
================================================================================
Step 3 (Extended): 카테고리별 변동성 분석
================================================================================
137개 전체 변수를 카테고리별로 분류하고, 각 카테고리의 대표 변수를 선정하여
확장된 GARCH 변동성 분석을 수행합니다.

카테고리:
1. 전통 자산 (Traditional Assets): SPX, QQQ, VIX, GOLD, TLT, DXY, etc.
2. 거시경제 (Macro): 금리, 통화, 인플레이션, GDP, 실업률
3. 온체인 (On-Chain): Hash Rate, Difficulty, Active Addresses, Fees
4. 기술지표 (Technical): RSI, MACD, BB, Volume, EMA
5. ETF 관련 (ETF): GBTC, IBIT, FBTC, ARKB, BITB
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("Step 3 (Extended): 카테고리별 변동성 분석")
print("="*80)

# ============================================================================
# 카테고리별 변수 정의
# ============================================================================
print("\n[Step 3E.1] 카테고리별 변수 정의")
print("-"*80)

categories = {
    '전통자산': {
        'variables': ['SPX', 'QQQ', 'VIX', 'GOLD', 'TLT', 'DXY', 'OIL', 'SILVER',
                     'ETH', 'IWM', 'DIA', 'HYG', 'LQD', 'GLD', 'UUP'],
        'representatives': ['SPX', 'QQQ', 'VIX', 'GOLD', 'TLT', 'DXY']
    },
    '거시경제': {
        'variables': ['DFF', 'SOFR', 'RRPONTSYD', 'CPIAUCSL', 'GDP', 'UNRATE',
                     'M2SL', 'WALCL', 'FED_NET_LIQUIDITY', 'DGS10', 'T10Y3M', 'T10Y2Y',
                     'BAMLC0A0CM', 'BAMLH0A0HYM2', 'DEXUSEU', 'EURUSD', 'DTWEXBGS'],
        'representatives': ['DFF', 'SOFR', 'RRPONTSYD', 'CPIAUCSL', 'GDP', 'UNRATE', 'M2SL']
    },
    '온체인': {
        'variables': ['bc_hash_rate', 'bc_difficulty', 'Hash_Ribbon_MA30', 'Hash_Ribbon_MA60',
                     'Difficulty_MA30', 'Difficulty_MA60', 'Difficulty_MA90',
                     'bc_n_transactions', 'bc_n_unique_addresses', 'Active_Addresses_MA90',
                     'bc_transaction_fees', 'Avg_Fee_Per_Tx', 'Avg_Fee_Per_Tx_MA30',
                     'bc_miners_revenue', 'Miner_Revenue_to_Cap', 'Hash_Price', 'Hash_Price_MA90',
                     'bc_mempool_size', 'Mempool_Stress', 'bc_market_cap', 'bc_total_bitcoins'],
        'representatives': ['bc_hash_rate', 'bc_difficulty', 'bc_n_transactions', 'bc_transaction_fees', 'bc_miners_revenue']
    },
    '밸류에이션': {
        'variables': ['NVT_Ratio', 'NVT_Ratio_MA90', 'MVRV', 'Puell_Multiple',
                     'Price_to_MA200', 'SOPR', 'fear_greed_index'],
        'representatives': ['NVT_Ratio', 'MVRV', 'Puell_Multiple', 'SOPR']
    },
    '기술지표': {
        'variables': ['RSI', 'MACD', 'MACD_signal', 'MACD_diff', 'BB_high', 'BB_mid', 'BB_low', 'BB_width',
                     'Volume', 'OBV', 'volatility_20d', 'ATR', 'ADX', 'CCI', 'ROC',
                     'Stoch_K', 'Stoch_D', 'Williams_R', 'MFI'],
        'representatives': ['RSI', 'MACD', 'Volume', 'volatility_20d', 'ATR']
    },
    'ETF관련': {
        'variables': ['GBTC_Premium', 'GBTC_Price', 'IBIT_Price', 'FBTC_Price', 'ARKB_Price', 'BITB_Price',
                     'Total_BTC_ETF_Volume', 'GBTC_Volume_Change_7d', 'IBIT_Volume_Change_7d',
                     'FBTC_Volume_Change_7d', 'ARKB_Volume_Change_7d', 'BITB_Volume_Change_7d'],
        'representatives': ['GBTC_Premium', 'IBIT_Price', 'Total_BTC_ETF_Volume']
    }
}

# 통계 출력
print("\n카테고리별 변수 수:")
for cat, info in categories.items():
    print(f"  {cat}: {len(info['variables'])}개 변수, {len(info['representatives'])}개 대표 변수")

total_vars = sum(len(info['variables']) for info in categories.values())
total_reps = sum(len(info['representatives']) for info in categories.values())
print(f"\n전체: {total_vars}개 변수, {total_reps}개 대표 변수")

# ============================================================================
# 데이터 로드
# ============================================================================
print("\n[Step 3E.2] 데이터 로드")
print("-"*80)

# Z-score 표준화된 전체 데이터 로드
data_df = pd.read_csv('bitcoin_data.csv', index_col=0, parse_dates=True)
print(f"전체 데이터: {data_df.shape}")

# BTC 가격 추가
btc_data = data_df[['Close']].copy()

# 대표 변수들만 추출
representative_vars = ['Close']  # BTC 포함
for cat, info in categories.items():
    for var in info['representatives']:
        if var in data_df.columns and var != 'Close':
            representative_vars.append(var)

# 사용 가능한 변수만 필터링
available_vars = [v for v in representative_vars if v in data_df.columns]
print(f"\n사용 가능한 대표 변수: {len(available_vars)}개")
print(f"변수 목록: {available_vars[:10]}... (총 {len(available_vars)}개)")

extended_df = data_df[available_vars].copy()

# 결측치 제거
extended_df = extended_df.dropna()
print(f"결측치 제거 후: {extended_df.shape}")

# ETF 승인일
etf_date = pd.Timestamp('2024-01-10')
pre_etf = extended_df[extended_df.index < etf_date]
post_etf = extended_df[extended_df.index >= etf_date]

print(f"\nETF 이전: {len(pre_etf)}일")
print(f"ETF 이후: {len(post_etf)}일")

# ============================================================================
# 수익률 계산
# ============================================================================
print("\n[Step 3E.3] 수익률 계산")
print("-"*80)

returns_df = extended_df.pct_change().dropna()
print(f"수익률 데이터: {returns_df.shape}")

pre_etf_returns = returns_df[returns_df.index < etf_date]
post_etf_returns = returns_df[returns_df.index >= etf_date]

# ============================================================================
# 카테고리별 Rolling Correlation
# ============================================================================
print("\n[Step 3E.4] 카테고리별 Rolling Correlation")
print("-"*80)

window = 60  # 60일 롤링윈도우

# BTC와 각 카테고리 대표 변수들의 상관계수
category_correlations = {}

for cat, info in categories.items():
    print(f"\n{cat} 카테고리:")
    cat_corrs = {}

    for var in info['representatives']:
        if var in returns_df.columns and var != 'Close':
            rolling_corr = returns_df['Close'].rolling(window=window).corr(returns_df[var])
            cat_corrs[var] = rolling_corr

            # ETF 전후 평균 상관계수
            pre_mean = rolling_corr[rolling_corr.index < etf_date].mean()
            post_mean = rolling_corr[rolling_corr.index >= etf_date].mean()
            change = post_mean - pre_mean

            print(f"  {var:25s}: {pre_mean:+.3f} → {post_mean:+.3f} ({change:+.3f})")

    category_correlations[cat] = pd.DataFrame(cat_corrs)

# 카테고리별 평균 상관계수
print("\n\n카테고리별 평균 절대 상관계수:")
for cat, corr_df in category_correlations.items():
    if len(corr_df.columns) > 0:
        pre_mean = corr_df[corr_df.index < etf_date].abs().mean().mean()
        post_mean = corr_df[corr_df.index >= etf_date].abs().mean().mean()
        change = post_mean - pre_mean
        print(f"  {cat:15s}: {pre_mean:.3f} → {post_mean:.3f} ({change:+.3f})")

# ============================================================================
# 카테고리별 변동성 분석
# ============================================================================
print("\n\n[Step 3E.5] 카테고리별 변동성 분석")
print("-"*80)

# 실현 변동성 (20일 window)
rv_window = 20
volatilities = {}

for cat, info in categories.items():
    print(f"\n{cat} 카테고리:")
    cat_vols = {}

    for var in info['representatives']:
        if var in returns_df.columns:
            rolling_vol = returns_df[var].rolling(window=rv_window).std() * np.sqrt(252)
            cat_vols[var] = rolling_vol

            # ETF 전후 평균 변동성
            pre_mean = rolling_vol[rolling_vol.index < etf_date].mean()
            post_mean = rolling_vol[rolling_vol.index >= etf_date].mean()
            change = post_mean - pre_mean
            change_pct = (change / pre_mean) * 100 if pre_mean != 0 else 0

            print(f"  {var:25s}: {pre_mean:.4f} → {post_mean:.4f} ({change:+.4f}, {change_pct:+.1f}%)")

    volatilities[cat] = pd.DataFrame(cat_vols)

# ============================================================================
# 결과 저장
# ============================================================================
print("\n\n[Step 3E.6] 결과 저장")
print("-"*80)

# 카테고리별 상관계수
for cat, corr_df in category_correlations.items():
    if len(corr_df.columns) > 0:
        filename = f'volatility_category_{cat}_correlation.csv'
        corr_df.to_csv(filename)
        print(f"  {filename}")

# 카테고리별 변동성
for cat, vol_df in volatilities.items():
    if len(vol_df.columns) > 0:
        filename = f'volatility_category_{cat}_volatility.csv'
        vol_df.to_csv(filename)
        print(f"  {filename}")

# ============================================================================
# 시각화
# ============================================================================
print("\n[Step 3E.7] 카테고리별 시각화")
print("-"*80)

# 1. 카테고리별 평균 상관계수 변화
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

for idx, (cat, corr_df) in enumerate(category_correlations.items()):
    if idx >= 6 or len(corr_df.columns) == 0:
        continue

    ax = axes[idx]

    # 각 변수의 상관계수 시계열
    for col in corr_df.columns:
        ax.plot(corr_df.index, corr_df[col], alpha=0.6, label=col)

    # ETF 승인일 표시
    ax.axvline(x=etf_date, color='red', linestyle='--', linewidth=2, alpha=0.7, label='ETF 승인')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    ax.set_title(f'{cat} - BTC 상관계수', fontsize=12, fontweight='bold')
    ax.set_xlabel('날짜')
    ax.set_ylabel('상관계수')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('volatility_category_correlations.png', dpi=300, bbox_inches='tight')
print("  volatility_category_correlations.png 저장 완료")
plt.close()

# 2. 카테고리별 변동성 변화
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

for idx, (cat, vol_df) in enumerate(volatilities.items()):
    if idx >= 6 or len(vol_df.columns) == 0:
        continue

    ax = axes[idx]

    # 각 변수의 변동성 시계열
    for col in vol_df.columns:
        ax.plot(vol_df.index, vol_df[col], alpha=0.6, label=col)

    # ETF 승인일 표시
    ax.axvline(x=etf_date, color='red', linestyle='--', linewidth=2, alpha=0.7, label='ETF 승인')

    ax.set_title(f'{cat} - 변동성 (20일 RV)', fontsize=12, fontweight='bold')
    ax.set_xlabel('날짜')
    ax.set_ylabel('연율화 변동성')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('volatility_category_volatilities.png', dpi=300, bbox_inches='tight')
print("  volatility_category_volatilities.png 저장 완료")
plt.close()

# 3. 카테고리별 ETF 전후 비교 (박스플롯)
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

for idx, (cat, corr_df) in enumerate(category_correlations.items()):
    if idx >= 6 or len(corr_df.columns) == 0:
        continue

    ax = axes[idx]

    # ETF 전후 상관계수 분포
    pre_data = []
    post_data = []
    labels = []

    for col in corr_df.columns:
        pre = corr_df.loc[corr_df.index < etf_date, col].dropna().abs()
        post = corr_df.loc[corr_df.index >= etf_date, col].dropna().abs()

        if len(pre) > 0 and len(post) > 0:
            pre_data.append(pre)
            post_data.append(post)
            labels.append(col)

    if len(labels) > 0:
        positions = np.arange(len(labels)) * 3
        bp1 = ax.boxplot(pre_data, positions=positions - 0.6, widths=0.5,
                         patch_artist=True, showfliers=False)
        bp2 = ax.boxplot(post_data, positions=positions + 0.6, widths=0.5,
                         patch_artist=True, showfliers=False)

        for patch in bp1['boxes']:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.7)
        for patch in bp2['boxes']:
            patch.set_facecolor('coral')
            patch.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('|상관계수|')
        ax.set_title(f'{cat} - ETF 전후 상관계수 분포', fontsize=12, fontweight='bold')
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['ETF 이전', 'ETF 이후'], loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('volatility_category_comparison.png', dpi=300, bbox_inches='tight')
print("  volatility_category_comparison.png 저장 완료")
plt.close()

# ============================================================================
# 요약
# ============================================================================
print("\n" + "="*80)
print("Step 3 (Extended): 카테고리별 변동성 분석 완료")
print("="*80)

print(f"""
분석 범위:
- 전체 변수: {len(available_vars)}개
- 카테고리: {len(categories)}개
- 기간: {extended_df.index[0].date()} ~ {extended_df.index[-1].date()}

카테고리별 대표 변수:
""")

for cat, info in categories.items():
    available_reps = [v for v in info['representatives'] if v in available_vars]
    print(f"  {cat}: {', '.join(available_reps)}")

print(f"""
생성된 파일:
- 카테고리별 상관계수 CSV: {len([c for c in category_correlations.values() if len(c.columns) > 0])}개
- 카테고리별 변동성 CSV: {len([v for v in volatilities.values() if len(v.columns) > 0])}개
- 시각화: 3개 PNG 파일
""")
