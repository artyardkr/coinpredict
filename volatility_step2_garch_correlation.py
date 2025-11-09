"""
Step 2: GARCH 변동성 추정 & Rolling Correlation 분석

방법:
1. GARCH(1,1) 모형으로 조건부 변동성 추정
2. 이동 상관계수 계산 (60일, 90일 창)
3. ETF 전후 비교
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("Step 2: GARCH 변동성 추정 & Rolling Correlation")
print("=" * 80)

# 데이터 로드
returns = pd.read_csv('returns_data.csv', index_col='Date', parse_dates=True)
rv_20 = pd.read_csv('realized_volatility_20d.csv', index_col='Date', parse_dates=True)

ETF_DATE = pd.to_datetime('2024-01-10')

# ============================================================================
# Step 2.1: GARCH(1,1) 변동성 추정
# ============================================================================

print("\n[Step 2.1] GARCH(1,1) 조건부 변동성 추정")
print("-" * 80)

# 핵심 자산 선택
CORE_ASSETS = ['Close', 'SPX', 'QQQ', 'VIX', 'GOLD', 'TLT', 'DXY']
core_assets = [a for a in CORE_ASSETS if a in returns.columns]

def estimate_garch(returns_series, name):
    """
    GARCH(1,1) 모형 추정

    Parameters:
    -----------
    returns_series : Series
        수익률 데이터
    name : str
        자산 이름

    Returns:
    --------
    conditional_vol : Series
        조건부 변동성 (연율화)
    """
    try:
        # 수익률을 퍼센트로 변환 (수렴 개선)
        returns_pct = returns_series * 100

        # GARCH(1,1) 모형
        model = arch_model(returns_pct, vol='Garch', p=1, q=1, rescale=False)
        result = model.fit(disp='off', show_warning=False)

        # 조건부 변동성 추출 (연율화)
        cond_vol = result.conditional_volatility / 100 * np.sqrt(252)

        return cond_vol

    except Exception as e:
        print(f"  {name}: GARCH 추정 실패 - {str(e)}")
        return None

# GARCH 변동성 추정
garch_vol = pd.DataFrame(index=returns.index)

print("GARCH(1,1) 추정 진행 중...")
for asset in core_assets:
    print(f"  {asset}...", end='')
    vol = estimate_garch(returns[asset], asset)
    if vol is not None:
        garch_vol[asset] = vol
        print(f" 완료 (평균={vol.mean():.4f})")
    else:
        print(" 실패")

# 결과 저장
garch_vol.to_csv('garch_volatility.csv')
print(f"\nGARCH 변동성 저장: {garch_vol.shape}")

# ============================================================================
# Step 2.2: Rolling Correlation 계산
# ============================================================================

print("\n[Step 2.2] Rolling Correlation 계산")
print("-" * 80)

def calculate_rolling_correlation(vol1, vol2, window=60):
    """
    이동 상관계수 계산

    Parameters:
    -----------
    vol1, vol2 : Series
        변동성 시계열
    window : int
        이동창 크기

    Returns:
    --------
    rolling_corr : Series
        이동 상관계수
    """
    # 두 시계열을 데이터프레임으로 결합
    df_temp = pd.DataFrame({'vol1': vol1, 'vol2': vol2})
    df_temp = df_temp.dropna()

    # 이동 상관계수
    rolling_corr = df_temp['vol1'].rolling(window=window).corr(df_temp['vol2'])

    return rolling_corr

# BTC vs 주요 자산 Rolling Correlation (60일)
print("BTC vs 주요 자산 60일 이동 상관계수 계산 중...")

rolling_corr_60 = pd.DataFrame(index=rv_20.index)

if 'Close' in rv_20.columns:
    btc_vol = rv_20['Close']

    for asset in ['SPX', 'VIX', 'GOLD', 'TLT', 'DXY']:
        if asset in rv_20.columns:
            corr = calculate_rolling_correlation(btc_vol, rv_20[asset], window=60)
            rolling_corr_60[f'BTC_vs_{asset}'] = corr
            print(f"  BTC vs {asset}: 완료")

# 결과 저장
rolling_corr_60.to_csv('rolling_correlation_60d.csv')
print(f"Rolling Correlation 저장: {rolling_corr_60.shape}")

# ETF 전후 상관계수 비교
print("\nETF 전후 평균 상관계수 비교:")
corr_pre = rolling_corr_60[rolling_corr_60.index < ETF_DATE]
corr_post = rolling_corr_60[rolling_corr_60.index >= ETF_DATE]

for col in rolling_corr_60.columns:
    mean_pre = corr_pre[col].mean()
    mean_post = corr_post[col].mean()
    change = mean_post - mean_pre
    print(f"  {col:20s}: {mean_pre:+.3f} → {mean_post:+.3f} ({change:+.3f})")

# ============================================================================
# Step 2.3: 시각화
# ============================================================================

print("\n[Step 2.3] 시각화 생성")
print("-" * 80)

# 시각화 1: 변동성 비교 (RV vs GARCH)
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

compare_assets = ['Close', 'SPX', 'VIX', 'GOLD']

for idx, asset in enumerate(compare_assets):
    if asset in rv_20.columns and asset in garch_vol.columns:
        ax = axes[idx]

        # RV
        ax.plot(rv_20.index, rv_20[asset], label='Realized Vol (20d)',
                alpha=0.7, linewidth=1.5)

        # GARCH
        ax.plot(garch_vol.index, garch_vol[asset], label='GARCH(1,1)',
                alpha=0.7, linewidth=1.5)

        # ETF 승인일
        ax.axvline(ETF_DATE, color='red', linestyle='--', linewidth=2,
                   alpha=0.7, label='ETF 승인')

        ax.set_title(f'{asset} 변동성 비교', fontsize=14, fontweight='bold')
        ax.set_xlabel('날짜', fontsize=11)
        ax.set_ylabel('변동성 (연율화)', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('volatility_rv_vs_garch.png', dpi=300, bbox_inches='tight')
plt.close()
print("  volatility_rv_vs_garch.png 저장 완료")

# 시각화 2: Rolling Correlation
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, col in enumerate(rolling_corr_60.columns):
    if idx < len(axes):
        ax = axes[idx]

        # Rolling Correlation
        ax.plot(rolling_corr_60.index, rolling_corr_60[col], linewidth=1.5,
                color='steelblue', alpha=0.8)

        # ETF 승인일
        ax.axvline(ETF_DATE, color='red', linestyle='--', linewidth=2,
                   alpha=0.7, label='ETF 승인')

        # 0선
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

        # ETF 전후 평균 (수평선)
        mean_pre = rolling_corr_60.loc[rolling_corr_60.index < ETF_DATE, col].mean()
        mean_post = rolling_corr_60.loc[rolling_corr_60.index >= ETF_DATE, col].mean()

        ax.axhline(mean_pre, color='blue', linestyle=':', linewidth=1.5,
                   alpha=0.6, label=f'ETF 전 평균: {mean_pre:.3f}')
        ax.axhline(mean_post, color='green', linestyle=':', linewidth=1.5,
                   alpha=0.6, label=f'ETF 후 평균: {mean_post:.3f}')

        ax.set_title(f'{col} (60일 이동 상관계수)', fontsize=13, fontweight='bold')
        ax.set_xlabel('날짜', fontsize=11)
        ax.set_ylabel('상관계수', fontsize=11)
        ax.set_ylim(-1, 1)
        ax.legend(fontsize=9, loc='best')
        ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('rolling_correlation_60d.png', dpi=300, bbox_inches='tight')
plt.close()
print("  rolling_correlation_60d.png 저장 완료")

print("\n" + "=" * 80)
print("Step 2 완료: GARCH 변동성 추정 & Rolling Correlation")
print("=" * 80)
print("\n생성된 파일:")
print("  - garch_volatility.csv")
print("  - rolling_correlation_60d.csv")
print("  - volatility_rv_vs_garch.png")
print("  - rolling_correlation_60d.png")
