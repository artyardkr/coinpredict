"""
================================================================================
Step 5: 계층적 DCC-GARCH 분석 (Hierarchical DCC-GARCH)
================================================================================
카테고리별로 DCC-GARCH를 수행하여 동적 상관관계(Dynamic Conditional Correlation)를 추정합니다.

DCC-GARCH란?
- Dynamic Conditional Correlation GARCH
- 시간에 따라 변하는 조건부 상관계수를 추정하는 다변량 GARCH 모형
- 정적인 상관계수가 아닌, 매일 변하는 동적 상관관계 파악 가능

카테고리:
1. 전통자산 (6개): BTC, SPX, QQQ, VIX, GOLD, TLT
2. 거시경제 (7개): DFF, SOFR, RRPONTSYD, CPIAUCSL, GDP, M2SL, DGS10
3. 온체인 (5개): bc_hash_rate, bc_difficulty, bc_n_transactions, bc_transaction_fees, bc_miners_revenue
4. 밸류에이션 (4개): NVT_Ratio, MVRV, SOPR, Puell_Multiple
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from arch.univariate import GARCH, ConstantMean
from arch.univariate.distribution import Normal
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("Step 5: 계층적 DCC-GARCH 분석")
print("="*80)

# ============================================================================
# 카테고리 정의
# ============================================================================
print("\n[Step 5.1] 카테고리별 변수 정의")
print("-"*80)

categories = {
    '전통자산': {
        'variables': ['Close', 'SPX', 'QQQ', 'VIX', 'GOLD', 'TLT'],
        'description': 'BTC와 전통 금융자산 간 동적 상관관계'
    },
    '거시경제': {
        'variables': ['Close', 'DFF', 'SOFR', 'RRPONTSYD', 'CPIAUCSL', 'M2SL', 'DGS10'],
        'description': 'BTC와 거시경제 지표 간 동적 상관관계'
    },
    '온체인': {
        'variables': ['Close', 'bc_hash_rate', 'bc_difficulty', 'bc_n_transactions',
                     'bc_transaction_fees', 'bc_miners_revenue'],
        'description': 'BTC 가격과 온체인 지표 간 동적 상관관계'
    },
    '밸류에이션': {
        'variables': ['Close', 'NVT_Ratio', 'MVRV', 'SOPR', 'Puell_Multiple'],
        'description': 'BTC 가격과 밸류에이션 지표 간 동적 상관관계'
    }
}

print("카테고리별 변수:")
for cat, info in categories.items():
    print(f"  {cat} ({len(info['variables'])}개): {', '.join(info['variables'])}")

# ============================================================================
# 데이터 로드
# ============================================================================
print("\n[Step 5.2] 데이터 로드")
print("-"*80)

# 전체 데이터 로드
df = pd.read_csv('integrated_data_full_v2.csv', index_col='Date', parse_dates=True)
print(f"전체 데이터: {df.shape}")

# ETF 승인일
etf_date = pd.Timestamp('2024-01-10')

# ============================================================================
# DCC-GARCH 함수 정의
# ============================================================================

def estimate_dcc_garch_simple(returns_df, category_name):
    """
    간소화된 DCC-GARCH 추정

    Python의 arch 패키지는 완전한 DCC-GARCH를 지원하지 않으므로,
    단계별로 구현:
    1. 각 자산별 GARCH(1,1) 추정 → 표준화 잔차
    2. 표준화 잔차의 Rolling Correlation 계산 (동적 상관관계 대용)

    Returns:
        dynamic_corr: DataFrame, BTC와 다른 자산 간 동적 상관계수
    """
    print(f"\n  [{category_name}] DCC-GARCH 추정 중...")

    n_assets = returns_df.shape[1]
    asset_names = returns_df.columns.tolist()

    # Step 1: 각 자산별 GARCH(1,1) 추정
    print(f"    Step 1: {n_assets}개 자산 GARCH(1,1) 추정...")
    standardized_residuals = pd.DataFrame(index=returns_df.index)

    for col in asset_names:
        try:
            # 수익률을 백분율로 변환 (수치 안정성)
            returns_pct = returns_df[col].dropna() * 100

            # GARCH(1,1) 모형
            model = arch_model(returns_pct, vol='Garch', p=1, q=1, rescale=False)
            result = model.fit(disp='off', show_warning=False)

            # 표준화 잔차 = 잔차 / 조건부 표준편차
            std_resid = result.resid / result.conditional_volatility

            standardized_residuals[col] = std_resid

        except Exception as e:
            print(f"      {col} GARCH 추정 실패: {e}")
            standardized_residuals[col] = np.nan

    # 결측치 제거
    standardized_residuals = standardized_residuals.dropna()
    print(f"    표준화 잔차: {standardized_residuals.shape}")

    # Step 2: 동적 상관관계 계산 (Rolling Correlation of Standardized Residuals)
    print(f"    Step 2: 동적 상관관계 계산 (60일 롤링)...")

    window = 60
    dynamic_corr = pd.DataFrame(index=standardized_residuals.index)

    # BTC(Close)와 다른 자산들 간 동적 상관계수
    btc_col = 'Close'
    if btc_col in standardized_residuals.columns:
        for col in standardized_residuals.columns:
            if col != btc_col:
                rolling_corr = standardized_residuals[btc_col].rolling(window=window).corr(
                    standardized_residuals[col]
                )
                dynamic_corr[f'BTC_vs_{col}'] = rolling_corr

    dynamic_corr = dynamic_corr.dropna()
    print(f"    동적 상관관계: {dynamic_corr.shape}")

    return dynamic_corr, standardized_residuals

# ============================================================================
# 카테고리별 DCC-GARCH 실행
# ============================================================================
print("\n[Step 5.3] 카테고리별 DCC-GARCH 실행")
print("-"*80)

dcc_results = {}

for cat, info in categories.items():
    print(f"\n{'='*80}")
    print(f"카테고리: {cat}")
    print(f"설명: {info['description']}")
    print(f"변수: {', '.join(info['variables'])}")
    print(f"{'='*80}")

    # 사용 가능한 변수만 선택
    available_vars = [v for v in info['variables'] if v in df.columns]

    if len(available_vars) < 2:
        print(f"  ⚠️ 사용 가능한 변수 부족 ({len(available_vars)}개), 건너뜀")
        continue

    print(f"  사용 가능한 변수: {len(available_vars)}개")

    # 데이터 추출
    df_cat = df[available_vars].copy()

    # 결측치 처리
    df_cat = df_cat.fillna(method='ffill').fillna(method='bfill').dropna()
    print(f"  데이터 shape: {df_cat.shape}")

    # 수익률 계산
    returns_cat = df_cat.pct_change().dropna()

    # 극단값 제거 (±10% 초과 변화는 윈저화)
    for col in returns_cat.columns:
        q01 = returns_cat[col].quantile(0.01)
        q99 = returns_cat[col].quantile(0.99)
        returns_cat[col] = returns_cat[col].clip(lower=q01, upper=q99)

    print(f"  수익률 데이터: {returns_cat.shape}")
    print(f"  기간: {returns_cat.index[0].date()} ~ {returns_cat.index[-1].date()}")

    # DCC-GARCH 추정
    try:
        dynamic_corr, std_resid = estimate_dcc_garch_simple(returns_cat, cat)

        dcc_results[cat] = {
            'dynamic_corr': dynamic_corr,
            'std_resid': std_resid,
            'variables': available_vars
        }

        print(f"  ✅ DCC-GARCH 추정 완료")

    except Exception as e:
        print(f"  ❌ DCC-GARCH 추정 실패: {e}")
        continue

# ============================================================================
# ETF 전후 동적 상관관계 비교
# ============================================================================
print("\n\n[Step 5.4] ETF 전후 동적 상관관계 비교")
print("-"*80)

comparison_results = []

for cat, result in dcc_results.items():
    print(f"\n{cat} 카테고리:")
    print("-"*60)

    dynamic_corr = result['dynamic_corr']

    # ETF 전후 분할
    pre_corr = dynamic_corr[dynamic_corr.index < etf_date]
    post_corr = dynamic_corr[dynamic_corr.index >= etf_date]

    for col in dynamic_corr.columns:
        pre_mean = pre_corr[col].mean()
        post_mean = post_corr[col].mean()
        change = post_mean - pre_mean

        # 변동성 (표준편차) 변화
        pre_std = pre_corr[col].std()
        post_std = post_corr[col].std()

        asset_name = col.replace('BTC_vs_', '')

        print(f"  {asset_name:20s}: 평균 {pre_mean:+.3f} → {post_mean:+.3f} ({change:+.3f}), "
              f"변동성 {pre_std:.3f} → {post_std:.3f}")

        comparison_results.append({
            'Category': cat,
            'Asset': asset_name,
            'Corr_Pre': pre_mean,
            'Corr_Post': post_mean,
            'Corr_Change': change,
            'Std_Pre': pre_std,
            'Std_Post': post_std
        })

comparison_df = pd.DataFrame(comparison_results)

# ============================================================================
# 결과 저장
# ============================================================================
print("\n\n[Step 5.5] 결과 저장")
print("-"*80)

# 카테고리별 동적 상관관계 저장
for cat, result in dcc_results.items():
    filename = f'dcc_garch_{cat}_dynamic_corr.csv'
    result['dynamic_corr'].to_csv(filename)
    print(f"  {filename}")

# 비교 결과 저장
comparison_df.to_csv('dcc_garch_comparison_summary.csv', index=False)
print(f"  dcc_garch_comparison_summary.csv")

# ============================================================================
# 시각화
# ============================================================================
print("\n[Step 5.6] 시각화 생성")
print("-"*80)

# 각 카테고리별 동적 상관관계 시계열 플롯
for cat, result in dcc_results.items():
    dynamic_corr = result['dynamic_corr']

    n_assets = dynamic_corr.shape[1]

    # 서브플롯 개수 계산
    n_cols = 2
    n_rows = (n_assets + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_assets > 1 else [axes]

    for idx, col in enumerate(dynamic_corr.columns):
        ax = axes[idx]

        # 동적 상관계수 플롯
        ax.plot(dynamic_corr.index, dynamic_corr[col], linewidth=1, alpha=0.7, label='동적 상관계수')

        # ETF 승인일 표시
        ax.axvline(x=etf_date, color='red', linestyle='--', linewidth=2, alpha=0.7, label='ETF 승인')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

        # ETF 전후 평균선
        pre = dynamic_corr.loc[dynamic_corr.index < etf_date, col].mean()
        post = dynamic_corr.loc[dynamic_corr.index >= etf_date, col].mean()

        ax.axhline(y=pre, color='blue', linestyle=':', linewidth=1.5, alpha=0.5,
                  label=f'ETF 이전 평균 ({pre:.3f})')
        ax.axhline(y=post, color='orange', linestyle=':', linewidth=1.5, alpha=0.5,
                  label=f'ETF 이후 평균 ({post:.3f})')

        asset_name = col.replace('BTC_vs_', '')
        ax.set_title(f'{asset_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('날짜')
        ax.set_ylabel('상관계수')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    # 빈 서브플롯 숨기기
    for idx in range(n_assets, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'{cat} - BTC 동적 상관관계 (DCC-GARCH)', fontsize=16, fontweight='bold')
    plt.tight_layout()

    filename = f'dcc_garch_{cat}_plot.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  {filename}")
    plt.close()

# 종합 비교 플롯
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

cat_idx = 0
for cat, result in dcc_results.items():
    if cat_idx >= 4:
        break

    ax = axes[cat_idx]
    dynamic_corr = result['dynamic_corr']

    # 평균 동적 상관계수 (카테고리 내 모든 자산 평균)
    mean_corr = dynamic_corr.mean(axis=1)

    ax.plot(mean_corr.index, mean_corr, linewidth=2, label='평균 동적 상관계수')
    ax.axvline(x=etf_date, color='red', linestyle='--', linewidth=2, alpha=0.7, label='ETF 승인')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    # ETF 전후 평균
    pre_mean = mean_corr[mean_corr.index < etf_date].mean()
    post_mean = mean_corr[mean_corr.index >= etf_date].mean()

    ax.axhline(y=pre_mean, color='blue', linestyle=':', linewidth=1.5, alpha=0.5,
              label=f'ETF 이전 평균 ({pre_mean:.3f})')
    ax.axhline(y=post_mean, color='orange', linestyle=':', linewidth=1.5, alpha=0.5,
              label=f'ETF 이후 평균 ({post_mean:.3f})')

    ax.set_title(f'{cat} 카테고리', fontsize=14, fontweight='bold')
    ax.set_xlabel('날짜')
    ax.set_ylabel('평균 상관계수')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    cat_idx += 1

plt.suptitle('카테고리별 BTC 평균 동적 상관관계', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('dcc_garch_category_comparison.png', dpi=300, bbox_inches='tight')
print(f"  dcc_garch_category_comparison.png")
plt.close()

# ============================================================================
# 요약
# ============================================================================
print("\n" + "="*80)
print("Step 5 완료: 계층적 DCC-GARCH 분석")
print("="*80)

print(f"""
분석된 카테고리: {len(dcc_results)}개

카테고리별 주요 발견:
""")

for cat, result in dcc_results.items():
    dynamic_corr = result['dynamic_corr']

    # ETF 전후 평균 상관계수
    pre = dynamic_corr[dynamic_corr.index < etf_date].mean().mean()
    post = dynamic_corr[dynamic_corr.index >= etf_date].mean().mean()
    change = post - pre

    print(f"\n{cat}:")
    print(f"  변수: {len(result['variables'])}개")
    print(f"  평균 동적 상관계수: {pre:.3f} → {post:.3f} ({change:+.3f})")

    # 가장 큰 변화를 보인 자산
    changes = []
    for col in dynamic_corr.columns:
        pre_val = dynamic_corr.loc[dynamic_corr.index < etf_date, col].mean()
        post_val = dynamic_corr.loc[dynamic_corr.index >= etf_date, col].mean()
        changes.append((col.replace('BTC_vs_', ''), post_val - pre_val))

    changes.sort(key=lambda x: abs(x[1]), reverse=True)
    top_change = changes[0]
    print(f"  최대 변화: {top_change[0]} ({top_change[1]:+.3f})")

print(f"""
생성된 파일:
  - 카테고리별 동적 상관관계 CSV: {len(dcc_results)}개
  - 비교 요약: dcc_garch_comparison_summary.csv
  - 시각화: {len(dcc_results) + 1}개 PNG 파일
""")
