"""
================================================================================
Step 3: VAR 기반 변동성 전이(Spillover) 분석
================================================================================
GARCH 변동성 간 상호 전이 효과를 VAR 모형으로 분석합니다.

방법론:
1. VAR(p) 모형 추정 (최적 차수 선택)
2. 분산분해(Variance Decomposition) - 변동성 기여도
3. 충격반응함수(IRF) - 충격 전파 경로
4. Spillover Index (Diebold-Yilmaz) - 시스템 전체 전이도
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("Step 3: VAR 기반 변동성 전이(Spillover) 분석")
print("="*80)

# ============================================================================
# 데이터 로드
# ============================================================================
print("\n[Step 3.1] GARCH 변동성 데이터 로드")
print("-"*80)

garch_df = pd.read_csv('garch_volatility.csv', index_col=0, parse_dates=True)
print(f"GARCH 변동성: {garch_df.shape}")
print(f"자산: {list(garch_df.columns)}")

# 결측치 제거
garch_df = garch_df.dropna()
print(f"결측치 제거 후: {garch_df.shape}")

# ETF 승인일
etf_date = pd.Timestamp('2024-01-10')
pre_etf = garch_df[garch_df.index < etf_date]
post_etf = garch_df[garch_df.index >= etf_date]

print(f"\nETF 이전: {len(pre_etf)}일")
print(f"ETF 이후: {len(post_etf)}일")

# ============================================================================
# VAR 모형 추정 및 최적 차수 선택
# ============================================================================
print("\n[Step 3.2] VAR 모형 추정")
print("-"*80)

from statsmodels.tsa.api import VAR

def estimate_var_model(data, maxlags=10):
    """VAR 모형 추정 및 최적 차수 선택"""
    model = VAR(data)

    # 최적 차수 선택 (AIC 기준)
    aic_results = []
    bic_results = []

    for p in range(1, min(maxlags + 1, len(data) // 3)):
        try:
            result = model.fit(p)
            aic_results.append((p, result.aic))
            bic_results.append((p, result.bic))
        except:
            continue

    if not aic_results:
        return None, None

    optimal_p_aic = min(aic_results, key=lambda x: x[1])[0]
    optimal_p_bic = min(bic_results, key=lambda x: x[1])[0]

    # BIC로 선택 (더 보수적)
    optimal_p = optimal_p_bic
    result = model.fit(optimal_p)

    return result, optimal_p

# ETF 이전 VAR 모형
print("ETF 이전 VAR 모형 추정 중...")
var_pre, p_pre = estimate_var_model(pre_etf, maxlags=10)
print(f"  최적 차수: p = {p_pre}")
print(f"  AIC: {var_pre.aic:.2f}")
print(f"  BIC: {var_pre.bic:.2f}")

# ETF 이후 VAR 모형
print("\nETF 이후 VAR 모형 추정 중...")
var_post, p_post = estimate_var_model(post_etf, maxlags=10)
print(f"  최적 차수: p = {p_post}")
print(f"  AIC: {var_post.aic:.2f}")
print(f"  BIC: {var_post.bic:.2f}")

# ============================================================================
# 분산분해 (Variance Decomposition)
# ============================================================================
print("\n[Step 3.3] 분산분해 (Variance Decomposition)")
print("-"*80)

def compute_variance_decomposition(var_result, steps=10):
    """분산분해 계산"""
    fevd = var_result.fevd(steps)
    return fevd

# ETF 이전 분산분해
print("ETF 이전 분산분해 계산 중...")
fevd_pre = compute_variance_decomposition(var_pre, steps=10)

# ETF 이후 분산분해
print("ETF 이후 분산분해 계산 중...")
fevd_post = compute_variance_decomposition(var_post, steps=10)

# BTC 변동성에 대한 각 자산의 기여도 (10일 후)
print("\n[BTC 변동성 분산분해 - 10일 후]")
# fevd.decomp shape: (n_vars, steps, n_vars)
# decomp[i, j, k]: i번째 변수의 변동성 중 k번째 변수의 충격이 설명하는 비율 (j일 후)
btc_fevd_pre = fevd_pre.decomp[0, -1, :]  # BTC(0번)에 대한 모든 변수의 기여도 (10일 후)
btc_fevd_post = fevd_post.decomp[0, -1, :]

btc_decomp_df = pd.DataFrame({
    'Pre_ETF': btc_fevd_pre,
    'Post_ETF': btc_fevd_post,
    'Change': btc_fevd_post - btc_fevd_pre
}, index=garch_df.columns)

btc_decomp_df = btc_decomp_df.sort_values('Post_ETF', ascending=False)
print(btc_decomp_df.round(4))

# ============================================================================
# Spillover Index (Diebold-Yilmaz, 2012)
# ============================================================================
print("\n[Step 3.4] Spillover Index 계산")
print("-"*80)

def calculate_spillover_index(fevd_result):
    """
    Diebold-Yilmaz Spillover Index 계산

    Spillover Index = (비대각선 요소 합 / 전체 합) × 100
    """
    # 10일 후 분산분해 행렬
    # fevd.decomp shape: (n_vars, steps, n_vars)
    # decomp[i, j, k]: i번째 변수의 변동성 중 k번째 변수의 충격이 설명하는 비율 (j일 후)
    # 10일 후(마지막 스텝)의 분산분해 행렬 추출
    decomp = fevd_result.decomp[:, -1, :]  # shape: (n_vars, n_vars)

    n = decomp.shape[0]

    # 총 전이(Total Spillover)
    total_sum = np.sum(decomp)
    off_diagonal_sum = total_sum - np.trace(decomp)
    spillover_index = (off_diagonal_sum / total_sum) * 100

    # 방향성 전이(Directional Spillover)
    # TO: i가 다른 변수들에게 주는 영향 (열 합계 - 자기 자신)
    spillover_to = np.sum(decomp, axis=0) - np.diag(decomp)

    # FROM: i가 다른 변수들로부터 받는 영향 (행 합계 - 자기 자신)
    spillover_from = np.sum(decomp, axis=1) - np.diag(decomp)

    # NET: TO - FROM
    spillover_net = spillover_to - spillover_from

    return {
        'spillover_index': spillover_index,
        'spillover_to': spillover_to,
        'spillover_from': spillover_from,
        'spillover_net': spillover_net,
        'decomp_matrix': decomp
    }

# ETF 이전 Spillover
spillover_pre = calculate_spillover_index(fevd_pre)
print(f"ETF 이전 Spillover Index: {spillover_pre['spillover_index']:.2f}%")

# ETF 이후 Spillover
spillover_post = calculate_spillover_index(fevd_post)
print(f"ETF 이후 Spillover Index: {spillover_post['spillover_index']:.2f}%")

print(f"\n변화: {spillover_post['spillover_index'] - spillover_pre['spillover_index']:+.2f}%p")

# 방향성 Spillover 비교
spillover_df = pd.DataFrame({
    'TO_Pre': spillover_pre['spillover_to'],
    'TO_Post': spillover_post['spillover_to'],
    'FROM_Pre': spillover_pre['spillover_from'],
    'FROM_Post': spillover_post['spillover_from'],
    'NET_Pre': spillover_pre['spillover_net'],
    'NET_Post': spillover_post['spillover_net']
}, index=garch_df.columns)

print("\n[방향성 Spillover]")
print("TO: 해당 자산이 다른 자산들에게 주는 영향")
print("FROM: 해당 자산이 다른 자산들로부터 받는 영향")
print("NET: TO - FROM (양수면 영향을 주는 쪽, 음수면 받는 쪽)")
print(spillover_df.round(4))

# ============================================================================
# 충격반응함수 (Impulse Response Function)
# ============================================================================
print("\n[Step 3.5] 충격반응함수 (IRF) 계산")
print("-"*80)

# ETF 이전 IRF
print("ETF 이전 IRF 계산 중...")
irf_pre = var_pre.irf(10)

# ETF 이후 IRF
print("ETF 이후 IRF 계산 중...")
irf_post = var_post.irf(10)

print("IRF 계산 완료")

# ============================================================================
# 결과 저장
# ============================================================================
print("\n[Step 3.6] 결과 저장")
print("-"*80)

# BTC 분산분해
btc_decomp_df.to_csv('volatility_btc_variance_decomposition.csv')
print("BTC 분산분해 저장: volatility_btc_variance_decomposition.csv")

# Spillover Index
spillover_df.to_csv('volatility_spillover_index.csv')
print("Spillover Index 저장: volatility_spillover_index.csv")

# ============================================================================
# 시각화
# ============================================================================
print("\n[Step 3.7] 시각화 생성")
print("-"*80)

fig = plt.figure(figsize=(20, 12))

# 1. BTC 분산분해 비교
ax1 = plt.subplot(2, 3, 1)
x = np.arange(len(btc_decomp_df))
width = 0.35
ax1.bar(x - width/2, btc_decomp_df['Pre_ETF'], width, label='ETF 이전', alpha=0.8)
ax1.bar(x + width/2, btc_decomp_df['Post_ETF'], width, label='ETF 이후', alpha=0.8)
ax1.set_xlabel('자산')
ax1.set_ylabel('기여도')
ax1.set_title('BTC 변동성 분산분해 (10일 후)')
ax1.set_xticks(x)
ax1.set_xticklabels(btc_decomp_df.index, rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Spillover TO 비교
ax2 = plt.subplot(2, 3, 2)
x = np.arange(len(spillover_df))
ax2.bar(x - width/2, spillover_df['TO_Pre'], width, label='ETF 이전', alpha=0.8)
ax2.bar(x + width/2, spillover_df['TO_Post'], width, label='ETF 이후', alpha=0.8)
ax2.set_xlabel('자산')
ax2.set_ylabel('Spillover TO')
ax2.set_title('변동성 전이 (TO others)')
ax2.set_xticks(x)
ax2.set_xticklabels(spillover_df.index, rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Spillover FROM 비교
ax3 = plt.subplot(2, 3, 3)
ax3.bar(x - width/2, spillover_df['FROM_Pre'], width, label='ETF 이전', alpha=0.8)
ax3.bar(x + width/2, spillover_df['FROM_Post'], width, label='ETF 이후', alpha=0.8)
ax3.set_xlabel('자산')
ax3.set_ylabel('Spillover FROM')
ax3.set_title('변동성 수용 (FROM others)')
ax3.set_xticks(x)
ax3.set_xticklabels(spillover_df.index, rotation=45)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. NET Spillover 비교
ax4 = plt.subplot(2, 3, 4)
ax4.bar(x - width/2, spillover_df['NET_Pre'], width, label='ETF 이전', alpha=0.8)
ax4.bar(x + width/2, spillover_df['NET_Post'], width, label='ETF 이후', alpha=0.8)
ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax4.set_xlabel('자산')
ax4.set_ylabel('NET Spillover')
ax4.set_title('순 변동성 전이 (TO - FROM)')
ax4.set_xticks(x)
ax4.set_xticklabels(spillover_df.index, rotation=45)
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. IRF: SPX → BTC (ETF 이전 vs 이후)
ax5 = plt.subplot(2, 3, 5)
# SPX가 BTC에 미치는 충격 (SPX는 인덱스 1, BTC는 인덱스 0)
spx_to_btc_pre = irf_pre.irfs[:, 0, 1]  # [steps, response_var, impulse_var]
spx_to_btc_post = irf_post.irfs[:, 0, 1]
ax5.plot(spx_to_btc_pre, marker='o', label='ETF 이전', linewidth=2)
ax5.plot(spx_to_btc_post, marker='s', label='ETF 이후', linewidth=2)
ax5.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
ax5.set_xlabel('기간 (일)')
ax5.set_ylabel('반응')
ax5.set_title('IRF: SPX 충격 → BTC 반응')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. IRF: VIX → BTC (ETF 이전 vs 이후)
ax6 = plt.subplot(2, 3, 6)
# VIX가 BTC에 미치는 충격 (VIX는 인덱스 3)
vix_to_btc_pre = irf_pre.irfs[:, 0, 3]
vix_to_btc_post = irf_post.irfs[:, 0, 3]
ax6.plot(vix_to_btc_pre, marker='o', label='ETF 이전', linewidth=2)
ax6.plot(vix_to_btc_post, marker='s', label='ETF 이후', linewidth=2)
ax6.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
ax6.set_xlabel('기간 (일)')
ax6.set_ylabel('반응')
ax6.set_title('IRF: VIX 충격 → BTC 반응')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('volatility_spillover_analysis.png', dpi=300, bbox_inches='tight')
print("시각화 저장: volatility_spillover_analysis.png")
plt.close()

# ============================================================================
# 요약
# ============================================================================
print("\n" + "="*80)
print("Step 3 완료: VAR 기반 변동성 전이(Spillover) 분석")
print("="*80)

print(f"""
주요 결과:

1. VAR 모형
   - ETF 이전: VAR({p_pre})
   - ETF 이후: VAR({p_post})

2. Spillover Index (시스템 전체 변동성 전이도)
   - ETF 이전: {spillover_pre['spillover_index']:.2f}%
   - ETF 이후: {spillover_post['spillover_index']:.2f}%
   - 변화: {spillover_post['spillover_index'] - spillover_pre['spillover_index']:+.2f}%p

3. BTC 변동성 분산분해 (10일 후 기여도)
   - 자체 기여: {btc_decomp_df.loc['Close', 'Pre_ETF']:.2%} → {btc_decomp_df.loc['Close', 'Post_ETF']:.2%}
   - SPX 기여: {btc_decomp_df.loc['SPX', 'Pre_ETF']:.2%} → {btc_decomp_df.loc['SPX', 'Post_ETF']:.2%}
   - VIX 기여: {btc_decomp_df.loc['VIX', 'Pre_ETF']:.2%} → {btc_decomp_df.loc['VIX', 'Post_ETF']:.2%}

생성된 파일:
  - volatility_btc_variance_decomposition.csv
  - volatility_spillover_index.csv
  - volatility_spillover_analysis.png
""")
