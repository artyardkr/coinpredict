"""
================================================================================
Step 4: Granger 인과관계 테스트
================================================================================
변동성 간 인과관계를 Granger Causality Test로 검증합니다.

방법론:
1. Granger Causality Test (각 자산 쌍에 대해)
2. ETF 전후 비교
3. 인과관계 네트워크 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("Step 4: Granger 인과관계 테스트")
print("="*80)

# ============================================================================
# 데이터 로드
# ============================================================================
print("\n[Step 4.1] GARCH 변동성 데이터 로드")
print("-"*80)

garch_df = pd.read_csv('garch_volatility.csv', index_col=0, parse_dates=True)
print(f"GARCH 변동성: {garch_df.shape}")

# 결측치 제거
garch_df = garch_df.dropna()
print(f"결측치 제거 후: {garch_df.shape}")

# ETF 승인일
etf_date = pd.Timestamp('2024-01-10')
pre_etf = garch_df[garch_df.index < etf_date]
post_etf = garch_df[garch_df.index >= etf_date]

print(f"\nETF 이전: {len(pre_etf)}일")
print(f"ETF 이후: {len(post_etf)}일")

assets = list(garch_df.columns)
print(f"\n자산: {assets}")

# ============================================================================
# Granger Causality Test
# ============================================================================
print("\n[Step 4.2] Granger Causality Test")
print("-"*80)

def granger_test_pairwise(data, var1, var2, maxlag=5):
    """
    var1 → var2 Granger 인과관계 테스트

    Returns:
        p_value: 최소 p-value (가장 유의한 lag의 p-value)
        best_lag: 최소 p-value를 보이는 lag
    """
    try:
        # var2를 종속변수, var1을 독립변수로 테스트
        test_data = data[[var2, var1]]

        # Granger causality test
        result = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)

        # 각 lag에 대한 F-test p-value 추출
        p_values = []
        for lag in range(1, maxlag + 1):
            # ssr_ftest: SSR F-test
            p_val = result[lag][0]['ssr_ftest'][1]
            p_values.append(p_val)

        min_p = min(p_values)
        best_lag = p_values.index(min_p) + 1

        return min_p, best_lag

    except Exception as e:
        return np.nan, np.nan

def test_all_pairs(data, assets, maxlag=5, alpha=0.05):
    """
    모든 자산 쌍에 대해 Granger 인과관계 테스트

    Returns:
        DataFrame: (cause, effect, p_value, best_lag, significant)
    """
    results = []

    for cause in assets:
        for effect in assets:
            if cause == effect:
                continue

            p_val, best_lag = granger_test_pairwise(data, cause, effect, maxlag)

            results.append({
                'cause': cause,
                'effect': effect,
                'p_value': p_val,
                'best_lag': best_lag,
                'significant': p_val < alpha if not np.isnan(p_val) else False
            })

    return pd.DataFrame(results)

# ETF 이전 Granger 테스트
print("ETF 이전 Granger 인과관계 테스트 중...")
granger_pre = test_all_pairs(pre_etf, assets, maxlag=5, alpha=0.05)
print(f"  완료: {len(granger_pre)}개 쌍 테스트")
print(f"  유의한 인과관계: {granger_pre['significant'].sum()}개")

# ETF 이후 Granger 테스트
print("\nETF 이후 Granger 인과관계 테스트 중...")
granger_post = test_all_pairs(post_etf, assets, maxlag=5, alpha=0.05)
print(f"  완료: {len(granger_post)}개 쌍 테스트")
print(f"  유의한 인과관계: {granger_post['significant'].sum()}개")

# ============================================================================
# BTC 관련 인과관계 분석
# ============================================================================
print("\n[Step 4.3] BTC 관련 인과관계")
print("-"*80)

# BTC를 effect로 하는 인과관계 (X → BTC)
print("\n[X → BTC (BTC에 영향을 주는 자산)]")
btc_in_pre = granger_pre[granger_pre['effect'] == 'Close'].sort_values('p_value')
btc_in_post = granger_post[granger_post['effect'] == 'Close'].sort_values('p_value')

print("\nETF 이전:")
print(btc_in_pre[['cause', 'p_value', 'best_lag', 'significant']].to_string(index=False))

print("\nETF 이후:")
print(btc_in_post[['cause', 'p_value', 'best_lag', 'significant']].to_string(index=False))

# BTC를 cause로 하는 인과관계 (BTC → X)
print("\n[BTC → X (BTC가 영향을 주는 자산)]")
btc_out_pre = granger_pre[granger_pre['cause'] == 'Close'].sort_values('p_value')
btc_out_post = granger_post[granger_post['cause'] == 'Close'].sort_values('p_value')

print("\nETF 이전:")
print(btc_out_pre[['effect', 'p_value', 'best_lag', 'significant']].to_string(index=False))

print("\nETF 이후:")
print(btc_out_post[['effect', 'p_value', 'best_lag', 'significant']].to_string(index=False))

# ============================================================================
# 인과관계 변화 분석
# ============================================================================
print("\n[Step 4.4] 인과관계 변화 분석")
print("-"*80)

# Pre vs Post 비교
granger_compare = granger_pre.merge(
    granger_post,
    on=['cause', 'effect'],
    suffixes=('_pre', '_post')
)

# 인과관계 방향별 변화
print("\n[인과관계 방향별 변화]")
print(f"ETF 이전 유의: {granger_pre['significant'].sum()}개")
print(f"ETF 이후 유의: {granger_post['significant'].sum()}개")

# 새로 생긴 인과관계 (Pre: 비유의 → Post: 유의)
new_causality = granger_compare[
    (~granger_compare['significant_pre']) & (granger_compare['significant_post'])
]
print(f"\n새로 생긴 인과관계: {len(new_causality)}개")
if len(new_causality) > 0:
    print(new_causality[['cause', 'effect', 'p_value_post']].to_string(index=False))

# 사라진 인과관계 (Pre: 유의 → Post: 비유의)
lost_causality = granger_compare[
    (granger_compare['significant_pre']) & (~granger_compare['significant_post'])
]
print(f"\n사라진 인과관계: {len(lost_causality)}개")
if len(lost_causality) > 0:
    print(lost_causality[['cause', 'effect', 'p_value_pre']].to_string(index=False))

# ============================================================================
# 결과 저장
# ============================================================================
print("\n[Step 4.5] 결과 저장")
print("-"*80)

# Granger 테스트 결과
granger_pre.to_csv('volatility_granger_causality_pre.csv', index=False)
granger_post.to_csv('volatility_granger_causality_post.csv', index=False)
print("Granger 인과관계 결과 저장:")
print("  - volatility_granger_causality_pre.csv")
print("  - volatility_granger_causality_post.csv")

# BTC 관련 인과관계
btc_causality = pd.DataFrame({
    'Direction': ['X→BTC'] * len(btc_in_pre) + ['BTC→X'] * len(btc_out_pre),
    'Asset_Pre': list(btc_in_pre['cause']) + list(btc_out_pre['effect']),
    'PValue_Pre': list(btc_in_pre['p_value']) + list(btc_out_pre['p_value']),
    'Significant_Pre': list(btc_in_pre['significant']) + list(btc_out_pre['significant']),
    'Asset_Post': list(btc_in_post['cause']) + list(btc_out_post['effect']),
    'PValue_Post': list(btc_in_post['p_value']) + list(btc_out_post['p_value']),
    'Significant_Post': list(btc_in_post['significant']) + list(btc_out_post['significant'])
})
btc_causality.to_csv('volatility_btc_granger_causality.csv', index=False)
print("  - volatility_btc_granger_causality.csv")

# ============================================================================
# 시각화
# ============================================================================
print("\n[Step 4.6] 인과관계 네트워크 시각화")
print("-"*80)

def plot_causality_network(granger_df, title, ax):
    """
    Granger 인과관계 네트워크 그래프
    """
    # 유의한 인과관계만 추출
    sig = granger_df[granger_df['significant']].copy()

    # NetworkX 그래프 생성
    G = nx.DiGraph()

    # 노드 추가
    for asset in assets:
        G.add_node(asset)

    # 엣지 추가 (유의한 인과관계)
    for _, row in sig.iterrows():
        G.add_edge(row['cause'], row['effect'], weight=1 - row['p_value'])

    # 레이아웃
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # 노드 색상 (BTC는 빨강, 나머지는 파랑)
    node_colors = ['red' if node == 'Close' else 'lightblue' for node in G.nodes()]

    # 그리기
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    nx.draw_networkx_edges(
        G, pos,
        edge_color='gray',
        arrows=True,
        arrowsize=20,
        arrowstyle='->',
        width=2,
        alpha=0.6,
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    # 통계
    n_edges = len(sig)
    ax.text(
        0.02, 0.98,
        f'유의한 인과관계: {n_edges}개',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

# 그래프 생성
fig, axes = plt.subplots(1, 2, figsize=(20, 9))

plot_causality_network(granger_pre, 'ETF 이전 Granger 인과관계', axes[0])
plot_causality_network(granger_post, 'ETF 이후 Granger 인과관계', axes[1])

plt.tight_layout()
plt.savefig('volatility_granger_causality_network.png', dpi=300, bbox_inches='tight')
print("시각화 저장: volatility_granger_causality_network.png")
plt.close()

# BTC 중심 인과관계 바차트
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# X → BTC (ETF 이전)
ax1 = axes[0, 0]
btc_in_pre_plot = btc_in_pre.sort_values('p_value')
ax1.barh(btc_in_pre_plot['cause'], -np.log10(btc_in_pre_plot['p_value']), color='steelblue', alpha=0.8)
ax1.axvline(x=-np.log10(0.05), color='red', linestyle='--', label='α=0.05')
ax1.set_xlabel('-log10(p-value)')
ax1.set_ylabel('자산')
ax1.set_title('X → BTC (ETF 이전)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# X → BTC (ETF 이후)
ax2 = axes[0, 1]
btc_in_post_plot = btc_in_post.sort_values('p_value')
ax2.barh(btc_in_post_plot['cause'], -np.log10(btc_in_post_plot['p_value']), color='coral', alpha=0.8)
ax2.axvline(x=-np.log10(0.05), color='red', linestyle='--', label='α=0.05')
ax2.set_xlabel('-log10(p-value)')
ax2.set_ylabel('자산')
ax2.set_title('X → BTC (ETF 이후)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# BTC → X (ETF 이전)
ax3 = axes[1, 0]
btc_out_pre_plot = btc_out_pre.sort_values('p_value')
ax3.barh(btc_out_pre_plot['effect'], -np.log10(btc_out_pre_plot['p_value']), color='steelblue', alpha=0.8)
ax3.axvline(x=-np.log10(0.05), color='red', linestyle='--', label='α=0.05')
ax3.set_xlabel('-log10(p-value)')
ax3.set_ylabel('자산')
ax3.set_title('BTC → X (ETF 이전)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# BTC → X (ETF 이후)
ax4 = axes[1, 1]
btc_out_post_plot = btc_out_post.sort_values('p_value')
ax4.barh(btc_out_post_plot['effect'], -np.log10(btc_out_post_plot['p_value']), color='coral', alpha=0.8)
ax4.axvline(x=-np.log10(0.05), color='red', linestyle='--', label='α=0.05')
ax4.set_xlabel('-log10(p-value)')
ax4.set_ylabel('자산')
ax4.set_title('BTC → X (ETF 이후)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('volatility_btc_granger_causality.png', dpi=300, bbox_inches='tight')
print("시각화 저장: volatility_btc_granger_causality.png")
plt.close()

# ============================================================================
# 요약
# ============================================================================
print("\n" + "="*80)
print("Step 4 완료: Granger 인과관계 테스트")
print("="*80)

# BTC 관련 유의한 인과관계
btc_in_sig_pre = btc_in_pre[btc_in_pre['significant']]
btc_in_sig_post = btc_in_post[btc_in_post['significant']]
btc_out_sig_pre = btc_out_pre[btc_out_pre['significant']]
btc_out_sig_post = btc_out_post[btc_out_post['significant']]

print(f"""
주요 결과:

1. 전체 인과관계
   - ETF 이전 유의: {granger_pre['significant'].sum()}/{len(granger_pre)}
   - ETF 이후 유의: {granger_post['significant'].sum()}/{len(granger_post)}
   - 변화: {granger_post['significant'].sum() - granger_pre['significant'].sum():+d}개

2. BTC 관련 인과관계 (α=0.05)

   [X → BTC] BTC에 영향을 주는 자산
   - ETF 이전: {len(btc_in_sig_pre)}개 ({', '.join(btc_in_sig_pre['cause'].tolist()) if len(btc_in_sig_pre) > 0 else 'None'})
   - ETF 이후: {len(btc_in_sig_post)}개 ({', '.join(btc_in_sig_post['cause'].tolist()) if len(btc_in_sig_post) > 0 else 'None'})

   [BTC → X] BTC가 영향을 주는 자산
   - ETF 이전: {len(btc_out_sig_pre)}개 ({', '.join(btc_out_sig_pre['effect'].tolist()) if len(btc_out_sig_pre) > 0 else 'None'})
   - ETF 이후: {len(btc_out_sig_post)}개 ({', '.join(btc_out_sig_post['effect'].tolist()) if len(btc_out_sig_post) > 0 else 'None'})

3. 인과관계 변화
   - 새로 생긴 인과관계: {len(new_causality)}개
   - 사라진 인과관계: {len(lost_causality)}개

생성된 파일:
  - volatility_granger_causality_pre.csv
  - volatility_granger_causality_post.csv
  - volatility_btc_granger_causality.csv
  - volatility_granger_causality_network.png
  - volatility_btc_granger_causality.png
""")
