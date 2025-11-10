"""
================================================================================
Step 6: Markov Switching Regime Analysis
================================================================================
Markov Switching Model (Hamilton, 1989)을 사용하여 BTC 시장의 레짐을 자동 감지합니다.

방법론:
- 2-State Markov Switching Model
- 각 레짐(State)마다 다른 평균 수익률과 변동성
- 레짐 간 전환 확률을 추정
- 실시간 레짐 확률 계산 (Filtering Probability)

레짐 해석:
- Regime 0: 저변동성 레짐 (보통 Risk-On, 강세장)
- Regime 1: 고변동성 레짐 (보통 Risk-Off, 약세장)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("Step 6: Markov Switching Regime Analysis")
print("="*80)

# ============================================================================
# 데이터 로드
# ============================================================================
print("\n[Step 6.1] 데이터 로드")
print("-"*80)

# 전체 데이터 로드
df = pd.read_csv('integrated_data_full_v2.csv', index_col='Date', parse_dates=True)
print(f"전체 데이터: {df.shape}")

# ETF 승인일
etf_date = pd.Timestamp('2024-01-10')

# BTC 가격 및 주요 변수
btc_price = df['Close'].copy()
vix = df['VIX'].copy() if 'VIX' in df.columns else None

# 결측치 처리
btc_price = btc_price.fillna(method='ffill').dropna()

# 수익률 계산
returns = btc_price.pct_change().dropna()
returns = returns * 100  # 백분율로 변환 (수치 안정성)

print(f"BTC 수익률 데이터: {len(returns)}일")
print(f"기간: {returns.index[0].date()} ~ {returns.index[-1].date()}")

# 기초 통계
print(f"\nBTC 수익률 통계:")
print(f"  평균: {returns.mean():.4f}%")
print(f"  표준편차: {returns.std():.4f}%")
print(f"  최소: {returns.min():.4f}%")
print(f"  최대: {returns.max():.4f}%")

# ETF 전후 분리
returns_pre = returns[returns.index < etf_date]
returns_post = returns[returns.index >= etf_date]

print(f"\nETF 이전: {len(returns_pre)}일")
print(f"ETF 이후: {len(returns_post)}일")

# ============================================================================
# Markov Switching Model 추정
# ============================================================================
print("\n[Step 6.2] Markov Switching Model 추정")
print("-"*80)

def estimate_markov_switching(returns_data, name="전체", k_regimes=2):
    """
    Markov Switching Model 추정

    Parameters:
    -----------
    returns_data : Series
        수익률 시계열
    name : str
        데이터 이름 (출력용)
    k_regimes : int
        레짐 개수 (기본값: 2)

    Returns:
    --------
    result : MarkovRegressionResults
        추정 결과
    """
    print(f"\n  [{name}] Markov Switching 추정 중...")
    print(f"    관측치: {len(returns_data)}일")

    try:
        # Markov Switching Model
        # switching_variance=True: 레짐별로 다른 분산
        model = MarkovRegression(
            returns_data,
            k_regimes=k_regimes,
            trend='c',  # constant (절편만)
            switching_variance=True
        )

        result = model.fit(maxiter=1000, disp=False)

        print(f"    ✅ 추정 완료")
        print(f"    Log-Likelihood: {result.llf:.2f}")
        print(f"    AIC: {result.aic:.2f}")
        print(f"    BIC: {result.bic:.2f}")

        return result

    except Exception as e:
        print(f"    ❌ 추정 실패: {e}")
        return None

# 전체 기간 Markov Switching
print("\n" + "="*80)
print("전체 기간 (2021-2025)")
print("="*80)
ms_full = estimate_markov_switching(returns, "전체 기간", k_regimes=2)

# ETF 이전 Markov Switching
print("\n" + "="*80)
print("ETF 이전 (2021-2024.01)")
print("="*80)
ms_pre = estimate_markov_switching(returns_pre, "ETF 이전", k_regimes=2)

# ETF 이후 Markov Switching
print("\n" + "="*80)
print("ETF 이후 (2024.01-2025)")
print("="*80)
ms_post = estimate_markov_switching(returns_post, "ETF 이후", k_regimes=2)

# ============================================================================
# 레짐 특성 분석
# ============================================================================
print("\n\n[Step 6.3] 레짐 특성 분석")
print("-"*80)

def analyze_regime_characteristics(result, name="전체"):
    """
    레짐별 특성 추출 및 출력
    """
    if result is None:
        return None

    print(f"\n{name} 레짐 특성:")
    print("-"*60)

    # 레짐별 평균 (절편)
    means = result.params[result.params.index.str.contains('const')]

    # 레짐별 분산 (표준편차로 변환)
    sigmas = result.params[result.params.index.str.contains('sigma2')]
    stds = np.sqrt(sigmas)

    regime_chars = []

    for i in range(len(means)):
        mean = means.iloc[i]
        std = stds.iloc[i]

        regime_chars.append({
            'Regime': i,
            'Mean_Return': mean,
            'Volatility': std,
            'Annualized_Return': mean * 252,
            'Annualized_Vol': std * np.sqrt(252)
        })

        print(f"\n  Regime {i}:")
        print(f"    평균 수익률: {mean:.4f}% (일별)")
        print(f"    변동성: {std:.4f}% (일별)")
        print(f"    연율화 수익률: {mean * 252:.2f}%")
        print(f"    연율화 변동성: {std * np.sqrt(252):.2f}%")

    # 전환 확률 행렬
    print(f"\n  전환 확률 행렬 (Transition Matrix):")
    transition_matrix = result.regime_transition

    for i in range(len(transition_matrix)):
        for j in range(len(transition_matrix)):
            prob = float(transition_matrix[i, j])
            print(f"    P(Regime {i} → Regime {j}): {prob:.4f}")

    # 평균 지속 기간
    print(f"\n  평균 레짐 지속 기간:")
    for i in range(len(transition_matrix)):
        # E[Duration] = 1 / (1 - P(i→i))
        prob_stay = float(transition_matrix[i, i])
        duration = 1 / (1 - prob_stay)
        print(f"    Regime {i}: {duration:.1f}일")

    return pd.DataFrame(regime_chars)

# 전체 기간 레짐 분석
chars_full = analyze_regime_characteristics(ms_full, "전체 기간")

# ETF 전후 레짐 분석
chars_pre = analyze_regime_characteristics(ms_pre, "ETF 이전")
chars_post = analyze_regime_characteristics(ms_post, "ETF 이후")

# ============================================================================
# 레짐 확률 (Filtered Probability) 추출
# ============================================================================
print("\n\n[Step 6.4] 레짐 확률 시계열 추출")
print("-"*80)

if ms_full is not None:
    # Filtered Probabilities: P(Regime=i | data up to t)
    filtered_probs = ms_full.filtered_marginal_probabilities

    # Smoothed Probabilities: P(Regime=i | all data)
    smoothed_probs = ms_full.smoothed_marginal_probabilities

    print(f"Filtered Probabilities: {filtered_probs.shape}")
    print(f"Smoothed Probabilities: {smoothed_probs.shape}")

    # 가장 가능성 높은 레짐
    most_likely_regime = smoothed_probs.idxmax(axis=1)

    # 레짐 분포
    regime_counts = most_likely_regime.value_counts()
    print(f"\n레짐 분포:")
    for regime, count in regime_counts.items():
        pct = count / len(most_likely_regime) * 100
        print(f"  Regime {regime}: {count}일 ({pct:.1f}%)")

    # ETF 전후 레짐 분포
    pre_regime = most_likely_regime[most_likely_regime.index < etf_date]
    post_regime = most_likely_regime[most_likely_regime.index >= etf_date]

    print(f"\nETF 이전 레짐 분포:")
    for regime, count in pre_regime.value_counts().items():
        pct = count / len(pre_regime) * 100
        print(f"  Regime {regime}: {count}일 ({pct:.1f}%)")

    print(f"\nETF 이후 레짐 분포:")
    for regime, count in post_regime.value_counts().items():
        pct = count / len(post_regime) * 100
        print(f"  Regime {regime}: {count}일 ({pct:.1f}%)")

# ============================================================================
# 레짐별 BTC 성과 분석
# ============================================================================
print("\n\n[Step 6.5] 레짐별 BTC 성과 분석")
print("-"*80)

if ms_full is not None:
    # 각 레짐에서의 수익률
    regime_performance = []

    for regime in range(2):
        # 해당 레짐일 확률 > 0.5인 날들
        regime_days = smoothed_probs[regime] > 0.5
        regime_returns = returns[regime_days]

        perf = {
            'Regime': regime,
            'N_Days': len(regime_returns),
            'Mean_Return': regime_returns.mean(),
            'Volatility': regime_returns.std(),
            'Sharpe': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0,
            'Max_Return': regime_returns.max(),
            'Min_Return': regime_returns.min(),
            'Positive_Days': (regime_returns > 0).sum(),
            'Negative_Days': (regime_returns < 0).sum()
        }

        regime_performance.append(perf)

        print(f"\nRegime {regime} 성과:")
        print(f"  거래일 수: {perf['N_Days']}일")
        print(f"  평균 수익률: {perf['Mean_Return']:.4f}%")
        print(f"  변동성: {perf['Volatility']:.4f}%")
        print(f"  샤프 비율: {perf['Sharpe']:.4f}")
        print(f"  양의 수익 일수: {perf['Positive_Days']}일 ({perf['Positive_Days']/perf['N_Days']*100:.1f}%)")
        print(f"  음의 수익 일수: {perf['Negative_Days']}일 ({perf['Negative_Days']/perf['N_Days']*100:.1f}%)")

    performance_df = pd.DataFrame(regime_performance)

# ============================================================================
# 결과 저장
# ============================================================================
print("\n\n[Step 6.6] 결과 저장")
print("-"*80)

if ms_full is not None:
    # 레짐 확률
    filtered_probs.to_csv('markov_switching_filtered_probabilities.csv')
    smoothed_probs.to_csv('markov_switching_smoothed_probabilities.csv')
    print("  markov_switching_filtered_probabilities.csv")
    print("  markov_switching_smoothed_probabilities.csv")

    # 레짐 분류
    regime_df = pd.DataFrame({
        'Date': most_likely_regime.index,
        'Regime': most_likely_regime.values,
        'Prob_Regime_0': smoothed_probs[0].values,
        'Prob_Regime_1': smoothed_probs[1].values,
        'Return': returns.loc[most_likely_regime.index].values
    })
    regime_df.to_csv('markov_switching_regime_classification.csv', index=False)
    print("  markov_switching_regime_classification.csv")

    # 레짐 특성
    if chars_full is not None:
        chars_full.to_csv('markov_switching_regime_characteristics.csv', index=False)
        print("  markov_switching_regime_characteristics.csv")

    # 레짐별 성과
    performance_df.to_csv('markov_switching_regime_performance.csv', index=False)
    print("  markov_switching_regime_performance.csv")

# ============================================================================
# 시각화
# ============================================================================
print("\n[Step 6.7] 시각화 생성")
print("-"*80)

if ms_full is not None:
    # 1. 레짐 확률 시계열
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # 1-1. BTC 가격
    ax1 = axes[0]
    ax1.plot(btc_price.index, btc_price, linewidth=1, color='black')
    ax1.axvline(x=etf_date, color='red', linestyle='--', linewidth=2, alpha=0.7, label='ETF 승인')
    ax1.set_ylabel('BTC 가격 ($)')
    ax1.set_title('Bitcoin 가격', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 1-2. 레짐 확률
    ax2 = axes[1]
    ax2.fill_between(smoothed_probs.index, 0, smoothed_probs[0], alpha=0.5, label='Regime 0 (저변동)', color='green')
    ax2.fill_between(smoothed_probs.index, 0, smoothed_probs[1], alpha=0.5, label='Regime 1 (고변동)', color='red')
    ax2.axvline(x=etf_date, color='red', linestyle='--', linewidth=2, alpha=0.7, label='ETF 승인')
    ax2.set_ylabel('레짐 확률')
    ax2.set_title('Markov Switching 레짐 확률', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # 1-3. 수익률
    ax3 = axes[2]
    colors = ['green' if r > 0 else 'red' for r in returns]
    ax3.bar(returns.index, returns, color=colors, alpha=0.6, width=1)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.axvline(x=etf_date, color='red', linestyle='--', linewidth=2, alpha=0.7, label='ETF 승인')
    ax3.set_ylabel('수익률 (%)')
    ax3.set_xlabel('날짜')
    ax3.set_title('BTC 일별 수익률', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('markov_switching_regime_probabilities.png', dpi=300, bbox_inches='tight')
    print("  markov_switching_regime_probabilities.png")
    plt.close()

    # 2. 레짐별 수익률 분포
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for regime in range(2):
        ax = axes[regime]
        regime_days = smoothed_probs[regime] > 0.5
        regime_returns_data = returns[regime_days]

        ax.hist(regime_returns_data, bins=50, alpha=0.7, edgecolor='black', color='green' if regime == 0 else 'red')
        ax.axvline(x=regime_returns_data.mean(), color='blue', linestyle='--', linewidth=2, label=f'평균 ({regime_returns_data.mean():.2f}%)')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('수익률 (%)')
        ax.set_ylabel('빈도')
        ax.set_title(f'Regime {regime} 수익률 분포', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('markov_switching_return_distributions.png', dpi=300, bbox_inches='tight')
    print("  markov_switching_return_distributions.png")
    plt.close()

    # 3. ETF 전후 레짐 비교
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 3-1. ETF 이전 레짐 확률
    ax1 = axes[0, 0]
    pre_probs = smoothed_probs[smoothed_probs.index < etf_date]
    ax1.fill_between(pre_probs.index, 0, pre_probs[0], alpha=0.5, label='Regime 0', color='green')
    ax1.fill_between(pre_probs.index, 0, pre_probs[1], alpha=0.5, label='Regime 1', color='red')
    ax1.set_ylabel('레짐 확률')
    ax1.set_title('ETF 이전 레짐 확률', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # 3-2. ETF 이후 레짐 확률
    ax2 = axes[0, 1]
    post_probs = smoothed_probs[smoothed_probs.index >= etf_date]
    ax2.fill_between(post_probs.index, 0, post_probs[0], alpha=0.5, label='Regime 0', color='green')
    ax2.fill_between(post_probs.index, 0, post_probs[1], alpha=0.5, label='Regime 1', color='red')
    ax2.set_ylabel('레짐 확률')
    ax2.set_title('ETF 이후 레짐 확률', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # 3-3. 레짐 분포 (ETF 이전)
    ax3 = axes[1, 0]
    pre_regime_counts = pre_regime.value_counts()
    ax3.bar(pre_regime_counts.index, pre_regime_counts.values, color=['green', 'red'], alpha=0.7)
    ax3.set_xlabel('Regime')
    ax3.set_ylabel('일수')
    ax3.set_title('ETF 이전 레짐 분포', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # 3-4. 레짐 분포 (ETF 이후)
    ax4 = axes[1, 1]
    post_regime_counts = post_regime.value_counts()
    ax4.bar(post_regime_counts.index, post_regime_counts.values, color=['green', 'red'], alpha=0.7)
    ax4.set_xlabel('Regime')
    ax4.set_ylabel('일수')
    ax4.set_title('ETF 이후 레짐 분포', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('markov_switching_etf_comparison.png', dpi=300, bbox_inches='tight')
    print("  markov_switching_etf_comparison.png")
    plt.close()

# ============================================================================
# 요약
# ============================================================================
print("\n" + "="*80)
print("Step 6 완료: Markov Switching Regime Analysis")
print("="*80)

if ms_full is not None and chars_full is not None:
    print(f"""
주요 발견:

1. 레짐 특성 (전체 기간)
   - Regime 0 (저변동): 평균 {chars_full.loc[0, 'Mean_Return']:.3f}%, 변동성 {chars_full.loc[0, 'Volatility']:.3f}%
   - Regime 1 (고변동): 평균 {chars_full.loc[1, 'Mean_Return']:.3f}%, 변동성 {chars_full.loc[1, 'Volatility']:.3f}%

2. 레짐 분포
   - 전체 기간: Regime 0 ({regime_counts.get(0, 0)}일), Regime 1 ({regime_counts.get(1, 0)}일)
   - ETF 이전: Regime 0 ({pre_regime.value_counts().get(0, 0)}일), Regime 1 ({pre_regime.value_counts().get(1, 0)}일)
   - ETF 이후: Regime 0 ({post_regime.value_counts().get(0, 0)}일), Regime 1 ({post_regime.value_counts().get(1, 0)}일)

3. 레짐별 성과
   - Regime 0: 샤프 {performance_df.loc[0, 'Sharpe']:.3f}, 양의 수익 {performance_df.loc[0, 'Positive_Days']/performance_df.loc[0, 'N_Days']*100:.1f}%
   - Regime 1: 샤프 {performance_df.loc[1, 'Sharpe']:.3f}, 양의 수익 {performance_df.loc[1, 'Positive_Days']/performance_df.loc[1, 'N_Days']*100:.1f}%

생성된 파일:
  - 레짐 확률 시계열: 2개 CSV
  - 레짐 분류: 1개 CSV
  - 레짐 특성: 1개 CSV
  - 레짐별 성과: 1개 CSV
  - 시각화: 3개 PNG
""")
else:
    print("모델 추정 실패")
