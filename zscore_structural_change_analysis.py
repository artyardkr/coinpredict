"""
Z-Score 표준화 구조변화 분석
(구조변화검정_표준_프로토콜.md 구현)

목적:
- 모든 변수를 Z-score로 표준화 (평균 0, 표준편차 1)
- 스케일 차이 제거하여 F-통계량 직접 비교 가능
- 3단계 검정: Chow → Quandt-Andrews → CUSUM

작성일: 2025-11-09
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# Step 0: 데이터 표준화 (Z-Score)
# ============================================================================

def preprocess_and_standardize(df):
    """
    표준 전처리 + Z-score 표준화

    Step 0.1: 결측치 처리
    Step 0.2: 이상치 처리 (Winsorization)
    Step 0.3: Z-score 표준화
    """

    print("Step 0: 데이터 표준화")
    print("-" * 80)

    # Step 0.1: 결측치 처리
    print("Step 0.1: 결측치 처리 중...")
    df_clean = df.fillna(method='ffill').fillna(method='bfill').fillna(df.median())
    print(f"  결측치: {df_clean.isnull().sum().sum()}개 (처리 완료)")

    # Step 0.2: 이상치 처리 (Winsorization)
    print("\nStep 0.2: 이상치 처리 (Winsorization) 중...")
    df_winsor = df_clean.copy()
    for col in df_winsor.select_dtypes(include=[np.number]).columns:
        df_winsor[col] = winsorize(df_winsor[col], limits=[0.01, 0.01])
    print("  상하위 1% 절사 완료")

    # Step 0.3: Z-score 표준화
    print("\nStep 0.3: Z-score 표준화 중...")
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_winsor),
        columns=df_winsor.columns,
        index=df_winsor.index
    )
    print("  평균 0, 표준편차 1로 변환 완료")

    # 검증
    print("\n검증:")
    print(f"  평균: {df_scaled.mean().mean():.6f} (≈ 0)")
    print(f"  표준편차: {df_scaled.std().mean():.6f} (≈ 1)")

    return df_scaled


# ============================================================================
# Step 1: Chow Test (Z-Score 기반)
# ============================================================================

def chow_test_zscore(y, X, breakpoint_idx):
    """
    Z-score 표준화 데이터로 Chow Test

    장점:
    - 모든 변수가 동일한 스케일 → F-통계량 직접 비교
    - 계수 해석 일관성: β는 표준편차 변화
    """

    # 전체 회귀
    model_full = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    rss_full = model_full.ssr

    # 분할 회귀
    X1, y1 = X[:breakpoint_idx], y[:breakpoint_idx]
    X2, y2 = X[breakpoint_idx:], y[breakpoint_idx:]

    model1 = sm.OLS(y1, X1).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    model2 = sm.OLS(y2, X2).fit(cov_type='HAC', cov_kwds={'maxlags': 5})

    rss1 = model1.ssr
    rss2 = model2.ssr

    # Chow F-통계량
    n1, n2 = len(y1), len(y2)
    k = X.shape[1]

    numerator = (rss_full - (rss1 + rss2)) / k
    denominator = (rss1 + rss2) / (n1 + n2 - 2*k)

    F_stat = numerator / denominator
    p_value = 1 - stats.f.cdf(F_stat, k, n1 + n2 - 2*k)

    return {
        'F_stat': F_stat,
        'p_value': p_value,
        'coef_pre': model1.params[1] if len(model1.params) > 1 else np.nan,
        'coef_post': model2.params[1] if len(model2.params) > 1 else np.nan,
        'coef_change': (model2.params[1] - model1.params[1]) if len(model1.params) > 1 else np.nan,
        'r2_pre': model1.rsquared,
        'r2_post': model2.rsquared
    }


def run_chow_test_all_variables(df_scaled, y_col, breakpoint_date, variables):
    """
    전체 변수에 대해 Chow Test 실행 + 다중 검정 보정
    """

    print("\nStep 1: Chow Test")
    print("-" * 80)

    breakpoint = pd.to_datetime(breakpoint_date)
    breakpoint_idx = df_scaled.index.get_loc(breakpoint)

    print(f"변화점: {breakpoint_date}")
    print(f"변화점 인덱스: {breakpoint_idx}")
    print(f"ETF 이전: {breakpoint_idx}일")
    print(f"ETF 이후: {len(df_scaled) - breakpoint_idx}일")

    results = []

    for var in variables:
        if var == y_col:
            continue

        y = df_scaled[y_col]
        X = sm.add_constant(df_scaled[[var]])

        result = chow_test_zscore(y, X, breakpoint_idx)
        result['Variable'] = var
        results.append(result)

    df_results = pd.DataFrame(results)

    # 다중 검정 보정
    print(f"\n다중 검정 보정 중... (변수 수: {len(df_results)})")

    # Bonferroni
    reject_bonf, pvals_bonf, _, _ = multipletests(
        df_results['p_value'], alpha=0.05, method='bonferroni'
    )

    # FDR
    reject_fdr, pvals_fdr, _, _ = multipletests(
        df_results['p_value'], alpha=0.05, method='fdr_bh'
    )

    df_results['Significant_Bonferroni'] = reject_bonf
    df_results['Significant_FDR'] = reject_fdr
    df_results['p_value_Bonferroni'] = pvals_bonf
    df_results['p_value_FDR'] = pvals_fdr

    # 정렬
    df_results = df_results.sort_values('F_stat', ascending=False)

    print(f"\nBonferroni 유의: {reject_bonf.sum()}/{len(df_results)} ({reject_bonf.sum()/len(df_results)*100:.1f}%)")
    print(f"FDR 유의: {reject_fdr.sum()}/{len(df_results)} ({reject_fdr.sum()/len(df_results)*100:.1f}%)")

    return df_results


# ============================================================================
# Step 2: Quandt-Andrews Test
# ============================================================================

def quandt_andrews_test_zscore(y, X, trim=0.15):
    """
    Z-score 표준화 데이터로 Quandt-Andrews Test
    """

    n = len(y)
    start = int(n * trim)
    end = int(n * (1 - trim))

    f_stats = []
    breakpoints = []

    for tau in range(start, end):
        result = chow_test_zscore(y, X, tau)
        f_stats.append(result['F_stat'])
        breakpoints.append(y.index[tau])

    sup_f = max(f_stats)
    sup_f_idx = f_stats.index(sup_f)
    sup_f_date = breakpoints[sup_f_idx]

    return {
        'sup_F': sup_f,
        'breakpoint_date': sup_f_date,
        'breakpoint_idx': start + sup_f_idx
    }


def run_qa_test_all_variables(df_scaled, y_col, variables, trim=0.15):
    """
    전체 변수에 대해 Quandt-Andrews Test 실행
    """

    print("\nStep 2: Quandt-Andrews Test")
    print("-" * 80)

    results = []

    for var in variables:
        if var == y_col:
            continue

        y = df_scaled[y_col]
        X = sm.add_constant(df_scaled[[var]])

        result = quandt_andrews_test_zscore(y, X, trim)
        result['Variable'] = var
        results.append(result)

    df_results = pd.DataFrame(results)

    # Andrews (1993) Critical Values
    CRITICAL_VALUES = {0.10: 7.78, 0.05: 9.21, 0.01: 12.16}

    df_results['Significant_5pct'] = df_results['sup_F'] > CRITICAL_VALUES[0.05]
    df_results['Significant_1pct'] = df_results['sup_F'] > CRITICAL_VALUES[0.01]

    df_results = df_results.sort_values('sup_F', ascending=False)

    print(f"5% 유의: {df_results['Significant_5pct'].sum()}/{len(df_results)}")
    print(f"1% 유의: {df_results['Significant_1pct'].sum()}/{len(df_results)}")

    return df_results


# ============================================================================
# Step 3: CUSUM Test
# ============================================================================

def cusum_test_zscore(y, X):
    """
    Z-score 표준화 데이터로 CUSUM Test
    """

    model = sm.OLS(y, X).fit()
    residuals = model.resid

    # 표준화된 잔차
    std_residuals = residuals / residuals.std()

    # 누적합
    cusum = std_residuals.cumsum()

    # 경계선
    n = len(y)
    boundary = 0.948 * np.sqrt(n)

    # 경계 이탈
    breaches = (cusum.abs() > boundary).sum()
    max_cusum = cusum.abs().max()

    return {
        'max_cusum': max_cusum,
        'boundary': boundary,
        'n_breaches': breaches,
        'breach_ratio': max_cusum / boundary
    }


def run_cusum_test_all_variables(df_scaled, y_col, variables):
    """
    전체 변수에 대해 CUSUM Test 실행
    """

    print("\nStep 3: CUSUM Test")
    print("-" * 80)

    results = []

    for var in variables:
        if var == y_col:
            continue

        y = df_scaled[y_col]
        X = sm.add_constant(df_scaled[[var]])

        result = cusum_test_zscore(y, X)
        result['Variable'] = var
        results.append(result)

    df_results = pd.DataFrame(results)

    # 변화 유형 분류
    def classify_change(ratio):
        if ratio > 1.0:
            return 'ABRUPT'
        elif ratio > 0.8:
            return 'RAPID'
        elif ratio > 0.5:
            return 'MODERATE'
        else:
            return 'GRADUAL'

    df_results['Change_Type'] = df_results['breach_ratio'].apply(classify_change)

    print(f"경계 이탈: {(df_results['n_breaches'] > 0).sum()}/{len(df_results)}")
    print(f"\n변화 유형:")
    print(df_results['Change_Type'].value_counts().to_string())

    return df_results


# ============================================================================
# Step 4: 표준화 계수 비교 분석 (Z-score의 진짜 장점!)
# ============================================================================

def analyze_standardized_coefficients(chow_results):
    """
    Z-score 표준화의 핵심 가치: 표준화된 계수 비교

    표준화된 계수(β)의 의미:
    - β = 0.5: X가 1 표준편차 증가 → Y가 0.5 표준편차 증가
    - 모든 변수가 같은 스케일 → 직접 비교 가능!
    - |β| 크기 = 영향력의 크기
    """

    print("\nStep 4: 표준화 계수 비교 분석")
    print("-" * 80)

    # ETF 전후 계수 절대값 (영향력)
    chow_results['impact_pre'] = chow_results['coef_pre'].abs()
    chow_results['impact_post'] = chow_results['coef_post'].abs()
    chow_results['impact_change'] = chow_results['impact_post'] - chow_results['impact_pre']
    chow_results['impact_change_pct'] = (chow_results['impact_change'] / (chow_results['impact_pre'] + 1e-10)) * 100

    # 영향력 분류
    def classify_impact(coef_abs):
        if coef_abs > 1.0:
            return 'VERY_HIGH'
        elif coef_abs > 0.5:
            return 'HIGH'
        elif coef_abs > 0.3:
            return 'MODERATE'
        elif coef_abs > 0.1:
            return 'LOW'
        else:
            return 'VERY_LOW'

    chow_results['impact_level_pre'] = chow_results['impact_pre'].apply(classify_impact)
    chow_results['impact_level_post'] = chow_results['impact_post'].apply(classify_impact)

    # 변화 방향
    def classify_direction_change(pre, post):
        if (pre > 0 and post > 0) or (pre < 0 and post < 0):
            if abs(post) > abs(pre) * 1.5:
                return 'AMPLIFIED'  # 같은 방향, 증폭
            elif abs(post) < abs(pre) * 0.5:
                return 'DAMPENED'  # 같은 방향, 감소
            else:
                return 'STABLE'  # 같은 방향, 비슷
        else:
            return 'REVERSED'  # 방향 전환

    chow_results['direction_change'] = chow_results.apply(
        lambda x: classify_direction_change(x['coef_pre'], x['coef_post']), axis=1
    )

    # 통계 요약
    print("\n[ETF 이전 영향력 분포]")
    print(chow_results['impact_level_pre'].value_counts().to_string())

    print("\n[ETF 이후 영향력 분포]")
    print(chow_results['impact_level_post'].value_counts().to_string())

    print("\n[변화 방향 분포]")
    print(chow_results['direction_change'].value_counts().to_string())

    # TOP 변화 변수들
    print("\n[영향력 증가 TOP 10]")
    top_increase = chow_results.nlargest(10, 'impact_change')[['Variable', 'impact_pre', 'impact_post', 'impact_change', 'direction_change']]
    for idx, row in top_increase.iterrows():
        print(f"  {row['Variable']:30s} {row['impact_pre']:6.3f} → {row['impact_post']:6.3f} (+{row['impact_change']:6.3f}) [{row['direction_change']}]")

    print("\n[영향력 감소 TOP 10]")
    top_decrease = chow_results.nsmallest(10, 'impact_change')[['Variable', 'impact_pre', 'impact_post', 'impact_change', 'direction_change']]
    for idx, row in top_decrease.iterrows():
        print(f"  {row['Variable']:30s} {row['impact_pre']:6.3f} → {row['impact_post']:6.3f} ({row['impact_change']:6.3f}) [{row['direction_change']}]")

    print("\n[방향 전환 변수들]")
    reversed_vars = chow_results[chow_results['direction_change'] == 'REVERSED'].nlargest(10, 'impact_post')[['Variable', 'coef_pre', 'coef_post', 'impact_post']]
    for idx, row in reversed_vars.iterrows():
        sign_pre = '+' if row['coef_pre'] > 0 else '-'
        sign_post = '+' if row['coef_post'] > 0 else '-'
        print(f"  {row['Variable']:30s} {sign_pre}{abs(row['coef_pre']):6.3f} → {sign_post}{abs(row['coef_post']):6.3f}")

    return chow_results


# ============================================================================
# 시각화
# ============================================================================

def plot_chow_test_results(chow_results, top_n=30):
    """Chow Test 결과 시각화"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # F-통계량 TOP 30
    top30 = chow_results.head(top_n)
    ax1.barh(range(len(top30)), top30['F_stat'].values, color='steelblue')
    ax1.set_yticks(range(len(top30)))
    ax1.set_yticklabels(top30['Variable'].values, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel('F-통계량', fontsize=12)
    ax1.set_title(f'Chow Test: F-통계량 TOP {top_n}', fontsize=14, fontweight='bold')
    ax1.axvline(x=10, color='red', linestyle='--', linewidth=1, label='일반적 유의 기준')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)

    # 계수 변화
    ax2.scatter(top30['coef_pre'], top30['coef_post'], s=100, alpha=0.6)
    for idx, row in top30.iterrows():
        ax2.annotate(row['Variable'], (row['coef_pre'], row['coef_post']),
                    fontsize=7, alpha=0.7)

    # 대각선 (변화 없음)
    lim = max(abs(top30['coef_pre']).max(), abs(top30['coef_post']).max()) * 1.1
    ax2.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1, label='변화 없음')
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.axvline(0, color='gray', linewidth=0.5)
    ax2.set_xlabel('ETF 이전 계수', fontsize=12)
    ax2.set_ylabel('ETF 이후 계수', fontsize=12)
    ax2.set_title(f'계수 변화 (TOP {top_n})', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('zscore_chow_test_results.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_qa_test_results(qa_results, breakpoint_date):
    """Quandt-Andrews Test 결과 시각화"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # sup F-통계량 TOP 30
    top30 = qa_results.head(30)
    ax1.barh(range(len(top30)), top30['sup_F'].values, color='coral')
    ax1.set_yticks(range(len(top30)))
    ax1.set_yticklabels(top30['Variable'].values, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel('sup F-통계량', fontsize=12)
    ax1.set_title('Quandt-Andrews: sup F TOP 30', fontsize=14, fontweight='bold')
    ax1.axvline(x=9.21, color='red', linestyle='--', linewidth=1, label='5% CV')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)

    # 변화점 날짜 분포
    qa_dates = pd.to_datetime(qa_results['breakpoint_date'])
    ax2.hist(qa_dates, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=pd.to_datetime(breakpoint_date), color='red',
                linestyle='--', linewidth=2, label='ETF 승인일')
    ax2.set_xlabel('변화점 날짜', fontsize=12)
    ax2.set_ylabel('변수 개수', fontsize=12)
    ax2.set_title('Quandt-Andrews: 변화점 날짜 분포', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('zscore_qa_test_results.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_coefficient_impact_analysis(chow_results):
    """표준화 계수 영향력 분석 시각화"""

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. 영향력 변화 TOP 20
    ax1 = fig.add_subplot(gs[0, :])
    top20_change = chow_results.nlargest(20, 'impact_change')
    colors = ['green' if x > 0 else 'red' for x in top20_change['impact_change']]
    ax1.barh(range(len(top20_change)), top20_change['impact_change'].values, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(top20_change)))
    ax1.set_yticklabels(top20_change['Variable'].values, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel('영향력 변화 (절대값)', fontsize=12)
    ax1.set_title('영향력 변화 TOP 20 (ETF 이후 - ETF 이전)', fontsize=14, fontweight='bold')
    ax1.axvline(0, color='black', linewidth=1)
    ax1.grid(axis='x', alpha=0.3)

    # 2. ETF 전후 영향력 산점도
    ax2 = fig.add_subplot(gs[1, 0])
    direction_colors = {
        'AMPLIFIED': 'green',
        'DAMPENED': 'orange',
        'STABLE': 'blue',
        'REVERSED': 'red'
    }
    for direction, color in direction_colors.items():
        mask = chow_results['direction_change'] == direction
        ax2.scatter(chow_results.loc[mask, 'impact_pre'],
                   chow_results.loc[mask, 'impact_post'],
                   c=color, label=direction, alpha=0.6, s=50)

    # 대각선
    max_val = max(chow_results['impact_pre'].max(), chow_results['impact_post'].max())
    ax2.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5, label='변화 없음')
    ax2.set_xlabel('ETF 이전 영향력', fontsize=12)
    ax2.set_ylabel('ETF 이후 영향력', fontsize=12)
    ax2.set_title('영향력 변화 패턴', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(alpha=0.3)

    # 3. 영향력 레벨 분포 (전)
    ax3 = fig.add_subplot(gs[1, 1])
    level_counts_pre = chow_results['impact_level_pre'].value_counts()
    level_order = ['VERY_HIGH', 'HIGH', 'MODERATE', 'LOW', 'VERY_LOW']
    level_counts_pre = level_counts_pre.reindex(level_order, fill_value=0)
    ax3.barh(range(len(level_counts_pre)), level_counts_pre.values, color='skyblue', alpha=0.7)
    ax3.set_yticks(range(len(level_counts_pre)))
    ax3.set_yticklabels(level_counts_pre.index, fontsize=10)
    ax3.invert_yaxis()
    ax3.set_xlabel('변수 개수', fontsize=12)
    ax3.set_title('ETF 이전 영향력 분포', fontsize=14, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)

    # 4. 영향력 레벨 분포 (후)
    ax4 = fig.add_subplot(gs[2, 0])
    level_counts_post = chow_results['impact_level_post'].value_counts()
    level_counts_post = level_counts_post.reindex(level_order, fill_value=0)
    ax4.barh(range(len(level_counts_post)), level_counts_post.values, color='coral', alpha=0.7)
    ax4.set_yticks(range(len(level_counts_post)))
    ax4.set_yticklabels(level_counts_post.index, fontsize=10)
    ax4.invert_yaxis()
    ax4.set_xlabel('변수 개수', fontsize=12)
    ax4.set_title('ETF 이후 영향력 분포', fontsize=14, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)

    # 5. 방향 전환 변수 TOP 10
    ax5 = fig.add_subplot(gs[2, 1])
    reversed_vars = chow_results[chow_results['direction_change'] == 'REVERSED'].nlargest(10, 'impact_post')
    if len(reversed_vars) > 0:
        x = range(len(reversed_vars))
        width = 0.35
        ax5.barh([i - width/2 for i in x], reversed_vars['impact_pre'].values,
                width, label='ETF 이전', color='lightblue', alpha=0.7)
        ax5.barh([i + width/2 for i in x], reversed_vars['impact_post'].values,
                width, label='ETF 이후', color='lightcoral', alpha=0.7)
        ax5.set_yticks(x)
        ax5.set_yticklabels(reversed_vars['Variable'].values, fontsize=9)
        ax5.invert_yaxis()
        ax5.set_xlabel('영향력 (절대값)', fontsize=12)
        ax5.set_title('방향 전환 변수 TOP 10', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(axis='x', alpha=0.3)

    plt.savefig('zscore_coefficient_impact_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# 메인 실행
# ============================================================================

if __name__ == "__main__":

    print("=" * 80)
    print("Z-Score 표준화 구조변화 분석")
    print("=" * 80)

    # 데이터 로드
    print("\n데이터 로드 중...")
    df = pd.read_csv('integrated_data_full_v2.csv', index_col='Date', parse_dates=True)
    print(f"데이터 shape: {df.shape}")

    # Step 0: 표준화
    df_scaled = preprocess_and_standardize(df)

    # 변수 설정
    y_col = 'Close'
    variables = [col for col in df_scaled.columns if col != y_col]
    BREAKPOINT_DATE = '2024-01-10'

    print(f"\n종속변수: {y_col}")
    print(f"독립변수: {len(variables)}개")

    # Step 1: Chow Test
    chow_results = run_chow_test_all_variables(df_scaled, y_col, BREAKPOINT_DATE, variables)

    # Step 2: Quandt-Andrews Test
    qa_results = run_qa_test_all_variables(df_scaled, y_col, variables, trim=0.15)

    # Step 3: CUSUM Test
    cusum_results = run_cusum_test_all_variables(df_scaled, y_col, variables)

    # Step 4: 표준화 계수 비교 분석 (Z-score의 핵심 가치!)
    chow_results = analyze_standardized_coefficients(chow_results)

    # 결과 저장
    print("\n결과 저장 중...")
    chow_results.to_csv('zscore_chow_test_results.csv', index=False, encoding='utf-8-sig')
    qa_results.to_csv('zscore_qa_test_results.csv', index=False, encoding='utf-8-sig')
    cusum_results.to_csv('zscore_cusum_test_results.csv', index=False, encoding='utf-8-sig')
    print("저장 완료!")

    # 시각화
    print("\n시각화 생성 중...")
    plot_chow_test_results(chow_results)
    plot_qa_test_results(qa_results, BREAKPOINT_DATE)
    plot_coefficient_impact_analysis(chow_results)
    print("시각화 완료!")

    # 요약
    print("\n" + "=" * 80)
    print("요약 보고서")
    print("=" * 80)
    print(f"\nChow Test: {chow_results['Significant_Bonferroni'].sum()}/{len(chow_results)} 변수 유의")
    print(f"QA Test: {qa_results['Significant_5pct'].sum()}/{len(qa_results)} 변수 유의")
    print(f"CUSUM: {(cusum_results['n_breaches'] > 0).sum()}/{len(cusum_results)} 변수 경계 이탈")

    print("\n생성된 파일:")
    print("  - zscore_chow_test_results.csv")
    print("  - zscore_qa_test_results.csv")
    print("  - zscore_cusum_test_results.csv")
    print("  - zscore_chow_test_results.png")
    print("  - zscore_qa_test_results.png")
    print("  - zscore_coefficient_impact_analysis.png")
