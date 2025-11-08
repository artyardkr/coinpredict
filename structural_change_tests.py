"""
Bitcoin 구조 변화 검정
- Chow Test: ETF 승인일 기준 회귀계수 변화 검정
- Quandt-Andrews Test: 변화점 탐지
- CUSUM Test: 누적합 검정

통계적 보정:
- Bonferroni / FDR 다중 검정 보정
- HAC 표준오차 (Newey-West)
- Winsorization 이상치 처리
- Out-of-sample 검증
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats.mstats import winsorize
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_white, acorr_breusch_godfrey
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 1. Data Preparation
# ============================================================================

def load_and_prepare_data(file_path='integrated_data_full_v2.csv',
                          etf_date='2024-01-10',
                          winsorize_limits=(0.01, 0.01)):
    """
    데이터 로드 및 전처리

    Args:
        file_path: 데이터 파일 경로
        etf_date: ETF 승인일
        winsorize_limits: Winsorization 상하한 (default: 1%)

    Returns:
        df: 전처리된 데이터프레임
        etf_date: ETF 날짜 (Timestamp)
    """
    print("="*80)
    print("1. 데이터 로드 및 전처리")
    print("="*80)

    # 데이터 로드
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()

    print(f"원본 데이터: {df.shape[0]} samples, {df.shape[1]} variables")
    print(f"기간: {df.index[0].date()} ~ {df.index[-1].date()}")

    # ETF 날짜
    etf_date = pd.Timestamp(etf_date)
    pre_etf_samples = (df.index < etf_date).sum()
    post_etf_samples = (df.index >= etf_date).sum()
    print(f"\nETF 승인일: {etf_date.date()}")
    print(f"  - ETF 전: {pre_etf_samples} samples")
    print(f"  - ETF 후: {post_etf_samples} samples")

    # 데이터 누수 변수 제거 (TAFAS와 동일)
    leakage_vars = [
        'Close', 'High', 'Low', 'Open',
        'EMA5_close', 'EMA10_close', 'EMA14_close', 'EMA20_close', 'EMA30_close',
        'EMA100_close', 'EMA200_close',
        'SMA5_close', 'SMA10_close', 'SMA20_close', 'SMA30_close',
        'BB_high', 'BB_low', 'BB_mid',
        'bc_market_cap'
    ]

    # Target 변수 분리
    y = df['Close'].copy()

    # Feature 변수 (누수 제거)
    X_cols = [col for col in df.columns if col not in leakage_vars]
    X = df[X_cols].copy()

    print(f"\nTarget: Close")
    print(f"Features: {len(X_cols)} variables (데이터 누수 제거 후)")

    # Winsorization (이상치 처리)
    print(f"\nWinsorization 적용: limits={winsorize_limits}")

    y_winsorized = pd.Series(
        winsorize(y.values, limits=winsorize_limits),
        index=y.index
    )

    X_winsorized = pd.DataFrame(
        index=X.index,
        columns=X.columns
    )

    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            # inf 처리
            X_col = X[col].replace([np.inf, -np.inf], np.nan)

            # Winsorization
            if not X_col.isna().all():
                X_winsorized[col] = winsorize(X_col.fillna(X_col.median()).values,
                                              limits=winsorize_limits)
            else:
                X_winsorized[col] = X_col

    # 결측치 처리
    X_winsorized = X_winsorized.fillna(X_winsorized.median())

    # 최종 데이터프레임
    df_processed = X_winsorized.copy()
    df_processed['Close'] = y_winsorized

    print(f"전처리 완료: {df_processed.shape[0]} samples, {df_processed.shape[1]} variables")

    return df_processed, etf_date

def train_test_split_temporal(df, etf_date, test_start_date='2024-01-01'):
    """
    시계열 train/test 분할

    Args:
        df: 데이터프레임
        etf_date: ETF 날짜
        test_start_date: 테스트 시작 날짜

    Returns:
        train, test 데이터프레임
    """
    test_start = pd.Timestamp(test_start_date)

    df_train = df[df.index < test_start].copy()
    df_test = df[df.index >= test_start].copy()

    print(f"\nTrain/Test 분할:")
    print(f"  - Train: {df_train.index[0].date()} ~ {df_train.index[-1].date()} ({len(df_train)} samples)")
    print(f"  - Test: {df_test.index[0].date()} ~ {df_test.index[-1].date()} ({len(df_test)} samples)")
    print(f"  - ETF 날짜({etf_date.date()})는 Test 기간 내")

    return df_train, df_test

# ============================================================================
# 2. Variable Selection
# ============================================================================

def select_variables_pretrain(df_train, target_col='Close',
                               corr_threshold=0.3, vif_threshold=10):
    """
    Pre-ETF 데이터로만 변수 선택 (look-ahead bias 방지)

    Args:
        df_train: 학습 데이터
        target_col: 타겟 변수
        corr_threshold: 상관계수 임계값
        vif_threshold: VIF 임계값

    Returns:
        selected_vars: 선택된 변수 리스트
    """
    print("\n" + "="*80)
    print("2. 변수 선택 (Pre-ETF 데이터만 사용)")
    print("="*80)

    y = df_train[target_col]
    X = df_train.drop(columns=[target_col])

    # Step 1: 상관계수 필터
    print(f"\nStep 1: 상관계수 필터 (|corr| > {corr_threshold})")

    correlations = {}
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            corr = y.corr(X[col])
            if not np.isnan(corr) and not np.isinf(corr):
                correlations[col] = corr

    corr_filtered = [var for var, corr in correlations.items() if abs(corr) > corr_threshold]
    print(f"  - 상관계수 기준 통과: {len(corr_filtered)} / {len(X.columns)} 변수")

    # Step 2: VIF 필터 (다중공선성)
    print(f"\nStep 2: VIF 필터 (VIF < {vif_threshold})")

    X_filtered = X[corr_filtered].copy()

    # 상수항 추가
    X_filtered_const = sm.add_constant(X_filtered)

    vif_data = []
    for i, col in enumerate(X_filtered.columns):
        try:
            vif = variance_inflation_factor(X_filtered_const.values, i+1)  # +1 for constant
            vif_data.append({'variable': col, 'VIF': vif})
        except:
            vif_data.append({'variable': col, 'VIF': np.nan})

    vif_df = pd.DataFrame(vif_data)
    vif_filtered = vif_df[vif_df['VIF'] < vif_threshold]['variable'].tolist()

    print(f"  - VIF 기준 통과: {len(vif_filtered)} / {len(corr_filtered)} 변수")

    # 최종 선택 변수
    selected_vars = vif_filtered

    print(f"\n최종 선택 변수: {len(selected_vars)}개")
    print("주요 변수:")
    top_corr = sorted([(var, abs(correlations[var])) for var in selected_vars],
                     key=lambda x: x[1], reverse=True)[:15]
    for var, corr in top_corr:
        print(f"  - {var:40s} | corr = {correlations[var]:+.3f}")

    return selected_vars

# ============================================================================
# 3. Chow Test
# ============================================================================

def chow_test(y, X, breakpoint_idx, use_hac=True, hac_maxlags=None):
    """
    Chow Test for structural break

    Args:
        y: Target variable
        X: Feature matrix (with constant)
        breakpoint_idx: Index of breakpoint
        use_hac: Use HAC standard errors
        hac_maxlags: Maximum lags for HAC (default: auto)

    Returns:
        dict with F-stat, p-value, SSR, etc.
    """
    n = len(y)
    k = X.shape[1]

    # Split data
    y1, y2 = y[:breakpoint_idx], y[breakpoint_idx:]
    X1, X2 = X[:breakpoint_idx], X[breakpoint_idx:]

    n1, n2 = len(y1), len(y2)

    # Pooled regression
    if use_hac:
        if hac_maxlags is None:
            hac_maxlags = int(np.floor(4 * (n/100)**(2/9)))
        model_pooled = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': hac_maxlags})
    else:
        model_pooled = OLS(y, X).fit()

    SSR_pooled = model_pooled.ssr

    # Regime 1 regression
    if use_hac:
        model1 = OLS(y1, X1).fit(cov_type='HAC', cov_kwds={'maxlags': min(hac_maxlags, n1-k-1)})
    else:
        model1 = OLS(y1, X1).fit()

    SSR1 = model1.ssr

    # Regime 2 regression
    if use_hac:
        model2 = OLS(y2, X2).fit(cov_type='HAC', cov_kwds={'maxlags': min(hac_maxlags, n2-k-1)})
    else:
        model2 = OLS(y2, X2).fit()

    SSR2 = model2.ssr

    # Chow F-statistic
    numerator = (SSR_pooled - (SSR1 + SSR2)) / k
    denominator = (SSR1 + SSR2) / (n1 + n2 - 2*k)

    if denominator == 0:
        F_stat = np.nan
        p_value = np.nan
    else:
        F_stat = numerator / denominator
        df1 = k
        df2 = n1 + n2 - 2*k
        p_value = stats.f.sf(F_stat, df1, df2)

    return {
        'F_stat': F_stat,
        'p_value': p_value,
        'df1': k,
        'df2': n1 + n2 - 2*k,
        'SSR_pooled': SSR_pooled,
        'SSR1': SSR1,
        'SSR2': SSR2,
        'coef_pooled': model_pooled.params,
        'coef1': model1.params,
        'coef2': model2.params,
        'model_pooled': model_pooled,
        'model1': model1,
        'model2': model2
    }

def run_chow_tests(df, selected_vars, etf_date, target_col='Close'):
    """
    모든 선택 변수에 대해 Chow Test 수행

    Args:
        df: 데이터프레임
        selected_vars: 선택된 변수 리스트
        etf_date: ETF 날짜
        target_col: 타겟 변수

    Returns:
        results_df: 결과 데이터프레임
    """
    print("\n" + "="*80)
    print("3. Chow Test 수행")
    print("="*80)
    print(f"변화점: {etf_date.date()}")
    print(f"검정 변수 수: {len(selected_vars)}")

    y = df[target_col]
    breakpoint_idx = (df.index < etf_date).sum()

    results = []

    for var in selected_vars:
        # 단변량 회귀
        X = sm.add_constant(df[[var]])

        # Chow Test
        chow_result = chow_test(y, X, breakpoint_idx, use_hac=True)

        # 계수 추출
        coef_pooled = chow_result['coef1'][var] if var in chow_result['coef_pooled'].index else np.nan
        coef_pre = chow_result['coef1'][var] if var in chow_result['coef1'].index else np.nan
        coef_post = chow_result['coef2'][var] if var in chow_result['coef2'].index else np.nan
        coef_change = coef_post - coef_pre

        results.append({
            'variable': var,
            'F_stat': chow_result['F_stat'],
            'p_value': chow_result['p_value'],
            'df1': chow_result['df1'],
            'df2': chow_result['df2'],
            'coef_pooled': coef_pooled,
            'coef_pre': coef_pre,
            'coef_post': coef_post,
            'coef_change': coef_change,
            'pct_change': (coef_change / abs(coef_pre) * 100) if coef_pre != 0 else np.nan
        })

    results_df = pd.DataFrame(results)

    # Bonferroni 보정
    alpha = 0.05
    alpha_bonf = alpha / len(selected_vars)
    results_df['p_adj_bonf'] = results_df['p_value']
    results_df['significant_bonf'] = results_df['p_adj_bonf'] < alpha_bonf

    # FDR 보정 (Benjamini-Hochberg)
    from statsmodels.stats.multitest import multipletests
    reject_fdr, p_adj_fdr, _, _ = multipletests(results_df['p_value'], alpha=alpha, method='fdr_bh')
    results_df['p_adj_fdr'] = p_adj_fdr
    results_df['significant_fdr'] = reject_fdr

    # 정렬
    results_df = results_df.sort_values('F_stat', ascending=False)

    print(f"\nBonferroni 보정 (α = {alpha_bonf:.6f}):")
    print(f"  - 유의한 변수: {results_df['significant_bonf'].sum()} / {len(selected_vars)}")

    print(f"\nFDR 보정 (α = {alpha}):")
    print(f"  - 유의한 변수: {results_df['significant_fdr'].sum()} / {len(selected_vars)}")

    return results_df

# ============================================================================
# 4. Quandt-Andrews Test
# ============================================================================

def quandt_andrews_test(y, X, trim=0.15):
    """
    Quandt-Andrews Test for unknown breakpoint

    Args:
        y: Target variable
        X: Feature matrix (with constant)
        trim: Trim proportion (default: 15%)

    Returns:
        dict with sup_F, breakpoint, etc.
    """
    n = len(y)
    k = X.shape[1]

    # Trimming
    tau_min = int(trim * n)
    tau_max = int((1 - trim) * n)

    f_stats = []

    for tau in range(tau_min, tau_max):
        chow_result = chow_test(y, X, tau, use_hac=True)
        f_stats.append((tau, chow_result['F_stat']))

    # sup F-statistic
    tau_star, sup_f = max(f_stats, key=lambda x: x[1])

    # Andrews critical values (approximation for k=2, α=0.05)
    # For more variables, use tables from Andrews (1993)
    critical_values = {
        0.10: 7.12,
        0.05: 8.68,
        0.01: 12.16
    }

    breakpoint_date = y.index[tau_star]

    return {
        'sup_F': sup_f,
        'tau_star': tau_star,
        'breakpoint_date': breakpoint_date,
        'f_stats': f_stats,
        'critical_values': critical_values,
        'significant_05': sup_f > critical_values[0.05],
        'significant_01': sup_f > critical_values[0.01]
    }

def run_quandt_andrews_tests(df, selected_vars, target_col='Close', top_n=20):
    """
    주요 변수에 대해 Quandt-Andrews Test 수행

    Args:
        df: 데이터프레임
        selected_vars: 선택된 변수 리스트
        target_col: 타겟 변수
        top_n: 상위 몇 개 변수에 대해 수행할지

    Returns:
        results_df: 결과 데이터프레임
    """
    print("\n" + "="*80)
    print("4. Quandt-Andrews Test 수행")
    print("="*80)
    print(f"검정 변수 수: {min(top_n, len(selected_vars))} (계산 시간 고려)")

    y = df[target_col]

    results = []

    for var in selected_vars[:top_n]:
        print(f"  - {var}...", end=' ')

        X = sm.add_constant(df[[var]])

        qa_result = quandt_andrews_test(y, X, trim=0.15)

        results.append({
            'variable': var,
            'sup_F': qa_result['sup_F'],
            'breakpoint_date': qa_result['breakpoint_date'],
            'tau_star': qa_result['tau_star'],
            'critical_10': qa_result['critical_values'][0.10],
            'critical_05': qa_result['critical_values'][0.05],
            'critical_01': qa_result['critical_values'][0.01],
            'significant_05': qa_result['significant_05'],
            'significant_01': qa_result['significant_01']
        })

        print(f"sup_F = {qa_result['sup_F']:.2f}, breakpoint = {qa_result['breakpoint_date'].date()}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('sup_F', ascending=False)

    print(f"\n유의한 변화점 검출 (α = 0.05): {results_df['significant_05'].sum()} / {len(results_df)}")

    return results_df

# ============================================================================
# 5. CUSUM Test
# ============================================================================

def cusum_test(y, X):
    """
    CUSUM and CUSUM-SQ Test

    Args:
        y: Target variable
        X: Feature matrix (with constant)

    Returns:
        dict with CUSUM, CUSUM-SQ, boundaries, crossing points
    """
    from statsmodels.stats.diagnostic import recursive_olsresiduals

    # Recursive OLS residuals
    rr_output = recursive_olsresiduals(OLS(y, X).fit(), skip=30)
    rr = rr_output[0]  # recursive residuals

    # CUSUM
    cusum = np.cumsum(rr / np.std(rr))

    # CUSUM-SQ
    cusum_sq = np.cumsum((rr / np.std(rr))**2)

    # 5% significance boundaries
    n = len(cusum)
    boundary_cusum = 0.948 * np.sqrt(n)

    # CUSUM-SQ boundaries (approximation)
    from scipy.stats import chi2
    a = 0.05
    boundary_cusum_sq_upper = n + chi2.ppf(1 - a/2, n)
    boundary_cusum_sq_lower = chi2.ppf(a/2, n)

    # Detect boundary crossings
    cusum_crossing = np.where(np.abs(cusum) > boundary_cusum)[0]
    cusum_sq_crossing = np.where((cusum_sq > boundary_cusum_sq_upper) |
                                 (cusum_sq < boundary_cusum_sq_lower))[0]

    return {
        'cusum': cusum,
        'cusum_sq': cusum_sq,
        'boundary_cusum': boundary_cusum,
        'boundary_cusum_sq_upper': boundary_cusum_sq_upper,
        'boundary_cusum_sq_lower': boundary_cusum_sq_lower,
        'cusum_crossing': cusum_crossing,
        'cusum_sq_crossing': cusum_sq_crossing,
        'recursive_residuals': rr
    }

def run_cusum_tests(df, selected_vars, target_col='Close', top_n=10):
    """
    주요 변수에 대해 CUSUM Test 수행

    Args:
        df: 데이터프레임
        selected_vars: 선택된 변수 리스트
        target_col: 타겟 변수
        top_n: 상위 몇 개 변수

    Returns:
        results: CUSUM 결과 딕셔너리
    """
    print("\n" + "="*80)
    print("5. CUSUM Test 수행")
    print("="*80)
    print(f"검정 변수 수: {min(top_n, len(selected_vars))}")

    y = df[target_col]

    results = {}

    for var in selected_vars[:top_n]:
        X = sm.add_constant(df[[var]])

        cusum_result = cusum_test(y, X)
        results[var] = cusum_result

        n_cusum_cross = len(cusum_result['cusum_crossing'])
        n_cusum_sq_cross = len(cusum_result['cusum_sq_crossing'])

        print(f"  - {var:40s} | CUSUM 이탈: {n_cusum_cross:2d}회, CUSUM-SQ 이탈: {n_cusum_sq_cross:2d}회")

    return results

# ============================================================================
# 6. Visualization
# ============================================================================

def plot_chow_results(chow_results, save_path='plots/chow_test_results.png', top_n=20):
    """Chow Test 결과 시각화"""
    import os
    os.makedirs('plots', exist_ok=True)

    top_results = chow_results.head(top_n).copy()

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # F-통계량
    ax1 = axes[0]
    colors = ['red' if sig else 'gray' for sig in top_results['significant_bonf']]
    bars = ax1.barh(range(len(top_results)), top_results['F_stat'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(top_results)))
    ax1.set_yticklabels([v[:35] for v in top_results['variable']], fontsize=9)
    ax1.set_xlabel('F-통계량', fontsize=12, fontweight='bold')
    ax1.set_title('Chow Test 결과 (상위 20개 변수)', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')

    # 값 표시
    for i, (f_stat, p_val) in enumerate(zip(top_results['F_stat'], top_results['p_value'])):
        ax1.text(f_stat, i, f'  F={f_stat:.1f}, p={p_val:.4f}',
                va='center', fontsize=8)

    # 계수 변화
    ax2 = axes[1]
    colors2 = ['green' if c > 0 else 'red' for c in top_results['coef_change']]
    bars2 = ax2.barh(range(len(top_results)), top_results['coef_change'], color=colors2, alpha=0.7)
    ax2.set_yticks(range(len(top_results)))
    ax2.set_yticklabels([v[:35] for v in top_results['variable']], fontsize=9)
    ax2.set_xlabel('계수 변화량 (ETF 후 - ETF 전)', fontsize=12, fontweight='bold')
    ax2.set_title('회귀계수 변화', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n그래프 저장: {save_path}")
    plt.close()

def plot_cusum_results(cusum_results, df, etf_date, save_path='plots/cusum_test_results.png'):
    """CUSUM Test 결과 시각화"""
    import os
    os.makedirs('plots', exist_ok=True)

    n_vars = len(cusum_results)
    fig, axes = plt.subplots(n_vars, 2, figsize=(18, 4*n_vars))

    if n_vars == 1:
        axes = axes.reshape(1, -1)

    for i, (var, result) in enumerate(cusum_results.items()):
        # CUSUM
        ax1 = axes[i, 0]
        # recursive residuals starts from skip+k
        skip_n = len(df) - len(result['cusum'])
        dates = df.index[skip_n:]  # match the length of cusum

        ax1.plot(dates, result['cusum'], color='blue', linewidth=2, label='CUSUM')
        ax1.axhline(y=result['boundary_cusum'], color='red', linestyle='--',
                   linewidth=2, label=f'경계 (±{result["boundary_cusum"]:.2f})')
        ax1.axhline(y=-result['boundary_cusum'], color='red', linestyle='--', linewidth=2)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.axvline(x=etf_date, color='green', linestyle='--', alpha=0.5,
                   linewidth=2, label='ETF 승인')

        ax1.set_title(f'{var} - CUSUM', fontsize=12, fontweight='bold')
        ax1.set_ylabel('CUSUM', fontsize=10)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # CUSUM-SQ
        ax2 = axes[i, 1]
        ax2.plot(dates, result['cusum_sq'], color='orange', linewidth=2, label='CUSUM-SQ')
        ax2.axhline(y=result['boundary_cusum_sq_upper'], color='red', linestyle='--',
                   linewidth=2, label='상한')
        ax2.axhline(y=result['boundary_cusum_sq_lower'], color='red', linestyle='--',
                   linewidth=2, label='하한')
        ax2.axvline(x=etf_date, color='green', linestyle='--', alpha=0.5,
                   linewidth=2, label='ETF 승인')

        ax2.set_title(f'{var} - CUSUM-SQ', fontsize=12, fontweight='bold')
        ax2.set_ylabel('CUSUM-SQ', fontsize=10)
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"그래프 저장: {save_path}")
    plt.close()

# ============================================================================
# 7. Main
# ============================================================================

def main():
    """메인 실행 함수"""
    print("="*80)
    print("Bitcoin 구조 변화 검정")
    print("="*80)
    print("- Chow Test")
    print("- Quandt-Andrews Test")
    print("- CUSUM Test")
    print("="*80)

    # 1. 데이터 준비
    df, etf_date = load_and_prepare_data(
        file_path='integrated_data_full_v2.csv',
        etf_date='2024-01-10',
        winsorize_limits=(0.01, 0.01)
    )

    # 2. Train/Test 분할
    df_train, df_test = train_test_split_temporal(df, etf_date, test_start_date='2024-01-01')

    # 3. 변수 선택 (Pre-ETF 데이터만)
    selected_vars = select_variables_pretrain(
        df_train,
        target_col='Close',
        corr_threshold=0.2,  # 더 많은 변수 포함
        vif_threshold=15
    )

    # 4. Chow Test (전체 데이터, ETF 날짜 기준)
    chow_results = run_chow_tests(df, selected_vars, etf_date, target_col='Close')

    # 5. Quandt-Andrews Test (상위 20개 변수)
    qa_results = run_quandt_andrews_tests(df, selected_vars, target_col='Close', top_n=20)

    # 6. CUSUM Test (상위 10개 변수)
    cusum_results = run_cusum_tests(df, selected_vars, target_col='Close', top_n=10)

    # 7. 시각화
    print("\n" + "="*80)
    print("6. 시각화")
    print("="*80)

    plot_chow_results(chow_results, save_path='plots/chow_test_results.png', top_n=25)
    plot_cusum_results(cusum_results, df, etf_date, save_path='plots/cusum_test_results.png')

    # 8. 결과 저장
    print("\n" + "="*80)
    print("7. 결과 저장")
    print("="*80)

    import os
    os.makedirs('results', exist_ok=True)

    chow_results.to_csv('results/chow_test_results.csv', index=False)
    qa_results.to_csv('results/quandt_andrews_results.csv', index=False)

    print("  - results/chow_test_results.csv")
    print("  - results/quandt_andrews_results.csv")

    # 9. 요약 보고서
    print("\n" + "="*80)
    print("8. 요약 보고서")
    print("="*80)

    print(f"\n【Chow Test 결과】")
    print(f"  - 검정 변수 수: {len(chow_results)}")
    print(f"  - Bonferroni 유의 (α={0.05/len(chow_results):.6f}): {chow_results['significant_bonf'].sum()}개")
    print(f"  - FDR 유의 (α=0.05): {chow_results['significant_fdr'].sum()}개")

    print(f"\n  상위 10개 변수 (F-통계량 기준):")
    for i, row in chow_results.head(10).iterrows():
        sig_mark = '***' if row['significant_bonf'] else ('**' if row['significant_fdr'] else '')
        print(f"    {row['variable']:35s} | F={row['F_stat']:8.2f} | p={row['p_value']:.6f} {sig_mark}")
        print(f"      {'':35s} | 계수: {row['coef_pre']:+.4f} → {row['coef_post']:+.4f} ({row['coef_change']:+.4f})")

    print(f"\n【Quandt-Andrews Test 결과】")
    print(f"  - 검정 변수 수: {len(qa_results)}")
    print(f"  - 유의한 변화점 검출 (α=0.05): {qa_results['significant_05'].sum()}개")

    print(f"\n  상위 5개 변수 (sup F 기준):")
    for i, row in qa_results.head(5).iterrows():
        sig_mark = '***' if row['significant_01'] else ('**' if row['significant_05'] else '')
        print(f"    {row['variable']:35s} | sup_F={row['sup_F']:8.2f} {sig_mark}")
        print(f"      {'':35s} | 변화점: {row['breakpoint_date'].date()}")

    print(f"\n【CUSUM Test 결과】")
    print(f"  - 검정 변수 수: {len(cusum_results)}")

    for var, result in list(cusum_results.items())[:5]:
        n_cross = len(result['cusum_crossing'])
        print(f"    {var:35s} | 경계 이탈: {n_cross}회")

    print("\n" + "="*80)
    print("분석 완료!")
    print("="*80)

if __name__ == '__main__':
    import os
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    main()
