"""
Bitcoin 구조 변화 검정 - 모든 변수 버전
- 119개 변수 전체 검정
- 더 엄격한 Bonferroni 보정 (α = 0.05/119)
- 병렬 처리로 속도 향상
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats.mstats import winsorize
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
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
    """데이터 로드 및 전처리"""
    print("="*80)
    print("1. 데이터 로드 및 전처리")
    print("="*80)

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()

    print(f"원본 데이터: {df.shape[0]} samples, {df.shape[1]} variables")
    print(f"기간: {df.index[0].date()} ~ {df.index[-1].date()}")

    etf_date = pd.Timestamp(etf_date)
    pre_etf_samples = (df.index < etf_date).sum()
    post_etf_samples = (df.index >= etf_date).sum()
    print(f"\nETF 승인일: {etf_date.date()}")
    print(f"  - ETF 전: {pre_etf_samples} samples")
    print(f"  - ETF 후: {post_etf_samples} samples")

    # 데이터 누수 변수 제거
    leakage_vars = [
        'Close', 'High', 'Low', 'Open',
        'EMA5_close', 'EMA10_close', 'EMA14_close', 'EMA20_close', 'EMA30_close',
        'EMA100_close', 'EMA200_close',
        'SMA5_close', 'SMA10_close', 'SMA20_close', 'SMA30_close',
        'BB_high', 'BB_low', 'BB_mid',
        'bc_market_cap'
    ]

    y = df['Close'].copy()
    X_cols = [col for col in df.columns if col not in leakage_vars]
    X = df[X_cols].copy()

    print(f"\nTarget: Close")
    print(f"Features: {len(X_cols)} variables (데이터 누수 제거 후)")

    # Winsorization
    print(f"\nWinsorization 적용: limits={winsorize_limits}")

    y_winsorized = pd.Series(
        winsorize(y.values, limits=winsorize_limits),
        index=y.index
    )

    X_winsorized = pd.DataFrame(index=X.index, columns=X.columns)

    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            X_col = X[col].replace([np.inf, -np.inf], np.nan)
            if not X_col.isna().all():
                X_winsorized[col] = winsorize(X_col.fillna(X_col.median()).values,
                                              limits=winsorize_limits)
            else:
                X_winsorized[col] = X_col

    X_winsorized = X_winsorized.fillna(X_winsorized.median())

    df_processed = X_winsorized.copy()
    df_processed['Close'] = y_winsorized

    print(f"전처리 완료: {df_processed.shape[0]} samples, {df_processed.shape[1]} variables")

    return df_processed, etf_date

# ============================================================================
# 2. Chow Test
# ============================================================================

def chow_test(y, X, breakpoint_idx, use_hac=True, hac_maxlags=None):
    """Chow Test for structural break"""
    n = len(y)
    k = X.shape[1]

    y1, y2 = y[:breakpoint_idx], y[breakpoint_idx:]
    X1, X2 = X[:breakpoint_idx], X[breakpoint_idx:]

    n1, n2 = len(y1), len(y2)

    # 충분한 샘플 확인
    if n1 < k + 10 or n2 < k + 10:
        return {
            'F_stat': np.nan,
            'p_value': np.nan,
            'df1': k,
            'df2': n1 + n2 - 2*k,
            'coef_pre': np.nan,
            'coef_post': np.nan,
            'coef_change': np.nan
        }

    try:
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

        if denominator == 0 or np.isnan(denominator):
            F_stat = np.nan
            p_value = np.nan
        else:
            F_stat = numerator / denominator
            df1 = k
            df2 = n1 + n2 - 2*k
            p_value = stats.f.sf(F_stat, df1, df2)

        # 계수 추출
        if X.shape[1] == 2:  # constant + 1 variable
            var_name = X.columns[1] if hasattr(X, 'columns') else 1
            coef_pre = model1.params[var_name] if var_name in model1.params.index else np.nan
            coef_post = model2.params[var_name] if var_name in model2.params.index else np.nan
            coef_change = coef_post - coef_pre
        else:
            coef_pre = np.nan
            coef_post = np.nan
            coef_change = np.nan

        return {
            'F_stat': F_stat,
            'p_value': p_value,
            'df1': k,
            'df2': n1 + n2 - 2*k,
            'coef_pre': coef_pre,
            'coef_post': coef_post,
            'coef_change': coef_change
        }

    except Exception as e:
        print(f"Error in chow_test: {e}")
        return {
            'F_stat': np.nan,
            'p_value': np.nan,
            'df1': k,
            'df2': n1 + n2 - 2*k,
            'coef_pre': np.nan,
            'coef_post': np.nan,
            'coef_change': np.nan
        }

def test_single_variable(args):
    """단일 변수 검정 (병렬 처리용)"""
    var, y, X_var, breakpoint_idx = args

    try:
        X = sm.add_constant(X_var)
        result = chow_test(y, X, breakpoint_idx, use_hac=True)

        return {
            'variable': var,
            'F_stat': result['F_stat'],
            'p_value': result['p_value'],
            'df1': result['df1'],
            'df2': result['df2'],
            'coef_pre': result['coef_pre'],
            'coef_post': result['coef_post'],
            'coef_change': result['coef_change'],
            'pct_change': (result['coef_change'] / abs(result['coef_pre']) * 100)
                         if result['coef_pre'] != 0 and not np.isnan(result['coef_pre']) else np.nan
        }
    except Exception as e:
        return {
            'variable': var,
            'F_stat': np.nan,
            'p_value': np.nan,
            'df1': np.nan,
            'df2': np.nan,
            'coef_pre': np.nan,
            'coef_post': np.nan,
            'coef_change': np.nan,
            'pct_change': np.nan
        }

def run_chow_tests_all_vars(df, etf_date, target_col='Close', n_workers=4):
    """
    모든 변수에 대해 Chow Test 수행 (병렬 처리)

    Args:
        df: 데이터프레임
        etf_date: ETF 날짜
        target_col: 타겟 변수
        n_workers: 병렬 처리 워커 수

    Returns:
        results_df: 결과 데이터프레임
    """
    print("\n" + "="*80)
    print("2. Chow Test 수행 (모든 변수)")
    print("="*80)
    print(f"변화점: {etf_date.date()}")

    y = df[target_col]
    X_cols = [col for col in df.columns if col != target_col]

    # 숫자형 변수만 선택
    numeric_cols = []
    for col in X_cols:
        if df[col].dtype in ['float64', 'int64']:
            # 분산이 0이 아닌지 확인
            if df[col].std() > 1e-10:
                numeric_cols.append(col)

    print(f"검정 변수 수: {len(numeric_cols)}")

    breakpoint_idx = (df.index < etf_date).sum()

    # 병렬 처리를 위한 arguments 준비
    args_list = [(var, y, df[[var]], breakpoint_idx) for var in numeric_cols]

    results = []

    # 병렬 처리
    print(f"\n병렬 처리 시작 (워커 {n_workers}개)...")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # tqdm으로 진행률 표시
        futures = {executor.submit(test_single_variable, args): args[0]
                  for args in args_list}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Chow Test"):
            result = future.result()
            results.append(result)

    results_df = pd.DataFrame(results)

    # NaN 제거
    results_df = results_df.dropna(subset=['F_stat', 'p_value'])

    # Bonferroni 보정
    alpha = 0.05
    n_tests = len(results_df)
    alpha_bonf = alpha / n_tests
    results_df['p_adj_bonf'] = results_df['p_value']
    results_df['significant_bonf'] = results_df['p_adj_bonf'] < alpha_bonf

    # FDR 보정 (Benjamini-Hochberg)
    from statsmodels.stats.multitest import multipletests
    reject_fdr, p_adj_fdr, _, _ = multipletests(results_df['p_value'], alpha=alpha, method='fdr_bh')
    results_df['p_adj_fdr'] = p_adj_fdr
    results_df['significant_fdr'] = reject_fdr

    # 정렬
    results_df = results_df.sort_values('F_stat', ascending=False)

    print(f"\n검정 완료: {len(results_df)} 변수")
    print(f"\nBonferroni 보정 (α = {alpha_bonf:.6f}):")
    print(f"  - 유의한 변수: {results_df['significant_bonf'].sum()} / {n_tests}")

    print(f"\nFDR 보정 (α = {alpha}):")
    print(f"  - 유의한 변수: {results_df['significant_fdr'].sum()} / {n_tests}")

    return results_df

# ============================================================================
# 3. Quandt-Andrews Test
# ============================================================================

def quandt_andrews_test(y, X, trim=0.15):
    """Quandt-Andrews Test for unknown breakpoint"""
    n = len(y)
    k = X.shape[1]

    tau_min = int(trim * n)
    tau_max = int((1 - trim) * n)

    f_stats = []

    for tau in range(tau_min, tau_max):
        if tau < k + 10 or (n - tau) < k + 10:
            continue

        result = chow_test(y, X, tau, use_hac=True)
        if not np.isnan(result['F_stat']):
            f_stats.append((tau, result['F_stat']))

    if len(f_stats) == 0:
        return {
            'sup_F': np.nan,
            'tau_star': np.nan,
            'breakpoint_date': None,
            'significant_05': False,
            'significant_01': False
        }

    tau_star, sup_f = max(f_stats, key=lambda x: x[1])

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
        'critical_values': critical_values,
        'significant_05': sup_f > critical_values[0.05],
        'significant_01': sup_f > critical_values[0.01]
    }

def test_single_variable_qa(args):
    """단일 변수 QA 검정 (병렬 처리용)"""
    var, y, X_var = args

    try:
        X = sm.add_constant(X_var)
        result = quandt_andrews_test(y, X, trim=0.15)

        return {
            'variable': var,
            'sup_F': result['sup_F'],
            'breakpoint_date': result['breakpoint_date'],
            'tau_star': result['tau_star'],
            'critical_10': 7.12,
            'critical_05': 8.68,
            'critical_01': 12.16,
            'significant_05': result['significant_05'],
            'significant_01': result['significant_01']
        }
    except Exception as e:
        return {
            'variable': var,
            'sup_F': np.nan,
            'breakpoint_date': None,
            'tau_star': np.nan,
            'critical_10': 7.12,
            'critical_05': 8.68,
            'critical_01': 12.16,
            'significant_05': False,
            'significant_01': False
        }

def run_quandt_andrews_tests_all_vars(df, chow_results, target_col='Close',
                                      top_n=50, n_workers=4):
    """
    상위 변수들에 대해 Quandt-Andrews Test 수행

    Args:
        df: 데이터프레임
        chow_results: Chow Test 결과 (상위 변수 선택용)
        target_col: 타겟 변수
        top_n: 상위 몇 개 변수
        n_workers: 병렬 처리 워커 수
    """
    print("\n" + "="*80)
    print("3. Quandt-Andrews Test 수행")
    print("="*80)

    # Chow Test에서 유의한 변수 중 상위 top_n개 선택
    significant_vars = chow_results[chow_results['significant_fdr']].head(top_n)['variable'].tolist()

    print(f"검정 변수 수: {len(significant_vars)} (Chow Test 유의 변수 중 상위)")

    y = df[target_col]

    # 병렬 처리를 위한 arguments 준비
    args_list = [(var, y, df[[var]]) for var in significant_vars]

    results = []

    print(f"\n병렬 처리 시작 (워커 {n_workers}개)...")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(test_single_variable_qa, args): args[0]
                  for args in args_list}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Q-A Test"):
            result = future.result()
            results.append(result)

    results_df = pd.DataFrame(results)
    results_df = results_df.dropna(subset=['sup_F'])
    results_df = results_df.sort_values('sup_F', ascending=False)

    print(f"\n검정 완료: {len(results_df)} 변수")
    print(f"유의한 변화점 검출 (α = 0.05): {results_df['significant_05'].sum()} / {len(results_df)}")

    return results_df

# ============================================================================
# 4. Visualization
# ============================================================================

def plot_chow_results_all(chow_results, save_path='plots/chow_test_all_vars.png', top_n=30):
    """Chow Test 결과 시각화 (상위 변수)"""
    import os
    os.makedirs('plots', exist_ok=True)

    top_results = chow_results.head(top_n).copy()

    fig, axes = plt.subplots(2, 1, figsize=(18, 14))

    # F-통계량
    ax1 = axes[0]
    colors = ['red' if sig else 'gray' for sig in top_results['significant_bonf']]
    bars = ax1.barh(range(len(top_results)), top_results['F_stat'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(top_results)))
    ax1.set_yticklabels([v[:40] for v in top_results['variable']], fontsize=8)
    ax1.set_xlabel('F-통계량', fontsize=12, fontweight='bold')
    ax1.set_title(f'Chow Test 결과 - 모든 변수 (상위 {top_n}개)', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')

    for i, (f_stat, p_val, sig) in enumerate(zip(top_results['F_stat'],
                                                   top_results['p_value'],
                                                   top_results['significant_bonf'])):
        sig_mark = '***' if sig else ''
        ax1.text(f_stat, i, f'  F={f_stat:.1f} {sig_mark}',
                va='center', fontsize=7)

    # 계수 변화
    ax2 = axes[1]
    # NaN이 아닌 값만 선택
    plot_data = top_results.dropna(subset=['coef_change']).head(top_n)

    if len(plot_data) > 0:
        colors2 = ['green' if c > 0 else 'red' for c in plot_data['coef_change']]
        bars2 = ax2.barh(range(len(plot_data)), plot_data['coef_change'], color=colors2, alpha=0.7)
        ax2.set_yticks(range(len(plot_data)))
        ax2.set_yticklabels([v[:40] for v in plot_data['variable']], fontsize=8)
        ax2.set_xlabel('계수 변화량 (ETF 후 - ETF 전)', fontsize=12, fontweight='bold')
        ax2.set_title('회귀계수 변화', fontsize=14, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n그래프 저장: {save_path}")
    plt.close()

def create_summary_report(chow_results, qa_results, save_path='results/all_vars_summary.txt'):
    """요약 보고서 생성"""
    import os
    os.makedirs('results', exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Bitcoin 구조 변화 검정 - 모든 변수 분석 결과\n")
        f.write("="*80 + "\n\n")

        # Chow Test 요약
        f.write("【Chow Test 결과】\n")
        f.write(f"  - 검정 변수 수: {len(chow_results)}\n")
        f.write(f"  - Bonferroni 유의: {chow_results['significant_bonf'].sum()}개\n")
        f.write(f"  - FDR 유의: {chow_results['significant_fdr'].sum()}개\n\n")

        f.write("  상위 30개 변수:\n")
        for i, row in chow_results.head(30).iterrows():
            sig_mark = '***' if row['significant_bonf'] else ('**' if row['significant_fdr'] else '')
            f.write(f"    {row['variable']:45s} | F={row['F_stat']:8.2f} | p={row['p_value']:.2e} {sig_mark}\n")

        f.write("\n" + "="*80 + "\n")

        # Quandt-Andrews 요약
        f.write("\n【Quandt-Andrews Test 결과】\n")
        f.write(f"  - 검정 변수 수: {len(qa_results)}\n")
        f.write(f"  - 유의한 변화점: {qa_results['significant_05'].sum()}개\n\n")

        f.write("  상위 20개 변수:\n")
        for i, row in qa_results.head(20).iterrows():
            sig_mark = '***' if row['significant_01'] else ('**' if row['significant_05'] else '')
            bp_date = row['breakpoint_date'].date() if pd.notna(row['breakpoint_date']) else 'N/A'
            f.write(f"    {row['variable']:45s} | sup_F={row['sup_F']:8.2f} | {bp_date} {sig_mark}\n")

        f.write("\n" + "="*80 + "\n")

    print(f"요약 보고서 저장: {save_path}")

# ============================================================================
# 5. Main
# ============================================================================

def main():
    """메인 실행 함수"""
    print("="*80)
    print("Bitcoin 구조 변화 검정 - 모든 변수 분석")
    print("="*80)
    print("- 119개 변수 전체 검정")
    print("- Bonferroni 보정 (α = 0.05/119 = 0.00042)")
    print("- 병렬 처리로 속도 향상")
    print("="*80)

    # 1. 데이터 준비
    df, etf_date = load_and_prepare_data(
        file_path='integrated_data_full_v2.csv',
        etf_date='2024-01-10',
        winsorize_limits=(0.01, 0.01)
    )

    # 2. Chow Test (모든 변수, 병렬 처리)
    chow_results = run_chow_tests_all_vars(
        df,
        etf_date,
        target_col='Close',
        n_workers=4  # CPU 코어 수에 맞게 조정
    )

    # 3. Quandt-Andrews Test (상위 50개 변수)
    qa_results = run_quandt_andrews_tests_all_vars(
        df,
        chow_results,
        target_col='Close',
        top_n=50,
        n_workers=4
    )

    # 4. 시각화
    print("\n" + "="*80)
    print("4. 시각화")
    print("="*80)

    plot_chow_results_all(chow_results, save_path='plots/chow_test_all_vars.png', top_n=30)

    # 5. 결과 저장
    print("\n" + "="*80)
    print("5. 결과 저장")
    print("="*80)

    import os
    os.makedirs('results', exist_ok=True)

    chow_results.to_csv('results/chow_test_all_vars.csv', index=False)
    qa_results.to_csv('results/quandt_andrews_all_vars.csv', index=False)

    print("  - results/chow_test_all_vars.csv")
    print("  - results/quandt_andrews_all_vars.csv")

    # 6. 요약 보고서
    create_summary_report(chow_results, qa_results)

    # 7. 카테고리별 분석
    print("\n" + "="*80)
    print("6. 카테고리별 분석")
    print("="*80)

    # 변수 카테고리 정의
    categories = {
        '가격/기술적': ['Volume', 'RSI', 'MACD', 'BB_width', 'volatility_20d', 'ATR', 'OBV',
                       'Stoch_K', 'Stoch_D', 'ADX', 'CCI', 'Williams_R', 'ROC', 'MFI'],
        '전통시장': ['SPX', 'QQQ', 'IWM', 'DIA', 'ETH', 'GOLD', 'SILVER', 'OIL',
                    'GLD', 'TLT', 'HYG', 'LQD'],
        '금리': ['T10Y3M', 'T10Y2Y', 'DFF', 'SOFR', 'DGS10', 'BAMLH0A0HYM2', 'BAMLC0A0CM'],
        'Fed유동성': ['WALCL', 'RRPONTSYD', 'WTREGEN', 'FED_NET_LIQUIDITY'],
        'VIX/변동성': ['VIX', 'VIXCLS'],
        '온체인': ['bc_hash_rate', 'bc_difficulty', 'bc_n_transactions', 'bc_n_unique_addresses',
                  'bc_miners_revenue', 'bc_transaction_fees', 'NVT_Ratio', 'Puell_Multiple',
                  'Hash_Ribbon_MA30', 'Hash_Ribbon_MA60', 'Miner_Revenue_to_Cap',
                  'Active_Addresses_Change', 'Hash_Price'],
        'ETF': ['IBIT_Price', 'FBTC_Price', 'GBTC_Price', 'ARKB_Price', 'BITB_Price',
                'Total_BTC_ETF_Volume', 'IBIT_Volume_Change_7d'],
        '심리지표': ['fear_greed_index', 'google_trends_btc']
    }

    for category, vars_list in categories.items():
        category_results = chow_results[chow_results['variable'].isin(vars_list)]
        if len(category_results) > 0:
            significant = category_results['significant_bonf'].sum()
            print(f"\n【{category}】")
            print(f"  - 검정 변수: {len(category_results)}개")
            print(f"  - 유의한 변수: {significant}개 ({significant/len(category_results)*100:.1f}%)")

            if significant > 0:
                print(f"  - 상위 5개:")
                for idx, row in category_results.head(5).iterrows():
                    sig_mark = '***' if row['significant_bonf'] else ''
                    print(f"      {row['variable']:35s} | F={row['F_stat']:8.2f} {sig_mark}")

    print("\n" + "="*80)
    print("분석 완료!")
    print("="*80)

if __name__ == '__main__':
    import os
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    main()
