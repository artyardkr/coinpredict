"""
다중 윈도우 롤링 상관계수 분석
- 여러 윈도우 크기로 롤링 상관계수 계산 및 비교
- ETF 전후 변화 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path='integrated_data_full_v2.csv'):
    """데이터 로드"""
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    return df

def calculate_multi_window_correlation(df, target_col='Close', windows=[30, 60, 90, 180]):
    """
    여러 윈도우 크기로 롤링 상관계수 계산

    Args:
        df: 데이터프레임
        target_col: 기준 변수
        windows: 윈도우 크기 리스트

    Returns:
        dict: {window: {variable: correlation_series}}
    """
    print(f"\n다중 윈도우 롤링 상관계수 계산 중...")
    print(f"기준 변수: {target_col}")
    print(f"윈도우 크기: {windows}")

    results = {window: {} for window in windows}

    # 숫자형 컬럼만 선택
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target_col]

    print(f"분석 대상 변수: {len(numeric_cols)}개")

    for window in windows:
        print(f"\n  윈도우 {window}일 계산 중...")
        for col in numeric_cols:
            try:
                # 결측치 및 inf 처리
                temp_df = df[[target_col, col]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(temp_df) > window:
                    rolling_corr = temp_df[target_col].rolling(window=window).corr(temp_df[col])
                    results[window][col] = rolling_corr
            except:
                continue

    return results

def analyze_etf_changes(results, etf_date='2024-01-10', pre_days=90, post_days=90):
    """
    ETF 전후 상관계수 변화 분석 (다중 윈도우)

    Args:
        results: calculate_multi_window_correlation 결과
        etf_date: ETF 승인일
        pre_days: ETF 전 분석 기간
        post_days: ETF 후 분석 기간

    Returns:
        DataFrame: 윈도우별 변화량
    """
    etf_date = pd.Timestamp(etf_date)
    pre_start = etf_date - pd.Timedelta(days=pre_days)
    post_end = etf_date + pd.Timedelta(days=post_days)

    change_summary = []

    for window in sorted(results.keys()):
        print(f"\n{'='*80}")
        print(f"윈도우 {window}일 - ETF 전후 상관계수 변화 (전후 {pre_days}일 평균 비교)")
        print(f"{'='*80}")

        window_data = results[window]

        for var, corr_series in window_data.items():
            # ETF 전후 마스크
            pre_mask = (corr_series.index >= pre_start) & (corr_series.index < etf_date)
            post_mask = (corr_series.index >= etf_date) & (corr_series.index <= post_end)

            pre_corr = corr_series[pre_mask].mean()
            post_corr = corr_series[post_mask].mean()

            if not (np.isnan(pre_corr) or np.isnan(post_corr)):
                change = post_corr - pre_corr
                change_summary.append({
                    'window': window,
                    'variable': var,
                    'pre_etf': pre_corr,
                    'post_etf': post_corr,
                    'change': change,
                    'abs_change': abs(change)
                })

    df_summary = pd.DataFrame(change_summary)
    return df_summary

def plot_multi_window_comparison(results, variables, save_path='plots/multi_window_comparison.png'):
    """
    주요 변수들의 다중 윈도우 롤링 상관계수 비교

    Args:
        results: calculate_multi_window_correlation 결과
        variables: 비교할 변수 리스트
        save_path: 저장 경로
    """
    import os
    os.makedirs('plots', exist_ok=True)

    windows = sorted(results.keys())
    n_vars = len(variables)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten() if n_vars > 1 else [axes]

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    etf_date = pd.Timestamp('2024-01-10')

    for idx, var in enumerate(variables):
        ax = axes[idx]

        for i, window in enumerate(windows):
            if var in results[window]:
                corr = results[window][var]
                ax.plot(corr.index, corr, label=f'{window}일',
                       color=colors[i % len(colors)], alpha=0.7, linewidth=2)

        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.axvline(x=etf_date, color='red', linestyle='--', alpha=0.5, linewidth=2, label='ETF 승인')
        ax.set_title(f'{var}', fontsize=12, fontweight='bold')
        ax.set_ylabel('상관계수', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    # 빈 subplot 제거
    for idx in range(n_vars, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n그래프 저장: {save_path}")
    plt.close()

def plot_window_sensitivity_heatmap(df_summary, top_n=20, save_path='plots/window_sensitivity_heatmap.png'):
    """
    윈도우별 상관계수 변화 히트맵

    Args:
        df_summary: analyze_etf_changes 결과
        top_n: 상위 몇 개 변수 표시
        save_path: 저장 경로
    """
    import os
    os.makedirs('plots', exist_ok=True)

    # 변화량이 큰 상위 변수 선택
    top_vars = (df_summary.groupby('variable')['abs_change'].mean()
                .sort_values(ascending=False).head(top_n).index.tolist())

    # 윈도우별 변화량 피벗
    pivot_data = df_summary[df_summary['variable'].isin(top_vars)].pivot(
        index='variable', columns='window', values='change'
    )

    # 변수명 길이 제한
    pivot_data.index = [idx[:30] for idx in pivot_data.index]

    fig, axes = plt.subplots(1, 3, figsize=(24, 10))

    # 1. 변화량 히트맵
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                vmin=-0.5, vmax=0.5, ax=axes[0], cbar_kws={'label': '변화량'})
    axes[0].set_title('ETF 전후 상관계수 변화 (윈도우별)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('윈도우 크기 (일)', fontsize=12)
    axes[0].set_ylabel('변수', fontsize=12)

    # 2. ETF 전 상관계수
    pivot_pre = df_summary[df_summary['variable'].isin(top_vars)].pivot(
        index='variable', columns='window', values='pre_etf'
    )
    pivot_pre.index = [idx[:30] for idx in pivot_pre.index]

    sns.heatmap(pivot_pre, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, ax=axes[1], cbar_kws={'label': '상관계수'})
    axes[1].set_title('ETF 전 상관계수 (윈도우별)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('윈도우 크기 (일)', fontsize=12)
    axes[1].set_ylabel('')

    # 3. ETF 후 상관계수
    pivot_post = df_summary[df_summary['variable'].isin(top_vars)].pivot(
        index='variable', columns='window', values='post_etf'
    )
    pivot_post.index = [idx[:30] for idx in pivot_post.index]

    sns.heatmap(pivot_post, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, ax=axes[2], cbar_kws={'label': '상관계수'})
    axes[2].set_title('ETF 후 상관계수 (윈도우별)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('윈도우 크기 (일)', fontsize=12)
    axes[2].set_ylabel('')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"히트맵 저장: {save_path}")
    plt.close()

def plot_top_changes_by_window(df_summary, top_n=15, save_path='plots/top_changes_by_window.png'):
    """
    윈도우별 상위 변화 변수 비교
    """
    import os
    os.makedirs('plots', exist_ok=True)

    windows = sorted(df_summary['window'].unique())
    n_windows = len(windows)

    fig, axes = plt.subplots(2, n_windows, figsize=(6*n_windows, 12))

    for i, window in enumerate(windows):
        window_data = df_summary[df_summary['window'] == window].copy()

        # 상위 증가
        top_increase = window_data.nlargest(top_n, 'change')
        ax1 = axes[0, i] if n_windows > 1 else axes[0]

        vars_inc = [v[:25] for v in top_increase['variable'].tolist()]
        changes_inc = top_increase['change'].tolist()

        ax1.barh(range(len(vars_inc)), changes_inc, color='#2ecc71', alpha=0.7)
        ax1.set_yticks(range(len(vars_inc)))
        ax1.set_yticklabels(vars_inc, fontsize=9)
        ax1.set_xlabel('변화량', fontsize=10)
        ax1.set_title(f'윈도우 {window}일 - 상관계수 증가 TOP {top_n}', fontsize=11, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')

        # 값 표시
        for j, change in enumerate(changes_inc):
            ax1.text(change, j, f' {change:+.3f}', va='center', fontsize=8)

        # 상위 감소
        top_decrease = window_data.nsmallest(top_n, 'change')
        ax2 = axes[1, i] if n_windows > 1 else axes[1]

        vars_dec = [v[:25] for v in top_decrease['variable'].tolist()]
        changes_dec = top_decrease['change'].tolist()

        ax2.barh(range(len(vars_dec)), changes_dec, color='#e74c3c', alpha=0.7)
        ax2.set_yticks(range(len(vars_dec)))
        ax2.set_yticklabels(vars_dec, fontsize=9)
        ax2.set_xlabel('변화량', fontsize=10)
        ax2.set_title(f'윈도우 {window}일 - 상관계수 감소 TOP {top_n}', fontsize=11, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3, axis='x')

        # 값 표시
        for j, change in enumerate(changes_dec):
            ax2.text(change, j, f' {change:+.3f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"상위 변화 그래프 저장: {save_path}")
    plt.close()

def main():
    """메인 실행 함수"""
    print("="*80)
    print("다중 윈도우 롤링 상관계수 분석")
    print("="*80)

    # 1. 데이터 로드
    print("\n1. 데이터 로딩 중...")
    df = load_data('integrated_data_full_v2.csv')
    print(f"   - 기간: {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"   - 샘플 수: {len(df):,}개")
    print(f"   - 변수 수: {len(df.columns)}개")

    # 2. 다중 윈도우 롤링 상관계수 계산
    windows = [30, 60, 90, 180, 365]  # 1개월, 2개월, 3개월, 6개월, 1년
    print(f"\n2. 다중 윈도우 롤링 상관계수 계산 중...")
    results = calculate_multi_window_correlation(df, target_col='Close', windows=windows)

    # 3. ETF 전후 변화 분석
    print(f"\n3. ETF 전후 변화 분석 중...")
    df_summary = analyze_etf_changes(results, etf_date='2024-01-10', pre_days=90, post_days=90)

    # 4. 주요 변수들의 다중 윈도우 비교 시각화
    print(f"\n4. 주요 변수 다중 윈도우 비교 시각화 중...")

    # 주요 변수 선택 (카테고리별)
    key_variables = [
        # 금리
        'T10Y3M', 'T10Y2Y', 'DFF',
        # 전통 시장
        'SPX', 'QQQ', 'IWM', 'ETH',
        # 온체인
        'NVT_Ratio', 'Hash_Price', 'Miner_Revenue_to_Cap_MA30',
        # Fed 유동성
        'FED_NET_LIQUIDITY', 'WALCL', 'RRPONTSYD',
        # 기술적
        'RSI', 'Volume', 'fear_greed_index',
        # ETF 관련
        'GBTC_Price', 'Total_BTC_ETF_Volume'
    ]

    # 존재하는 변수만 필터링
    key_variables = [v for v in key_variables if v in df.columns]

    plot_multi_window_comparison(results, key_variables,
                                 save_path='plots/multi_window_comparison.png')

    # 5. 윈도우 민감도 히트맵
    print(f"\n5. 윈도우 민감도 히트맵 생성 중...")
    plot_window_sensitivity_heatmap(df_summary, top_n=25,
                                   save_path='plots/window_sensitivity_heatmap.png')

    # 6. 윈도우별 상위 변화 변수
    print(f"\n6. 윈도우별 상위 변화 변수 시각화 중...")
    plot_top_changes_by_window(df_summary, top_n=12,
                               save_path='plots/top_changes_by_window.png')

    # 7. 윈도우별 요약 통계
    print(f"\n7. 윈도우별 요약 통계:")
    print("="*80)

    for window in sorted(windows):
        window_data = df_summary[df_summary['window'] == window]
        print(f"\n윈도우 {window}일:")
        print(f"  - 분석 변수 수: {len(window_data)}개")
        print(f"  - 평균 변화량: {window_data['change'].mean():.4f}")
        print(f"  - 변화량 표준편차: {window_data['change'].std():.4f}")
        print(f"  - 최대 증가: {window_data['change'].max():.4f} ({window_data.loc[window_data['change'].idxmax(), 'variable']})")
        print(f"  - 최대 감소: {window_data['change'].min():.4f} ({window_data.loc[window_data['change'].idxmin(), 'variable']})")

    # 8. 윈도우 일관성 분석
    print(f"\n8. 윈도우 일관성 분석:")
    print("="*80)
    print("모든 윈도우에서 일관되게 증가/감소한 변수들")

    # 각 변수별로 모든 윈도우에서의 변화 방향 확인
    var_consistency = {}
    for var in df_summary['variable'].unique():
        var_data = df_summary[df_summary['variable'] == var]
        if len(var_data) == len(windows):  # 모든 윈도우에 데이터가 있는 경우만
            changes = var_data['change'].values
            if all(changes > 0):
                var_consistency[var] = ('증가', changes.mean())
            elif all(changes < 0):
                var_consistency[var] = ('감소', changes.mean())

    # 일관된 증가
    consistent_increase = [(var, avg) for var, (direction, avg) in var_consistency.items() if direction == '증가']
    consistent_increase.sort(key=lambda x: x[1], reverse=True)

    print(f"\n모든 윈도우에서 일관되게 증가한 변수 ({len(consistent_increase)}개):")
    for i, (var, avg_change) in enumerate(consistent_increase[:15], 1):
        print(f"  {i:2d}. {var:40s} | 평균 변화: {avg_change:+.4f}")

    # 일관된 감소
    consistent_decrease = [(var, avg) for var, (direction, avg) in var_consistency.items() if direction == '감소']
    consistent_decrease.sort(key=lambda x: x[1])

    print(f"\n모든 윈도우에서 일관되게 감소한 변수 ({len(consistent_decrease)}개):")
    for i, (var, avg_change) in enumerate(consistent_decrease[:15], 1):
        print(f"  {i:2d}. {var:40s} | 평균 변화: {avg_change:+.4f}")

    # 9. 결과 저장
    print(f"\n9. 결과 저장 중...")
    df_summary.to_csv('results/multi_window_correlation_summary.csv', index=False)
    print("   - results/multi_window_correlation_summary.csv")

    # 윈도우별 상세 결과 저장
    for window in windows:
        window_data = df_summary[df_summary['window'] == window].copy()
        window_data = window_data.sort_values('abs_change', ascending=False)
        window_data.to_csv(f'results/correlation_window_{window}d.csv', index=False)
        print(f"   - results/correlation_window_{window}d.csv")

    print("\n" + "="*80)
    print("분석 완료!")
    print("="*80)
    print("\n생성된 파일:")
    print("  - plots/multi_window_comparison.png       : 주요 변수 다중 윈도우 비교")
    print("  - plots/window_sensitivity_heatmap.png    : 윈도우별 변화 히트맵")
    print("  - plots/top_changes_by_window.png         : 윈도우별 상위 변화 변수")
    print("  - results/multi_window_correlation_summary.csv  : 전체 요약")
    for window in windows:
        print(f"  - results/correlation_window_{window}d.csv")
    print()

if __name__ == '__main__':
    import os
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    main()
