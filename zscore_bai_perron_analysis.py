"""
Z-Score 표준화 Bai-Perron 다중 구조변화 검정
(구조변화검정_표준_프로토콜.md 구현)

목적:
- 여러 개의 구조변화 시점을 자동으로 탐지
- Chow/QA는 단일 변화점만 찾음 → Bai-Perron은 여러 변화점 탐지
- 모든 변수를 Z-score로 표준화하여 직접 비교 가능
- BIC를 사용한 최적 변화점 개수 자동 선택

작성일: 2025-11-11
작성자: Song Hyowon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
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
# Bai-Perron Multiple Breakpoint Test
# ============================================================================

def calculate_ssr(y, X):
    """잔차제곱합 계산"""
    model = sm.OLS(y, X).fit()
    return model.ssr


def bai_perron_test(y, X, max_breaks=5, trim=0.15, min_segment_size=None):
    """
    Bai-Perron 다중 구조변화 검정

    Parameters:
    -----------
    y : array-like
        종속변수
    X : array-like
        독립변수 (상수항 포함)
    max_breaks : int
        최대 변화점 개수 (기본값: 5)
    trim : float
        양 끝단 트림 비율 (기본값: 0.15)
    min_segment_size : int
        최소 세그먼트 크기 (기본값: 전체 샘플의 15%)

    Returns:
    --------
    dict : 검정 결과
        - optimal_breaks: 최적 변화점 개수
        - breakpoints: 변화점 인덱스 리스트
        - breakpoint_dates: 변화점 날짜 리스트
        - bic_values: 각 변화점 개수별 BIC
        - ssr_values: 각 변화점 개수별 SSR
        - f_statistics: 각 변화점별 F-통계량
    """

    n = len(y)
    k = X.shape[1]

    # 최소 세그먼트 크기 설정
    if min_segment_size is None:
        min_segment_size = max(int(n * trim), k + 5)

    print(f"\n  전체 샘플: {n}개")
    print(f"  최소 세그먼트 크기: {min_segment_size}개")
    print(f"  최대 변화점: {max_breaks}개")

    # 0개 변화점 (단일 체제) - 기준 모델
    ssr_0 = calculate_ssr(y, X)
    bic_0 = n * np.log(ssr_0 / n) + k * np.log(n)

    results = {
        0: {
            'ssr': ssr_0,
            'bic': bic_0,
            'breakpoints': [],
            'breakpoint_dates': []
        }
    }

    # 1개부터 max_breaks개까지 변화점 탐색
    for m in range(1, max_breaks + 1):
        print(f"\n  {m}개 변화점 탐색 중...")

        # 동적 프로그래밍으로 최적 변화점 조합 찾기
        best_ssr = np.inf
        best_breakpoints = None

        # 가능한 모든 변화점 조합 탐색
        breakpoint_candidates = list(range(min_segment_size, n - min_segment_size))

        if m == 1:
            # 1개 변화점: 단순 탐색
            for bp in breakpoint_candidates:
                # 세그먼트가 최소 크기 이상인지 확인
                if bp < min_segment_size or (n - bp) < min_segment_size:
                    continue

                ssr1 = calculate_ssr(y[:bp], X[:bp])
                ssr2 = calculate_ssr(y[bp:], X[bp:])
                total_ssr = ssr1 + ssr2

                if total_ssr < best_ssr:
                    best_ssr = total_ssr
                    best_breakpoints = [bp]

        elif m == 2:
            # 2개 변화점: 이중 루프
            for bp1 in breakpoint_candidates:
                if bp1 < min_segment_size:
                    continue

                for bp2 in breakpoint_candidates:
                    if bp2 <= bp1 + min_segment_size or (n - bp2) < min_segment_size:
                        continue

                    ssr1 = calculate_ssr(y[:bp1], X[:bp1])
                    ssr2 = calculate_ssr(y[bp1:bp2], X[bp1:bp2])
                    ssr3 = calculate_ssr(y[bp2:], X[bp2:])
                    total_ssr = ssr1 + ssr2 + ssr3

                    if total_ssr < best_ssr:
                        best_ssr = total_ssr
                        best_breakpoints = [bp1, bp2]

        elif m == 3:
            # 3개 변화점: 삼중 루프
            for bp1 in breakpoint_candidates[::2]:  # 속도를 위해 간격 2로
                if bp1 < min_segment_size:
                    continue

                for bp2 in breakpoint_candidates[::2]:
                    if bp2 <= bp1 + min_segment_size:
                        continue

                    for bp3 in breakpoint_candidates[::2]:
                        if bp3 <= bp2 + min_segment_size or (n - bp3) < min_segment_size:
                            continue

                        ssr1 = calculate_ssr(y[:bp1], X[:bp1])
                        ssr2 = calculate_ssr(y[bp1:bp2], X[bp1:bp2])
                        ssr3 = calculate_ssr(y[bp2:bp3], X[bp2:bp3])
                        ssr4 = calculate_ssr(y[bp3:], X[bp3:])
                        total_ssr = ssr1 + ssr2 + ssr3 + ssr4

                        if total_ssr < best_ssr:
                            best_ssr = total_ssr
                            best_breakpoints = [bp1, bp2, bp3]

        else:
            # 4개 이상: 균등 분할 근사법 (계산 복잡도 때문)
            # 전체 기간을 m+1개 구간으로 균등 분할한 후 각 변화점을 미세 조정
            segment_size = n // (m + 1)
            initial_breakpoints = [segment_size * (i + 1) for i in range(m)]

            # 각 변화점 주변 탐색
            search_range = min(segment_size // 4, 50)  # 탐색 범위

            for offsets in np.ndindex(*([3] * m)):  # -1, 0, +1 조합
                breakpoints = []
                valid = True

                for i, offset in enumerate(offsets):
                    bp = initial_breakpoints[i] + (offset - 1) * search_range

                    # 유효성 검사
                    if i == 0:
                        if bp < min_segment_size or bp > n - min_segment_size * m:
                            valid = False
                            break
                    else:
                        if bp <= breakpoints[-1] + min_segment_size or bp > n - min_segment_size * (m - i):
                            valid = False
                            break

                    breakpoints.append(bp)

                if not valid:
                    continue

                # SSR 계산
                total_ssr = 0
                prev_bp = 0

                for bp in breakpoints + [n]:
                    if bp - prev_bp >= k + 1:  # 최소한 추정 가능한 크기
                        total_ssr += calculate_ssr(y[prev_bp:bp], X[prev_bp:bp])
                    else:
                        total_ssr = np.inf
                        break
                    prev_bp = bp

                if total_ssr < best_ssr:
                    best_ssr = total_ssr
                    best_breakpoints = breakpoints

        # 결과 저장
        if best_breakpoints is not None:
            # BIC 계산
            bic = n * np.log(best_ssr / n) + (m + 1) * k * np.log(n)

            # 변화점 날짜 변환
            breakpoint_dates = [y.index[bp] for bp in best_breakpoints]

            results[m] = {
                'ssr': best_ssr,
                'bic': bic,
                'breakpoints': best_breakpoints,
                'breakpoint_dates': breakpoint_dates
            }

            print(f"    SSR: {best_ssr:.2f}")
            print(f"    BIC: {bic:.2f}")
            print(f"    변화점: {breakpoint_dates}")

    # 최적 변화점 개수 선택 (최소 BIC)
    bic_values = {m: results[m]['bic'] for m in results.keys()}
    optimal_breaks = min(bic_values, key=bic_values.get)

    print(f"\n  최적 변화점 개수: {optimal_breaks}개 (BIC 기준)")

    # F-통계량 계산 (각 변화점에 대해)
    f_statistics = []
    if optimal_breaks > 0:
        ssr_restricted = results[0]['ssr']
        ssr_unrestricted = results[optimal_breaks]['ssr']

        numerator = (ssr_restricted - ssr_unrestricted) / (optimal_breaks * k)
        denominator = ssr_unrestricted / (n - (optimal_breaks + 1) * k)

        f_stat = numerator / denominator
        df1 = optimal_breaks * k
        df2 = n - (optimal_breaks + 1) * k
        p_value = 1 - stats.f.cdf(f_stat, df1, df2)

        f_statistics = {
            'F_stat': f_stat,
            'p_value': p_value,
            'df1': df1,
            'df2': df2
        }

    return {
        'optimal_breaks': optimal_breaks,
        'breakpoints': results[optimal_breaks]['breakpoints'],
        'breakpoint_dates': results[optimal_breaks]['breakpoint_dates'],
        'bic_values': bic_values,
        'ssr_values': {m: results[m]['ssr'] for m in results.keys()},
        'f_statistics': f_statistics,
        'all_results': results
    }


def run_bai_perron_all_variables(df_scaled, y_col, variables, max_breaks=5, trim=0.15):
    """
    전체 변수에 대해 Bai-Perron Test 실행
    """

    print("\nBai-Perron 다중 구조변화 검정")
    print("=" * 80)

    results = []

    for i, var in enumerate(variables, 1):
        if var == y_col:
            continue

        print(f"\n[{i}/{len(variables)}] {var}")
        print("-" * 80)

        y = df_scaled[y_col]
        X = sm.add_constant(df_scaled[[var]])

        try:
            result = bai_perron_test(y, X, max_breaks=max_breaks, trim=trim)

            result['Variable'] = var
            results.append(result)

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            continue

    # 결과를 DataFrame으로 변환
    summary_data = []

    for r in results:
        summary_data.append({
            'Variable': r['Variable'],
            'Optimal_Breaks': r['optimal_breaks'],
            'Breakpoints': ', '.join([str(d.date()) for d in r['breakpoint_dates']]),
            'BIC_0breaks': r['bic_values'][0],
            'BIC_optimal': r['bic_values'][r['optimal_breaks']],
            'BIC_improvement': r['bic_values'][0] - r['bic_values'][r['optimal_breaks']],
            'F_stat': r['f_statistics'].get('F_stat', np.nan) if r['f_statistics'] else np.nan,
            'p_value': r['f_statistics'].get('p_value', np.nan) if r['f_statistics'] else np.nan
        })

    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.sort_values('BIC_improvement', ascending=False)

    # 통계 요약
    print("\n" + "=" * 80)
    print("전체 요약")
    print("=" * 80)

    print(f"\n총 변수 수: {len(df_summary)}")
    print("\n변화점 개수 분포:")
    print(df_summary['Optimal_Breaks'].value_counts().sort_index().to_string())

    if (df_summary['Optimal_Breaks'] > 0).sum() > 0:
        print(f"\n구조변화 감지 변수: {(df_summary['Optimal_Breaks'] > 0).sum()}/{len(df_summary)} ({(df_summary['Optimal_Breaks'] > 0).sum()/len(df_summary)*100:.1f}%)")
        print(f"p < 0.05 변수: {(df_summary['p_value'] < 0.05).sum()}/{len(df_summary)}")
        print(f"p < 0.01 변수: {(df_summary['p_value'] < 0.01).sum()}/{len(df_summary)}")

    print("\n[BIC 개선 TOP 20]")
    print(df_summary.head(20)[['Variable', 'Optimal_Breaks', 'Breakpoints', 'BIC_improvement', 'p_value']].to_string(index=False))

    return df_summary, results


# ============================================================================
# 시각화
# ============================================================================

def plot_bai_perron_summary(df_summary):
    """Bai-Perron 검정 전체 요약 시각화"""

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. 변화점 개수 분포
    ax1 = fig.add_subplot(gs[0, 0])
    breaks_dist = df_summary['Optimal_Breaks'].value_counts().sort_index()
    ax1.bar(breaks_dist.index, breaks_dist.values, color='steelblue', alpha=0.7)
    ax1.set_xlabel('변화점 개수', fontsize=12)
    ax1.set_ylabel('변수 개수', fontsize=12)
    ax1.set_title('변화점 개수 분포', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    for i, v in enumerate(breaks_dist.values):
        ax1.text(breaks_dist.index[i], v + 0.5, str(v), ha='center', fontsize=10)

    # 2. BIC 개선 TOP 30
    ax2 = fig.add_subplot(gs[0, 1])
    top30_bic = df_summary.nlargest(30, 'BIC_improvement')
    colors = ['red' if x > 0 else 'blue' for x in top30_bic['Optimal_Breaks']]
    ax2.barh(range(len(top30_bic)), top30_bic['BIC_improvement'].values, color=colors, alpha=0.6)
    ax2.set_yticks(range(len(top30_bic)))
    ax2.set_yticklabels(top30_bic['Variable'].values, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel('BIC 개선', fontsize=12)
    ax2.set_title('BIC 개선 TOP 30 (빨강: 변화점 있음, 파랑: 없음)', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    # 3. F-통계량 vs p-value (변화점 있는 변수만)
    ax3 = fig.add_subplot(gs[1, 0])
    with_breaks = df_summary[df_summary['Optimal_Breaks'] > 0].copy()

    if len(with_breaks) > 0:
        scatter = ax3.scatter(with_breaks['F_stat'], -np.log10(with_breaks['p_value']),
                             c=with_breaks['Optimal_Breaks'], cmap='viridis',
                             s=100, alpha=0.6)
        ax3.axhline(-np.log10(0.05), color='red', linestyle='--', linewidth=1, label='p=0.05')
        ax3.axhline(-np.log10(0.01), color='darkred', linestyle='--', linewidth=1, label='p=0.01')
        ax3.set_xlabel('F-통계량', fontsize=12)
        ax3.set_ylabel('-log10(p-value)', fontsize=12)
        ax3.set_title('F-통계량 vs p-value (변화점 있는 변수)', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='변화점 개수')

    # 4. 변화점 개수별 BIC 개선
    ax4 = fig.add_subplot(gs[1, 1])
    bic_by_breaks = df_summary.groupby('Optimal_Breaks')['BIC_improvement'].agg(['mean', 'median', 'std'])
    x = bic_by_breaks.index
    ax4.errorbar(x, bic_by_breaks['mean'], yerr=bic_by_breaks['std'],
                fmt='o-', capsize=5, linewidth=2, markersize=8, label='평균 ± 표준편차')
    ax4.plot(x, bic_by_breaks['median'], 's--', linewidth=2, markersize=8, label='중앙값', alpha=0.7)
    ax4.set_xlabel('변화점 개수', fontsize=12)
    ax4.set_ylabel('BIC 개선', fontsize=12)
    ax4.set_title('변화점 개수별 BIC 개선 통계', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)

    # 5. 변화점 타임라인 (TOP 20)
    ax5 = fig.add_subplot(gs[2, :])
    top20_vars = df_summary.nlargest(20, 'BIC_improvement')

    for i, (idx, row) in enumerate(top20_vars.iterrows()):
        if row['Optimal_Breaks'] > 0 and row['Breakpoints']:
            breakpoint_strs = row['Breakpoints'].split(', ')
            breakpoints = [pd.to_datetime(bp) for bp in breakpoint_strs]

            for bp in breakpoints:
                ax5.scatter(bp, i, s=100, alpha=0.7,
                          c=f'C{int(row["Optimal_Breaks"]) % 10}')

    ax5.set_yticks(range(len(top20_vars)))
    ax5.set_yticklabels(top20_vars['Variable'].values, fontsize=9)
    ax5.invert_yaxis()
    ax5.set_xlabel('날짜', fontsize=12)
    ax5.set_title('변화점 타임라인 (BIC 개선 TOP 20)', fontsize=14, fontweight='bold')
    ax5.grid(axis='x', alpha=0.3)

    plt.savefig('zscore_bai_perron_summary.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_individual_breakpoints(df_scaled, results, y_col, top_n=10):
    """개별 변수의 변화점 시각화 (TOP N)"""

    # BIC 개선 기준 TOP N 변수 선택
    sorted_results = sorted(results,
                          key=lambda x: x['bic_values'][0] - x['bic_values'][x['optimal_breaks']],
                          reverse=True)
    top_results = sorted_results[:top_n]

    n_cols = 2
    n_rows = (top_n + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten()

    for i, result in enumerate(top_results):
        ax = axes[i]
        var = result['Variable']

        # 데이터
        y = df_scaled[y_col]
        x = df_scaled[var]

        # 산점도
        ax.scatter(x, y, alpha=0.3, s=10, c='gray')

        # 변화점 표시
        if result['optimal_breaks'] > 0:
            breakpoints = [0] + result['breakpoints'] + [len(y)]
            colors = plt.cm.Set1(np.linspace(0, 1, result['optimal_breaks'] + 1))

            for j in range(len(breakpoints) - 1):
                start = breakpoints[j]
                end = breakpoints[j + 1]

                # 세그먼트별 회귀선
                X_seg = sm.add_constant(x.iloc[start:end])
                y_seg = y.iloc[start:end]

                if len(X_seg) > 2:
                    model = sm.OLS(y_seg, X_seg).fit()
                    x_sorted = np.sort(x.iloc[start:end])
                    y_pred = model.params[0] + model.params[1] * x_sorted

                    ax.plot(x_sorted, y_pred, linewidth=2, color=colors[j],
                           label=f'Regime {j+1}')

            # 변화점 날짜 표시
            title = f'{var}\n{result["optimal_breaks"]}개 변화점: {", ".join([d.strftime("%Y-%m-%d") for d in result["breakpoint_dates"]])}'
        else:
            # 단일 회귀선
            X_full = sm.add_constant(x)
            model = sm.OLS(y, X_full).fit()
            x_sorted = np.sort(x)
            y_pred = model.params[0] + model.params[1] * x_sorted
            ax.plot(x_sorted, y_pred, linewidth=2, color='blue', label='단일 체제')

            title = f'{var}\n변화점 없음'

        ax.set_xlabel(f'{var} (Z-score)', fontsize=10)
        ax.set_ylabel(f'{y_col} (Z-score)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # 빈 subplot 제거
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig('zscore_bai_perron_individual_breakpoints.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_breakpoint_distribution(results, df_scaled):
    """모든 변수의 변화점 분포 히트맵"""

    # 날짜별 변화점 빈도 계산
    all_dates = df_scaled.index
    breakpoint_counts = pd.Series(0, index=all_dates)

    for result in results:
        if result['optimal_breaks'] > 0:
            for bp_date in result['breakpoint_dates']:
                if bp_date in breakpoint_counts.index:
                    breakpoint_counts[bp_date] += 1

    # 시각화
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))

    # 1. 시계열 막대 그래프
    ax1.bar(breakpoint_counts.index, breakpoint_counts.values, width=1, alpha=0.7, color='steelblue')
    ax1.set_xlabel('날짜', fontsize=12)
    ax1.set_ylabel('변화점 개수', fontsize=12)
    ax1.set_title('날짜별 변화점 빈도', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # 주요 변화점 표시
    top_dates = breakpoint_counts.nlargest(10)
    for date, count in top_dates.items():
        ax1.axvline(date, color='red', linestyle='--', alpha=0.3, linewidth=1)
        ax1.text(date, count, f'{count}', ha='center', va='bottom', fontsize=8, color='red')

    # 2. 월별 집계
    monthly_counts = breakpoint_counts.resample('M').sum()
    ax2.bar(monthly_counts.index, monthly_counts.values, width=20, alpha=0.7, color='coral')
    ax2.set_xlabel('월', fontsize=12)
    ax2.set_ylabel('변화점 개수', fontsize=12)
    ax2.set_title('월별 변화점 빈도', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('zscore_bai_perron_breakpoint_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 주요 변화점 날짜 출력
    print("\n[주요 변화점 날짜 TOP 10]")
    for date, count in top_dates.items():
        print(f"  {date.date()}: {count}개 변수에서 변화점 감지")


# ============================================================================
# 메인 실행
# ============================================================================

if __name__ == "__main__":

    print("=" * 80)
    print("Z-Score 표준화 Bai-Perron 다중 구조변화 검정")
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

    print(f"\n종속변수: {y_col}")
    print(f"독립변수: {len(variables)}개")

    # Bai-Perron Test
    df_summary, results = run_bai_perron_all_variables(
        df_scaled, y_col, variables,
        max_breaks=5,  # 최대 5개 변화점
        trim=0.15      # 양 끝단 15% 트림
    )

    # 결과 저장
    print("\n결과 저장 중...")
    df_summary.to_csv('zscore_bai_perron_summary.csv', index=False, encoding='utf-8-sig')

    # 상세 결과 저장
    detailed_results = []
    for r in results:
        detailed_results.append({
            'Variable': r['Variable'],
            'Optimal_Breaks': r['optimal_breaks'],
            'Breakpoints_Indices': str(r['breakpoints']),
            'Breakpoints_Dates': str([d.strftime('%Y-%m-%d') for d in r['breakpoint_dates']]),
            **{f'BIC_{i}breaks': r['bic_values'][i] for i in r['bic_values'].keys()},
            **{f'SSR_{i}breaks': r['ssr_values'][i] for i in r['ssr_values'].keys()},
            'F_stat': r['f_statistics'].get('F_stat', np.nan) if r['f_statistics'] else np.nan,
            'p_value': r['f_statistics'].get('p_value', np.nan) if r['f_statistics'] else np.nan
        })

    pd.DataFrame(detailed_results).to_csv('zscore_bai_perron_detailed.csv', index=False, encoding='utf-8-sig')
    print("저장 완료!")

    # 시각화
    print("\n시각화 생성 중...")
    plot_bai_perron_summary(df_summary)
    plot_individual_breakpoints(df_scaled, results, y_col, top_n=12)
    plot_breakpoint_distribution(results, df_scaled)
    print("시각화 완료!")

    # 최종 요약
    print("\n" + "=" * 80)
    print("최종 요약")
    print("=" * 80)

    print(f"\n총 변수 수: {len(df_summary)}")
    print(f"구조변화 감지 변수: {(df_summary['Optimal_Breaks'] > 0).sum()}/{len(df_summary)} ({(df_summary['Optimal_Breaks'] > 0).sum()/len(df_summary)*100:.1f}%)")
    print(f"유의한 변수 (p<0.05): {(df_summary['p_value'] < 0.05).sum()}/{len(df_summary)}")

    print("\n생성된 파일:")
    print("  - zscore_bai_perron_summary.csv")
    print("  - zscore_bai_perron_detailed.csv")
    print("  - zscore_bai_perron_summary.png")
    print("  - zscore_bai_perron_individual_breakpoints.png")
    print("  - zscore_bai_perron_breakpoint_distribution.png")
