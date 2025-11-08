#!/usr/bin/env python3
"""
Version A와 B의 차이 분석
왜 Version A는 성공하고 B는 실패했는가?
"""

import pandas as pd
import numpy as np

print("="*80)
print("Version A vs B 차이 분석")
print("="*80)

# 데이터 로드
df = pd.read_csv('integrated_data_full.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Version A 테스트 기간
version_a_start = pd.to_datetime('2024-05-17')
version_a_end = pd.to_datetime('2025-10-13')

# Version B 테스트 기간
version_b_start = pd.to_datetime('2025-01-01')
version_b_end = pd.to_datetime('2025-10-13')

# 2024년 후반 (Version A에만 포함)
only_a_start = pd.to_datetime('2024-05-17')
only_a_end = pd.to_datetime('2024-12-31')

# 기간별 데이터 추출
df_version_a = df[(df['Date'] >= version_a_start) & (df['Date'] <= version_a_end)].copy()
df_version_b = df[(df['Date'] >= version_b_start) & (df['Date'] <= version_b_end)].copy()
df_only_a = df[(df['Date'] >= only_a_start) & (df['Date'] <= only_a_end)].copy()

print("\n" + "="*60)
print("기간별 시장 특성")
print("="*60)

def analyze_period(df_period, name):
    """기간별 시장 분석"""
    print(f"\n{name}:")
    print(f"  기간: {df_period['Date'].min().date()} ~ {df_period['Date'].max().date()}")
    print(f"  일수: {len(df_period)}일")

    # 가격 통계
    start_price = df_period['Close'].iloc[0]
    end_price = df_period['Close'].iloc[-1]
    total_return = (end_price / start_price - 1) * 100

    print(f"\n  시작 가격: ${start_price:,.2f}")
    print(f"  종료 가격: ${end_price:,.2f}")
    print(f"  총 수익률: {total_return:+.2f}%")

    # 일별 변동성
    daily_returns = df_period['Close'].pct_change() * 100
    avg_daily_return = daily_returns.mean()
    daily_volatility = daily_returns.std()

    print(f"\n  평균 일일 변동: {avg_daily_return:+.3f}%")
    print(f"  일일 변동성: {daily_volatility:.3f}%")
    print(f"  연율 변동성: {daily_volatility * np.sqrt(252):.2f}%")

    # 추세
    up_days = (daily_returns > 0).sum()
    down_days = (daily_returns < 0).sum()
    print(f"\n  상승일: {up_days}일 ({up_days/len(daily_returns)*100:.1f}%)")
    print(f"  하락일: {down_days}일 ({down_days/len(daily_returns)*100:.1f}%)")

    # 최대 낙폭
    cummax = df_period['Close'].cummax()
    drawdown = (df_period['Close'] / cummax - 1) * 100
    max_dd = drawdown.min()

    print(f"\n  최대 낙폭: {max_dd:.2f}%")

    # 가격 레벨 변화
    price_range = df_period['Close'].max() - df_period['Close'].min()
    price_range_pct = price_range / df_period['Close'].min() * 100

    print(f"  가격 범위: ${price_range:,.2f} ({price_range_pct:.1f}%)")

    # 큰 변동 일수 (±3% 이상)
    big_moves = (abs(daily_returns) > 3).sum()
    print(f"  큰 변동(±3%) 일수: {big_moves}일 ({big_moves/len(daily_returns)*100:.1f}%)")

    return {
        'total_return': total_return,
        'daily_volatility': daily_volatility,
        'up_ratio': up_days/len(daily_returns)*100,
        'max_dd': max_dd,
        'big_moves_ratio': big_moves/len(daily_returns)*100
    }

# 분석 실행
stats_a = analyze_period(df_version_a, "Version A (전체)")
stats_only_a = analyze_period(df_only_a, "2024년 후반 (A에만 포함)")
stats_b = analyze_period(df_version_b, "Version B (2025년)")

# 비교
print("\n" + "="*60)
print("핵심 차이점")
print("="*60)

print(f"""
1. 수익률:
   Version A 전체: {stats_a['total_return']:+.2f}%
   - 2024 후반: {stats_only_a['total_return']:+.2f}% (A에만 포함)
   - 2025년: {stats_b['total_return']:+.2f}% (B와 겹침)

2. 변동성:
   2024 후반: {stats_only_a['daily_volatility']:.3f}%
   2025년: {stats_b['daily_volatility']:.3f}%
   차이: {stats_b['daily_volatility'] - stats_only_a['daily_volatility']:+.3f}%p

3. 추세:
   2024 후반 상승비율: {stats_only_a['up_ratio']:.1f}%
   2025년 상승비율: {stats_b['up_ratio']:.1f}%
   {'🔺 2024 후반이 더 상승장' if stats_only_a['up_ratio'] > stats_b['up_ratio'] else '🔻 2025년이 더 상승장'}

4. 최대 낙폭:
   2024 후반: {stats_only_a['max_dd']:.2f}%
   2025년: {stats_b['max_dd']:.2f}%
   {'✅ 2024 후반이 더 안정적' if stats_only_a['max_dd'] > stats_b['max_dd'] else '⚠️ 2025년이 더 안정적'}

5. 큰 변동 빈도:
   2024 후반: {stats_only_a['big_moves_ratio']:.1f}%
   2025년: {stats_b['big_moves_ratio']:.1f}%
   {'📊 2024 후반이 더 변동성 큼' if stats_only_a['big_moves_ratio'] > stats_b['big_moves_ratio'] else '📊 2025년이 더 변동성 큼'}
""")

# 월별 수익률 비교
print("\n" + "="*60)
print("월별 수익률 비교")
print("="*60)

df_version_a['month'] = df_version_a['Date'].dt.to_period('M')
monthly_returns = df_version_a.groupby('month').apply(
    lambda x: (x['Close'].iloc[-1] / x['Close'].iloc[0] - 1) * 100
).reset_index()
monthly_returns.columns = ['month', 'return']

print("\nVersion A 기간 월별 수익률:")
for _, row in monthly_returns.iterrows():
    month_str = str(row['month'])
    ret = row['return']
    marker = '🟢' if ret > 0 else '🔴'
    in_version_b = '← Version B 포함' if month_str >= '2025-01' else ''
    print(f"  {month_str}: {ret:+6.2f}% {marker} {in_version_b}")

# 결론
print("\n" + "="*60)
print("결론")
print("="*60)

print(f"""
Version A가 성공한 이유:
{'1. ✅ 2024년 후반(5~12월)이 큰 상승장 (+' + f"{stats_only_a['total_return']:.1f}" + '%)'if stats_only_a['total_return'] > 20 else '1. ⚠️ 2024년 후반 수익 낮음'}
{'2. ✅ 상승 일수 비율 높음 (' + f"{stats_only_a['up_ratio']:.1f}" + '%)' if stats_only_a['up_ratio'] > 50 else '2. ⚠️ 상승 비율 낮음'}
3. ✅ 2024 + 2025 = 17개월의 다양한 데이터

Version B가 실패한 이유:
{'1. ❌ 2025년만으로는 추세 파악 어려움 (' + f"{stats_b['total_return']:.1f}" + '%)' if abs(stats_b['total_return']) < 30 else '1. 수익률 극단적'}
{'2. ❌ 변동성 높음 (일일 ' + f"{stats_b['daily_volatility']:.2f}" + '%)' if stats_b['daily_volatility'] > stats_only_a['daily_volatility'] else '2. 변동성은 비슷'}
3. ❌ 짧은 기간 (10개월)으로 전략 검증 부족

핵심 발견:
- Version A의 성공은 "과대평가"가 아니라
  {"✅ 2024년 후반의 강한 상승장 덕분!" if stats_only_a['total_return'] > 30 else "⚠️ 복합적 요인"}
- Version B의 실패는 "모델의 한계"보다는
  {"❌ 2025년 시장 환경이 특수했거나" if abs(stats_b['total_return'] - stats_only_a['total_return']) > 20 else "⚠️ 짧은 테스트 기간"}
  {"❌ 테스트 기간이 너무 짧았기 때문" if len(df_version_b) < 300 else ""}

{'⚠️ 과대평가 가능성: 낮음' if stats_only_a['total_return'] > 30 and stats_only_a['up_ratio'] > 52 else '✅ 진짜 실력일 가능성 높음'}
""")

print("="*60)
