import pandas as pd
import numpy as np
from datetime import datetime

# 데이터 로드
print("데이터 로딩 중...\n")
close_prices = pd.read_csv('crypto_close_prices_2021_2025.csv', index_col=0, parse_dates=True)
volumes = pd.read_csv('crypto_volumes_2021_2025.csv', index_col=0, parse_dates=True)

# ETF 승인일: 2024년 1월 10일
etf_date = pd.Timestamp('2024-01-10', tz='UTC')

# 기간 분할
pre_etf = close_prices[close_prices.index < etf_date]
post_etf = close_prices[close_prices.index >= etf_date]

pre_etf_vol = volumes[volumes.index < etf_date]
post_etf_vol = volumes[volumes.index >= etf_date]

print("=" * 70)
print("BTC ETF 승인 전후 비교 분석 (2024년 1월 10일 기준)")
print("=" * 70)
print(f"\nETF 이전 기간: {pre_etf.index[0].strftime('%Y-%m-%d')} ~ {pre_etf.index[-1].strftime('%Y-%m-%d')} ({len(pre_etf)}일)")
print(f"ETF 이후 기간: {post_etf.index[0].strftime('%Y-%m-%d')} ~ {post_etf.index[-1].strftime('%Y-%m-%d')} ({len(post_etf)}일)")

# 1. 평균 가격 비교
print("\n" + "=" * 70)
print("1. 평균 가격 비교")
print("=" * 70)
print(f"{'코인':<8} {'ETF 이전':>15} {'ETF 이후':>15} {'변화율':>12}")
print("-" * 70)

for col in close_prices.columns:
    coin = col.replace('_Close', '')
    pre_mean = pre_etf[col].mean()
    post_mean = post_etf[col].mean()
    change = ((post_mean - pre_mean) / pre_mean) * 100
    print(f"{coin:<8} ${pre_mean:>14,.2f} ${post_mean:>14,.2f} {change:>11.1f}%")

# 2. 일별 변동성 (표준편차) 비교
print("\n" + "=" * 70)
print("2. 일별 변동성 비교 (일별 수익률의 표준편차, %)")
print("=" * 70)

returns_pre = pre_etf.pct_change().dropna() * 100
returns_post = post_etf.pct_change().dropna() * 100

print(f"{'코인':<8} {'ETF 이전':>15} {'ETF 이후':>15} {'변화':>12}")
print("-" * 70)

for col in close_prices.columns:
    coin = col.replace('_Close', '')
    pre_vol = returns_pre[col].std()
    post_vol = returns_post[col].std()
    change = post_vol - pre_vol
    print(f"{coin:<8} {pre_vol:>14.2f}% {post_vol:>14.2f}% {change:>11.2f}%p")

# 3. 최대 상승/하락 비교
print("\n" + "=" * 70)
print("3. 최대 일간 수익률 비교 (%)")
print("=" * 70)

print(f"{'코인':<8} {'기간':<8} {'최대 상승':>12} {'최대 하락':>12}")
print("-" * 70)

for col in close_prices.columns:
    coin = col.replace('_Close', '')
    pre_max = returns_pre[col].max()
    pre_min = returns_pre[col].min()
    post_max = returns_post[col].max()
    post_min = returns_post[col].min()

    print(f"{coin:<8} {'이전':<8} {pre_max:>11.2f}% {pre_min:>11.2f}%")
    print(f"{coin:<8} {'이후':<8} {post_max:>11.2f}% {post_min:>11.2f}%")
    print("-" * 70)

# 4. 거래량 변화
print("\n" + "=" * 70)
print("4. 평균 거래량 비교")
print("=" * 70)
print(f"{'코인':<8} {'ETF 이전':>20} {'ETF 이후':>20} {'변화율':>12}")
print("-" * 70)

for col in volumes.columns:
    coin = col.replace('_Volume', '')
    pre_vol_mean = pre_etf_vol[col].mean()
    post_vol_mean = post_etf_vol[col].mean()
    change = ((post_vol_mean - pre_vol_mean) / pre_vol_mean) * 100
    print(f"{coin:<8} {pre_vol_mean:>20,.0f} {post_vol_mean:>20,.0f} {change:>11.1f}%")

# 5. 상관관계 변화
print("\n" + "=" * 70)
print("5. 암호화폐 간 상관관계 변화")
print("=" * 70)

corr_pre = pre_etf.corr()
corr_post = post_etf.corr()

print("\n[ETF 이전 상관관계]")
print(corr_pre.round(3))

print("\n[ETF 이후 상관관계]")
print(corr_post.round(3))

print("\n[상관관계 변화 (이후 - 이전)]")
corr_diff = corr_post - corr_pre
print(corr_diff.round(3))

# 6. 샤프 비율 비교 (간단 버전, 무위험 수익률 = 0 가정)
print("\n" + "=" * 70)
print("6. 샤프 비율 비교 (연율화, 무위험 수익률 = 0)")
print("=" * 70)
print(f"{'코인':<8} {'ETF 이전':>15} {'ETF 이후':>15} {'변화':>12}")
print("-" * 70)

for col in close_prices.columns:
    coin = col.replace('_Close', '')

    # 연평균 수익률 / 연변동성
    pre_return = returns_pre[col].mean() * 252  # 연율화
    pre_std = returns_pre[col].std() * np.sqrt(252)  # 연율화
    pre_sharpe = pre_return / pre_std if pre_std != 0 else 0

    post_return = returns_post[col].mean() * 252
    post_std = returns_post[col].std() * np.sqrt(252)
    post_sharpe = post_return / post_std if post_std != 0 else 0

    change = post_sharpe - pre_sharpe

    print(f"{coin:<8} {pre_sharpe:>15.3f} {post_sharpe:>15.3f} {change:>11.3f}")

# 7. 누적 수익률
print("\n" + "=" * 70)
print("7. 누적 수익률 비교")
print("=" * 70)
print(f"{'코인':<8} {'ETF 이전':>15} {'ETF 이후':>15}")
print("-" * 70)

for col in close_prices.columns:
    coin = col.replace('_Close', '')

    pre_cumulative = ((pre_etf[col].iloc[-1] / pre_etf[col].iloc[0]) - 1) * 100
    post_cumulative = ((post_etf[col].iloc[-1] / post_etf[col].iloc[0]) - 1) * 100

    print(f"{coin:<8} {pre_cumulative:>14.1f}% {post_cumulative:>14.1f}%")

# 결과를 CSV로 저장
print("\n" + "=" * 70)
print("분석 결과 저장 중...")
print("=" * 70)

# 요약 데이터프레임 생성
summary_data = []
for col in close_prices.columns:
    coin = col.replace('_Close', '')

    summary_data.append({
        'Coin': coin,
        'Pre_ETF_Avg_Price': pre_etf[col].mean(),
        'Post_ETF_Avg_Price': post_etf[col].mean(),
        'Price_Change_%': ((post_etf[col].mean() - pre_etf[col].mean()) / pre_etf[col].mean()) * 100,
        'Pre_ETF_Volatility_%': returns_pre[col].std(),
        'Post_ETF_Volatility_%': returns_post[col].std(),
        'Pre_ETF_Sharpe': (returns_pre[col].mean() * 252) / (returns_pre[col].std() * np.sqrt(252)),
        'Post_ETF_Sharpe': (returns_post[col].mean() * 252) / (returns_post[col].std() * np.sqrt(252)),
        'Pre_ETF_Cumulative_%': ((pre_etf[col].iloc[-1] / pre_etf[col].iloc[0]) - 1) * 100,
        'Post_ETF_Cumulative_%': ((post_etf[col].iloc[-1] / post_etf[col].iloc[0]) - 1) * 100,
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('etf_impact_summary.csv', index=False)
print("✓ 요약 저장: etf_impact_summary.csv")

# 상관관계 변화 저장
corr_diff.to_csv('correlation_change_etf.csv')
print("✓ 상관관계 변화 저장: correlation_change_etf.csv")

print("\n분석 완료!")
