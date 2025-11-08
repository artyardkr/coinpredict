import yfinance as yf
import pandas as pd
from datetime import datetime

# 수집할 암호화폐 목록
cryptos = {
    'BTC': 'BTC-USD',
    'ETH': 'ETH-USD',
    'SOL': 'SOL-USD',
    'DOGE': 'DOGE-USD',
    'XRP': 'XRP-USD'
}

# 날짜 범위
start_date = "2021-01-01"
end_date = "2025-10-15"

print(f"암호화폐 데이터 수집 중... ({start_date} ~ {end_date})\n")

# 각 암호화폐별 데이터 저장
all_data = {}

for name, ticker in cryptos.items():
    print(f"{name} 데이터 수집 중...")
    try:
        crypto = yf.Ticker(ticker)
        df = crypto.history(start=start_date, end=end_date)

        if len(df) > 0:
            all_data[name] = df

            # 개별 CSV 저장
            filename = f"{name.lower()}_data_2021_2025.csv"
            df.to_csv(filename)

            print(f"  ✓ {name}: {len(df)}개 데이터 수집 완료")
            print(f"    첫 거래일: {df.index[0].strftime('%Y-%m-%d')}")
            print(f"    마지막 거래일: {df.index[-1].strftime('%Y-%m-%d')}")
            print(f"    최저가: ${df['Low'].min():,.2f}")
            print(f"    최고가: ${df['High'].max():,.2f}")
        else:
            print(f"  ✗ {name}: 데이터를 가져올 수 없습니다")
    except Exception as e:
        print(f"  ✗ {name} 오류: {e}")
    print()

# 종가만 모아서 하나의 DataFrame으로 통합
if all_data:
    print("\n=== 통합 데이터 생성 중 ===")
    close_prices = pd.DataFrame()
    volumes = pd.DataFrame()

    for name, df in all_data.items():
        close_prices[f'{name}_Close'] = df['Close']
        volumes[f'{name}_Volume'] = df['Volume']

    # 통합 CSV 저장
    close_prices.to_csv('crypto_close_prices_2021_2025.csv')
    volumes.to_csv('crypto_volumes_2021_2025.csv')

    print(f"✓ 통합 종가 데이터 저장: crypto_close_prices_2021_2025.csv")
    print(f"✓ 통합 거래량 데이터 저장: crypto_volumes_2021_2025.csv")

    # 상관관계 분석
    print("\n=== 암호화폐 간 종가 상관관계 (전체 기간) ===")
    correlation = close_prices.corr()
    print(correlation.round(3))
    correlation.to_csv('crypto_correlation.csv')
    print("\n✓ 상관관계 저장: crypto_correlation.csv")

    # 기본 통계
    print("\n=== 기본 통계 ===")
    print(close_prices.describe())

    print(f"\n총 {len(close_prices)}일 데이터 수집 완료!")
    print(f"\n생성된 파일:")
    print("  개별 데이터: btc_data_2021_2025.csv, eth_data_2021_2025.csv, ...")
    print("  통합 종가: crypto_close_prices_2021_2025.csv")
    print("  통합 거래량: crypto_volumes_2021_2025.csv")
    print("  상관관계: crypto_correlation.csv")
