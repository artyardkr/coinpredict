import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

print("=" * 70)
print("Step 2b: 추가 전통시장 데이터 수집")
print("=" * 70)

# 데이터 기간 설정 (BTC 데이터와 동일하게)
end_date = datetime.now() - timedelta(days=1)  # DST 문제 회피
start_date = datetime(2021, 1, 1)

print(f"\n데이터 기간: {start_date.date()} ~ {end_date.date()}")
print("-" * 70)

# 추가할 시장 지수 및 자산
tickers = {
    'DIA': '다우존스 산업평균 (Dow Jones Industrial Average)',
    'IWM': '러셀 2000 소형주 (Russell 2000)',
    'TLT': '20년 장기 국채 ETF (20+ Year Treasury Bond)',
    'DX-Y.NYB': '달러 지수 (US Dollar Index)',
    'ETH-USD': '이더리움 (Ethereum)',
    'GLD': '금 ETF (Gold ETF)',
    'HYG': '하이일드 회사채 ETF (High Yield Corporate Bond)',
    'LQD': '투자등급 회사채 ETF (Investment Grade Corporate Bond)',
    '^VIX': 'VIX 지수 (Volatility Index)',
}

# 데이터 수집
market_data = pd.DataFrame()

for ticker, description in tickers.items():
    print(f"\n{ticker:12} - {description}")
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if len(data) > 0:
            # Close 가격만 사용
            if ticker == 'DX-Y.NYB':
                # 달러 지수는 DXY로 저장
                market_data['DXY'] = data['Close']
                print(f"  ✓ 성공: {len(data)}개 데이터 (DXY로 저장)")
            elif ticker == 'ETH-USD':
                # 이더리움은 ETH로 저장
                market_data['ETH'] = data['Close']
                print(f"  ✓ 성공: {len(data)}개 데이터 (ETH로 저장)")
            elif ticker == '^VIX':
                # VIX는 VIX로 저장
                market_data['VIX'] = data['Close']
                print(f"  ✓ 성공: {len(data)}개 데이터 (VIX로 저장)")
            else:
                market_data[ticker] = data['Close']
                print(f"  ✓ 성공: {len(data)}개 데이터")
        else:
            print(f"  ✗ 실패: 데이터 없음")

    except Exception as e:
        print(f"  ✗ 에러: {e}")

# 데이터 정보
print("\n" + "=" * 70)
print("수집 완료")
print("=" * 70)
print(f"총 변수: {len(market_data.columns)}개")
print(f"총 행: {len(market_data):,}개")
print(f"기간: {market_data.index[0].date()} ~ {market_data.index[-1].date()}")

# 결측치 확인
print("\n결측치 확인:")
null_counts = market_data.isnull().sum()
for col in market_data.columns:
    null_count = null_counts[col]
    null_pct = (null_count / len(market_data) * 100)
    print(f"  {col:12} : {null_count:4}개 ({null_pct:5.2f}%)")

# 파일 저장
market_data.to_csv('additional_market_data.csv')
print("\n✓ 저장 완료: additional_market_data.csv")

# 기본 통계
print("\n" + "=" * 70)
print("기본 통계")
print("=" * 70)
print(market_data.describe())

print("\n" + "=" * 70)
print("Step 2b 완료!")
print("=" * 70)
