import yfinance as yf
import pandas as pd

print("=" * 70)
print("Step 1.2: 전통 시장 지수 수집")
print("=" * 70)

# 수집 기간
start_date = "2021-01-01"
end_date = "2025-10-15"

print(f"\n수집 기간: {start_date} ~ {end_date}")

# 전통 시장 지수 티커
tickers = {
    'QQQ': 'Nasdaq-100 ETF',
    '^GSPC': 'S&P 500',
    'UUP': 'USD Index (Dollar Index)',
    'EURUSD=X': 'EUR/USD 환율',
    'GC=F': '금 선물',
    'SI=F': '은 선물',
    'CL=F': 'WTI 원유 선물',
    'BSV': 'Vanguard 단기채권 ETF',
}

print(f"\n수집할 지수: {len(tickers)}개")
print("-" * 70)

traditional_data = {}
success_count = 0
fail_count = 0

for ticker, name in tickers.items():
    try:
        print(f"다운로드 중: {ticker:12} - {name}")
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if len(data) > 0:
            # 종가 데이터만 사용 (squeeze로 Series 변환)
            close_data = data['Close'].squeeze()
            traditional_data[ticker] = close_data
            success_count += 1
            print(f"  ✓ 성공: {len(data)}개 데이터")
        else:
            print(f"  ✗ 실패: 데이터 없음")
            fail_count += 1
    except Exception as e:
        print(f"  ✗ 오류: {e}")
        fail_count += 1

print("\n" + "=" * 70)
print("다운로드 완료")
print("=" * 70)
print(f"성공: {success_count}개")
print(f"실패: {fail_count}개")

# DataFrame으로 통합
traditional_df = pd.DataFrame(traditional_data)

# 컬럼명 정리 (심볼 특수문자 제거)
column_mapping = {
    '^GSPC': 'SPX',
    'EURUSD=X': 'EURUSD',
    'GC=F': 'GOLD',
    'SI=F': 'SILVER',
    'CL=F': 'OIL'
}
traditional_df = traditional_df.rename(columns=column_mapping)

print(f"\n통합 데이터 형태: {traditional_df.shape}")
print(f"기간: {traditional_df.index[0]} ~ {traditional_df.index[-1]}")

# 결측치 확인
null_counts = traditional_df.isnull().sum()
print(f"\n결측치:")
for col in traditional_df.columns:
    if null_counts[col] > 0:
        print(f"  {col}: {null_counts[col]}개")

# Forward fill (주말/공휴일 데이터)
traditional_df = traditional_df.fillna(method='ffill')

# 남은 결측치 제거
traditional_df = traditional_df.dropna()

print(f"\n결측치 처리 후: {traditional_df.shape}")

# 파일 저장
traditional_df.to_csv('traditional_market_indices.csv')
print("\n✓ 저장 완료: traditional_market_indices.csv")

print("\n" + "=" * 70)
print("수집된 지수 요약")
print("=" * 70)

# 각 지수 기본 통계
for col in traditional_df.columns:
    print(f"\n{col}:")
    print(f"  평균: ${traditional_df[col].mean():,.2f}")
    print(f"  최소: ${traditional_df[col].min():,.2f}")
    print(f"  최대: ${traditional_df[col].max():,.2f}")
    print(f"  표준편차: ${traditional_df[col].std():,.2f}")

print("\n" + "=" * 70)
print("Step 1.2 완료!")
print("=" * 70)
