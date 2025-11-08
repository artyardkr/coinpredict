import yfinance as yf
import pandas as pd
from datetime import datetime

# BTC-USD 데이터 가져오기
print("BTC 데이터를 가져오는 중...")

btc = yf.Ticker("BTC-USD")

# 2021년 1월 1일부터 2025년 10월 15일까지의 데이터
start_date = "2021-01-01"
end_date = "2025-10-15"

# 데이터 다운로드
df = btc.history(start=start_date, end=end_date)

# 데이터 정보 출력
print(f"\n데이터 기간: {start_date} ~ {end_date}")
print(f"전체 데이터 개수: {len(df)}개")
print(f"\n첫 5개 데이터:")
print(df.head())
print(f"\n마지막 5개 데이터:")
print(df.tail())

# 기본 통계
print("\n기본 통계:")
print(df.describe())

# CSV 파일로 저장
csv_filename = "btc_data_2021_2025.csv"
df.to_csv(csv_filename)
print(f"\n데이터가 '{csv_filename}' 파일로 저장되었습니다.")

# 컬럼 정보
print(f"\n컬럼: {list(df.columns)}")
