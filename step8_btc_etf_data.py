import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

print("=" * 70)
print("Step 8: Bitcoin ETF 데이터 수집")
print("=" * 70)

# 데이터 기간 설정
end_date = datetime.now() - timedelta(days=1)  # DST 문제 회피
start_date = datetime(2024, 1, 1)  # Bitcoin Spot ETF는 2024년 1월부터

print(f"\n데이터 기간: {start_date.date()} ~ {end_date.date()}")
print("(Bitcoin Spot ETF는 2024년 1월 출시)")
print("-" * 70)

# Bitcoin ETF 티커
etf_tickers = {
    'IBIT': 'BlackRock iShares Bitcoin Trust',
    'FBTC': 'Fidelity Wise Origin Bitcoin Fund',
    'GBTC': 'Grayscale Bitcoin Trust',
    'ARKB': 'ARK 21Shares Bitcoin ETF',
    'BITB': 'Bitwise Bitcoin ETF',
}

# BTC 가격도 수집 (Premium 계산용)
print("\nBTC-USD 가격 수집 중...")
btc_data = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
if len(btc_data) > 0:
    # Series로 변환
    if isinstance(btc_data['Close'], pd.DataFrame):
        btc_price = btc_data['Close'].squeeze()
    else:
        btc_price = btc_data['Close']
    print(f"  ✓ BTC 가격: {len(btc_price)}개 데이터")
else:
    print("  ✗ BTC 가격 수집 실패")
    btc_price = None

# ETF 데이터 수집
etf_data = pd.DataFrame()
etf_volume = pd.DataFrame()

for ticker, description in etf_tickers.items():
    print(f"\n{ticker:6} - {description}")
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if len(data) > 0:
            # Close 가격과 거래량
            etf_data[f'{ticker}_Price'] = data['Close']
            etf_volume[f'{ticker}_Volume'] = data['Volume']
            print(f"  ✓ 성공: {len(data)}개 데이터")
        else:
            print(f"  ✗ 실패: 데이터 없음")

    except Exception as e:
        print(f"  ✗ 에러: {e}")

# ===== GBTC Premium/Discount 계산 =====
if 'GBTC_Price' in etf_data.columns and btc_price is not None:
    print("\n" + "=" * 70)
    print("GBTC Premium/Discount 계산")
    print("=" * 70)

    # GBTC는 약 0.00092 BTC per share (이 비율은 변할 수 있으니 확인 필요)
    # 정확한 비율은 Grayscale 웹사이트 참조
    GBTC_BTC_PER_SHARE = 0.00092  # 근사치

    # 날짜 정렬 및 일치
    gbtc_aligned = etf_data['GBTC_Price'].reindex(btc_price.index)
    implied_btc_price = gbtc_aligned / GBTC_BTC_PER_SHARE
    actual_btc_price = btc_price

    # Premium/Discount 계산
    etf_data['GBTC_Premium'] = ((implied_btc_price - actual_btc_price) / actual_btc_price * 100).fillna(0)

    print(f"✓ GBTC Premium 계산 완료")
    print(f"  평균 Premium: {etf_data['GBTC_Premium'].mean():.2f}%")
    print(f"  최대 Premium: {etf_data['GBTC_Premium'].max():.2f}%")
    print(f"  최소 Premium: {etf_data['GBTC_Premium'].min():.2f}%")
    print(f"  최근 Premium: {etf_data['GBTC_Premium'].iloc[-1]:.2f}%")

# ===== 총 ETF 거래량 계산 =====
if len(etf_volume.columns) > 0:
    etf_data['Total_BTC_ETF_Volume'] = etf_volume.sum(axis=1)
    print(f"\n✓ 총 BTC ETF 거래량 계산 완료")

# ===== 거래량 변화율 =====
for col in etf_volume.columns:
    ticker = col.replace('_Volume', '')
    etf_data[f'{ticker}_Volume_Change_7d'] = etf_volume[col].pct_change(7) * 100

# 데이터 정보
print("\n" + "=" * 70)
print("수집 완료")
print("=" * 70)
print(f"총 ETF: {len(etf_tickers)}개")
print(f"총 변수: {len(etf_data.columns)}개")
print(f"총 행: {len(etf_data):,}개")
if len(etf_data) > 0:
    print(f"기간: {etf_data.index[0].date()} ~ {etf_data.index[-1].date()}")

# 결측치 확인
print("\n결측치 확인 (상위 10개):")
null_counts = etf_data.isnull().sum().sort_values(ascending=False).head(10)
for col, count in null_counts.items():
    pct = (count / len(etf_data) * 100)
    print(f"  {col:30} : {count:4}개 ({pct:5.2f}%)")

# 파일 저장
etf_data.to_csv('bitcoin_etf_data.csv')
print("\n✓ 저장 완료: bitcoin_etf_data.csv")

# 기본 통계
if len(etf_data.columns) > 0:
    print("\n" + "=" * 70)
    print("최근 ETF 가격 (최신 데이터)")
    print("=" * 70)
    price_cols = [c for c in etf_data.columns if '_Price' in c]
    if price_cols and len(etf_data) > 0:
        latest = etf_data[price_cols].iloc[-1]
        print(latest)

    print("\n" + "=" * 70)
    print("최근 7일 거래량 변화")
    print("=" * 70)
    volume_change_cols = [c for c in etf_data.columns if 'Volume_Change_7d' in c]
    if volume_change_cols and len(etf_data) > 7:
        latest_changes = etf_data[volume_change_cols].iloc[-1]
        print(latest_changes)

print("\n" + "=" * 70)
print("Step 8 완료!")
print("=" * 70)
print(f"\n✅ {len(etf_tickers)}개 Bitcoin ETF 데이터 수집 완료")
print("✅ GBTC Premium 계산 완료")
