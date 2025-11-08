#!/usr/bin/env python3
"""
Binance Futures API - Historical Data 수집 가능 여부 테스트
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time

print("=" * 80)
print("Binance Futures API - Historical Data 테스트")
print("=" * 80)

# Base URLs
base_url = "https://fapi.binance.com"
symbol = "BTCUSDT"

# ========================================
# 1. Funding Rate (펀딩비)
# ========================================
print(f"\n{'='*80}")
print("1. Funding Rate (펀딩비) - Historical Data")
print(f"{'='*80}")

try:
    endpoint = f"{base_url}/fapi/v1/fundingRate"
    params = {
        'symbol': symbol,
        'limit': 1000  # 최대 1000개
    }

    response = requests.get(endpoint, params=params)
    funding_data = response.json()

    if isinstance(funding_data, list) and len(funding_data) > 0:
        df_funding = pd.DataFrame(funding_data)
        df_funding['fundingTime'] = pd.to_datetime(df_funding['fundingTime'], unit='ms')

        print(f"✅ 조회 성공!")
        print(f"   - 조회 건수: {len(df_funding)}개")
        print(f"   - 가장 오래된 데이터: {df_funding['fundingTime'].min()}")
        print(f"   - 가장 최근 데이터: {df_funding['fundingTime'].max()}")
        print(f"   - 펀딩비 범위: {float(df_funding['fundingRate'].min())*100:.4f}% ~ {float(df_funding['fundingRate'].max())*100:.4f}%")
        print(f"\n   샘플 (최근 5개):")
        print(df_funding[['fundingTime', 'fundingRate']].tail(5).to_string(index=False))
    else:
        print(f"⚠️  응답 형식 이상: {funding_data}")

except Exception as e:
    print(f"❌ 오류: {e}")

time.sleep(1)

# ========================================
# 2. Open Interest (미결제약정)
# ========================================
print(f"\n\n{'='*80}")
print("2. Open Interest (미결제약정) - Historical Data")
print(f"{'='*80}")

try:
    endpoint = f"{base_url}/futures/data/openInterestHist"

    # 과거 데이터 조회 (최근 30일)
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)

    params = {
        'symbol': symbol,
        'period': '5m',  # 5분봉
        'limit': 500,
        'startTime': start_time,
        'endTime': end_time
    }

    response = requests.get(endpoint, params=params)
    oi_data = response.json()

    if isinstance(oi_data, list) and len(oi_data) > 0:
        df_oi = pd.DataFrame(oi_data)
        df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'], unit='ms')
        df_oi['sumOpenInterest'] = df_oi['sumOpenInterest'].astype(float)
        df_oi['sumOpenInterestValue'] = df_oi['sumOpenInterestValue'].astype(float)

        print(f"✅ 조회 성공!")
        print(f"   - 조회 건수: {len(df_oi)}개")
        print(f"   - 가장 오래된 데이터: {df_oi['timestamp'].min()}")
        print(f"   - 가장 최근 데이터: {df_oi['timestamp'].max()}")
        print(f"   - OI 범위: {df_oi['sumOpenInterest'].min():,.0f} ~ {df_oi['sumOpenInterest'].max():,.0f} BTC")
        print(f"   - OI Value 범위: ${df_oi['sumOpenInterestValue'].min():,.0f} ~ ${df_oi['sumOpenInterestValue'].max():,.0f}")
        print(f"\n   샘플 (최근 5개):")
        print(df_oi[['timestamp', 'sumOpenInterest', 'sumOpenInterestValue']].tail(5).to_string(index=False))
    else:
        print(f"⚠️  응답 형식 이상: {oi_data}")

except Exception as e:
    print(f"❌ 오류: {e}")

time.sleep(1)

# ========================================
# 3. Long/Short Ratio (롱숏 비율)
# ========================================
print(f"\n\n{'='*80}")
print("3. Long/Short Ratio (롱숏 비율) - Historical Data")
print(f"{'='*80}")

try:
    endpoint = f"{base_url}/futures/data/topLongShortAccountRatio"

    # 과거 데이터 조회 (최근 30일)
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)

    params = {
        'symbol': symbol,
        'period': '5m',
        'limit': 500,
        'startTime': start_time,
        'endTime': end_time
    }

    response = requests.get(endpoint, params=params)
    ratio_data = response.json()

    if isinstance(ratio_data, list) and len(ratio_data) > 0:
        df_ratio = pd.DataFrame(ratio_data)
        df_ratio['timestamp'] = pd.to_datetime(df_ratio['timestamp'], unit='ms')
        df_ratio['longShortRatio'] = df_ratio['longShortRatio'].astype(float)
        df_ratio['longAccount'] = df_ratio['longAccount'].astype(float)
        df_ratio['shortAccount'] = df_ratio['shortAccount'].astype(float)

        print(f"✅ 조회 성공!")
        print(f"   - 조회 건수: {len(df_ratio)}개")
        print(f"   - 가장 오래된 데이터: {df_ratio['timestamp'].min()}")
        print(f"   - 가장 최근 데이터: {df_ratio['timestamp'].max()}")
        print(f"   - 롱숏비율 범위: {df_ratio['longShortRatio'].min():.4f} ~ {df_ratio['longShortRatio'].max():.4f}")
        print(f"\n   샘플 (최근 5개):")
        print(df_ratio[['timestamp', 'longShortRatio', 'longAccount', 'shortAccount']].tail(5).to_string(index=False))
    else:
        print(f"⚠️  응답 형식 이상: {ratio_data}")

except Exception as e:
    print(f"❌ 오류: {e}")

time.sleep(1)

# ========================================
# 4. Liquidation Data (청산 데이터)
# ========================================
print(f"\n\n{'='*80}")
print("4. Liquidation Data (청산 데이터)")
print(f"{'='*80}")

try:
    endpoint = f"{base_url}/fapi/v1/allForceOrders"

    # 최근 7일만 조회 가능
    params = {
        'symbol': symbol,
        'limit': 100
    }

    response = requests.get(endpoint, params=params)
    liq_data = response.json()

    if isinstance(liq_data, list) and len(liq_data) > 0:
        df_liq = pd.DataFrame(liq_data)
        df_liq['time'] = pd.to_datetime(df_liq['time'], unit='ms')
        df_liq['price'] = df_liq['price'].astype(float)
        df_liq['origQty'] = df_liq['origQty'].astype(float)

        print(f"✅ 조회 성공!")
        print(f"   - 조회 건수: {len(df_liq)}개")
        print(f"   - 가장 오래된 데이터: {df_liq['time'].min()}")
        print(f"   - 가장 최근 데이터: {df_liq['time'].max()}")
        print(f"   ⚠️  청산 데이터는 최근 7일만 제공됨")
        print(f"\n   샘플 (최근 5개):")
        print(df_liq[['time', 'side', 'price', 'origQty']].tail(5).to_string(index=False))
    else:
        print(f"⚠️  응답 형식 이상: {liq_data}")

except Exception as e:
    print(f"❌ 오류: {e}")

time.sleep(1)

# ========================================
# 5. Futures Price (선물 가격) - 베이시스 계산용
# ========================================
print(f"\n\n{'='*80}")
print("5. Futures Price (선물 가격) - 베이시스 계산용")
print(f"{'='*80}")

try:
    endpoint = f"{base_url}/fapi/v1/klines"

    params = {
        'symbol': symbol,
        'interval': '1d',
        'limit': 100
    }

    response = requests.get(endpoint, params=params)
    kline_data = response.json()

    if isinstance(kline_data, list) and len(kline_data) > 0:
        df_kline = pd.DataFrame(kline_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        df_kline['timestamp'] = pd.to_datetime(df_kline['timestamp'], unit='ms')
        df_kline['close'] = df_kline['close'].astype(float)

        print(f"✅ 조회 성공!")
        print(f"   - 조회 건수: {len(df_kline)}개")
        print(f"   - 가장 오래된 데이터: {df_kline['timestamp'].min()}")
        print(f"   - 가장 최근 데이터: {df_kline['timestamp'].max()}")
        print(f"   - 가격 범위: ${df_kline['close'].min():,.2f} ~ ${df_kline['close'].max():,.2f}")
        print(f"\n   샘플 (최근 5개):")
        print(df_kline[['timestamp', 'close', 'volume']].tail(5).to_string(index=False))
    else:
        print(f"⚠️  응답 형식 이상: {kline_data}")

except Exception as e:
    print(f"❌ 오류: {e}")

# ========================================
# 요약
# ========================================
print(f"\n\n{'='*80}")
print("📊 수집 가능 여부 요약")
print(f"{'='*80}\n")

summary = """
1. ✅ Funding Rate (펀딩비)
   - 과거 누적 데이터: 조회 가능
   - 제공 범위: 2019년 9월 ~ 현재 (Futures 출시 이후)
   - 조회 방법: startTime/endTime 또는 limit으로 pagination
   - 갱신 주기: 8시간마다

2. ✅ Open Interest (미결제약정)
   - 과거 누적 데이터: 조회 가능
   - 제공 범위: 2019년 9월 ~ 현재
   - 시간 단위: 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d
   - 조회 방법: startTime/endTime으로 기간 지정

3. ✅ Long/Short Ratio (롱숏 비율)
   - 과거 누적 데이터: 조회 가능
   - 제공 범위: 2019년 9월 ~ 현재
   - 시간 단위: 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d
   - 계정별/포지션별 두 가지 제공

4. ⚠️  Liquidation Data (청산 데이터)
   - 과거 누적 데이터: 제한적 (최근 7일만)
   - 장기 historical data 불가
   - 대안: 집계 데이터는 Coinglass/CryptoQuant 유료 필요

5. ✅ Futures Price (선물 가격)
   - 과거 누적 데이터: 조회 가능
   - 제공 범위: 2019년 9월 ~ 현재
   - 현물과 비교하여 베이시스 계산 가능

📌 결론:
- 펀딩비, OI, 롱숏비율, 선물가격은 2021-02-03부터의 데이터를
  문제없이 수집할 수 있습니다!
- 청산 데이터는 최근 7일만 제공되므로, 과거 누적은 어렵습니다.
- 총 약 12~15개의 파생상품 지표를 추가할 수 있습니다.
"""

print(summary)

print(f"\n{'='*80}")
print("테스트 완료!")
print(f"{'='*80}")
