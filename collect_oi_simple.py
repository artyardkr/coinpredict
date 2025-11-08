#!/usr/bin/env python3
"""
Simple OI & Long/Short Ratio collection (without startTime)
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time

print("=" * 80)
print("OI & Long/Short Ratio - Simple Collection")
print("=" * 80)

base_url = "https://fapi.binance.com"
symbol = "BTCUSDT"

# ========================================
# Test without startTime
# ========================================
print(f"\n{'='*80}")
print("1. Testing OI without startTime")
print(f"{'='*80}")

try:
    endpoint = f"{base_url}/futures/data/openInterestHist"
    params = {
        'symbol': symbol,
        'period': '1d',
        'limit': 500
    }

    response = requests.get(endpoint, params=params)

    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            df_oi = pd.DataFrame(data)
            df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'], unit='ms')
            df_oi['sumOpenInterest'] = df_oi['sumOpenInterest'].astype(float)

            print(f"✅ OI 조회 성공!")
            print(f"   - 총 {len(df_oi)}개")
            print(f"   - 기간: {df_oi['timestamp'].min()} ~ {df_oi['timestamp'].max()}")

            df_oi.to_csv('oi_data_recent.csv', index=False)
            print(f"   - 저장: oi_data_recent.csv")
        else:
            print(f"⚠️  데이터 없음")
    else:
        print(f"❌ Error {response.status_code}: {response.text}")

except Exception as e:
    print(f"❌ Exception: {e}")

time.sleep(1)

# ========================================
# Long/Short Ratio
# ========================================
print(f"\n{'='*80}")
print("2. Testing Long/Short Ratio without startTime")
print(f"{'='*80}")

try:
    endpoint = f"{base_url}/futures/data/topLongShortAccountRatio"
    params = {
        'symbol': symbol,
        'period': '1d',
        'limit': 500
    }

    response = requests.get(endpoint, params=params)

    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            df_ratio = pd.DataFrame(data)
            df_ratio['timestamp'] = pd.to_datetime(df_ratio['timestamp'], unit='ms')
            df_ratio['longShortRatio'] = df_ratio['longShortRatio'].astype(float)

            print(f"✅ Long/Short Ratio 조회 성공!")
            print(f"   - 총 {len(df_ratio)}개")
            print(f"   - 기간: {df_ratio['timestamp'].min()} ~ {df_ratio['timestamp'].max()}")

            df_ratio.to_csv('longshortratio_data_recent.csv', index=False)
            print(f"   - 저장: longshortratio_data_recent.csv")
        else:
            print(f"⚠️  데이터 없음")
    else:
        print(f"❌ Error {response.status_code}: {response.text}")

except Exception as e:
    print(f"❌ Exception: {e}")

print(f"\n{'='*80}")
print("Complete!")
print(f"{'='*80}")

print(f"\n📌 결론:")
print(f"   - Binance API는 OI/Long Short Ratio의 과거 데이터를 startTime으로 조회 불가")
print(f"   - 최근 500개(약 500일)만 조회 가능")
print(f"   - 2020년부터의 전체 historical data는 제공하지 않음")
print(f"   - 대안: Coinglass, CryptoQuant 등 유료 서비스 필요")
