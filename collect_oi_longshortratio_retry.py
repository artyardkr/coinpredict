#!/usr/bin/env python3
"""
Retry collecting OI and Long/Short Ratio with different start dates
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time

print("=" * 80)
print("OI & Long/Short Ratio Collection - Testing Different Start Dates")
print("=" * 80)

base_url = "https://fapi.binance.com"
symbol = "BTCUSDT"

# Try different start dates
test_dates = [
    datetime(2019, 9, 1),   # Futures launch
    datetime(2020, 1, 1),   # 2020
    datetime(2021, 1, 1),   # 2021
    datetime(2021, 6, 1),   # Mid 2021
    datetime(2022, 1, 1),   # 2022
    datetime(2023, 1, 1),   # 2023
]

for test_date in test_dates:
    print(f"\n{'='*80}")
    print(f"Testing start date: {test_date.date()}")
    print(f"{'='*80}")

    start_ms = int(test_date.timestamp() * 1000)
    end_ms = int((test_date + timedelta(days=30)).timestamp() * 1000)

    # Test OI
    try:
        endpoint = f"{base_url}/futures/data/openInterestHist"
        params = {
            'symbol': symbol,
            'period': '1d',
            'startTime': start_ms,
            'endTime': end_ms,
            'limit': 30
        }

        response = requests.get(endpoint, params=params)

        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                print(f"✅ OI: {len(data)} records from {test_date.date()}")
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                print(f"   First record: {df['timestamp'].min()}")
                print(f"   Last record: {df['timestamp'].max()}")
                break  # Found the earliest date
            else:
                print(f"⚠️  OI: No data for {test_date.date()}")
        else:
            print(f"❌ OI Error {response.status_code}: {response.text[:100]}")

        time.sleep(0.5)

    except Exception as e:
        print(f"❌ OI Exception: {e}")

    # Test Long/Short Ratio
    try:
        endpoint = f"{base_url}/futures/data/topLongShortAccountRatio"
        params = {
            'symbol': symbol,
            'period': '1d',
            'startTime': start_ms,
            'endTime': end_ms,
            'limit': 30
        }

        response = requests.get(endpoint, params=params)

        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                print(f"✅ Long/Short: {len(data)} records from {test_date.date()}")
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                print(f"   First record: {df['timestamp'].min()}")
                print(f"   Last record: {df['timestamp'].max()}")
                break  # Found the earliest date
            else:
                print(f"⚠️  Long/Short: No data for {test_date.date()}")
        else:
            print(f"❌ Long/Short Error {response.status_code}: {response.text[:100]}")

        time.sleep(0.5)

    except Exception as e:
        print(f"❌ Long/Short Exception: {e}")

print(f"\n{'='*80}")
print("Testing Complete")
print(f"{'='*80}")

# Now collect from the earliest working date
print(f"\n{'='*80}")
print("Collecting OI from 2021-01-01")
print(f"{'='*80}")

all_oi = []
start_date = datetime(2021, 1, 1)
end_date = datetime(2025, 11, 1)
current_start = start_date

while current_start < end_date:
    try:
        start_ms = int(current_start.timestamp() * 1000)
        end_ms = int(min(current_start + timedelta(days=500), end_date).timestamp() * 1000)

        endpoint = f"{base_url}/futures/data/openInterestHist"
        params = {
            'symbol': symbol,
            'period': '1d',
            'startTime': start_ms,
            'endTime': end_ms,
            'limit': 500
        }

        response = requests.get(endpoint, params=params)

        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                all_oi.extend(data)
                last_ts = datetime.fromtimestamp(int(data[-1]['timestamp'])/1000)
                print(f"   ✓ {current_start.date()} ~ {last_ts.date()}: {len(data)} records")
                current_start = last_ts + timedelta(days=1)
            else:
                print(f"   ⚠️  No data, skipping...")
                current_start += timedelta(days=500)
        else:
            print(f"   ❌ Error {response.status_code}: {response.text[:100]}")
            current_start += timedelta(days=500)

        time.sleep(0.5)

    except Exception as e:
        print(f"   ❌ Exception: {e}")
        current_start += timedelta(days=500)

if len(all_oi) > 0:
    df_oi = pd.DataFrame(all_oi)
    df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'], unit='ms')
    df_oi['sumOpenInterest'] = df_oi['sumOpenInterest'].astype(float)
    df_oi['sumOpenInterestValue'] = df_oi['sumOpenInterestValue'].astype(float)
    df_oi = df_oi.sort_values('timestamp').reset_index(drop=True)
    df_oi.to_csv('oi_data_2021_2025.csv', index=False)

    print(f"\n✅ OI 수집 완료!")
    print(f"   - 총 {len(df_oi)}개")
    print(f"   - 기간: {df_oi['timestamp'].min()} ~ {df_oi['timestamp'].max()}")
    print(f"   - 저장: oi_data_2021_2025.csv")
else:
    print(f"\n❌ OI 수집 실패")

# Long/Short Ratio
print(f"\n{'='*80}")
print("Collecting Long/Short Ratio from 2021-01-01")
print(f"{'='*80}")

all_ratio = []
current_start = start_date

while current_start < end_date:
    try:
        start_ms = int(current_start.timestamp() * 1000)
        end_ms = int(min(current_start + timedelta(days=500), end_date).timestamp() * 1000)

        endpoint = f"{base_url}/futures/data/topLongShortAccountRatio"
        params = {
            'symbol': symbol,
            'period': '1d',
            'startTime': start_ms,
            'endTime': end_ms,
            'limit': 500
        }

        response = requests.get(endpoint, params=params)

        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                all_ratio.extend(data)
                last_ts = datetime.fromtimestamp(int(data[-1]['timestamp'])/1000)
                print(f"   ✓ {current_start.date()} ~ {last_ts.date()}: {len(data)} records")
                current_start = last_ts + timedelta(days=1)
            else:
                print(f"   ⚠️  No data, skipping...")
                current_start += timedelta(days=500)
        else:
            print(f"   ❌ Error {response.status_code}: {response.text[:100]}")
            current_start += timedelta(days=500)

        time.sleep(0.5)

    except Exception as e:
        print(f"   ❌ Exception: {e}")
        current_start += timedelta(days=500)

if len(all_ratio) > 0:
    df_ratio = pd.DataFrame(all_ratio)
    df_ratio['timestamp'] = pd.to_datetime(df_ratio['timestamp'], unit='ms')
    df_ratio['longShortRatio'] = df_ratio['longShortRatio'].astype(float)
    df_ratio['longAccount'] = df_ratio['longAccount'].astype(float)
    df_ratio['shortAccount'] = df_ratio['shortAccount'].astype(float)
    df_ratio = df_ratio.sort_values('timestamp').reset_index(drop=True)
    df_ratio.to_csv('longshortratio_data_2021_2025.csv', index=False)

    print(f"\n✅ Long/Short Ratio 수집 완료!")
    print(f"   - 총 {len(df_ratio)}개")
    print(f"   - 기간: {df_ratio['timestamp'].min()} ~ {df_ratio['timestamp'].max()}")
    print(f"   - 저장: longshortratio_data_2021_2025.csv")
else:
    print(f"\n❌ Long/Short Ratio 수집 실패")

print(f"\n{'='*80}")
print("Complete!")
print(f"{'='*80}")
