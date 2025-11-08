#!/usr/bin/env python3
"""
Binance Futures Derivatives Data Collection (2020-01-01 ~ 2025-11-01)
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np

print("=" * 80)
print("Binance Futures Derivatives Data Collection")
print("Period: 2020-01-01 ~ 2025-11-01")
print("=" * 80)

# Configuration
base_url = "https://fapi.binance.com"
symbol = "BTCUSDT"
start_date = datetime(2020, 1, 1)
end_date = datetime(2025, 11, 1)

# ========================================
# 1. Funding Rate (펀딩비)
# ========================================
print(f"\n{'='*80}")
print("1. Collecting Funding Rate (펀딩비)")
print(f"{'='*80}")

all_funding = []
current_start = start_date

while current_start < end_date:
    try:
        start_ms = int(current_start.timestamp() * 1000)
        end_ms = int(min(current_start + timedelta(days=300), end_date).timestamp() * 1000)

        endpoint = f"{base_url}/fapi/v1/fundingRate"
        params = {
            'symbol': symbol,
            'startTime': start_ms,
            'endTime': end_ms,
            'limit': 1000
        }

        response = requests.get(endpoint, params=params)

        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                all_funding.extend(data)
                print(f"   ✓ {current_start.date()} ~ {datetime.fromtimestamp(end_ms/1000).date()}: {len(data)} records")
                current_start = datetime.fromtimestamp(int(data[-1]['fundingTime'])/1000) + timedelta(hours=8)
            else:
                current_start += timedelta(days=300)
        else:
            print(f"   ⚠️  Error {response.status_code}: {response.text}")
            current_start += timedelta(days=300)

        time.sleep(0.5)  # Rate limit

    except Exception as e:
        print(f"   ❌ Error: {e}")
        current_start += timedelta(days=300)

if len(all_funding) > 0:
    df_funding = pd.DataFrame(all_funding)
    df_funding['fundingTime'] = pd.to_datetime(df_funding['fundingTime'], unit='ms')
    df_funding['fundingRate'] = df_funding['fundingRate'].astype(float)
    df_funding = df_funding.sort_values('fundingTime').reset_index(drop=True)

    print(f"\n✅ Funding Rate 수집 완료!")
    print(f"   - 총 {len(df_funding)}개")
    print(f"   - 기간: {df_funding['fundingTime'].min()} ~ {df_funding['fundingTime'].max()}")
    print(f"   - 펀딩비 범위: {df_funding['fundingRate'].min()*100:.4f}% ~ {df_funding['fundingRate'].max()*100:.4f}%")
else:
    print(f"\n❌ Funding Rate 수집 실패")
    df_funding = pd.DataFrame()

# ========================================
# 2. Open Interest (미결제약정) - Daily
# ========================================
print(f"\n{'='*80}")
print("2. Collecting Open Interest (미결제약정) - Daily")
print(f"{'='*80}")

all_oi = []
current_start = start_date

while current_start < end_date:
    try:
        start_ms = int(current_start.timestamp() * 1000)
        end_ms = int(min(current_start + timedelta(days=500), end_date).timestamp() * 1000)

        endpoint = f"{base_url}/futures/data/openInterestHist"
        params = {
            'symbol': symbol,
            'period': '1d',  # Daily
            'startTime': start_ms,
            'endTime': end_ms,
            'limit': 500
        }

        response = requests.get(endpoint, params=params)

        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                all_oi.extend(data)
                print(f"   ✓ {current_start.date()} ~ {datetime.fromtimestamp(end_ms/1000).date()}: {len(data)} records")
                current_start = datetime.fromtimestamp(int(data[-1]['timestamp'])/1000) + timedelta(days=1)
            else:
                current_start += timedelta(days=500)
        else:
            print(f"   ⚠️  Error {response.status_code}: {response.text}")
            current_start += timedelta(days=500)

        time.sleep(0.5)

    except Exception as e:
        print(f"   ❌ Error: {e}")
        current_start += timedelta(days=500)

if len(all_oi) > 0:
    df_oi = pd.DataFrame(all_oi)
    df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'], unit='ms')
    df_oi['sumOpenInterest'] = df_oi['sumOpenInterest'].astype(float)
    df_oi['sumOpenInterestValue'] = df_oi['sumOpenInterestValue'].astype(float)
    df_oi = df_oi.sort_values('timestamp').reset_index(drop=True)

    print(f"\n✅ Open Interest 수집 완료!")
    print(f"   - 총 {len(df_oi)}개")
    print(f"   - 기간: {df_oi['timestamp'].min()} ~ {df_oi['timestamp'].max()}")
    print(f"   - OI 범위: {df_oi['sumOpenInterest'].min():,.0f} ~ {df_oi['sumOpenInterest'].max():,.0f} BTC")
else:
    print(f"\n❌ Open Interest 수집 실패")
    df_oi = pd.DataFrame()

# ========================================
# 3. Long/Short Ratio (롱숏 비율) - Daily
# ========================================
print(f"\n{'='*80}")
print("3. Collecting Long/Short Ratio (롱숏 비율) - Daily")
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
            'period': '1d',  # Daily
            'startTime': start_ms,
            'endTime': end_ms,
            'limit': 500
        }

        response = requests.get(endpoint, params=params)

        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                all_ratio.extend(data)
                print(f"   ✓ {current_start.date()} ~ {datetime.fromtimestamp(end_ms/1000).date()}: {len(data)} records")
                current_start = datetime.fromtimestamp(int(data[-1]['timestamp'])/1000) + timedelta(days=1)
            else:
                current_start += timedelta(days=500)
        else:
            print(f"   ⚠️  Error {response.status_code}: {response.text}")
            current_start += timedelta(days=500)

        time.sleep(0.5)

    except Exception as e:
        print(f"   ❌ Error: {e}")
        current_start += timedelta(days=500)

if len(all_ratio) > 0:
    df_ratio = pd.DataFrame(all_ratio)
    df_ratio['timestamp'] = pd.to_datetime(df_ratio['timestamp'], unit='ms')
    df_ratio['longShortRatio'] = df_ratio['longShortRatio'].astype(float)
    df_ratio['longAccount'] = df_ratio['longAccount'].astype(float)
    df_ratio['shortAccount'] = df_ratio['shortAccount'].astype(float)
    df_ratio = df_ratio.sort_values('timestamp').reset_index(drop=True)

    print(f"\n✅ Long/Short Ratio 수집 완료!")
    print(f"   - 총 {len(df_ratio)}개")
    print(f"   - 기간: {df_ratio['timestamp'].min()} ~ {df_ratio['timestamp'].max()}")
    print(f"   - 롱숏비율 범위: {df_ratio['longShortRatio'].min():.4f} ~ {df_ratio['longShortRatio'].max():.4f}")
else:
    print(f"\n❌ Long/Short Ratio 수집 실패")
    df_ratio = pd.DataFrame()

# ========================================
# 4. Futures Price (선물 가격) - Daily
# ========================================
print(f"\n{'='*80}")
print("4. Collecting Futures Price (선물 가격) - Daily")
print(f"{'='*80}")

all_futures = []
current_start = start_date

while current_start < end_date:
    try:
        start_ms = int(current_start.timestamp() * 1000)

        endpoint = f"{base_url}/fapi/v1/klines"
        params = {
            'symbol': symbol,
            'interval': '1d',
            'startTime': start_ms,
            'limit': 1000
        }

        response = requests.get(endpoint, params=params)

        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                all_futures.extend(data)
                print(f"   ✓ {current_start.date()}: {len(data)} records")
                current_start = datetime.fromtimestamp(int(data[-1][0])/1000) + timedelta(days=1)

                if current_start >= end_date:
                    break
            else:
                current_start += timedelta(days=1000)
        else:
            print(f"   ⚠️  Error {response.status_code}: {response.text}")
            current_start += timedelta(days=1000)

        time.sleep(0.5)

    except Exception as e:
        print(f"   ❌ Error: {e}")
        current_start += timedelta(days=1000)

if len(all_futures) > 0:
    df_futures = pd.DataFrame(all_futures, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    df_futures['timestamp'] = pd.to_datetime(df_futures['timestamp'], unit='ms')
    df_futures['close'] = df_futures['close'].astype(float)
    df_futures['volume'] = df_futures['volume'].astype(float)
    df_futures = df_futures.sort_values('timestamp').reset_index(drop=True)

    print(f"\n✅ Futures Price 수집 완료!")
    print(f"   - 총 {len(df_futures)}개")
    print(f"   - 기간: {df_futures['timestamp'].min()} ~ {df_futures['timestamp'].max()}")
    print(f"   - 가격 범위: ${df_futures['close'].min():,.2f} ~ ${df_futures['close'].max():,.2f}")
else:
    print(f"\n❌ Futures Price 수집 실패")
    df_futures = pd.DataFrame()

# ========================================
# 5. Process and Save
# ========================================
print(f"\n{'='*80}")
print("5. Processing and Saving Data")
print(f"{'='*80}")

# Funding Rate: Convert to daily (use last value of the day)
if len(df_funding) > 0:
    df_funding['Date'] = df_funding['fundingTime'].dt.date
    df_funding_daily = df_funding.groupby('Date').agg({
        'fundingRate': 'last'
    }).reset_index()
    df_funding_daily['Date'] = pd.to_datetime(df_funding_daily['Date'])
    df_funding_daily.columns = ['Date', 'Funding_Rate']
    print(f"✅ Funding Rate (Daily): {len(df_funding_daily)} records")
else:
    df_funding_daily = pd.DataFrame()

# Open Interest: Already daily
if len(df_oi) > 0:
    df_oi_daily = df_oi.copy()
    df_oi_daily['Date'] = pd.to_datetime(df_oi_daily['timestamp'].dt.date)
    df_oi_daily = df_oi_daily[['Date', 'sumOpenInterest', 'sumOpenInterestValue']]
    df_oi_daily.columns = ['Date', 'OI', 'OI_Value']
    print(f"✅ Open Interest (Daily): {len(df_oi_daily)} records")
else:
    df_oi_daily = pd.DataFrame()

# Long/Short Ratio: Already daily
if len(df_ratio) > 0:
    df_ratio_daily = df_ratio.copy()
    df_ratio_daily['Date'] = pd.to_datetime(df_ratio_daily['timestamp'].dt.date)
    df_ratio_daily = df_ratio_daily[['Date', 'longShortRatio', 'longAccount', 'shortAccount']]
    df_ratio_daily.columns = ['Date', 'Long_Short_Ratio', 'Long_Account_Pct', 'Short_Account_Pct']
    print(f"✅ Long/Short Ratio (Daily): {len(df_ratio_daily)} records")
else:
    df_ratio_daily = pd.DataFrame()

# Futures Price: Already daily
if len(df_futures) > 0:
    df_futures_daily = df_futures.copy()
    df_futures_daily['Date'] = pd.to_datetime(df_futures_daily['timestamp'].dt.date)
    df_futures_daily = df_futures_daily[['Date', 'close', 'volume']]
    df_futures_daily.columns = ['Date', 'Futures_Close', 'Futures_Volume']
    print(f"✅ Futures Price (Daily): {len(df_futures_daily)} records")
else:
    df_futures_daily = pd.DataFrame()

# Merge all
print(f"\n{'='*80}")
print("6. Merging All Data")
print(f"{'='*80}")

# Start with an empty dataframe with date range
date_range = pd.date_range(start=start_date, end=end_date, freq='D')
df_final = pd.DataFrame({'Date': date_range})

# Merge each dataset
if len(df_funding_daily) > 0:
    df_final = df_final.merge(df_funding_daily, on='Date', how='left')
    print(f"✅ Merged Funding Rate")

if len(df_oi_daily) > 0:
    df_final = df_final.merge(df_oi_daily, on='Date', how='left')
    print(f"✅ Merged Open Interest")

if len(df_ratio_daily) > 0:
    df_final = df_final.merge(df_ratio_daily, on='Date', how='left')
    print(f"✅ Merged Long/Short Ratio")

if len(df_futures_daily) > 0:
    df_final = df_final.merge(df_futures_daily, on='Date', how='left')
    print(f"✅ Merged Futures Price")

# Calculate derived features
print(f"\n{'='*80}")
print("7. Calculating Derived Features")
print(f"{'='*80}")

if 'Funding_Rate' in df_final.columns:
    df_final['Funding_Rate_MA7'] = df_final['Funding_Rate'].rolling(7).mean()
    df_final['Funding_Rate_MA30'] = df_final['Funding_Rate'].rolling(30).mean()
    print(f"✅ Funding Rate MA7, MA30")

if 'OI' in df_final.columns:
    df_final['OI_Change_1D'] = df_final['OI'].pct_change(1) * 100
    df_final['OI_Change_7D'] = df_final['OI'].pct_change(7) * 100
    df_final['OI_Momentum'] = df_final['OI'].pct_change(1) - df_final['OI'].pct_change(7)
    print(f"✅ OI Change 1D, 7D, Momentum")

if 'Funding_Rate' in df_final.columns and 'OI' in df_final.columns:
    df_final['Funding_OI_Product'] = df_final['Funding_Rate'] * df_final['OI']
    print(f"✅ Funding × OI Product")

if 'Long_Short_Ratio' in df_final.columns:
    df_final['Long_Short_Divergence'] = df_final['Long_Short_Ratio'] - df_final['Long_Short_Ratio'].rolling(7).mean()
    print(f"✅ Long/Short Divergence")

# Load spot price to calculate basis
print(f"\n{'='*80}")
print("8. Loading Spot Price for Basis Calculation")
print(f"{'='*80}")

try:
    # Try to load existing integrated data
    spot_df = pd.read_csv('integrated_data_full_v2.csv')
    spot_df['Date'] = pd.to_datetime(spot_df['Date'])
    spot_price = spot_df[['Date', 'Close']].copy()
    spot_price.columns = ['Date', 'Spot_Close']

    df_final = df_final.merge(spot_price, on='Date', how='left')

    if 'Futures_Close' in df_final.columns and 'Spot_Close' in df_final.columns:
        df_final['Basis'] = ((df_final['Futures_Close'] - df_final['Spot_Close']) / df_final['Spot_Close']) * 100
        df_final['Basis_MA7'] = df_final['Basis'].rolling(7).mean()
        print(f"✅ Calculated Basis (Futures - Spot) and MA7")
    else:
        print(f"⚠️  Cannot calculate Basis (missing price data)")

except Exception as e:
    print(f"⚠️  Could not load spot price: {e}")
    print(f"   Basis calculation skipped")

# Save
print(f"\n{'='*80}")
print("9. Saving to CSV")
print(f"{'='*80}")

output_file = 'binance_derivatives_2020_2025.csv'
df_final.to_csv(output_file, index=False)

print(f"✅ Saved: {output_file}")
print(f"   - Total records: {len(df_final)}")
print(f"   - Date range: {df_final['Date'].min()} ~ {df_final['Date'].max()}")
print(f"   - Columns: {len(df_final.columns)}")
print(f"\n   Column list:")
for col in df_final.columns:
    null_count = df_final[col].isnull().sum()
    null_pct = (null_count / len(df_final)) * 100
    print(f"      - {col:<30} (Null: {null_count:>4} / {null_pct:>5.1f}%)")

# Summary statistics
print(f"\n{'='*80}")
print("10. Summary Statistics")
print(f"{'='*80}\n")

print(df_final.describe().to_string())

print(f"\n{'='*80}")
print("✅ Collection Complete!")
print(f"{'='*80}")
