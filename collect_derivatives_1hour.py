#!/usr/bin/env python3
"""
Binance Futures Derivatives Data Collection - 1 Hour Interval
Period: 2020-01-01 ~ 2025-11-01
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time

print("=" * 80)
print("Binance Futures Derivatives Data Collection (1 Hour Interval)")
print("Period: 2020-01-01 ~ 2025-11-01")
print("=" * 80)

base_url = "https://fapi.binance.com"
symbol = "BTCUSDT"
start_date = datetime(2020, 1, 1)
end_date = datetime(2025, 11, 1)

# ========================================
# 1. Open Interest (1시간) - 최근 500개만 조회
# ========================================
print(f"\n{'='*80}")
print("1. Collecting Open Interest (1시간 단위) - Recent 500 records")
print(f"{'='*80}")

try:
    endpoint = f"{base_url}/futures/data/openInterestHist"
    params = {
        'symbol': symbol,
        'period': '1h',  # 1시간
        'limit': 500
    }

    response = requests.get(endpoint, params=params)

    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            df_oi = pd.DataFrame(data)
            df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'], unit='ms')
            df_oi['sumOpenInterest'] = df_oi['sumOpenInterest'].astype(float)
            df_oi['sumOpenInterestValue'] = df_oi['sumOpenInterestValue'].astype(float)

            print(f"✅ OI 조회 성공!")
            print(f"   - 총 {len(df_oi)}개")
            print(f"   - 기간: {df_oi['timestamp'].min()} ~ {df_oi['timestamp'].max()}")
            print(f"   - OI 범위: {df_oi['sumOpenInterest'].min():,.0f} ~ {df_oi['sumOpenInterest'].max():,.0f} BTC")
        else:
            print(f"⚠️  데이터 없음")
            df_oi = pd.DataFrame()
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
        df_oi = pd.DataFrame()

except Exception as e:
    print(f"❌ Exception: {e}")
    df_oi = pd.DataFrame()

time.sleep(1)

# ========================================
# 2. Long/Short Ratio (1시간) - 최근 500개만 조회
# ========================================
print(f"\n{'='*80}")
print("2. Collecting Long/Short Ratio (1시간 단위) - Recent 500 records")
print(f"{'='*80}")

try:
    endpoint = f"{base_url}/futures/data/topLongShortAccountRatio"
    params = {
        'symbol': symbol,
        'period': '1h',  # 1시간
        'limit': 500
    }

    response = requests.get(endpoint, params=params)

    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            df_ratio = pd.DataFrame(data)
            df_ratio['timestamp'] = pd.to_datetime(df_ratio['timestamp'], unit='ms')
            df_ratio['longShortRatio'] = df_ratio['longShortRatio'].astype(float)
            df_ratio['longAccount'] = df_ratio['longAccount'].astype(float)
            df_ratio['shortAccount'] = df_ratio['shortAccount'].astype(float)

            print(f"✅ Long/Short Ratio 조회 성공!")
            print(f"   - 총 {len(df_ratio)}개")
            print(f"   - 기간: {df_ratio['timestamp'].min()} ~ {df_ratio['timestamp'].max()}")
            print(f"   - 롱숏비율 범위: {df_ratio['longShortRatio'].min():.4f} ~ {df_ratio['longShortRatio'].max():.4f}")
        else:
            print(f"⚠️  데이터 없음")
            df_ratio = pd.DataFrame()
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
        df_ratio = pd.DataFrame()

except Exception as e:
    print(f"❌ Exception: {e}")
    df_ratio = pd.DataFrame()

time.sleep(1)

# ========================================
# 3. Convert to Daily (1시간 -> 1일)
# ========================================
print(f"\n{'='*80}")
print("3. Converting to Daily Data")
print(f"{'='*80}")

# OI: Daily average
if len(df_oi) > 0:
    df_oi['Date'] = df_oi['timestamp'].dt.date
    df_oi_daily = df_oi.groupby('Date').agg({
        'sumOpenInterest': 'mean',
        'sumOpenInterestValue': 'mean'
    }).reset_index()
    df_oi_daily['Date'] = pd.to_datetime(df_oi_daily['Date'])
    df_oi_daily.columns = ['Date', 'OI', 'OI_Value']
    print(f"✅ OI Daily: {len(df_oi_daily)} days")
else:
    df_oi_daily = pd.DataFrame()
    print(f"⚠️  OI Daily: No data")

# Long/Short Ratio: Daily average
if len(df_ratio) > 0:
    df_ratio['Date'] = df_ratio['timestamp'].dt.date
    df_ratio_daily = df_ratio.groupby('Date').agg({
        'longShortRatio': 'mean',
        'longAccount': 'mean',
        'shortAccount': 'mean'
    }).reset_index()
    df_ratio_daily['Date'] = pd.to_datetime(df_ratio_daily['Date'])
    df_ratio_daily.columns = ['Date', 'Long_Short_Ratio', 'Long_Account_Pct', 'Short_Account_Pct']
    print(f"✅ Long/Short Ratio Daily: {len(df_ratio_daily)} days")
else:
    df_ratio_daily = pd.DataFrame()
    print(f"⚠️  Long/Short Ratio Daily: No data")

# ========================================
# 4. Load existing Funding Rate and Futures Price data
# ========================================
print(f"\n{'='*80}")
print("4. Loading Existing Derivatives Data")
print(f"{'='*80}")

try:
    df_existing = pd.read_csv('binance_derivatives_2020_2025.csv')
    df_existing['Date'] = pd.to_datetime(df_existing['Date'])
    print(f"✅ Loaded: binance_derivatives_2020_2025.csv ({len(df_existing)} records)")
except Exception as e:
    print(f"❌ Could not load existing data: {e}")
    df_existing = pd.DataFrame()

# ========================================
# 5. Merge all data
# ========================================
print(f"\n{'='*80}")
print("5. Merging All Data")
print(f"{'='*80}")

if len(df_existing) > 0:
    df_final = df_existing.copy()
else:
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df_final = pd.DataFrame({'Date': date_range})

# Merge OI
if len(df_oi_daily) > 0:
    # Remove existing OI columns if present
    oi_cols = ['OI', 'OI_Value', 'OI_Change_1D', 'OI_Change_7D', 'OI_Momentum', 'Funding_OI_Product']
    for col in oi_cols:
        if col in df_final.columns:
            df_final = df_final.drop(columns=[col])

    df_final = df_final.merge(df_oi_daily, on='Date', how='left')
    print(f"✅ Merged OI")

# Merge Long/Short Ratio
if len(df_ratio_daily) > 0:
    # Remove existing ratio columns if present
    ratio_cols = ['Long_Short_Ratio', 'Long_Account_Pct', 'Short_Account_Pct', 'Long_Short_Divergence']
    for col in ratio_cols:
        if col in df_final.columns:
            df_final = df_final.drop(columns=[col])

    df_final = df_final.merge(df_ratio_daily, on='Date', how='left')
    print(f"✅ Merged Long/Short Ratio")

# ========================================
# 6. Calculate Derived Features
# ========================================
print(f"\n{'='*80}")
print("6. Calculating Derived Features")
print(f"{'='*80}")

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

# ========================================
# 7. Save
# ========================================
print(f"\n{'='*80}")
print("7. Saving to CSV")
print(f"{'='*80}")

output_file = 'binance_derivatives_2020_2025_full.csv'
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

# ========================================
# 8. Summary
# ========================================
print(f"\n{'='*80}")
print("📊 Summary")
print(f"{'='*80}\n")

print(f"✅ Funding Rate: 2020-01-01 ~ 2025-10-31 (전체 기간)")
print(f"✅ Futures Price: 2020-01-01 ~ 2025-11-02 (전체 기간)")
print(f"✅ Basis: 2021-02-03 ~ 2025-10-13 (현물 데이터 기간)")

if len(df_oi_daily) > 0:
    print(f"✅ OI: {df_oi_daily['Date'].min().date()} ~ {df_oi_daily['Date'].max().date()} (최근 ~500시간)")
else:
    print(f"❌ OI: No data")

if len(df_ratio_daily) > 0:
    print(f"✅ Long/Short Ratio: {df_ratio_daily['Date'].min().date()} ~ {df_ratio_daily['Date'].max().date()} (최근 ~500시간)")
else:
    print(f"❌ Long/Short Ratio: No data")

print(f"\n⚠️  주의: Binance API는 OI/Long Short Ratio의 과거 전체 데이터를 제공하지 않습니다.")
print(f"   - 최근 500개 데이터포인트만 조회 가능 (1시간 단위 = 약 20일)")
print(f"   - 과거 전체 데이터는 Coinglass, CryptoQuant 등 유료 서비스 필요")

print(f"\n{'='*80}")
print("✅ Collection Complete!")
print(f"{'='*80}")
