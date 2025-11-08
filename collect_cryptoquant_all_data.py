#!/usr/bin/env python3
"""
CryptoQuant API - Full Historical Data Collection
Period: 2020-01-01 ~ 2025-11-01

Required: CryptoQuant Advanced Plan (7-day free trial available)
API Docs: https://docs.cryptoquant.com/
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# ========================================
# Configuration
# ========================================
print("=" * 80)
print("CryptoQuant API - Full Historical Data Collection")
print("=" * 80)

# ⚠️  API 키를 여기에 입력하세요
API_KEY = "YOUR_API_KEY_HERE"  # 🔑 CryptoQuant에서 발급받은 API 키

if API_KEY == "YOUR_API_KEY_HERE":
    print("\n❌ API 키를 입력해주세요!")
    print("\n📌 CryptoQuant API 키 발급 방법:")
    print("   1. https://cryptoquant.com 회원가입")
    print("   2. Advanced 7일 무료 체험 시작")
    print("   3. Settings → API Keys → Create New Key")
    print("   4. 발급받은 키를 이 스크립트의 API_KEY 변수에 입력")
    print("\n💡 7일 안에 모든 데이터를 수집한 후 구독 취소 가능합니다!")
    exit()

BASE_URL = "https://api.cryptoquant.com/v1"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}

symbol = "btc"
start_date = "2020-01-01"
end_date = "2025-11-01"

print(f"\nPeriod: {start_date} ~ {end_date}")
print(f"Symbol: {symbol.upper()}")

# ========================================
# Helper Functions
# ========================================

def get_cryptoquant_data(endpoint, params, max_retries=3):
    """
    CryptoQuant API 호출 함수
    """
    url = f"{BASE_URL}/{endpoint}"

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, params=params)

            if response.status_code == 200:
                data = response.json()
                if 'result' in data and 'data' in data['result']:
                    return data['result']['data']
                else:
                    print(f"      ⚠️  Unexpected response format")
                    return []
            elif response.status_code == 401:
                print(f"      ❌ Authentication failed. Check your API key.")
                return []
            elif response.status_code == 429:
                print(f"      ⚠️  Rate limit exceeded. Waiting 60s...")
                time.sleep(60)
            else:
                print(f"      ⚠️  Error {response.status_code}: {response.text[:100]}")
                return []

        except Exception as e:
            print(f"      ❌ Exception: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return []

    return []

# ========================================
# 1. Open Interest (All Exchanges)
# ========================================
print(f"\n{'='*80}")
print("1. Collecting Open Interest (All Exchanges)")
print(f"{'='*80}")

oi_data = get_cryptoquant_data(
    f"btc/market-data/open-interest",
    {
        'from': start_date,
        'to': end_date,
        'exchange': 'all_exchange'  # 전체 거래소 합계
    }
)

if len(oi_data) > 0:
    df_oi = pd.DataFrame(oi_data)
    df_oi['date'] = pd.to_datetime(df_oi['date'])
    df_oi = df_oi.sort_values('date').reset_index(drop=True)
    print(f"✅ Collected: {len(df_oi)} records")
    print(f"   Range: {df_oi['date'].min()} ~ {df_oi['date'].max()}")
else:
    print(f"❌ Failed to collect Open Interest")
    df_oi = pd.DataFrame()

time.sleep(2)

# ========================================
# 2. Funding Rate (All Exchanges)
# ========================================
print(f"\n{'='*80}")
print("2. Collecting Funding Rate (All Exchanges)")
print(f"{'='*80}")

funding_data = get_cryptoquant_data(
    f"btc/market-data/funding-rates",
    {
        'from': start_date,
        'to': end_date,
        'exchange': 'all_exchange'
    }
)

if len(funding_data) > 0:
    df_funding = pd.DataFrame(funding_data)
    df_funding['date'] = pd.to_datetime(df_funding['date'])
    df_funding = df_funding.sort_values('date').reset_index(drop=True)
    print(f"✅ Collected: {len(df_funding)} records")
    print(f"   Range: {df_funding['date'].min()} ~ {df_funding['date'].max()}")
else:
    print(f"❌ Failed to collect Funding Rate")
    df_funding = pd.DataFrame()

time.sleep(2)

# ========================================
# 3. Liquidations (Long + Short)
# ========================================
print(f"\n{'='*80}")
print("3. Collecting Liquidations Data")
print(f"{'='*80}")

# Long Liquidations
long_liq_data = get_cryptoquant_data(
    f"btc/market-data/liquidations",
    {
        'from': start_date,
        'to': end_date,
        'exchange': 'all_exchange',
        'side': 'long'
    }
)

# Short Liquidations
short_liq_data = get_cryptoquant_data(
    f"btc/market-data/liquidations",
    {
        'from': start_date,
        'to': end_date,
        'exchange': 'all_exchange',
        'side': 'short'
    }
)

if len(long_liq_data) > 0 and len(short_liq_data) > 0:
    df_long_liq = pd.DataFrame(long_liq_data)
    df_short_liq = pd.DataFrame(short_liq_data)

    df_long_liq['date'] = pd.to_datetime(df_long_liq['date'])
    df_short_liq['date'] = pd.to_datetime(df_short_liq['date'])

    df_liquidations = df_long_liq.merge(df_short_liq, on='date', how='outer', suffixes=('_long', '_short'))
    df_liquidations = df_liquidations.sort_values('date').reset_index(drop=True)

    print(f"✅ Collected: {len(df_liquidations)} records")
    print(f"   Range: {df_liquidations['date'].min()} ~ {df_liquidations['date'].max()}")
else:
    print(f"❌ Failed to collect Liquidations")
    df_liquidations = pd.DataFrame()

time.sleep(2)

# ========================================
# 4. Exchange Netflow
# ========================================
print(f"\n{'='*80}")
print("4. Collecting Exchange Netflow")
print(f"{'='*80}")

netflow_data = get_cryptoquant_data(
    f"btc/exchange-flows/netflow-total",
    {
        'from': start_date,
        'to': end_date,
        'exchange': 'all_exchange'
    }
)

if len(netflow_data) > 0:
    df_netflow = pd.DataFrame(netflow_data)
    df_netflow['date'] = pd.to_datetime(df_netflow['date'])
    df_netflow = df_netflow.sort_values('date').reset_index(drop=True)
    print(f"✅ Collected: {len(df_netflow)} records")
    print(f"   Range: {df_netflow['date'].min()} ~ {df_netflow['date'].max()}")
else:
    print(f"❌ Failed to collect Exchange Netflow")
    df_netflow = pd.DataFrame()

time.sleep(2)

# ========================================
# 5. Miner Netflow
# ========================================
print(f"\n{'='*80}")
print("5. Collecting Miner Netflow")
print(f"{'='*80}")

miner_data = get_cryptoquant_data(
    f"btc/miner-flows/miner-to-exchange",
    {
        'from': start_date,
        'to': end_date
    }
)

if len(miner_data) > 0:
    df_miner = pd.DataFrame(miner_data)
    df_miner['date'] = pd.to_datetime(df_miner['date'])
    df_miner = df_miner.sort_values('date').reset_index(drop=True)
    print(f"✅ Collected: {len(df_miner)} records")
    print(f"   Range: {df_miner['date'].min()} ~ {df_miner['date'].max()}")
else:
    print(f"❌ Failed to collect Miner Netflow")
    df_miner = pd.DataFrame()

time.sleep(2)

# ========================================
# 6. Taker Buy/Sell Volume Ratio
# ========================================
print(f"\n{'='*80}")
print("6. Collecting Taker Buy/Sell Volume")
print(f"{'='*80}")

taker_data = get_cryptoquant_data(
    f"btc/market-data/taker-buy-sell-ratio",
    {
        'from': start_date,
        'to': end_date,
        'exchange': 'binance'  # Binance만 (all_exchange 지원 안함)
    }
)

if len(taker_data) > 0:
    df_taker = pd.DataFrame(taker_data)
    df_taker['date'] = pd.to_datetime(df_taker['date'])
    df_taker = df_taker.sort_values('date').reset_index(drop=True)
    print(f"✅ Collected: {len(df_taker)} records")
    print(f"   Range: {df_taker['date'].min()} ~ {df_taker['date'].max()}")
else:
    print(f"❌ Failed to collect Taker Volume")
    df_taker = pd.DataFrame()

# ========================================
# 7. Merge All Data
# ========================================
print(f"\n{'='*80}")
print("7. Merging All Data")
print(f"{'='*80}")

# Start with date range
date_range = pd.date_range(start=start_date, end=end_date, freq='D')
df_final = pd.DataFrame({'Date': date_range})

# Merge each dataset
datasets = [
    (df_oi, 'OI', 'date'),
    (df_funding, 'Funding_Rate', 'date'),
    (df_liquidations, 'Liquidations', 'date'),
    (df_netflow, 'Exchange_Netflow', 'date'),
    (df_miner, 'Miner_Netflow', 'date'),
    (df_taker, 'Taker_Volume', 'date')
]

for df, name, date_col in datasets:
    if len(df) > 0:
        df_renamed = df.copy()
        df_renamed['Date'] = pd.to_datetime(df_renamed[date_col])
        df_renamed = df_renamed.drop(columns=[date_col])

        # Rename columns to avoid conflicts
        df_renamed.columns = ['Date'] + [f"CQ_{name}_{col}" for col in df_renamed.columns if col != 'Date']

        df_final = df_final.merge(df_renamed, on='Date', how='left')
        print(f"✅ Merged {name}")
    else:
        print(f"⚠️  Skipped {name} (no data)")

# ========================================
# 8. Calculate Derived Features
# ========================================
print(f"\n{'='*80}")
print("8. Calculating Derived Features")
print(f"{'='*80}")

# Add any calculated features here
if 'CQ_OI_value' in df_final.columns:
    df_final['CQ_OI_Change_1D'] = df_final['CQ_OI_value'].pct_change(1) * 100
    df_final['CQ_OI_Change_7D'] = df_final['CQ_OI_value'].pct_change(7) * 100
    print(f"✅ OI Change calculations")

if 'CQ_Exchange_Netflow_value' in df_final.columns:
    df_final['CQ_Exchange_Netflow_MA7'] = df_final['CQ_Exchange_Netflow_value'].rolling(7).mean()
    print(f"✅ Exchange Netflow MA7")

# ========================================
# 9. Save
# ========================================
print(f"\n{'='*80}")
print("9. Saving to CSV")
print(f"{'='*80}")

output_file = 'cryptoquant_full_data_2020_2025.csv'
df_final.to_csv(output_file, index=False)

print(f"✅ Saved: {output_file}")
print(f"   - Total records: {len(df_final)}")
print(f"   - Date range: {df_final['Date'].min()} ~ {df_final['Date'].max()}")
print(f"   - Columns: {len(df_final.columns)}")

print(f"\n   Column list:")
for col in df_final.columns:
    null_count = df_final[col].isnull().sum()
    null_pct = (null_count / len(df_final)) * 100
    print(f"      - {col:<40} (Null: {null_count:>4} / {null_pct:>5.1f}%)")

# ========================================
# 10. Summary
# ========================================
print(f"\n{'='*80}")
print("📊 Collection Summary")
print(f"{'='*80}\n")

summary = f"""
✅ CryptoQuant 데이터 수집 완료!

수집된 데이터:
- Open Interest: {len(df_oi) if len(df_oi) > 0 else 0} records
- Funding Rate: {len(df_funding) if len(df_funding) > 0 else 0} records
- Liquidations: {len(df_liquidations) if len(df_liquidations) > 0 else 0} records
- Exchange Netflow: {len(df_netflow) if len(df_netflow) > 0 else 0} records
- Miner Netflow: {len(df_miner) if len(df_miner) > 0 else 0} records
- Taker Volume: {len(df_taker) if len(df_taker) > 0 else 0} records

💾 저장된 파일: {output_file}

🎯 다음 단계:
1. 기존 integrated_data_full_v2.csv와 병합
2. V3 데이터셋 생성 (138 + CryptoQuant 변수)
3. ElasticNet V3 학습 및 백테스팅

⚠️  7일 무료 체험 기간 내에:
- 모든 필요한 데이터를 지금 수집하세요
- 추가로 필요한 지표가 있다면 지금 가져오세요
- 데이터 수집 완료 후 구독 취소 가능
"""

print(summary)

print(f"\n{'='*80}")
print("✅ Complete!")
print(f"{'='*80}")
