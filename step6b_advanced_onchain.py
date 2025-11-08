import pandas as pd
import numpy as np

print("=" * 70)
print("Step 6b: 고급 온체인 지표 계산")
print("=" * 70)

# 기존 데이터 로드
print("\n기존 데이터 로드 중...")
btc_data = pd.read_csv('btc_technical_indicators.csv', index_col=0, parse_dates=True)
onchain_data = pd.read_csv('onchain_data.csv', index_col=0, parse_dates=True)

# 타임존 제거
btc_data.index = pd.to_datetime(btc_data.index).tz_localize(None)
onchain_data.index = pd.to_datetime(onchain_data.index).tz_localize(None)

# 데이터 결합
df = btc_data[['Close', 'Volume']].join(onchain_data, how='left')

print(f"✓ BTC 데이터: {btc_data.shape}")
print(f"✓ 온체인 데이터: {onchain_data.shape}")
print(f"✓ 결합 데이터: {df.shape}")

print("\n" + "=" * 70)
print("고급 온체인 지표 계산 중...")
print("=" * 70)

advanced_onchain = pd.DataFrame(index=df.index)

# ===== 1. NVT Ratio (Network Value to Transactions) =====
if 'bc_market_cap' in df.columns and 'bc_n_transactions' in df.columns:
    # 일일 거래량 USD (근사치: transactions * avg value)
    # bc_market_cap / 거래 수
    advanced_onchain['NVT_Ratio'] = df['bc_market_cap'] / (df['bc_n_transactions'] + 1)
    advanced_onchain['NVT_Ratio_MA90'] = advanced_onchain['NVT_Ratio'].rolling(90).mean()
    print("✓ NVT Ratio 계산 완료")

# ===== 2. Puell Multiple (채굴 수익성) =====
if 'bc_miners_revenue' in df.columns:
    revenue_ma365 = df['bc_miners_revenue'].rolling(365).mean()
    advanced_onchain['Puell_Multiple'] = df['bc_miners_revenue'] / (revenue_ma365 + 1)
    print("✓ Puell Multiple 계산 완료")

# ===== 3. Hash Ribbon (해시레이트 추세) =====
if 'bc_hash_rate' in df.columns:
    # 30일 / 60일 MA 교차
    hash_ma30 = df['bc_hash_rate'].rolling(30).mean()
    hash_ma60 = df['bc_hash_rate'].rolling(60).mean()
    advanced_onchain['Hash_Ribbon_MA30'] = hash_ma30
    advanced_onchain['Hash_Ribbon_MA60'] = hash_ma60
    advanced_onchain['Hash_Ribbon_Spread'] = (hash_ma30 - hash_ma60) / (hash_ma60 + 1) * 100
    print("✓ Hash Ribbon 계산 완료")

# ===== 4. Miner Position Index (채굴자 포지션) =====
if 'bc_miners_revenue' in df.columns and 'bc_market_cap' in df.columns:
    # 채굴 수익 / 시가총액 비율
    advanced_onchain['Miner_Revenue_to_Cap'] = df['bc_miners_revenue'] / (df['bc_market_cap'] + 1) * 100
    advanced_onchain['Miner_Revenue_to_Cap_MA30'] = advanced_onchain['Miner_Revenue_to_Cap'].rolling(30).mean()
    print("✓ Miner Position Index 계산 완료")

# ===== 5. Network Momentum (네트워크 모멘텀) =====
if 'bc_n_unique_addresses' in df.columns:
    # 활성 주소 변화율
    advanced_onchain['Active_Addresses_Change'] = df['bc_n_unique_addresses'].pct_change(30) * 100
    advanced_onchain['Active_Addresses_MA90'] = df['bc_n_unique_addresses'].rolling(90).mean()
    print("✓ Network Momentum 계산 완료")

# ===== 6. Transaction Value Momentum =====
if 'bc_transaction_fees' in df.columns and 'bc_n_transactions' in df.columns:
    # 평균 거래 수수료
    advanced_onchain['Avg_Fee_Per_Tx'] = df['bc_transaction_fees'] / (df['bc_n_transactions'] + 1)
    advanced_onchain['Avg_Fee_Per_Tx_MA30'] = advanced_onchain['Avg_Fee_Per_Tx'].rolling(30).mean()
    print("✓ Transaction Value Momentum 계산 완료")

# ===== 7. Difficulty Ribbon (난이도 추세) =====
if 'bc_difficulty' in df.columns:
    diff_ma30 = df['bc_difficulty'].rolling(30).mean()
    diff_ma60 = df['bc_difficulty'].rolling(60).mean()
    diff_ma90 = df['bc_difficulty'].rolling(90).mean()
    advanced_onchain['Difficulty_MA30'] = diff_ma30
    advanced_onchain['Difficulty_MA60'] = diff_ma60
    advanced_onchain['Difficulty_MA90'] = diff_ma90
    advanced_onchain['Difficulty_Compression'] = (diff_ma30 - diff_ma90) / (diff_ma90 + 1) * 100
    print("✓ Difficulty Ribbon 계산 완료")

# ===== 8. Simplified MVRV (간이 MVRV) =====
# 진짜 MVRV는 realized price가 필요하지만, 200일 MA를 "fair value"로 근사
if 'Close' in df.columns:
    price_ma200 = df['Close'].rolling(200).mean()
    advanced_onchain['Price_to_MA200'] = df['Close'] / (price_ma200 + 1)
    advanced_onchain['Price_MA200'] = price_ma200
    print("✓ Simplified MVRV (Price/MA200) 계산 완료")

# ===== 9. Mempool Stress (멤풀 압력) =====
if 'bc_mempool_size' in df.columns:
    mempool_ma30 = df['bc_mempool_size'].rolling(30).mean()
    advanced_onchain['Mempool_Stress'] = df['bc_mempool_size'] / (mempool_ma30 + 1)
    print("✓ Mempool Stress 계산 완료")

# ===== 10. Hash Price (해시당 수익) =====
if 'bc_miners_revenue' in df.columns and 'bc_hash_rate' in df.columns:
    # USD per TH/s
    advanced_onchain['Hash_Price'] = df['bc_miners_revenue'] / (df['bc_hash_rate'] + 1)
    advanced_onchain['Hash_Price_MA90'] = advanced_onchain['Hash_Price'].rolling(90).mean()
    print("✓ Hash Price 계산 완료")

# 데이터 정보
print("\n" + "=" * 70)
print("계산 완료")
print("=" * 70)
print(f"총 새로운 지표: {len(advanced_onchain.columns)}개")
print(f"총 행: {len(advanced_onchain):,}개")
print(f"기간: {advanced_onchain.index[0].date()} ~ {advanced_onchain.index[-1].date()}")

# 결측치 확인
print("\n결측치 확인 (상위 10개):")
null_counts = advanced_onchain.isnull().sum().sort_values(ascending=False).head(10)
for col, count in null_counts.items():
    pct = (count / len(advanced_onchain) * 100)
    print(f"  {col:35} : {count:4}개 ({pct:5.2f}%)")

# 파일 저장
advanced_onchain.to_csv('advanced_onchain_data.csv')
print("\n✓ 저장 완료: advanced_onchain_data.csv")

# 컬럼 목록
print("\n" + "=" * 70)
print("계산된 지표 목록")
print("=" * 70)

categories = {
    'NVT Ratio': [c for c in advanced_onchain.columns if 'NVT' in c],
    'Puell Multiple': [c for c in advanced_onchain.columns if 'Puell' in c],
    'Hash Ribbon': [c for c in advanced_onchain.columns if 'Hash_Ribbon' in c],
    'Difficulty Ribbon': [c for c in advanced_onchain.columns if 'Difficulty' in c],
    'Miner Position': [c for c in advanced_onchain.columns if 'Miner' in c or 'Hash_Price' in c],
    'Network Activity': [c for c in advanced_onchain.columns if 'Active_Addresses' in c or 'Fee' in c],
    'Mempool': [c for c in advanced_onchain.columns if 'Mempool' in c],
    'Valuation': [c for c in advanced_onchain.columns if 'Price_to' in c or 'Price_MA' in c],
}

for category, cols in categories.items():
    if cols:
        print(f"\n{category} ({len(cols)}개):")
        for col in cols:
            print(f"  - {col}")

# 기본 통계 (주요 지표만)
key_indicators = ['NVT_Ratio', 'Puell_Multiple', 'Price_to_MA200', 'Hash_Price', 'Active_Addresses_Change']
existing_indicators = [col for col in key_indicators if col in advanced_onchain.columns]

if existing_indicators:
    print("\n" + "=" * 70)
    print("주요 지표 통계")
    print("=" * 70)
    print(advanced_onchain[existing_indicators].describe())

print("\n" + "=" * 70)
print("Step 6b 완료!")
print("=" * 70)
print(f"\n✅ {len(advanced_onchain.columns)}개의 고급 온체인 지표 생성 완료")
