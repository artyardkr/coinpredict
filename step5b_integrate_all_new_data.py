import pandas as pd
import numpy as np

print("=" * 70)
print("Step 5b: 모든 신규 데이터 통합")
print("=" * 70)

# ===== 기존 데이터 로드 =====
print("\n1. 기존 통합 데이터 로드 중...")
print("-" * 70)

integrated_df = pd.read_csv('integrated_data_full.csv', index_col=0, parse_dates=True)
integrated_df.index = pd.to_datetime(integrated_df.index).tz_localize(None)
print(f"✓ 기존 데이터: {integrated_df.shape}")

# ===== 신규 데이터 로드 =====
print("\n2. 신규 데이터 로드 중...")
print("-" * 70)

# 추가 전통시장 데이터
try:
    additional_markets = pd.read_csv('additional_market_data.csv', index_col=0, parse_dates=True)
    additional_markets.index = pd.to_datetime(additional_markets.index).tz_localize(None)
    print(f"✓ 추가 전통시장: {additional_markets.shape} - {list(additional_markets.columns)}")
except Exception as e:
    print(f"✗ 추가 전통시장 로드 실패: {e}")
    additional_markets = None

# Fed 유동성 지표
try:
    fed_liquidity = pd.read_csv('fed_liquidity_data.csv', index_col=0, parse_dates=True)
    fed_liquidity.index = pd.to_datetime(fed_liquidity.index).tz_localize(None)
    print(f"✓ Fed 유동성: {fed_liquidity.shape} - {list(fed_liquidity.columns)}")
except Exception as e:
    print(f"✗ Fed 유동성 로드 실패: {e}")
    fed_liquidity = None

# 고급 온체인 지표
try:
    advanced_onchain = pd.read_csv('advanced_onchain_data.csv', index_col=0, parse_dates=True)
    advanced_onchain.index = pd.to_datetime(advanced_onchain.index).tz_localize(None)
    print(f"✓ 고급 온체인: {advanced_onchain.shape} - {advanced_onchain.columns.tolist()[:5]}...")
except Exception as e:
    print(f"✗ 고급 온체인 로드 실패: {e}")
    advanced_onchain = None

# Bitcoin ETF 데이터
try:
    btc_etf = pd.read_csv('bitcoin_etf_data.csv', index_col=0, parse_dates=True)
    btc_etf.index = pd.to_datetime(btc_etf.index).tz_localize(None)
    print(f"✓ Bitcoin ETF: {btc_etf.shape} - {list(btc_etf.columns)}")
except Exception as e:
    print(f"✗ Bitcoin ETF 로드 실패: {e}")
    btc_etf = None

# ===== 데이터 통합 =====
print("\n3. 데이터 통합 중...")
print("-" * 70)

original_cols = len(integrated_df.columns)

# 추가 전통시장 결합
if additional_markets is not None:
    integrated_df = integrated_df.join(additional_markets, how='left')
    print(f"+ 추가 전통시장: {integrated_df.shape} (+{additional_markets.shape[1]}개 변수)")

# Fed 유동성 결합
if fed_liquidity is not None:
    integrated_df = integrated_df.join(fed_liquidity, how='left')
    print(f"+ Fed 유동성: {integrated_df.shape} (+{fed_liquidity.shape[1]}개 변수)")

# 고급 온체인 결합
if advanced_onchain is not None:
    integrated_df = integrated_df.join(advanced_onchain, how='left')
    print(f"+ 고급 온체인: {integrated_df.shape} (+{advanced_onchain.shape[1]}개 변수)")

# Bitcoin ETF 결합
if btc_etf is not None:
    integrated_df = integrated_df.join(btc_etf, how='left')
    print(f"+ Bitcoin ETF: {integrated_df.shape} (+{btc_etf.shape[1]}개 변수)")

new_cols = len(integrated_df.columns)
added_cols = new_cols - original_cols

print(f"\n통합 완료: {integrated_df.shape}")
print(f"  - 기존 변수: {original_cols}개")
print(f"  - 신규 변수: {added_cols}개")
print(f"  - 총 변수: {new_cols}개")
print(f"  - 기간: {integrated_df.index[0].date()} ~ {integrated_df.index[-1].date()}")

# ===== 결측치 확인 =====
print("\n4. 결측치 확인")
print("-" * 70)

null_counts = integrated_df.isnull().sum()
null_pct = (null_counts / len(integrated_df) * 100).round(2)

high_null_cols = null_counts[null_counts > 0].sort_values(ascending=False).head(30)
if len(high_null_cols) > 0:
    print(f"\n결측치 상위 30개 컬럼:")
    for col, count in high_null_cols.items():
        pct = null_pct[col]
        print(f"  {col:40} : {count:4}개 ({pct:5.1f}%)")

# ===== 결측치 처리 =====
print("\n5. 결측치 처리 중...")
print("-" * 70)

before_len = len(integrated_df)

# Forward fill (금융 데이터는 주말/공휴일에 값이 동결됨)
print("  - Forward fill 적용...")
integrated_df = integrated_df.ffill()

# Backward fill (초기 데이터 처리)
print("  - Backward fill 적용...")
integrated_df = integrated_df.bfill()

# 남은 결측치 확인
remaining_nulls = integrated_df.isnull().sum().sum()
print(f"  - 남은 결측치: {remaining_nulls}개")

if remaining_nulls > 0:
    # 결측치가 많은 컬럼 제거 (80% 이상)
    null_threshold = 0.8
    cols_to_drop = null_counts[null_counts / len(integrated_df) > null_threshold].index.tolist()

    if cols_to_drop:
        print(f"\n  - 결측치 {null_threshold*100}% 이상 컬럼 제거: {len(cols_to_drop)}개")
        for col in cols_to_drop:
            print(f"    {col} ({null_counts[col]/len(integrated_df)*100:.1f}%)")
        integrated_df = integrated_df.drop(columns=cols_to_drop)

    # 남은 결측치가 있는 행 제거
    remaining_nulls_after = integrated_df.isnull().sum().sum()
    if remaining_nulls_after > 0:
        print(f"  - 결측치가 있는 행 제거...")
        integrated_df = integrated_df.dropna()
        after_len = len(integrated_df)
        print(f"    {before_len - after_len}개 행 제거됨")

print(f"\n결측치 처리 후: {integrated_df.shape}")

# ===== 데이터 검증 =====
print("\n6. 데이터 검증")
print("-" * 70)

# 무한대/비정상 값 확인
inf_check = np.isinf(integrated_df.select_dtypes(include=[np.number])).sum().sum()
print(f"무한대 값: {inf_check}개")

if inf_check > 0:
    print("  - 무한대 값을 NaN으로 변환...")
    integrated_df = integrated_df.replace([np.inf, -np.inf], np.nan)
    print("  - NaN 행 제거...")
    integrated_df = integrated_df.dropna()
    print(f"    최종 데이터: {integrated_df.shape}")

# 기본 통계
print(f"\n데이터 통계:")
print(f"  - 총 행: {len(integrated_df):,}개")
print(f"  - 총 컬럼: {len(integrated_df.columns)}개")
print(f"  - 기간: {integrated_df.index[0].date()} ~ {integrated_df.index[-1].date()}")
print(f"  - 일수: {(integrated_df.index[-1] - integrated_df.index[0]).days}일")

# ===== 파일 저장 =====
print("\n7. 파일 저장 중...")
print("-" * 70)

# 최종 통합 데이터 저장
integrated_df.to_csv('integrated_data_full_v2.csv')
print("✓ 저장 완료: integrated_data_full_v2.csv")

# 변수 증가 요약
print("\n" + "=" * 70)
print("신규 변수 요약")
print("=" * 70)

new_var_categories = {
    '추가 전통시장': ['DIA', 'IWM', 'TLT', 'DXY', 'ETH', 'GLD', 'HYG', 'LQD', 'VIX'],
    'Fed 유동성': ['WALCL', 'RRPONTSYD', 'WTREGEN', 'T10Y3M', 'SOFR', 'BAMLH0A0HYM2', 'BAMLC0A0CM', 'FED_NET_LIQUIDITY'],
    '고급 온체인': [c for c in integrated_df.columns if any(x in c for x in ['NVT', 'Puell', 'Hash_Ribbon', 'Difficulty', 'Miner', 'Hash_Price', 'Active_Addresses', 'Fee_Per_Tx', 'Mempool_Stress', 'Price_to_MA200'])],
    'Bitcoin ETF': [c for c in integrated_df.columns if any(x in c for x in ['IBIT', 'FBTC', 'GBTC', 'ARKB', 'BITB', 'ETF'])],
}

total_new_vars = 0
for category, vars in new_var_categories.items():
    actual_vars = [v for v in vars if v in integrated_df.columns]
    count = len(actual_vars)
    total_new_vars += count
    print(f"  {category:20} : {count:3}개")

print(f"\n  {'총 신규 변수':20} : {total_new_vars:3}개")
print(f"  {'기존 변수':20} : {original_cols:3}개")
print(f"  {'최종 변수':20} : {len(integrated_df.columns):3}개")

# ===== 완료 =====
print("\n" + "=" * 70)
print("Step 5b 완료!")
print("=" * 70)
print(f"\n✅ {added_cols}개의 신규 변수 추가 완료!")
print(f"✅ 최종 데이터셋: {integrated_df.shape}")
print(f"   - {len(integrated_df):,}개 샘플")
print(f"   - {len(integrated_df.columns)}개 특성")
print(f"   - 기간: {integrated_df.index[0].date()} ~ {integrated_df.index[-1].date()}")
print("\n다음 단계: 새로운 integrated_data_full_v2.csv로 모델 재학습")
print("=" * 70)
