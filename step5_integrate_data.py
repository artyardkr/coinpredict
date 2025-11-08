import pandas as pd
import numpy as np

print("=" * 70)
print("Step 1.5: 모든 데이터 통합 및 전처리")
print("=" * 70)

# ===== 데이터 로드 =====
print("\n1. 데이터 로드 중...")
print("-" * 70)

# BTC 기술적 지표
btc_tech = pd.read_csv('btc_technical_indicators.csv', index_col=0, parse_dates=True)
btc_tech.index = pd.to_datetime(btc_tech.index).tz_localize(None)
print(f"✓ BTC 기술적 지표: {btc_tech.shape} (컬럼 {len(btc_tech.columns)}개)")

# 전통 시장 지수
traditional = pd.read_csv('traditional_market_indices.csv', index_col=0, parse_dates=True)
traditional.index = pd.to_datetime(traditional.index).tz_localize(None)
print(f"✓ 전통 시장 지수: {traditional.shape} (컬럼 {len(traditional.columns)}개)")

# 거시경제 데이터
macro = pd.read_csv('fred_macro_data.csv', index_col=0, parse_dates=True)
macro.index = pd.to_datetime(macro.index).tz_localize(None)
print(f"✓ 거시경제 데이터: {macro.shape} (컬럼 {len(macro.columns)}개)")

# 감정/관심 지표
sentiment = pd.read_csv('sentiment_data.csv', index_col=0, parse_dates=True)
sentiment.index = pd.to_datetime(sentiment.index).tz_localize(None)
print(f"✓ 감정/관심 지표: {sentiment.shape} (컬럼 {len(sentiment.columns)}개)")

# 온체인 데이터
onchain = pd.read_csv('onchain_data.csv', index_col=0, parse_dates=True)
onchain.index = pd.to_datetime(onchain.index).tz_localize(None)
print(f"✓ 온체인 데이터: {onchain.shape} (컬럼 {len(onchain.columns)}개)")

# ===== 데이터 통합 =====
print("\n2. 데이터 통합 중...")
print("-" * 70)

# BTC 기술적 지표를 기준으로 시작 (outer join)
integrated_df = btc_tech.copy()
print(f"기준 데이터: {integrated_df.shape}")

# 전통 시장 지수 결합
integrated_df = integrated_df.join(traditional, how='left')
print(f"+ 전통 시장: {integrated_df.shape}")

# 거시경제 데이터 결합
integrated_df = integrated_df.join(macro, how='left')
print(f"+ 거시경제: {integrated_df.shape}")

# 감정/관심 지표 결합 (fear_greed_classification 제외 - 텍스트 데이터)
sentiment_numeric = sentiment[['fear_greed_index', 'google_trends_btc']]
integrated_df = integrated_df.join(sentiment_numeric, how='left')
print(f"+ 감정/관심: {integrated_df.shape}")

# 온체인 데이터 결합
integrated_df = integrated_df.join(onchain, how='left')
print(f"+ 온체인: {integrated_df.shape}")

print(f"\n통합 완료: {integrated_df.shape}")
print(f"  - 총 컬럼: {len(integrated_df.columns)}개")
print(f"  - 기간: {integrated_df.index[0].date()} ~ {integrated_df.index[-1].date()}")

# ===== 결측치 확인 =====
print("\n3. 결측치 확인")
print("-" * 70)

null_counts = integrated_df.isnull().sum()
null_pct = (null_counts / len(integrated_df) * 100).round(2)

print(f"결측치가 있는 컬럼: {(null_counts > 0).sum()}개")

# 결측치가 많은 컬럼 출력
high_null_cols = null_counts[null_counts > 0].sort_values(ascending=False).head(20)
if len(high_null_cols) > 0:
    print("\n결측치 상위 20개 컬럼:")
    for col, count in high_null_cols.items():
        pct = null_pct[col]
        print(f"  {col:30} : {count:4}개 ({pct:5.1f}%)")

# ===== 결측치 처리 =====
print("\n4. 결측치 처리 중...")
print("-" * 70)

# 기본 전략:
# 1. Forward fill (금융 시장 데이터는 주말/공휴일에 동결)
# 2. Backward fill (초기 데이터 처리)
# 3. 여전히 NaN이 남으면 행 제거

print("  - Forward fill 적용...")
integrated_df = integrated_df.ffill()

print("  - Backward fill 적용...")
integrated_df = integrated_df.bfill()

# 남은 결측치 확인
remaining_nulls = integrated_df.isnull().sum().sum()
print(f"  - 남은 결측치: {remaining_nulls}개")

if remaining_nulls > 0:
    print("  - 결측치가 있는 행 제거...")
    before_len = len(integrated_df)
    integrated_df = integrated_df.dropna()
    after_len = len(integrated_df)
    print(f"    {before_len - after_len}개 행 제거됨")

print(f"\n결측치 처리 후: {integrated_df.shape}")

# ===== 데이터 검증 =====
print("\n5. 데이터 검증")
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
print("\n6. 파일 저장 중...")
print("-" * 70)

# 최종 통합 데이터 저장
integrated_df.to_csv('integrated_data_full.csv')
print("✓ 저장 완료: integrated_data_full.csv")

# 컬럼 목록 저장
with open('integrated_data_columns.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("통합 데이터 컬럼 목록\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"총 {len(integrated_df.columns)}개 컬럼\n\n")

    # 카테고리별 컬럼
    categories = {
        'BTC 원본 데이터': ['Open', 'High', 'Low', 'Close', 'Volume'],
        '이동평균 (EMA/SMA)': [c for c in integrated_df.columns if 'EMA' in c or 'SMA' in c],
        '모멘텀 지표': [c for c in integrated_df.columns if any(x in c for x in ['RSI', 'Stoch', 'Williams', 'ROC', 'MFI'])],
        '트렌드 지표': [c for c in integrated_df.columns if any(x in c for x in ['MACD', 'ADX', 'CCI'])],
        '변동성 지표': [c for c in integrated_df.columns if any(x in c for x in ['BB', 'ATR', 'volatility'])],
        '거래량 지표': [c for c in integrated_df.columns if 'OBV' in c or ('volume' in c and 'EMA' not in c)],
        '수익률': [c for c in integrated_df.columns if 'return' in c],
        '전통 시장': ['QQQ', 'SPX', 'UUP', 'EURUSD', 'GOLD', 'SILVER', 'OIL', 'BSV'],
        '거시경제': [c for c in integrated_df.columns if c in ['DGS10', 'DFF', 'CPIAUCSL', 'UNRATE', 'M2SL', 'GDP', 'DEXUSEU', 'DTWEXBGS', 'T10Y2Y', 'VIXCLS']],
        '감정/관심': [c for c in integrated_df.columns if any(x in c for x in ['fear_greed', 'google_trends'])],
        '온체인': [c for c in integrated_df.columns if c.startswith('bc_') or c.startswith('cm_') or c.startswith('gn_')]
    }

    for category, cols in categories.items():
        if cols:
            f.write(f"\n{category} ({len(cols)}개):\n")
            for col in cols:
                f.write(f"  - {col}\n")

print("✓ 저장 완료: integrated_data_columns.txt")

# ===== 요약 통계 =====
print("\n" + "=" * 70)
print("카테고리별 컬럼 수")
print("=" * 70)

categories_summary = {
    'BTC 원본 데이터': 5,
    '기술적 지표': len([c for c in integrated_df.columns if any(x in c for x in ['EMA', 'SMA', 'RSI', 'MACD', 'BB', 'ATR', 'OBV', 'Stoch', 'ADX', 'CCI', 'Williams', 'ROC', 'MFI'])]),
    '수익률/변동성': len([c for c in integrated_df.columns if any(x in c for x in ['return', 'volatility', 'volume_change', 'market_cap_approx'])]),
    '전통 시장': 8,
    '거시경제': 10,
    '감정/관심': 2,
    '온체인': len([c for c in integrated_df.columns if c.startswith('bc_') or c.startswith('cm_') or c.startswith('gn_')])
}

for category, count in categories_summary.items():
    print(f"  {category:20} : {count:3}개")

print(f"\n  {'총 컬럼':20} : {len(integrated_df.columns):3}개")

# ===== Phase 1 완료 =====
print("\n" + "=" * 70)
print("Step 1.5 완료!")
print("=" * 70)
print("\n✅ Phase 1: 데이터 수집 및 전처리 완료!")
print(f"\n최종 데이터셋: {integrated_df.shape}")
print(f"  - {len(integrated_df):,}개 샘플")
print(f"  - {len(integrated_df.columns)}개 특성")
print(f"  - 기간: {integrated_df.index[0].date()} ~ {integrated_df.index[-1].date()}")
print("\n다음 단계: Phase 2 - 특성 엔지니어링 (FRA 알고리즘)")
print("=" * 70)
