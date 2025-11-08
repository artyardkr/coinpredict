import pandas as pd
from fredapi import Fred
import os
import ssl
import certifi

# SSL 인증서 설정
ssl._create_default_https_context = ssl._create_unverified_context

print("=" * 70)
print("Step 1.3: 거시경제 데이터 수집 (FRED API)")
print("=" * 70)

# FRED API 키 설정
# 방법 1: 환경변수에서 가져오기
# 방법 2: 직접 입력 (보안 주의)
FRED_API_KEY = 'cab424ac8d2ceb949264c8dd49b606f7'

if FRED_API_KEY == 'YOUR_API_KEY_HERE':
    print("\n⚠️  경고: FRED API 키가 설정되지 않았습니다.")
    print("다음 중 하나를 선택하세요:")
    print("  1. 환경변수 설정: export FRED_API_KEY='your_key_here'")
    print("  2. 스크립트에서 직접 수정: FRED_API_KEY = 'your_key_here'")
    print("\nFRED API 키 발급: https://fred.stlouisfed.org/docs/api/api_key.html")
    exit(1)

# FRED API 초기화
fred = Fred(api_key=FRED_API_KEY)

# 수집 기간
start_date = "2021-01-01"
end_date = "2025-10-15"

print(f"\n수집 기간: {start_date} ~ {end_date}")

# 수집할 거시경제 지표
indicators = {
    'DGS10': '10년 만기 국채 수익률',
    'DFF': '연방기금금리 (Federal Funds Rate)',
    'CPIAUCSL': '소비자물가지수 (CPI)',
    'UNRATE': '실업률',
    'M2SL': 'M2 통화공급량',
    'GDP': '미국 GDP',
    'DEXUSEU': 'USD/EUR 환율',
    'DTWEXBGS': '달러 인덱스 (Broad)',
    'T10Y2Y': '10년-2년 국채 스프레드',
    'VIXCLS': 'VIX 변동성 지수',
}

print(f"\n수집할 지표: {len(indicators)}개")
print("-" * 70)

macro_data = {}
success_count = 0
fail_count = 0

for series_id, name in indicators.items():
    try:
        print(f"다운로드 중: {series_id:12} - {name}")

        # FRED에서 데이터 가져오기
        data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)

        if len(data) > 0:
            macro_data[series_id] = data
            success_count += 1
            print(f"  ✓ 성공: {len(data)}개 데이터")
        else:
            print(f"  ✗ 실패: 데이터 없음")
            fail_count += 1

    except Exception as e:
        print(f"  ✗ 오류: {e}")
        fail_count += 1

print("\n" + "=" * 70)
print("다운로드 완료")
print("=" * 70)
print(f"성공: {success_count}개")
print(f"실패: {fail_count}개")

# DataFrame으로 통합
macro_df = pd.DataFrame(macro_data)

print(f"\n통합 데이터 형태: {macro_df.shape}")
print(f"기간: {macro_df.index[0]} ~ {macro_df.index[-1]}")

# 결측치 확인
null_counts = macro_df.isnull().sum()
print(f"\n결측치:")
for col in macro_df.columns:
    if null_counts[col] > 0:
        print(f"  {col}: {null_counts[col]}개 ({null_counts[col]/len(macro_df)*100:.1f}%)")

# Forward fill (일별 데이터로 변환 - 월별/분기별 데이터 보간)
print("\n일별 데이터로 리샘플링 중...")
macro_df = macro_df.resample('D').ffill()

# 남은 결측치 처리
macro_df = macro_df.fillna(method='ffill')
macro_df = macro_df.fillna(method='bfill')

print(f"리샘플링 후: {macro_df.shape}")

# 파일 저장
macro_df.to_csv('fred_macro_data.csv')
print("\n✓ 저장 완료: fred_macro_data.csv")

print("\n" + "=" * 70)
print("수집된 거시경제 지표 요약")
print("=" * 70)

# 각 지표 기본 통계
for col in macro_df.columns:
    print(f"\n{col} ({indicators.get(col, col)}):")
    print(f"  평균: {macro_df[col].mean():.2f}")
    print(f"  최소: {macro_df[col].min():.2f}")
    print(f"  최대: {macro_df[col].max():.2f}")
    print(f"  표준편차: {macro_df[col].std():.2f}")
    print(f"  최신값 ({macro_df.index[-1].date()}): {macro_df[col].iloc[-1]:.2f}")

print("\n" + "=" * 70)
print("Step 1.3 완료!")
print("=" * 70)
