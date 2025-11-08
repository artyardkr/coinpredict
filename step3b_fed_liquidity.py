import pandas as pd
from fredapi import Fred
from datetime import datetime
import ssl

# SSL 인증서 설정
ssl._create_default_https_context = ssl._create_unverified_context

print("=" * 70)
print("Step 3b: Fed 유동성 지표 수집")
print("=" * 70)

# FRED API 키 (기존 step3에서 사용한 것과 동일)
api_key = 'cab424ac8d2ceb949264c8dd49b606f7'
fred = Fred(api_key=api_key)

# 데이터 기간
start_date = '2021-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')

print(f"\n데이터 기간: {start_date} ~ {end_date}")
print("-" * 70)

# Fed 유동성 관련 지표
fed_indicators = {
    'WALCL': 'Fed 총자산 (Total Assets)',
    'RRPONTSYD': '역레포 (Reverse Repo)',
    'WTREGEN': '재무부 계정 TGA (Treasury General Account)',
    'T10Y3M': '10년-3개월 스프레드 (10Y-3M Spread)',
    'SOFR': '담보부 익일물 금리 (Secured Overnight Financing Rate)',
    'BAMLH0A0HYM2': '하이일드 스프레드 (High Yield Spread)',
    'BAMLC0A0CM': '투자등급 스프레드 (Investment Grade Spread)',
}

# 데이터 수집
fed_data = pd.DataFrame()

for code, description in fed_indicators.items():
    print(f"\n{code:15} - {description}")
    try:
        data = fred.get_series(code, observation_start=start_date, observation_end=end_date)

        if len(data) > 0:
            fed_data[code] = data
            print(f"  ✓ 성공: {len(data)}개 데이터")
        else:
            print(f"  ✗ 실패: 데이터 없음")

    except Exception as e:
        print(f"  ✗ 에러: {e}")

# 실제 유동성 계산 (WALCL - RRPONTSYD - WTREGEN)
if 'WALCL' in fed_data.columns and 'RRPONTSYD' in fed_data.columns:
    if 'WTREGEN' in fed_data.columns:
        fed_data['FED_NET_LIQUIDITY'] = fed_data['WALCL'] - fed_data['RRPONTSYD'] - fed_data['WTREGEN']
        print(f"\n{'FED_NET_LIQUIDITY':15} - Fed 순유동성 (WALCL - RRP - TGA)")
        print(f"  ✓ 계산 완료: {fed_data['FED_NET_LIQUIDITY'].notna().sum()}개 데이터")
    else:
        fed_data['FED_NET_LIQUIDITY'] = fed_data['WALCL'] - fed_data['RRPONTSYD']
        print(f"\n{'FED_NET_LIQUIDITY':15} - Fed 순유동성 (WALCL - RRP)")
        print(f"  ✓ 계산 완료: {fed_data['FED_NET_LIQUIDITY'].notna().sum()}개 데이터")

# 데이터 정보
print("\n" + "=" * 70)
print("수집 완료")
print("=" * 70)
print(f"총 변수: {len(fed_data.columns)}개")
print(f"총 행: {len(fed_data):,}개")
if len(fed_data) > 0:
    print(f"기간: {fed_data.index[0].date()} ~ {fed_data.index[-1].date()}")
else:
    print("❌ 데이터 수집 실패")

# 결측치 확인
print("\n결측치 확인:")
null_counts = fed_data.isnull().sum()
for col in fed_data.columns:
    null_count = null_counts[col]
    null_pct = (null_count / len(fed_data) * 100)
    print(f"  {col:20} : {null_count:4}개 ({null_pct:5.2f}%)")

# 파일 저장
fed_data.to_csv('fed_liquidity_data.csv')
print("\n✓ 저장 완료: fed_liquidity_data.csv")

# 기본 통계
if len(fed_data.columns) > 0:
    print("\n" + "=" * 70)
    print("기본 통계")
    print("=" * 70)
    print(fed_data.describe())

# 최근 유동성 변화 (최근 30일)
if 'FED_NET_LIQUIDITY' in fed_data.columns:
    recent_liquidity = fed_data['FED_NET_LIQUIDITY'].dropna()
    if len(recent_liquidity) > 30:
        latest = recent_liquidity.iloc[-1]
        month_ago = recent_liquidity.iloc[-30]
        change = latest - month_ago
        change_pct = (change / month_ago * 100)

        print("\n" + "=" * 70)
        print("최근 Fed 순유동성 변화 (30일)")
        print("=" * 70)
        print(f"현재:     ${latest/1e12:.2f}T")
        print(f"30일 전:  ${month_ago/1e12:.2f}T")
        print(f"변화:     ${change/1e12:.2f}T ({change_pct:+.2f}%)")
        if change > 0:
            print("📈 유동성 증가 (BTC 강세 요인)")
        else:
            print("📉 유동성 감소 (BTC 약세 요인)")

print("\n" + "=" * 70)
print("Step 3b 완료!")
print("=" * 70)
