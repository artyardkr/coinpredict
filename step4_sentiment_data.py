import pandas as pd
import requests
from pytrends.request import TrendReq
import time
from datetime import datetime, timedelta

print("=" * 70)
print("Step 1.4: 감정/관심 지표 수집")
print("=" * 70)

# 수집 기간
start_date = "2021-01-01"
end_date = "2025-10-15"

print(f"\n수집 기간: {start_date} ~ {end_date}")
print("-" * 70)

# ===== Part 1: Fear & Greed Index =====
print("\n[1/2] Fear & Greed Index 수집 중...")

try:
    # Alternative.me API
    # limit=0 은 모든 데이터를 가져옴
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        fng_data = data['data']

        # DataFrame으로 변환
        fng_df = pd.DataFrame(fng_data)

        # timestamp를 datetime으로 변환
        fng_df['date'] = pd.to_datetime(fng_df['timestamp'].astype(int), unit='s')
        fng_df['fear_greed_index'] = fng_df['value'].astype(int)
        fng_df['fear_greed_classification'] = fng_df['value_classification']

        # 필요한 컬럼만 선택
        fng_df = fng_df[['date', 'fear_greed_index', 'fear_greed_classification']]
        fng_df = fng_df.set_index('date').sort_index()

        # 기간 필터링
        fng_df = fng_df[start_date:end_date]

        print(f"  ✓ 성공: {len(fng_df)}개 데이터")
        print(f"    기간: {fng_df.index[0].date()} ~ {fng_df.index[-1].date()}")
        print(f"    평균: {fng_df['fear_greed_index'].mean():.2f}")
        print(f"    최소: {fng_df['fear_greed_index'].min()}")
        print(f"    최대: {fng_df['fear_greed_index'].max()}")
    else:
        print(f"  ✗ 실패: HTTP {response.status_code}")
        fng_df = pd.DataFrame()

except Exception as e:
    print(f"  ✗ 오류: {e}")
    fng_df = pd.DataFrame()

# ===== Part 2: Google Trends =====
print("\n[2/2] Google Trends (BTC 검색량) 수집 중...")

try:
    # pytrends 초기화
    pytrends = TrendReq(hl='en-US', tz=0)

    # 검색 키워드
    keywords = ['Bitcoin', 'BTC']

    all_trends = []

    # 날짜 범위를 분할 (Google Trends API 제약)
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    # 1년 단위로 분할하여 수집 (더 세밀한 데이터)
    current_start = start_dt

    while current_start < end_dt:
        current_end = min(current_start + timedelta(days=365), end_dt)

        timeframe = f"{current_start.strftime('%Y-%m-%d')} {current_end.strftime('%Y-%m-%d')}"

        print(f"  수집 중: {timeframe}")

        try:
            # Bitcoin 키워드로 검색
            pytrends.build_payload(
                kw_list=['Bitcoin'],
                cat=0,
                timeframe=timeframe,
                geo='',
                gprop=''
            )

            # 일별 데이터 가져오기
            trends_data = pytrends.interest_over_time()

            if not trends_data.empty:
                # 'isPartial' 컬럼 제거
                if 'isPartial' in trends_data.columns:
                    trends_data = trends_data.drop('isPartial', axis=1)

                all_trends.append(trends_data)
                print(f"    ✓ {len(trends_data)}개 데이터")

            # API rate limit 방지
            time.sleep(2)

        except Exception as e:
            print(f"    ✗ 오류: {e}")

        current_start = current_end + timedelta(days=1)

    # 모든 데이터 합치기
    if all_trends:
        trends_df = pd.concat(all_trends)

        # 중복 제거 (날짜 경계에서 발생할 수 있음)
        trends_df = trends_df[~trends_df.index.duplicated(keep='first')]

        # 컬럼명 변경
        trends_df = trends_df.rename(columns={'Bitcoin': 'google_trends_btc'})

        print(f"\n  ✓ 전체 성공: {len(trends_df)}개 데이터")
        print(f"    기간: {trends_df.index[0].date()} ~ {trends_df.index[-1].date()}")
        print(f"    평균: {trends_df['google_trends_btc'].mean():.2f}")
        print(f"    최소: {trends_df['google_trends_btc'].min()}")
        print(f"    최대: {trends_df['google_trends_btc'].max()}")
    else:
        print("  ✗ 실패: 데이터 없음")
        trends_df = pd.DataFrame()

except Exception as e:
    print(f"  ✗ 오류: {e}")
    trends_df = pd.DataFrame()

# ===== 데이터 통합 =====
print("\n" + "=" * 70)
print("데이터 통합 중...")
print("=" * 70)

# Fear & Greed는 일별, Google Trends도 일별
sentiment_df = pd.DataFrame()

if not fng_df.empty:
    sentiment_df = fng_df.copy()

if not trends_df.empty:
    if sentiment_df.empty:
        sentiment_df = trends_df.copy()
    else:
        sentiment_df = sentiment_df.join(trends_df, how='outer')

# 결측치 처리
if not sentiment_df.empty:
    print(f"\n통합 전 데이터 형태: {sentiment_df.shape}")

    # 결측치 확인
    null_counts = sentiment_df.isnull().sum()
    print(f"\n결측치:")
    for col in sentiment_df.columns:
        if null_counts[col] > 0:
            print(f"  {col}: {null_counts[col]}개 ({null_counts[col]/len(sentiment_df)*100:.1f}%)")

    # Forward fill
    sentiment_df = sentiment_df.fillna(method='ffill')
    sentiment_df = sentiment_df.fillna(method='bfill')

    # fear_greed_classification은 수치가 아니므로 NaN 처리
    if 'fear_greed_classification' in sentiment_df.columns:
        sentiment_df['fear_greed_classification'] = sentiment_df['fear_greed_classification'].fillna('Unknown')

    print(f"\n통합 후 데이터 형태: {sentiment_df.shape}")
    print(f"기간: {sentiment_df.index[0].date()} ~ {sentiment_df.index[-1].date()}")

    # 파일 저장
    sentiment_df.to_csv('sentiment_data.csv')
    print("\n✓ 저장 완료: sentiment_data.csv")

    # 요약 통계
    print("\n" + "=" * 70)
    print("감정/관심 지표 요약")
    print("=" * 70)

    if 'fear_greed_index' in sentiment_df.columns:
        print(f"\nFear & Greed Index:")
        print(f"  평균: {sentiment_df['fear_greed_index'].mean():.2f}")
        print(f"  최소: {sentiment_df['fear_greed_index'].min()}")
        print(f"  최대: {sentiment_df['fear_greed_index'].max()}")
        print(f"  표준편차: {sentiment_df['fear_greed_index'].std():.2f}")
        print(f"\n  분류 분포:")
        if 'fear_greed_classification' in sentiment_df.columns:
            for cat, count in sentiment_df['fear_greed_classification'].value_counts().items():
                print(f"    {cat}: {count}개 ({count/len(sentiment_df)*100:.1f}%)")

    if 'google_trends_btc' in sentiment_df.columns:
        print(f"\nGoogle Trends (Bitcoin):")
        print(f"  평균: {sentiment_df['google_trends_btc'].mean():.2f}")
        print(f"  최소: {sentiment_df['google_trends_btc'].min()}")
        print(f"  최대: {sentiment_df['google_trends_btc'].max()}")
        print(f"  표준편차: {sentiment_df['google_trends_btc'].std():.2f}")
else:
    print("\n⚠️  경고: 수집된 데이터가 없습니다.")

print("\n" + "=" * 70)
print("Step 1.4 완료!")
print("=" * 70)
