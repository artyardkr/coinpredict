import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import json

print("=" * 70)
print("Step 1.6: 온체인 데이터 수집")
print("=" * 70)

# 수집 기간
start_date = "2021-01-01"
end_date = "2025-10-15"

print(f"\n수집 기간: {start_date} ~ {end_date}")
print("-" * 70)

onchain_data = {}

# ===== Part 1: Blockchain.com API =====
print("\n[1/3] Blockchain.com API - 기본 온체인 지표")
print("-" * 70)

# Blockchain.com의 무료 API 엔드포인트
blockchain_metrics = {
    'market-price': 'BTC 시장 가격 (USD)',
    'hash-rate': '해시레이트 (TH/s)',
    'difficulty': '채굴 난이도',
    'n-transactions': '일일 트랜잭션 수',
    'n-unique-addresses': '고유 주소 수',
    'total-bitcoins': '총 BTC 공급량',
    'market-cap': '시가총액',
    'miners-revenue': '채굴자 수익',
    'transaction-fees': '트랜잭션 수수료',
    'mempool-size': '멤풀 크기',
}

for metric, description in blockchain_metrics.items():
    try:
        print(f"수집 중: {metric:25} - {description}")

        # Blockchain.com API
        url = f"https://api.blockchain.info/charts/{metric}?timespan=5years&format=json&cors=true"
        response = requests.get(url, timeout=30)

        if response.status_code == 200:
            data = response.json()
            values = data['values']

            # DataFrame으로 변환
            df = pd.DataFrame(values)
            df['date'] = pd.to_datetime(df['x'], unit='s')
            df = df.set_index('date')

            # 타임존 제거 (tz-aware인 경우)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            df = df.rename(columns={'y': f'bc_{metric.replace("-", "_")}'})
            df = df[[f'bc_{metric.replace("-", "_")}']]

            # 기간 필터링
            df = df[start_date:end_date]

            if len(df) > 0:
                onchain_data[f'bc_{metric.replace("-", "_")}'] = df
                print(f"  ✓ 성공: {len(df)}개 데이터")
            else:
                print(f"  ✗ 실패: 기간 내 데이터 없음")
        else:
            print(f"  ✗ 실패: HTTP {response.status_code}")

        time.sleep(1)  # Rate limit 방지

    except Exception as e:
        print(f"  ✗ 오류: {e}")

# ===== Part 2: Glassnode API (무료 tier) =====
print("\n[2/3] Glassnode API - 고급 온체인 지표")
print("-" * 70)

# Glassnode 무료 API (일부 지표만 무료)
GLASSNODE_API_KEY = "YOUR_GLASSNODE_API_KEY"  # 필요시 입력

glassnode_metrics = {
    'addresses/active_count': '활성 주소 수',
    'addresses/new_non_zero_count': '신규 non-zero 주소',
    'transactions/count': '트랜잭션 수',
    'transactions/transfers_volume_sum': '전송 볼륨',
}

if GLASSNODE_API_KEY != "YOUR_GLASSNODE_API_KEY":
    for metric, description in glassnode_metrics.items():
        try:
            print(f"수집 중: {metric:35} - {description}")

            url = f"https://api.glassnode.com/v1/metrics/{metric}"
            params = {
                'a': 'BTC',
                'api_key': GLASSNODE_API_KEY,
                's': int(datetime.strptime(start_date, '%Y-%m-%d').timestamp()),
                'u': int(datetime.strptime(end_date, '%Y-%m-%d').timestamp()),
            }

            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()

                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['t'], unit='s').tz_localize(None)
                df = df.set_index('date')

                metric_name = f'gn_{metric.replace("/", "_").replace("_", "_")}'
                df = df.rename(columns={'v': metric_name})
                df = df[[metric_name]]

                if len(df) > 0:
                    onchain_data[metric_name] = df
                    print(f"  ✓ 성공: {len(df)}개 데이터")
                else:
                    print(f"  ✗ 실패: 데이터 없음")
            else:
                print(f"  ✗ 실패: HTTP {response.status_code}")

            time.sleep(2)

        except Exception as e:
            print(f"  ✗ 오류: {e}")
else:
    print("  ⚠️  Glassnode API 키가 설정되지 않음 (선택사항)")
    print("  발급: https://studio.glassnode.com/settings/api")

# ===== Part 3: CoinMetrics (Community API - 무료) =====
print("\n[3/3] CoinMetrics Community API - 네트워크 지표")
print("-" * 70)

coinmetrics_metrics = {
    'AdrActCnt': '활성 주소 수',
    'TxCnt': '트랜잭션 수',
    'TxTfrValAdjUSD': '전송 가치 (USD)',
    'TxTfrValMeanUSD': '평균 전송 가치',
    'HashRate': '해시레이트',
    'IssTotUSD': '신규 발행 (USD)',
    'FeeTotUSD': '총 수수료 (USD)',
    'DiffMean': '평균 난이도',
}

try:
    print("수집 중: CoinMetrics 네트워크 지표...")

    # CoinMetrics Community API
    metrics_list = ','.join(coinmetrics_metrics.keys())
    url = f"https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    params = {
        'assets': 'btc',
        'metrics': metrics_list,
        'start_time': start_date,
        'end_time': end_date,
        'frequency': '1d',
    }

    response = requests.get(url, params=params, timeout=60)

    if response.status_code == 200:
        data = response.json()

        if 'data' in data and len(data['data']) > 0:
            df = pd.DataFrame(data['data'])
            df['date'] = pd.to_datetime(df['time']).dt.tz_localize(None)
            df = df.set_index('date')
            df = df.drop(['time', 'asset'], axis=1, errors='ignore')

            # 컬럼명 변경
            for metric in coinmetrics_metrics.keys():
                if metric in df.columns:
                    new_name = f'cm_{metric.lower()}'
                    df = df.rename(columns={metric: new_name})
                    onchain_data[new_name] = df[[new_name]]
                    print(f"  ✓ {metric}: {len(df)}개 데이터")
        else:
            print("  ✗ 실패: 데이터 없음")
    else:
        print(f"  ✗ 실패: HTTP {response.status_code}")

except Exception as e:
    print(f"  ✗ 오류: {e}")

# ===== 데이터 통합 =====
print("\n" + "=" * 70)
print("온체인 데이터 통합 중...")
print("=" * 70)

if onchain_data:
    # 모든 데이터프레임 합치기
    onchain_df = pd.DataFrame()

    for name, df in onchain_data.items():
        if onchain_df.empty:
            onchain_df = df.copy()
        else:
            onchain_df = onchain_df.join(df, how='outer')

    print(f"\n통합 전 데이터 형태: {onchain_df.shape}")
    print(f"기간: {onchain_df.index[0].date()} ~ {onchain_df.index[-1].date()}")

    # 결측치 확인
    null_counts = onchain_df.isnull().sum()
    print(f"\n결측치가 있는 컬럼: {(null_counts > 0).sum()}개")

    if (null_counts > 0).sum() > 0:
        print("\n상위 10개:")
        for col in null_counts[null_counts > 0].sort_values(ascending=False).head(10).index:
            count = null_counts[col]
            pct = count / len(onchain_df) * 100
            print(f"  {col:40} : {count:4}개 ({pct:5.1f}%)")

    # 결측치 처리
    print("\n결측치 처리 중...")
    onchain_df = onchain_df.ffill()
    onchain_df = onchain_df.bfill()

    remaining_nulls = onchain_df.isnull().sum().sum()
    if remaining_nulls > 0:
        print(f"  - 남은 결측치 {remaining_nulls}개, 행 제거...")
        onchain_df = onchain_df.dropna()

    print(f"\n최종 데이터 형태: {onchain_df.shape}")

    # 파일 저장
    onchain_df.to_csv('onchain_data.csv')
    print("\n✓ 저장 완료: onchain_data.csv")

    # 요약 통계
    print("\n" + "=" * 70)
    print("온체인 데이터 요약")
    print("=" * 70)

    print(f"\n총 {len(onchain_df.columns)}개 온체인 지표 수집:")
    for col in onchain_df.columns:
        # 숫자형 컬럼만 처리
        if pd.api.types.is_numeric_dtype(onchain_df[col]):
            print(f"\n{col}:")
            print(f"  평균: {onchain_df[col].mean():.2e}")
            print(f"  최소: {onchain_df[col].min():.2e}")
            print(f"  최대: {onchain_df[col].max():.2e}")
            print(f"  최신값: {onchain_df[col].iloc[-1]:.2e}")
        else:
            # 문자열 데이터는 숫자로 변환 시도
            try:
                numeric_col = pd.to_numeric(onchain_df[col], errors='coerce')
                onchain_df[col] = numeric_col
                print(f"\n{col}:")
                print(f"  평균: {numeric_col.mean():.2e}")
                print(f"  최소: {numeric_col.min():.2e}")
                print(f"  최대: {numeric_col.max():.2e}")
                print(f"  최신값: {numeric_col.iloc[-1]:.2e}")
            except:
                print(f"\n{col}: (변환 실패, 스킵)")

else:
    print("\n⚠️  경고: 수집된 온체인 데이터가 없습니다.")
    print("API 키를 설정하거나 네트워크 연결을 확인하세요.")

print("\n" + "=" * 70)
print("Step 1.6 완료!")
print("=" * 70)
