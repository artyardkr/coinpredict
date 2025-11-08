import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

print("=" * 70)
print("롱/숏 비율 데이터 수집 (Binance Futures)")
print("=" * 70)

# ===== 1. Binance Futures API 설정 =====
BASE_URL = "https://fapi.binance.com"

def get_long_short_ratio(symbol='BTCUSDT', period='1d', start_time=None, end_time=None, limit=500):
    """
    Binance Futures API로 롱/숏 비율 가져오기

    Parameters:
    - symbol: 심볼 (BTCUSDT)
    - period: 기간 (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
    - start_time: 시작 시간 (밀리초 timestamp)
    - end_time: 종료 시간 (밀리초 timestamp)
    - limit: 최대 개수 (기본 30, 최대 500)

    Returns:
    - DataFrame
    """

    # 3가지 타입의 롱/숏 비율
    endpoints = {
        'global': '/futures/data/globalLongShortAccountRatio',  # 전체 계정
        'top_trader_account': '/futures/data/topLongShortAccountRatio',  # 상위 트레이더 계정
        'top_trader_position': '/futures/data/topLongShortPositionRatio'  # 상위 트레이더 포지션
    }

    results = {}

    for name, endpoint in endpoints.items():
        url = BASE_URL + endpoint

        params = {
            'symbol': symbol,
            'period': period,
            'limit': limit
        }

        if start_time:
            params['startTime'] = int(start_time)
        if end_time:
            params['endTime'] = int(end_time)

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('timestamp')

                # 컬럼명 변경
                df.columns = [f'{name}_{col}' for col in df.columns]

                results[name] = df
                print(f"  ✓ {name}: {len(df)}개")
            else:
                print(f"  ✗ {name}: 데이터 없음")

        except Exception as e:
            print(f"  ✗ {name} 오류: {e}")

    return results

# ===== 2. 전체 기간 데이터 수집 (2021-01-01부터) =====
print("\n1. 롱/숏 비율 데이터 수집 (일별)")
print("-" * 70)

# 날짜 범위 설정
start_date = datetime(2021, 1, 1)
end_date = datetime.now()
total_days = (end_date - start_date).days

print(f"수집 기간: {start_date.date()} ~ {end_date.date()} ({total_days}일)")
print(f"API 호출 횟수: 약 {(total_days // 500) + 1}회 × 3 타입 = {((total_days // 500) + 1) * 3}회")

all_data = {
    'global': [],
    'top_trader_account': [],
    'top_trader_position': []
}

# 500개씩 반복 수집
iterations = (total_days // 500) + 1

for i in range(iterations):
    # 시작/종료 시간 계산
    current_start = start_date + timedelta(days=i * 500)
    current_end = min(current_start + timedelta(days=500), end_date)

    start_ms = int(current_start.timestamp() * 1000)
    end_ms = int(current_end.timestamp() * 1000)

    print(f"\n수집 중... {i+1}/{iterations} ({current_start.date()} ~ {current_end.date()})")

    # 데이터 수집
    results = get_long_short_ratio(
        symbol='BTCUSDT',
        period='1d',
        start_time=start_ms,
        end_time=end_ms,
        limit=500
    )

    # 결과 저장
    for name, df in results.items():
        if not df.empty:
            all_data[name].append(df)

    # API Rate limit 방지
    time.sleep(0.5)

# ===== 3. 데이터 병합 =====
print("\n2. 데이터 병합")
print("-" * 70)

merged_data = {}

for name, dfs in all_data.items():
    if dfs:
        df = pd.concat(dfs)

        # 중복 제거 (같은 timestamp)
        df = df[~df.index.duplicated(keep='first')]

        # 정렬
        df = df.sort_index()

        merged_data[name] = df

        print(f"✓ {name}: {len(df)}개 ({df.index[0].date()} ~ {df.index[-1].date()})")
    else:
        print(f"✗ {name}: 데이터 없음")

# ===== 4. 통합 DataFrame 생성 =====
print("\n3. 통합 데이터 생성")
print("-" * 70)

if merged_data:
    # 모든 타입 병합
    long_short_df = pd.DataFrame()

    for name, df in merged_data.items():
        if long_short_df.empty:
            long_short_df = df
        else:
            long_short_df = long_short_df.join(df, how='outer')

    # 타임존 제거
    long_short_df.index = pd.to_datetime(long_short_df.index).tz_localize(None)

    # 날짜만 남기기 (시간 제거)
    long_short_df.index = long_short_df.index.normalize()

    # 숫자형 변환
    for col in long_short_df.columns:
        long_short_df[col] = pd.to_numeric(long_short_df[col], errors='coerce')

    print(f"✓ 통합 완료: {long_short_df.shape}")
    print(f"  기간: {long_short_df.index[0].date()} ~ {long_short_df.index[-1].date()}")
    print(f"  컬럼: {len(long_short_df.columns)}개")

    # 컬럼 확인
    print(f"\n컬럼 목록:")
    for col in long_short_df.columns:
        print(f"  - {col}")

    # ===== 5. 통계 =====
    print("\n4. 롱/숏 비율 통계")
    print("-" * 70)

    # 주요 지표만 선택
    key_metrics = [col for col in long_short_df.columns if 'longShortRatio' in col or 'longAccount' in col]

    for metric in key_metrics:
        if metric in long_short_df.columns:
            values = long_short_df[metric].dropna()
            if len(values) > 0:
                print(f"\n{metric}:")
                print(f"  평균: {values.mean():.4f}")
                print(f"  중앙값: {values.median():.4f}")
                print(f"  최소: {values.min():.4f}")
                print(f"  최대: {values.max():.4f}")
                print(f"  표준편차: {values.std():.4f}")

    # ===== 6. 저장 =====
    print("\n5. 데이터 저장")
    print("-" * 70)

    # 전체 데이터 저장
    long_short_df.to_csv('long_short_ratio_data.csv')
    print("✓ long_short_ratio_data.csv")

    # 통합용 (주요 지표만)
    key_columns = [col for col in long_short_df.columns if 'longShortRatio' in col]
    if key_columns:
        long_short_features = long_short_df[key_columns].copy()
        long_short_features.to_csv('long_short_ratio_features.csv')
        print("✓ long_short_ratio_features.csv (통합용)")

    # ===== 7. 시각화 =====
    print("\n6. 시각화")
    print("-" * 70)

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # 1. Global Long/Short Ratio
    if 'global_longShortRatio' in long_short_df.columns:
        ax1 = axes[0]
        ax1.plot(long_short_df.index, long_short_df['global_longShortRatio'],
                linewidth=1, color='blue', label='Global L/S Ratio')
        ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Neutral (1.0)')
        ax1.fill_between(long_short_df.index, 1.0, long_short_df['global_longShortRatio'],
                        where=(long_short_df['global_longShortRatio'] > 1.0),
                        color='green', alpha=0.3, label='Long > Short')
        ax1.fill_between(long_short_df.index, 1.0, long_short_df['global_longShortRatio'],
                        where=(long_short_df['global_longShortRatio'] < 1.0),
                        color='red', alpha=0.3, label='Short > Long')
        ax1.set_xlabel('Date', fontsize=11)
        ax1.set_ylabel('Long/Short Ratio', fontsize=11)
        ax1.set_title('Global Long/Short Account Ratio (All Traders)', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

    # 2. Top Trader Account L/S Ratio
    if 'top_trader_account_longShortRatio' in long_short_df.columns:
        ax2 = axes[1]
        ax2.plot(long_short_df.index, long_short_df['top_trader_account_longShortRatio'],
                linewidth=1, color='orange', label='Top Trader L/S Ratio')
        ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Neutral (1.0)')
        ax2.fill_between(long_short_df.index, 1.0, long_short_df['top_trader_account_longShortRatio'],
                        where=(long_short_df['top_trader_account_longShortRatio'] > 1.0),
                        color='green', alpha=0.3)
        ax2.fill_between(long_short_df.index, 1.0, long_short_df['top_trader_account_longShortRatio'],
                        where=(long_short_df['top_trader_account_longShortRatio'] < 1.0),
                        color='red', alpha=0.3)
        ax2.set_xlabel('Date', fontsize=11)
        ax2.set_ylabel('Long/Short Ratio', fontsize=11)
        ax2.set_title('Top Trader Long/Short Account Ratio', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

    # 3. Top Trader Position L/S Ratio
    if 'top_trader_position_longShortRatio' in long_short_df.columns:
        ax3 = axes[2]
        ax3.plot(long_short_df.index, long_short_df['top_trader_position_longShortRatio'],
                linewidth=1, color='purple', label='Top Trader Position L/S Ratio')
        ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Neutral (1.0)')
        ax3.fill_between(long_short_df.index, 1.0, long_short_df['top_trader_position_longShortRatio'],
                        where=(long_short_df['top_trader_position_longShortRatio'] > 1.0),
                        color='green', alpha=0.3)
        ax3.fill_between(long_short_df.index, 1.0, long_short_df['top_trader_position_longShortRatio'],
                        where=(long_short_df['top_trader_position_longShortRatio'] < 1.0),
                        color='red', alpha=0.3)
        ax3.set_xlabel('Date', fontsize=11)
        ax3.set_ylabel('Long/Short Ratio', fontsize=11)
        ax3.set_title('Top Trader Long/Short Position Ratio', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('long_short_ratio_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ long_short_ratio_analysis.png")

    plt.close()

    # ===== 8. 기존 통합 데이터에 추가 =====
    print("\n7. 기존 통합 데이터에 추가")
    print("-" * 70)

    try:
        integrated = pd.read_csv('integrated_data_full.csv', index_col=0, parse_dates=True)
        integrated.index = pd.to_datetime(integrated.index).tz_localize(None)

        print(f"기존 통합 데이터: {integrated.shape}")

        # 롱/숏 비율 데이터 병합
        integrated = integrated.join(long_short_features, how='left')

        # Forward fill
        for col in key_columns:
            integrated[col] = integrated[col].fillna(method='ffill')

        print(f"롱/숏 비율 추가 후: {integrated.shape}")

        # 결측치 확인
        null_counts = integrated[key_columns].isnull().sum()
        print(f"\n결측치:")
        for col in key_columns:
            if col in null_counts.index:
                print(f"  {col}: {null_counts[col]}개")

        # 저장
        integrated.to_csv('integrated_data_full.csv')
        print("\n✓ integrated_data_full.csv 업데이트 완료!")

    except Exception as e:
        print(f"✗ 통합 데이터 업데이트 실패: {e}")

else:
    print("✗ 데이터 수집 실패")

print("\n" + "=" * 70)
print("롱/숏 비율 데이터 수집 완료!")
print("=" * 70)

print("\n📊 데이터 설명:")
print("-" * 70)
print("1. global_longShortRatio:")
print("   전체 트레이더의 롱/숏 계정 비율")
print("   > 1.0: 롱 포지션이 많음 (낙관)")
print("   < 1.0: 숏 포지션이 많음 (비관)")
print()
print("2. top_trader_account_longShortRatio:")
print("   상위 트레이더의 롱/숏 계정 비율")
print("   (전문 트레이더들의 시장 전망)")
print()
print("3. top_trader_position_longShortRatio:")
print("   상위 트레이더의 롱/숏 포지션 비율")
print("   (전문 트레이더들의 실제 포지션)")
print()
print("💡 활용:")
print("  - 시장 심리 파악")
print("  - 역발상 지표 (과도한 롱 = 조정 신호)")
print("  - 전문가 vs 일반인 비교")
