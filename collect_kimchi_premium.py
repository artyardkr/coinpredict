import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import yfinance as yf

print("=" * 70)
print("김치프리미엄 데이터 수집")
print("=" * 70)

# ===== 1. 글로벌 BTC 가격 (이미 있는 데이터) =====
print("\n1. 글로벌 BTC 가격 로드 (USD)")
print("-" * 70)

# 기존 BTC-USD 데이터 사용
btc_usd = pd.read_csv('btc_data_2021_2025.csv', index_col=0, parse_dates=True)
btc_usd.index = pd.to_datetime(btc_usd.index).tz_localize(None)

print(f"✓ 글로벌 BTC (USD): {len(btc_usd)}개")
print(f"  기간: {btc_usd.index[0].date()} ~ {btc_usd.index[-1].date()}")
print(f"  최근 가격: ${btc_usd['Close'].iloc[-1]:,.2f}")

# ===== 2. 한국 거래소 BTC 가격 수집 (Upbit) =====
print("\n2. 한국 거래소 BTC 가격 수집 (Upbit)")
print("-" * 70)

def get_upbit_candles(market='KRW-BTC', count=200, to=None):
    """
    Upbit API로 일봉 데이터 가져오기

    Parameters:
    - market: 마켓 코드 (KRW-BTC)
    - count: 캔들 개수 (최대 200)
    - to: 마지막 캔들 시각 (ISO 8601 format)

    Returns:
    - DataFrame
    """
    url = "https://api.upbit.com/v1/candles/days"

    params = {
        'market': market,
        'count': count
    }

    if to:
        params['to'] = to

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if data:
            df = pd.DataFrame(data)
            return df
        else:
            return pd.DataFrame()

    except Exception as e:
        print(f"  ✗ 오류: {e}")
        return pd.DataFrame()

# 전체 기간 데이터 수집 (2021-01-01부터)
print("Upbit KRW-BTC 데이터 수집 중...")

all_candles = []

# Upbit는 한 번에 최대 200개만 가져올 수 있으므로 반복
target_date = datetime(2021, 1, 1)
current_date = datetime.now()
total_days = (current_date - target_date).days

iterations = (total_days // 200) + 2  # 여유있게

for i in range(iterations):
    print(f"  수집 중... {i+1}/{iterations}")

    # to_date 설정: 이전 데이터의 가장 오래된 날짜
    if all_candles:
        # 이전에 수집한 데이터의 가장 오래된 날짜를 기준으로
        oldest_date = all_candles[-1]['candle_date_time_kst'].iloc[-1]
        candles = get_upbit_candles(market='KRW-BTC', count=200, to=oldest_date)
    else:
        # 첫 번째: 현재부터 200개
        candles = get_upbit_candles(market='KRW-BTC', count=200)

    if candles.empty:
        print(f"  데이터 없음 - 중단")
        break

    all_candles.append(candles)

    # 2021-01-01 이전이면 중단
    oldest_datetime = pd.to_datetime(candles['candle_date_time_kst']).min()
    if oldest_datetime <= target_date:
        print(f"  목표 날짜 도달: {oldest_datetime.date()}")
        break

    time.sleep(0.15)  # API Rate limit 방지

if all_candles:
    upbit_df = pd.concat(all_candles, ignore_index=True)

    # 중복 제거
    upbit_df = upbit_df.drop_duplicates(subset=['candle_date_time_kst'])

    # 날짜 변환
    upbit_df['date'] = pd.to_datetime(upbit_df['candle_date_time_kst'])
    upbit_df = upbit_df.set_index('date')
    upbit_df = upbit_df.sort_index()

    # 2021-01-01 이후만 필터링
    upbit_df = upbit_df[upbit_df.index >= '2021-01-01']

    # 필요한 컬럼만 선택
    upbit_krw = upbit_df[['trade_price']].copy()
    upbit_krw.columns = ['krw_price']

    print(f"\n✓ Upbit KRW-BTC: {len(upbit_krw)}개")
    print(f"  기간: {upbit_krw.index[0].date()} ~ {upbit_krw.index[-1].date()}")
    print(f"  최근 가격: ₩{upbit_krw['krw_price'].iloc[-1]:,.0f}")

    upbit_krw.to_csv('upbit_krw_btc.csv')
    print("  저장: upbit_krw_btc.csv")
else:
    print("✗ Upbit 데이터 수집 실패")
    upbit_krw = pd.DataFrame()

# ===== 3. USD/KRW 환율 =====
print("\n3. USD/KRW 환율 수집")
print("-" * 70)

# Yahoo Finance에서 USD/KRW 환율
usdkrw = yf.download(
    'KRW=X',
    start='2021-01-01',
    end=datetime.now().strftime('%Y-%m-%d'),
    progress=False
)

if not usdkrw.empty:
    usdkrw.index = pd.to_datetime(usdkrw.index).tz_localize(None)
    exchange_rate = usdkrw[['Close']].copy()
    exchange_rate.columns = ['usd_krw_rate']

    print(f"✓ USD/KRW 환율: {len(exchange_rate)}개")
    print(f"  기간: {exchange_rate.index[0].date()} ~ {exchange_rate.index[-1].date()}")
    print(f"  최근 환율: ₩{exchange_rate['usd_krw_rate'].iloc[-1]:,.2f}")

    exchange_rate.to_csv('usd_krw_exchange_rate.csv')
    print("  저장: usd_krw_exchange_rate.csv")
else:
    print("✗ 환율 데이터 수집 실패")
    exchange_rate = pd.DataFrame()

# ===== 4. 김치프리미엄 계산 =====
print("\n4. 김치프리미엄 계산")
print("-" * 70)

if not upbit_krw.empty and not exchange_rate.empty:
    # 데이터 병합
    kimchi_data = btc_usd[['Close']].copy()
    kimchi_data.columns = ['btc_usd']

    # Upbit KRW 가격 병합
    kimchi_data = kimchi_data.join(upbit_krw, how='left')

    # 환율 병합
    kimchi_data = kimchi_data.join(exchange_rate, how='left')

    # Forward fill로 결측치 처리 (주말/공휴일)
    kimchi_data = kimchi_data.fillna(method='ffill')
    kimchi_data = kimchi_data.fillna(method='bfill')

    # BTC USD 가격을 원화로 환산
    kimchi_data['btc_usd_in_krw'] = kimchi_data['btc_usd'] * kimchi_data['usd_krw_rate']

    # 김치프리미엄 계산
    # Kimchi Premium (%) = (한국가격 / 글로벌가격(원화환산) - 1) × 100
    kimchi_data['kimchi_premium'] = (
        (kimchi_data['krw_price'] / kimchi_data['btc_usd_in_krw'] - 1) * 100
    )

    # 절대 가격 차이 (원화)
    kimchi_data['kimchi_premium_krw'] = kimchi_data['krw_price'] - kimchi_data['btc_usd_in_krw']

    # NaN 제거
    kimchi_data = kimchi_data.dropna()

    print(f"✓ 김치프리미엄 계산 완료: {len(kimchi_data)}개")

    if len(kimchi_data) == 0:
        print("  ✗ 데이터가 없습니다. 날짜 범위를 확인하세요.")
        kimchi_data = None
    else:
        print(f"  기간: {kimchi_data.index[0].date()} ~ {kimchi_data.index[-1].date()}")

        # 통계
        print(f"\n김치프리미엄 통계:")
        print(f"  평균: {kimchi_data['kimchi_premium'].mean():+.2f}%")
        print(f"  표준편차: {kimchi_data['kimchi_premium'].std():.2f}%")
        print(f"  최소: {kimchi_data['kimchi_premium'].min():+.2f}%")
        print(f"  최대: {kimchi_data['kimchi_premium'].max():+.2f}%")
        print(f"  최근: {kimchi_data['kimchi_premium'].iloc[-1]:+.2f}%")

        # 양수/음수 비율
        positive_count = (kimchi_data['kimchi_premium'] > 0).sum()
        negative_count = (kimchi_data['kimchi_premium'] < 0).sum()
        total = len(kimchi_data)

        print(f"\n프리미엄 분포:")
        print(f"  양수 (한국 > 글로벌): {positive_count}일 ({positive_count/total*100:.1f}%)")
        print(f"  음수 (한국 < 글로벌): {negative_count}일 ({negative_count/total*100:.1f}%)")

        # 저장
        kimchi_data.to_csv('kimchi_premium_data.csv')
        print("\n✓ 저장: kimchi_premium_data.csv")

        # 통합 데이터에 추가할 컬럼만 선택
        kimchi_features = kimchi_data[['kimchi_premium', 'kimchi_premium_krw']].copy()
        kimchi_features.to_csv('kimchi_premium_features.csv')
        print("✓ 저장: kimchi_premium_features.csv (통합용)")

    # ===== 5. 시각화 =====
    if kimchi_data is not None and len(kimchi_data) > 0:
        print("\n5. 시각화")
        print("-" * 70)

        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. 김치프리미엄 시계열
    ax1 = axes[0, 0]
    ax1.plot(kimchi_data.index, kimchi_data['kimchi_premium'], linewidth=1, color='blue')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax1.fill_between(kimchi_data.index, 0, kimchi_data['kimchi_premium'],
                      where=(kimchi_data['kimchi_premium'] > 0), color='green', alpha=0.3, label='프리미엄')
    ax1.fill_between(kimchi_data.index, 0, kimchi_data['kimchi_premium'],
                      where=(kimchi_data['kimchi_premium'] < 0), color='red', alpha=0.3, label='디스카운트')
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Kimchi Premium (%)', fontsize=11)
    ax1.set_title('Kimchi Premium Over Time (2021-2025)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # 2. 가격 비교
    ax2 = axes[0, 1]
    ax2.plot(kimchi_data.index, kimchi_data['krw_price'], label='Upbit (KRW)', linewidth=1.5)
    ax2.plot(kimchi_data.index, kimchi_data['btc_usd_in_krw'], label='Global (USD→KRW)',
             linewidth=1.5, linestyle='--')
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('BTC Price (KRW)', fontsize=11)
    ax2.set_title('BTC Price: Korea vs Global (in KRW)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    # 3. 김치프리미엄 분포
    ax3 = axes[1, 0]
    ax3.hist(kimchi_data['kimchi_premium'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='0%')
    ax3.axvline(x=kimchi_data['kimchi_premium'].mean(), color='green',
                linestyle='--', linewidth=2, label=f'평균 ({kimchi_data["kimchi_premium"].mean():.2f}%)')
    ax3.set_xlabel('Kimchi Premium (%)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Kimchi Premium Distribution', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. 김치프리미엄 vs BTC 가격
    ax4 = axes[1, 1]
    scatter = ax4.scatter(kimchi_data['btc_usd'], kimchi_data['kimchi_premium'],
                         c=kimchi_data.index.astype(np.int64), cmap='viridis', alpha=0.5, s=10)
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_xlabel('BTC Price (USD)', fontsize=11)
    ax4.set_ylabel('Kimchi Premium (%)', fontsize=11)
    ax4.set_title('Kimchi Premium vs BTC Price', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Time')

    plt.tight_layout()
    plt.savefig('kimchi_premium_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ 시각화: kimchi_premium_analysis.png")

    plt.close()

else:
    print("✗ 데이터 부족으로 김치프리미엄 계산 실패")

print("\n" + "=" * 70)
print("김치프리미엄 데이터 수집 완료!")
print("=" * 70)

# ===== 6. 통합 데이터에 추가 =====
print("\n6. 기존 통합 데이터에 김치프리미엄 추가")
print("-" * 70)

try:
    integrated = pd.read_csv('integrated_data_full.csv', index_col=0, parse_dates=True)
    integrated.index = pd.to_datetime(integrated.index).tz_localize(None)

    print(f"기존 통합 데이터: {integrated.shape}")

    # 김치프리미엄 데이터 병합
    integrated = integrated.join(kimchi_features, how='left')

    # Forward fill
    integrated['kimchi_premium'] = integrated['kimchi_premium'].fillna(method='ffill')
    integrated['kimchi_premium_krw'] = integrated['kimchi_premium_krw'].fillna(method='ffill')

    print(f"김치프리미엄 추가 후: {integrated.shape}")
    print(f"  김치프리미엄 결측치: {integrated['kimchi_premium'].isnull().sum()}개")

    # 저장
    integrated.to_csv('integrated_data_full.csv')
    print("✓ integrated_data_full.csv 업데이트 완료!")

    print(f"\n최종 데이터셋:")
    print(f"  크기: {integrated.shape}")
    print(f"  새로운 컬럼: kimchi_premium, kimchi_premium_krw")

except Exception as e:
    print(f"✗ 통합 데이터 업데이트 실패: {e}")

print("\n" + "=" * 70)
print("모든 작업 완료!")
print("=" * 70)
