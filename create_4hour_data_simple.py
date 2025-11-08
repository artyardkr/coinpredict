#!/usr/bin/env python3
"""
4시간 단위 통합 데이터 생성 (pandas_ta 없이)
integrated_data_full.csv와 동일한 구조, 시간 단위만 4시간으로 변경
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 기술적 지표 계산 함수들
def calculate_rsi(prices, period=14):
    """RSI 계산"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """MACD 계산"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_diff = macd - macd_signal
    return macd, macd_signal, macd_diff

def calculate_obv(close, volume):
    """OBV 계산"""
    obv = np.where(close > close.shift(1), volume,
           np.where(close < close.shift(1), -volume, 0))
    return pd.Series(obv, index=close.index).cumsum()

def calculate_atr(high, low, close, period=14):
    """ATR 계산"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

print("="*80)
print("4시간 단위 통합 데이터 생성")
print("="*80)

# ========================================
# 1. BTC 4시간 가격 데이터 수집
# ========================================
print("\n[1/5] BTC 4시간 가격 데이터 수집 중...")

ticker = "BTC-USD"
start_date = "2021-02-01"
end_date = datetime.now().strftime("%Y-%m-%d")

print(f"  기간: {start_date} ~ {end_date}")
print(f"  데이터 소스: Yahoo Finance (1시간 → 4시간 리샘플링)")

# 1시간 데이터 수집
btc_1h = yf.download(ticker, start=start_date, end=end_date, interval="1h", progress=False)

if btc_1h.empty:
    print("  ❌ 데이터 수집 실패!")
    raise ValueError("Yahoo Finance에서 데이터를 가져올 수 없습니다.")

# 4시간으로 리샘플링
print("  리샘플링: 1시간 → 4시간...")
btc_4h = btc_1h.resample('4h').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})

btc_4h = btc_4h.dropna()
btc_4h.index.name = 'Date'
btc_4h = btc_4h.reset_index()

print(f"  ✅ BTC 4시간 데이터: {len(btc_4h)} rows")
print(f"     기간: {btc_4h['Date'].min()} ~ {btc_4h['Date'].max()}")

# ========================================
# 2. 일별 데이터 로드
# ========================================
print("\n[2/5] 기존 일별 데이터 로드 중...")

try:
    df_daily = pd.read_csv('integrated_data_full.csv')
    df_daily['Date'] = pd.to_datetime(df_daily['Date'])
    df_daily = df_daily.sort_values('Date').reset_index(drop=True)
    print(f"  ✅ 일별 데이터: {len(df_daily)} rows, {len(df_daily.columns)} columns")
except FileNotFoundError:
    print("  ❌ integrated_data_full.csv 파일이 없습니다!")
    raise

# ========================================
# 3. 일별 데이터를 4시간으로 보간
# ========================================
print("\n[3/5] 일별 데이터를 4시간으로 보간 중...")

# BTC 가격 관련 열 제외
exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

daily_features = df_daily.copy()
daily_features['Date'] = pd.to_datetime(daily_features['Date'])
daily_features = daily_features.set_index('Date')

# 4시간 간격으로 reindex
start_4h = btc_4h['Date'].min()
end_4h = btc_4h['Date'].max()
freq_4h = pd.date_range(start=start_4h, end=end_4h, freq='4h')

# Reindex & Forward Fill
daily_features_4h = daily_features.reindex(freq_4h, method='ffill')
daily_features_4h.index.name = 'Date'
daily_features_4h = daily_features_4h.reset_index()

print(f"  ✅ 보간 완료: {len(daily_features_4h)} rows")

# 기존 기술적 지표 제거 (재계산할 것)
tech_cols_to_remove = []
for col in daily_features_4h.columns:
    if any(x in col for x in ['RSI', 'MACD', 'ATR', 'OBV', 'MFI', 'ADX', 'CCI',
                               'Stoch', 'Williams', 'ROC', 'EMA', 'SMA', 'BB_',
                               'volatility', 'daily_return', 'volume_change',
                               'market_cap']):
        tech_cols_to_remove.append(col)

for col in tech_cols_to_remove:
    if col in daily_features_4h.columns:
        daily_features_4h = daily_features_4h.drop(columns=[col])

print(f"  제거한 기술지표: {len(tech_cols_to_remove)}개")

# ========================================
# 4. 기술적 지표 계산 (4시간 데이터)
# ========================================
print("\n[4/5] 기술적 지표 계산 중 (4시간 데이터)...")

btc_calc = btc_4h.copy()

# RSI
print("  RSI...")
btc_calc['RSI'] = calculate_rsi(btc_calc['Close'], 14)

# MACD
print("  MACD...")
macd, macd_signal, macd_diff = calculate_macd(btc_calc['Close'], 12, 26, 9)
btc_calc['MACD'] = macd
btc_calc['MACD_signal'] = macd_signal
btc_calc['MACD_diff'] = macd_diff

# OBV
print("  OBV...")
btc_calc['OBV'] = calculate_obv(btc_calc['Close'], btc_calc['Volume'])

# ATR
print("  ATR...")
btc_calc['ATR'] = calculate_atr(btc_calc['High'], btc_calc['Low'], btc_calc['Close'], 14)

# 변동성
print("  변동성...")
btc_calc['volatility_20d'] = btc_calc['Close'].pct_change().rolling(20).std()

# 수익률
btc_calc['daily_return'] = btc_calc['Close'].pct_change()
btc_calc['volume_change'] = btc_calc['Volume'].pct_change()

# EMA (간단한 버전만)
print("  EMA/SMA...")
periods = {
    'EMA5_volume': 8,
    'EMA10_volume': 15,
    'EMA20_volume': 30,
    'EMA30_volume': 45,
    'EMA100_volume': 150,
    'EMA200_volume': 300,
}

for name, period in periods.items():
    btc_calc[name] = btc_calc['Volume'].ewm(span=period, adjust=False).mean()

# 시가총액 근사치
btc_calc['market_cap_approx'] = btc_calc['Close'] * 19000000

# SMA
for period in [5, 10, 20, 30]:
    period_4h = period * 6  # 일별 → 4시간 변환
    btc_calc[f'SMA{period}_marketcap'] = btc_calc['market_cap_approx'].rolling(period_4h).mean()
    btc_calc[f'EMA{period}_marketcap'] = btc_calc['market_cap_approx'].ewm(span=period_4h, adjust=False).mean()

print(f"  ✅ 기술적 지표 계산 완료")

# ========================================
# 5. 데이터 병합 및 저장
# ========================================
print("\n[5/5] 데이터 병합 및 저장 중...")

btc_calc['Date'] = pd.to_datetime(btc_calc['Date'])
daily_features_4h['Date'] = pd.to_datetime(daily_features_4h['Date'])

# 병합
df_4h = pd.merge(btc_calc, daily_features_4h, on='Date', how='inner')

print(f"  병합 결과: {len(df_4h)} rows, {len(df_4h.columns)} columns")

# NaN/Inf 처리
print("  NaN/Inf 처리 중...")
for col in df_4h.columns:
    if col == 'Date':
        continue
    df_4h[col] = df_4h[col].replace([np.inf, -np.inf], np.nan)
    df_4h[col] = df_4h[col].fillna(method='ffill').fillna(method='bfill')

# 정렬
df_4h = df_4h.sort_values('Date').reset_index(drop=True)

# 저장
output_file = 'integrated_data_4hour.csv'
df_4h.to_csv(output_file, index=False)

print(f"  ✅ 저장 완료: {output_file}")

# ========================================
# 결과 요약
# ========================================
print("\n" + "="*80)
print("생성 완료!")
print("="*80)

print(f"\n파일: {output_file}")
print(f"샘플 수: {len(df_4h):,} (일별 {len(df_daily)}개의 약 {len(df_4h)/len(df_daily):.1f}배)")
print(f"변수 수: {len(df_4h.columns)}")
print(f"기간: {df_4h['Date'].min()} ~ {df_4h['Date'].max()}")

# 샘플
print("\n처음 3개:")
print(df_4h[['Date', 'Close', 'Volume', 'RSI', 'MACD', 'OBV']].head(3))

print("\n마지막 3개:")
print(df_4h[['Date', 'Close', 'Volume', 'RSI', 'MACD', 'OBV']].tail(3))

# 변수 리스트 저장
with open('4hour_data_columns.txt', 'w') as f:
    for col in df_4h.columns:
        f.write(f"{col}\n")

print(f"\n변수 리스트: 4hour_data_columns.txt")

print("\n" + "="*80)
print("⚠️  주의사항")
print("="*80)
print("""
1. 보간된 데이터 (4시간마다 같은 값 반복):
   - 전통 시장: SPX, QQQ, GOLD 등
   - 거시경제: DFF, M2SL, GDP 등
   - 온체인: bc_hash_rate, bc_difficulty 등
   - 감성: fear_greed_index

2. 진짜 4시간 데이터:
   - BTC 가격: Open, High, Low, Close, Volume
   - 기술적 지표: RSI, MACD, OBV 등

3. 예측 대상:
   - 일별: "24시간 후" 종가
   - 4시간: "4시간 후" 종가

4. 사용 시 주의:
   - 보간 데이터로 인한 과적합 가능
   - 실전 자동매매에는 부적합
   - 기술적 분석/백테스팅 용도 권장
""")

print("="*80)
