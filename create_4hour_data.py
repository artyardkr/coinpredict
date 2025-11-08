#!/usr/bin/env python3
"""
4시간 단위 통합 데이터 생성
integrated_data_full.csv와 동일한 구조, 시간 단위만 4시간으로 변경
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("4시간 단위 통합 데이터 생성")
print("="*80)

# ========================================
# 1. BTC 4시간 가격 데이터 수집
# ========================================
print("\n[1/6] BTC 4시간 가격 데이터 수집 중...")

# Yahoo Finance에서 BTC-USD 1시간 데이터 수집 (4시간으로 리샘플링)
ticker = "BTC-USD"
start_date = "2021-02-01"
end_date = datetime.now().strftime("%Y-%m-%d")

print(f"  기간: {start_date} ~ {end_date}")
print(f"  데이터 소스: Yahoo Finance (1시간 → 4시간 리샘플링)")

# 1시간 데이터 수집
btc_1h = yf.download(ticker, start=start_date, end=end_date, interval="1h", progress=False)

if btc_1h.empty:
    print("  ❌ 데이터 수집 실패! 대안: Binance API 사용 권장")
    raise ValueError("Yahoo Finance에서 데이터를 가져올 수 없습니다.")

# 4시간으로 리샘플링
print("  리샘플링: 1시간 → 4시간...")
btc_4h = btc_1h.resample('4H').agg({
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
# 2. 일별 데이터 로드 (기존 integrated_data_full.csv)
# ========================================
print("\n[2/6] 기존 일별 데이터 로드 중...")

try:
    df_daily = pd.read_csv('integrated_data_full.csv')
    df_daily['Date'] = pd.to_datetime(df_daily['Date'])
    df_daily = df_daily.sort_values('Date').reset_index(drop=True)
    print(f"  ✅ 일별 데이터: {len(df_daily)} rows, {len(df_daily.columns)} columns")
except FileNotFoundError:
    print("  ❌ integrated_data_full.csv 파일이 없습니다!")
    raise

# 일별 데이터의 변수 확인
daily_cols = df_daily.columns.tolist()
print(f"  변수 수: {len(daily_cols)}")

# ========================================
# 3. BTC 가격 데이터 외 변수 추출 (일별 → 4시간 보간)
# ========================================
print("\n[3/6] 일별 데이터를 4시간으로 보간 중...")

# BTC 가격 관련 열 제외 (4시간 데이터로 대체할 것)
exclude_btc_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

# 일별 데이터에서 BTC 가격 외 변수만 추출
daily_features = df_daily.copy()

# Date를 datetime으로
daily_features['Date'] = pd.to_datetime(daily_features['Date'])
daily_features = daily_features.set_index('Date')

# 4시간 간격으로 reindex
print("  4시간 간격으로 reindex...")
start_4h = btc_4h['Date'].min()
end_4h = btc_4h['Date'].max()

# 4시간 DatetimeIndex 생성
freq_4h = pd.date_range(start=start_4h, end=end_4h, freq='4H')

# Reindex & Forward Fill
daily_features_4h = daily_features.reindex(freq_4h, method='ffill')
daily_features_4h.index.name = 'Date'
daily_features_4h = daily_features_4h.reset_index()

print(f"  ✅ 보간 완료: {len(daily_features_4h)} rows")

# ========================================
# 4. 기술적 지표 재계산 (4시간 데이터 기반)
# ========================================
print("\n[4/6] 기술적 지표 재계산 중 (4시간 데이터)...")

btc_4h_calc = btc_4h.copy()

# 기존 기술적 지표 열 제거 (재계산할 것)
tech_indicator_cols = [
    'RSI', 'MACD', 'MACD_signal', 'MACD_diff',
    'ATR', 'OBV', 'MFI', 'ADX', 'CCI',
    'Stoch_K', 'Stoch_D', 'Williams_R', 'ROC',
    'volatility_20d', 'daily_return', 'volume_change'
]

for col in tech_indicator_cols:
    if col in daily_features_4h.columns:
        daily_features_4h = daily_features_4h.drop(columns=[col])

# EMA/SMA 관련 제거 (Close 기반이므로 재계산)
ema_sma_cols = [col for col in daily_features_4h.columns
                if 'EMA' in col or 'SMA' in col]
for col in ema_sma_cols:
    daily_features_4h = daily_features_4h.drop(columns=[col])

# 기술적 지표 계산
print("  계산 중: RSI, MACD, ATR, OBV...")

# RSI
btc_4h_calc['RSI'] = ta.rsi(btc_4h_calc['Close'], length=14)

# MACD
macd = ta.macd(btc_4h_calc['Close'], fast=12, slow=26, signal=9)
if macd is not None:
    btc_4h_calc['MACD'] = macd['MACD_12_26_9']
    btc_4h_calc['MACD_signal'] = macd['MACDs_12_26_9']
    btc_4h_calc['MACD_diff'] = macd['MACDh_12_26_9']

# ATR
atr = ta.atr(btc_4h_calc['High'], btc_4h_calc['Low'], btc_4h_calc['Close'], length=14)
if atr is not None:
    btc_4h_calc['ATR'] = atr

# OBV
obv = ta.obv(btc_4h_calc['Close'], btc_4h_calc['Volume'])
if obv is not None:
    btc_4h_calc['OBV'] = obv

# MFI
mfi = ta.mfi(btc_4h_calc['High'], btc_4h_calc['Low'], btc_4h_calc['Close'],
             btc_4h_calc['Volume'], length=14)
if mfi is not None:
    btc_4h_calc['MFI'] = mfi

# ADX
adx = ta.adx(btc_4h_calc['High'], btc_4h_calc['Low'], btc_4h_calc['Close'], length=14)
if adx is not None and 'ADX_14' in adx.columns:
    btc_4h_calc['ADX'] = adx['ADX_14']

# CCI
cci = ta.cci(btc_4h_calc['High'], btc_4h_calc['Low'], btc_4h_calc['Close'], length=20)
if cci is not None:
    btc_4h_calc['CCI'] = cci

# Stochastic
stoch = ta.stoch(btc_4h_calc['High'], btc_4h_calc['Low'], btc_4h_calc['Close'],
                 k=14, d=3)
if stoch is not None:
    btc_4h_calc['Stoch_K'] = stoch['STOCHk_14_3_3']
    btc_4h_calc['Stoch_D'] = stoch['STOCHd_14_3_3']

# Williams %R
willr = ta.willr(btc_4h_calc['High'], btc_4h_calc['Low'], btc_4h_calc['Close'], length=14)
if willr is not None:
    btc_4h_calc['Williams_R'] = willr

# ROC
roc = ta.roc(btc_4h_calc['Close'], length=10)
if roc is not None:
    btc_4h_calc['ROC'] = roc

# 변동성 (20주기 = 20*4시간 = 80시간 = 3.3일)
btc_4h_calc['volatility_20d'] = btc_4h_calc['Close'].pct_change().rolling(20).std()

# 수익률
btc_4h_calc['daily_return'] = btc_4h_calc['Close'].pct_change()

# 거래량 변화
btc_4h_calc['volume_change'] = btc_4h_calc['Volume'].pct_change()

# EMA/SMA (주기 조정: 일별 → 4시간)
# 일별 12일 = 4시간 12*6 = 72주기 (너무 길어서 조정)
# 대신: 일별 12 → 4시간 18 정도로 매핑
periods_mapping = {
    5: 8,    # 5일 → 8*4h = 32h (1.3일)
    10: 15,  # 10일 → 15*4h = 60h (2.5일)
    12: 18,  # 12일 → 18*4h = 72h (3일)
    14: 21,  # 14일 → 21*4h = 84h (3.5일)
    20: 30,  # 20일 → 30*4h = 120h (5일)
    30: 45,  # 30일 → 45*4h = 180h (7.5일)
    100: 150, # 100일 → 150*4h = 600h (25일)
    200: 300, # 200일 → 300*4h = 1200h (50일)
}

print("  계산 중: EMA/SMA...")
for period_daily, period_4h in periods_mapping.items():
    # EMA
    btc_4h_calc[f'EMA{period_daily}_volume'] = ta.ema(btc_4h_calc['Volume'], length=period_4h)
    # SMA (일부만)
    if period_daily in [5, 10, 20, 30]:
        btc_4h_calc[f'SMA{period_daily}_marketcap'] = ta.sma(
            btc_4h_calc['Close'] * 19000000,  # 근사 시가총액
            length=period_4h
        )

# market_cap_approx
btc_4h_calc['market_cap_approx'] = btc_4h_calc['Close'] * 19000000

print(f"  ✅ 기술적 지표 재계산 완료")

# ========================================
# 5. 데이터 병합
# ========================================
print("\n[5/6] 데이터 병합 중...")

# btc_4h_calc와 daily_features_4h 병합
btc_4h_calc['Date'] = pd.to_datetime(btc_4h_calc['Date'])
daily_features_4h['Date'] = pd.to_datetime(daily_features_4h['Date'])

# Date 기준 병합
df_4h = pd.merge(btc_4h_calc, daily_features_4h, on='Date', how='inner')

print(f"  병합 결과: {len(df_4h)} rows, {len(df_4h.columns)} columns")

# ========================================
# 6. 후처리 및 저장
# ========================================
print("\n[6/6] 후처리 및 저장 중...")

# NaN/Inf 처리
print("  NaN/Inf 처리 중...")
for col in df_4h.columns:
    if col == 'Date':
        continue
    df_4h[col] = df_4h[col].replace([np.inf, -np.inf], np.nan)
    df_4h[col] = df_4h[col].fillna(method='ffill').fillna(method='bfill')

# 날짜 정렬
df_4h = df_4h.sort_values('Date').reset_index(drop=True)

# 저장
output_file = 'integrated_data_4hour.csv'
df_4h.to_csv(output_file, index=False)

print(f"  ✅ 저장 완료: {output_file}")

# ========================================
# 7. 결과 요약
# ========================================
print("\n" + "="*80)
print("생성 완료!")
print("="*80)

print(f"\n파일: {output_file}")
print(f"샘플 수: {len(df_4h):,} (일별 대비 약 6배)")
print(f"변수 수: {len(df_4h.columns)}")
print(f"기간: {df_4h['Date'].min()} ~ {df_4h['Date'].max()}")
print(f"주기: 4시간")

# 일별 데이터와 비교
if len(df_daily) > 0:
    ratio = len(df_4h) / len(df_daily)
    print(f"\n일별 데이터: {len(df_daily)} rows")
    print(f"4시간 데이터: {len(df_4h)} rows")
    print(f"비율: {ratio:.2f}배")

# 샘플 출력
print("\n샘플 데이터 (처음 5개):")
print(df_4h[['Date', 'Close', 'Volume', 'RSI', 'MACD']].head())

print("\n샘플 데이터 (마지막 5개):")
print(df_4h[['Date', 'Close', 'Volume', 'RSI', 'MACD']].tail())

# 변수 리스트 저장
with open('4hour_data_columns.txt', 'w') as f:
    for col in df_4h.columns:
        f.write(f"{col}\n")
print(f"\n변수 리스트 저장: 4hour_data_columns.txt")

# ========================================
# 8. 주의사항 출력
# ========================================
print("\n" + "="*80)
print("⚠️  주의사항")
print("="*80)

print("""
1. 보간된 데이터:
   - 전통 시장 (SPX, QQQ, GOLD 등): 일별 → 4시간 (forward fill)
   - 거시경제 (DFF, M2SL 등): 월별 → 4시간 (forward fill)
   - 온체인 (bc_hash_rate 등): 일별 → 4시간 (forward fill)
   - 감성 (fear_greed_index): 일별 → 4시간 (forward fill)

   → 실제 4시간마다 변하지 않고, 같은 값이 6회 반복됩니다!

2. 진짜 4시간 데이터:
   - BTC 가격 (Open, High, Low, Close, Volume)
   - 기술적 지표 (RSI, MACD, OBV 등) - 4시간 데이터로 재계산

3. 예측 대상:
   - 일별: "다음날" 종가 (24시간 후)
   - 4시간: "4시간 후" 종가

4. 과적합 위험:
   - 보간된 변수(SPX 등)가 4시간마다 같은 값 반복
   - ElasticNet이 이를 학습하면 과적합 가능
   - 순수 BTC + 기술지표만 사용하는 것도 고려

5. 사용 권장:
   ✅ 단기 트레이딩 시뮬레이션
   ✅ 기술적 지표 효과 분석
   ⚠️ 전통시장/거시경제 영향 분석 (주의)
   ❌ 실전 자동매매 (보간 데이터 문제)
""")

print("="*80)
print("다음 단계: step25와 동일한 방식으로 분석")
print("  python step25_4hour_version.py")
print("="*80)
