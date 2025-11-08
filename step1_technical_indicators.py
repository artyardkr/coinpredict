import pandas as pd
import numpy as np
import ta

print("=" * 70)
print("Step 1.1: BTC 기술적 지표 생성")
print("=" * 70)

# BTC 데이터 로드
print("\n1. BTC 데이터 로딩...")
btc = pd.read_csv('btc_data_2021_2025.csv', index_col=0, parse_dates=True)
print(f"   데이터 형태: {btc.shape}")
print(f"   기간: {btc.index[0]} ~ {btc.index[-1]}")

# 원본 데이터 복사
btc_with_indicators = btc.copy()

print("\n2. 이동평균 (EMA, SMA) 생성 중...")

# EMA - 종가 기준
ema_periods = [5, 10, 14, 20, 30, 100, 200]
for period in ema_periods:
    btc_with_indicators[f'EMA{period}_close'] = btc['Close'].ewm(span=period, adjust=False).mean()
    print(f"   ✓ EMA{period}_close")

# EMA - 거래량 기준
for period in [100, 200]:
    btc_with_indicators[f'EMA{period}_volume'] = btc['Volume'].ewm(span=period, adjust=False).mean()
    print(f"   ✓ EMA{period}_volume")

# SMA - 종가 기준
sma_periods = [5, 10, 20, 30]
for period in sma_periods:
    btc_with_indicators[f'SMA{period}_close'] = btc['Close'].rolling(window=period).mean()
    print(f"   ✓ SMA{period}_close")

# 시가총액 근사 (Close * Volume)
btc_with_indicators['market_cap_approx'] = btc['Close'] * btc['Volume']

# EMA - 시가총액 기준
for period in ema_periods:
    btc_with_indicators[f'EMA{period}_marketcap'] = btc_with_indicators['market_cap_approx'].ewm(span=period, adjust=False).mean()
    print(f"   ✓ EMA{period}_marketcap")

# SMA - 시가총액 기준
for period in sma_periods:
    btc_with_indicators[f'SMA{period}_marketcap'] = btc_with_indicators['market_cap_approx'].rolling(window=period).mean()
    print(f"   ✓ SMA{period}_marketcap")

print("\n3. 기술적 지표 (RSI, MACD, Bollinger Bands 등) 생성 중...")

# RSI (Relative Strength Index)
btc_with_indicators['RSI'] = ta.momentum.RSIIndicator(btc['Close'], window=14).rsi()
print("   ✓ RSI")

# MACD
macd = ta.trend.MACD(btc['Close'])
btc_with_indicators['MACD'] = macd.macd()
btc_with_indicators['MACD_signal'] = macd.macd_signal()
btc_with_indicators['MACD_diff'] = macd.macd_diff()
print("   ✓ MACD, MACD_signal, MACD_diff")

# Bollinger Bands
bollinger = ta.volatility.BollingerBands(btc['Close'], window=20, window_dev=2)
btc_with_indicators['BB_high'] = bollinger.bollinger_hband()
btc_with_indicators['BB_low'] = bollinger.bollinger_lband()
btc_with_indicators['BB_mid'] = bollinger.bollinger_mavg()
btc_with_indicators['BB_width'] = bollinger.bollinger_wband()
print("   ✓ Bollinger Bands (high, low, mid, width)")

# ATR (Average True Range)
btc_with_indicators['ATR'] = ta.volatility.AverageTrueRange(
    btc['High'], btc['Low'], btc['Close'], window=14
).average_true_range()
print("   ✓ ATR")

# OBV (On Balance Volume)
btc_with_indicators['OBV'] = ta.volume.OnBalanceVolumeIndicator(
    btc['Close'], btc['Volume']
).on_balance_volume()
print("   ✓ OBV")

# Stochastic Oscillator
stoch = ta.momentum.StochasticOscillator(btc['High'], btc['Low'], btc['Close'])
btc_with_indicators['Stoch_K'] = stoch.stoch()
btc_with_indicators['Stoch_D'] = stoch.stoch_signal()
print("   ✓ Stochastic (K, D)")

# ADX (Average Directional Index)
btc_with_indicators['ADX'] = ta.trend.ADXIndicator(
    btc['High'], btc['Low'], btc['Close'], window=14
).adx()
print("   ✓ ADX")

# CCI (Commodity Channel Index)
btc_with_indicators['CCI'] = ta.trend.CCIIndicator(
    btc['High'], btc['Low'], btc['Close'], window=20
).cci()
print("   ✓ CCI")

# Williams %R
btc_with_indicators['Williams_R'] = ta.momentum.WilliamsRIndicator(
    btc['High'], btc['Low'], btc['Close'], lbp=14
).williams_r()
print("   ✓ Williams %R")

# ROC (Rate of Change)
btc_with_indicators['ROC'] = ta.momentum.ROCIndicator(btc['Close'], window=12).roc()
print("   ✓ ROC")

# MFI (Money Flow Index)
btc_with_indicators['MFI'] = ta.volume.MFIIndicator(
    btc['High'], btc['Low'], btc['Close'], btc['Volume'], window=14
).money_flow_index()
print("   ✓ MFI")

print("\n4. 수익률 및 변동성 지표 생성 중...")

# 일별 수익률
btc_with_indicators['daily_return'] = btc['Close'].pct_change() * 100
print("   ✓ daily_return")

# 누적 수익률
btc_with_indicators['cumulative_return'] = ((btc['Close'] / btc['Close'].iloc[0]) - 1) * 100
print("   ✓ cumulative_return")

# 변동성 (20일 표준편차)
btc_with_indicators['volatility_20d'] = btc['Close'].pct_change().rolling(window=20).std() * 100
print("   ✓ volatility_20d")

# 거래량 변화율
btc_with_indicators['volume_change'] = btc['Volume'].pct_change() * 100
print("   ✓ volume_change")

print("\n5. 결측치 처리 중...")
# 결측치 확인
null_counts = btc_with_indicators.isnull().sum()
print(f"   결측치가 있는 컬럼 수: {(null_counts > 0).sum()}")

# Forward fill (초기 이동평균 계산에서 생긴 NaN)
btc_with_indicators = btc_with_indicators.fillna(method='ffill')

# 남은 결측치 제거 (첫 200일 정도)
btc_with_indicators = btc_with_indicators.dropna()

print(f"   처리 후 데이터 형태: {btc_with_indicators.shape}")

print("\n6. 파일 저장 중...")
btc_with_indicators.to_csv('btc_technical_indicators.csv')
print("   ✓ 저장 완료: btc_technical_indicators.csv")

print("\n" + "=" * 70)
print("통계 요약")
print("=" * 70)
print(f"총 컬럼 수: {len(btc_with_indicators.columns)}")
print(f"원본 컬럼: {len(btc.columns)}")
print(f"추가된 기술적 지표: {len(btc_with_indicators.columns) - len(btc.columns)}")
print(f"\n데이터 기간: {btc_with_indicators.index[0]} ~ {btc_with_indicators.index[-1]}")
print(f"총 데이터 수: {len(btc_with_indicators)}")

print("\n생성된 지표 카테고리:")
print(f"  - 이동평균 (EMA/SMA): {len([c for c in btc_with_indicators.columns if 'EMA' in c or 'SMA' in c])}개")
print(f"  - 모멘텀 지표 (RSI, Stoch 등): {len([c for c in btc_with_indicators.columns if any(x in c for x in ['RSI', 'Stoch', 'Williams', 'ROC', 'MFI'])])}개")
print(f"  - 트렌드 지표 (MACD, ADX 등): {len([c for c in btc_with_indicators.columns if any(x in c for x in ['MACD', 'ADX', 'CCI'])])}개")
print(f"  - 변동성 지표 (BB, ATR 등): {len([c for c in btc_with_indicators.columns if any(x in c for x in ['BB', 'ATR', 'volatility'])])}개")
print(f"  - 거래량 지표 (OBV 등): {len([c for c in btc_with_indicators.columns if 'OBV' in c or 'volume' in c])}개")

print("\n" + "=" * 70)
print("Step 1.1 완료!")
print("=" * 70)
