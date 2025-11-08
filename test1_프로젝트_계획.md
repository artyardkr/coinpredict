# Test1: 논문1 방법론 기반 BTC 종가 예측 프로젝트

## 프로젝트 개요

**목표**: 논문 1의 방법론을 따라 다양한 데이터 소스를 활용한 BTC 종가 예측

**핵심 방법론**:
- Feature Reduction Algorithm (FRA)
- Random Forest & XGBoost
- 다양한 데이터 카테고리 통합
- ETF 전후 비교 분석

**예측 대상**: Crypto100 대신 **BTC 종가** 직접 예측

---

## Phase 0: 현재 상태 확인 ✓

### 이미 완료된 작업
- ✅ BTC, ETH, SOL, DOGE, XRP 가격 데이터 수집 (2021-2025)
- ✅ ETF 전후 비교 분석 (2024.01.10 기준)
- ✅ 기본 통계 분석 완료
- ✅ 논문 1, 2 분석 완료

### 현재 데이터
```python
# 파일 목록
btc_data_2021_2025.csv
eth_data_2021_2025.csv
sol_data_2021_2025.csv
doge_data_2021_2025.csv
xrp_data_2021_2025.csv
crypto_close_prices_2021_2025.csv
crypto_volumes_2021_2025.csv
```

---

## Phase 1: 데이터 준비 (1주)

### Step 1.1: 기술적 지표 생성 (1일)

**목표**: BTC 가격 데이터로부터 기술적 지표 계산

#### 구현할 지표

**이동평균 (Moving Averages)**:
```python
import pandas as pd
import ta  # Technical Analysis 라이브러리

# EMA (Exponential Moving Average)
ema_periods = [5, 10, 14, 20, 30, 100, 200]
for period in ema_periods:
    df[f'EMA{period}_close'] = df['Close'].ewm(span=period).mean()
    df[f'EMA{period}_volume'] = df['Volume'].ewm(span=period).mean()

# SMA (Simple Moving Average)
sma_periods = [5, 10, 20, 30]
for period in sma_periods:
    df[f'SMA{period}_close'] = df['Close'].rolling(window=period).mean()

# 시가총액 계산 (Close * Volume로 근사)
df['market_cap'] = df['Close'] * df['Volume']
for period in ema_periods:
    df[f'EMA{period}_marketcap'] = df['market_cap'].ewm(span=period).mean()
```

**기타 기술적 지표**:
```python
# RSI (Relative Strength Index)
df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

# MACD
macd = ta.trend.MACD(df['Close'])
df['MACD'] = macd.macd()
df['MACD_signal'] = macd.macd_signal()
df['MACD_diff'] = macd.macd_diff()

# Bollinger Bands
bollinger = ta.volatility.BollingerBands(df['Close'])
df['BB_high'] = bollinger.bollinger_hband()
df['BB_low'] = bollinger.bollinger_lband()
df['BB_mid'] = bollinger.bollinger_mavg()
df['BB_width'] = bollinger.bollinger_wband()

# ATR (Average True Range)
df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()

# OBV (On Balance Volume)
df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
```

**결과 파일**:
```
btc_technical_indicators.csv  # 약 50개 컬럼
```

### Step 1.2: 전통 시장 지수 수집 (0.5일)

**목표**: yfinance로 전통 시장 데이터 수집

```python
import yfinance as yf

# 수집 기간
start_date = "2021-01-01"
end_date = "2025-10-15"

# 전통 시장 지수
tickers = {
    'QQQ': 'Nasdaq-100',
    '^GSPC': 'S&P 500',
    'UUP': '달러 인덱스',
    'EURUSD=X': '유로/달러',
    'GC=F': '금 선물',
    'SI=F': '은 선물',
    'CL=F': '원유 선물',
    'BSV': '단기채권 ETF',
}

traditional_data = {}
for ticker, name in tickers.items():
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        traditional_data[ticker] = data['Close']
    except:
        print(f"Failed to download {ticker}")

# 통합
traditional_df = pd.DataFrame(traditional_data)
traditional_df.to_csv('traditional_market_indices.csv')
```

**결과 파일**:
```
traditional_market_indices.csv  # 8개 컬럼
```

### Step 1.3: 거시경제 데이터 수집 (FRED API) (0.5일)

**목표**: 미국 연방준비제도 경제 데이터

```python
from fredapi import Fred
import os

# FRED API Key 필요 (무료 등록)
# https://fred.stlouisfed.org/docs/api/api_key.html
fred = Fred(api_key='YOUR_API_KEY')

# 거시경제 지표
indicators = {
    'FEDFUNDS': 'Fed Funds Rate',
    'DGS10': '10년 국채 수익률',
    'CPIAUCSL': 'CPI (인플레이션)',
    'UNRATE': '실업률',
    'M2SL': 'M2 통화량',
    'DEXUSEU': 'USD/EUR',
}

macro_data = {}
for code, name in indicators.items():
    try:
        data = fred.get_series(code, start_date, end_date)
        macro_data[code] = data
    except:
        print(f"Failed to download {code}")

# 통합 (일별 데이터로 리샘플링)
macro_df = pd.DataFrame(macro_data)
macro_df = macro_df.resample('D').ffill()  # 주말/공휴일 forward fill
macro_df.to_csv('macro_indicators.csv')
```

**결과 파일**:
```
macro_indicators.csv  # 6개 컬럼
```

### Step 1.4: 감정/관심 지표 수집 (선택) (0.5일)

**목표**: Fear & Greed Index, Google Trends

#### Fear & Greed Index (무료)
```python
import requests

# Alternative.me API (무료)
def get_fear_greed_index(days=365*5):
    url = f"https://api.alternative.me/fng/?limit={days}"
    response = requests.get(url)
    data = response.json()['data']

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['value'] = df['value'].astype(int)
    df.set_index('timestamp', inplace=True)
    return df[['value', 'value_classification']]

fg_index = get_fear_greed_index()
fg_index.to_csv('fear_greed_index.csv')
```

#### Google Trends (무료)
```python
from pytrends.request import TrendReq

pytrends = TrendReq(hl='en-US', tz=360)

# 검색어
keywords = ['Bitcoin', 'Cryptocurrency', 'Ethereum']

# 기간별 수집 (5년치)
trends_data = {}
for keyword in keywords:
    pytrends.build_payload([keyword], timeframe='2021-01-01 2025-10-15')
    trends_data[f'gt_{keyword}'] = pytrends.interest_over_time()[keyword]

trends_df = pd.DataFrame(trends_data)
trends_df.to_csv('google_trends.csv')
```

**결과 파일**:
```
fear_greed_index.csv     # 2개 컬럼
google_trends.csv        # 3개 컬럼
```

### Step 1.5: 온체인 데이터 수집 (선택/제한적) (1일)

#### Option A: Blockchain.com API (완전 무료)

**기본 지표만 가능**:
```python
import requests

def get_blockchain_data(stat='n-transactions', timespan='2years'):
    """
    Available stats:
    - n-transactions: 거래 건수
    - hash-rate: 해시레이트
    - difficulty: 채굴 난이도
    - market-price: 시장 가격
    - total-bitcoins: 총 발행량
    - trade-volume: 거래량
    """
    url = f"https://api.blockchain.info/charts/{stat}"
    params = {'timespan': timespan, 'format': 'json'}
    response = requests.get(url, params=params)
    data = response.json()

    df = pd.DataFrame(data['values'])
    df['x'] = pd.to_datetime(df['x'], unit='s')
    df.set_index('x', inplace=True)
    return df['y']

# 수집
onchain_data = {}
stats = ['n-transactions', 'hash-rate', 'difficulty', 'trade-volume']
for stat in stats:
    onchain_data[stat] = get_blockchain_data(stat, timespan='5years')

onchain_df = pd.DataFrame(onchain_data)
onchain_df.to_csv('onchain_basic.csv')
```

#### Option B: Glassnode 무료 티어 (제한적)

**설치**:
```bash
pip install glassnode
```

**사용**:
```python
from glassnode import GlassnodeClient

# API Key 필요 (무료 가입)
client = GlassnodeClient()
client.set_api_key('YOUR_API_KEY')

# 무료 티어 지표 (제한적)
metrics = [
    'addresses/active_count',
    'transactions/count',
    # 유료: 'supply/distribution', 'market/mvrv', etc.
]

# 데이터 수집 생략 (API 제한)
```

**결과 파일**:
```
onchain_basic.csv        # 4개 컬럼 (기본만)
```

### Step 1.6: 데이터 통합 (1일)

**목표**: 모든 데이터를 하나의 DataFrame으로 통합

```python
import pandas as pd

# 1. BTC 가격 데이터 (베이스)
btc = pd.read_csv('btc_data_2021_2025.csv', index_col=0, parse_dates=True)

# 2. 기술적 지표
technical = pd.read_csv('btc_technical_indicators.csv', index_col=0, parse_dates=True)

# 3. 전통 시장 지수
traditional = pd.read_csv('traditional_market_indices.csv', index_col=0, parse_dates=True)

# 4. 거시경제 지표
macro = pd.read_csv('macro_indicators.csv', index_col=0, parse_dates=True)

# 5. 감정/관심 지표 (선택)
fear_greed = pd.read_csv('fear_greed_index.csv', index_col=0, parse_dates=True)
trends = pd.read_csv('google_trends.csv', index_col=0, parse_dates=True)

# 6. 온체인 데이터 (선택)
onchain = pd.read_csv('onchain_basic.csv', index_col=0, parse_dates=True)

# 통합
master_df = btc.copy()
master_df = master_df.join(technical, how='left')
master_df = master_df.join(traditional, how='left')
master_df = master_df.join(macro, how='left')
master_df = master_df.join(fear_greed, how='left')
master_df = master_df.join(trends, how='left')
master_df = master_df.join(onchain, how='left')

# Forward fill (주말/공휴일 데이터)
master_df = master_df.fillna(method='ffill')

# 결측치 처리
master_df = master_df.dropna()

print(f"통합 데이터 shape: {master_df.shape}")
print(f"변수 개수: {len(master_df.columns)}")

master_df.to_csv('master_dataset.csv')
```

**예상 변수 개수**:
- BTC 가격: 7개 (Open, High, Low, Close, Volume, Dividends, Stock Splits)
- 기술적 지표: 50개
- 전통 시장: 8개
- 거시경제: 6개
- 감정/관심: 5개
- 온체인: 4개
- **총: 약 80개 변수**

**결과 파일**:
```
master_dataset.csv       # 80개 컬럼, 1700+ rows
```

---

## Phase 2: Feature Engineering (3일)

### Step 2.1: 타겟 변수 생성 (0.5일)

**목표**: 여러 예측 기간에 대한 타겟 생성

```python
# 예측 기간 (논문1 따라)
horizons = [1, 7, 30, 90, 180]

for h in horizons:
    # 미래 종가
    master_df[f'target_{h}d'] = master_df['Close'].shift(-h)

    # 수익률 (선택)
    master_df[f'target_return_{h}d'] = (master_df['Close'].shift(-h) / master_df['Close'] - 1) * 100

    # 방향 (상승/하락) - 분류 문제
    master_df[f'target_direction_{h}d'] = (master_df['Close'].shift(-h) > master_df['Close']).astype(int)

# 결측치 제거 (마지막 180일)
master_df = master_df.dropna()

master_df.to_csv('master_dataset_with_targets.csv')
```

### Step 2.2: Feature Selection - Pearson Correlation (0.5일)

**목표**: 타겟과의 상관관계 계산

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1일 예측 타겟으로 분석
target = 'target_1d'

# 상관관계 계산
correlations = {}
for col in master_df.columns:
    if col not in [target] and col not in ['Open', 'High', 'Low', 'Close']:
        corr = master_df[col].corr(master_df[target])
        correlations[col] = abs(corr)

# 정렬
corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['correlation'])
corr_df = corr_df.sort_values('correlation', ascending=False)

# 상위 50개 시각화
plt.figure(figsize=(10, 12))
corr_df.head(50).plot(kind='barh')
plt.title('Top 50 Features by Correlation with Target')
plt.xlabel('Absolute Correlation')
plt.tight_layout()
plt.savefig('correlation_analysis.png')

corr_df.to_csv('feature_correlations.csv')
```

### Step 2.3: Feature Selection - Tree-based Importance (1일)

**목표**: Random Forest와 XGBoost로 Feature Importance 계산

```python
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# 데이터 준비
X = master_df.drop(['target_1d', 'target_return_1d', 'target_direction_1d'], axis=1)
y = master_df['target_1d']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Random Forest
print("Training Random Forest...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Feature Importance (MDI)
rf_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# XGBoost
print("Training XGBoost...")
xgb = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train)

# Feature Importance
xgb_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb.feature_importances_
}).sort_values('importance', ascending=False)

# 저장
rf_importance.to_csv('rf_feature_importance.csv', index=False)
xgb_importance.to_csv('xgb_feature_importance.csv', index=False)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
rf_importance.head(30).plot(x='feature', y='importance', kind='barh', ax=axes[0])
axes[0].set_title('Random Forest - Top 30 Features')
xgb_importance.head(30).plot(x='feature', y='importance', kind='barh', ax=axes[1])
axes[1].set_title('XGBoost - Top 30 Features')
plt.tight_layout()
plt.savefig('tree_importance.png')
```

### Step 2.4: Feature Selection - Permutation Importance (0.5일)

**목표**: PFI 계산

```python
from sklearn.inspection import permutation_importance

# Random Forest PFI
print("Calculating RF Permutation Importance...")
rf_perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
rf_perm_df = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_perm.importances_mean
}).sort_values('importance', ascending=False)

# XGBoost PFI
print("Calculating XGB Permutation Importance...")
xgb_perm = permutation_importance(xgb, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
xgb_perm_df = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_perm.importances_mean
}).sort_values('importance', ascending=False)

# 저장
rf_perm_df.to_csv('rf_permutation_importance.csv', index=False)
xgb_perm_df.to_csv('xgb_permutation_importance.csv', index=False)
```

### Step 2.5: Feature Reduction Algorithm (FRA) 구현 (1일)

**목표**: 논문1의 FRA 알고리즘 구현

```python
def feature_reduction_algorithm(X, y, target_features=50, corr_threshold_start=0.5, corr_increment=0.025):
    """
    논문1의 FRA 알고리즘 구현

    Parameters:
    - X: Feature DataFrame
    - y: Target Series
    - target_features: 목표 feature 개수
    - corr_threshold_start: 초기 상관계수 임계값
    - corr_increment: 임계값 증가량
    """

    current_features = X.columns.tolist()
    corr_threshold = corr_threshold_start

    while len(current_features) > target_features:
        print(f"Current features: {len(current_features)}, Threshold: {corr_threshold:.3f}")

        X_current = X[current_features]

        # 1. Pearson Correlation
        correlations = X_current.corrwith(y).abs()

        # 2. Random Forest MDI
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X_current, y)
        rf_importance = pd.Series(rf.feature_importances_, index=current_features)

        # 3. XGBoost MDI
        xgb = XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        xgb.fit(X_current, y)
        xgb_importance = pd.Series(xgb.feature_importances_, index=current_features)

        # 4. Rank 계산 (각 방법에서의 순위)
        corr_rank = correlations.rank(ascending=False)
        rf_rank = rf_importance.rank(ascending=False)
        xgb_rank = xgb_importance.rank(ascending=False)

        # 평균 순위
        avg_rank = (corr_rank + rf_rank + xgb_rank) / 3

        # 하위 50% 중 상관계수 낮은 feature 제거
        bottom_50_pct = avg_rank > (len(current_features) / 2)
        low_corr = correlations < corr_threshold

        to_remove = current_features[bottom_50_pct & low_corr]

        if len(to_remove) == 0:
            corr_threshold += corr_increment
            continue

        current_features = [f for f in current_features if f not in to_remove]

        if len(current_features) <= target_features:
            break

    return current_features

# 실행 (1일 예측 기준)
X = master_df.drop(['target_1d', 'target_return_1d', 'target_direction_1d'], axis=1)
y = master_df['target_1d']

selected_features = feature_reduction_algorithm(X, y, target_features=50)

print(f"\nSelected {len(selected_features)} features:")
print(selected_features)

# 저장
pd.DataFrame({'feature': selected_features}).to_csv('selected_features_fra.csv', index=False)
```

### Step 2.6: SHAP 분석 (선택) (0.5일)

**목표**: SHAP으로 feature importance 검증

```python
import shap

# XGBoost로 SHAP 계산
X_selected = X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, shuffle=False)

xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_train, y_train)

# SHAP values
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png')

# Feature importance
shap_importance = pd.DataFrame({
    'feature': X_test.columns,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

shap_importance.to_csv('shap_importance.csv', index=False)
```

---

## Phase 3: 모델 학습 및 평가 (2주)

### Step 3.1: 베이스라인 모델 (1일)

**목표**: 단순 모델로 베이스라인 설정

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 데이터 준비
X = master_df[selected_features]
y = master_df['target_1d']

# ETF 전후 분리
etf_date = pd.Timestamp('2024-01-10', tz='UTC')
train_mask = master_df.index < etf_date
test_mask = master_df.index >= etf_date

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

# 베이스라인 1: Last Value (Naive)
y_pred_naive = master_df['Close'][train_mask].iloc[-1]
mse_naive = mean_squared_error(y_test, [y_pred_naive] * len(y_test))
print(f"Naive MSE: {mse_naive:.2f}")

# 베이스라인 2: Moving Average
y_pred_ma = master_df['Close'][train_mask].rolling(7).mean().iloc[-1]
mse_ma = mean_squared_error(y_test, [y_pred_ma] * len(y_test))
print(f"Moving Average MSE: {mse_ma:.2f}")

# 베이스라인 3: Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
print(f"Linear Regression MSE: {mse_lr:.2f}")

# 저장
baseline_results = pd.DataFrame({
    'model': ['Naive', 'Moving Average', 'Linear Regression'],
    'mse': [mse_naive, mse_ma, mse_lr]
})
baseline_results.to_csv('baseline_results.csv', index=False)
```

### Step 3.2: Random Forest 학습 (2일)

**목표**: Grid Search로 최적 하이퍼파라미터 찾기

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# Time Series Split (시계열 데이터용)
tscv = TimeSeriesSplit(n_splits=5)

# 하이퍼파라미터 그리드
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Grid Search
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    rf, param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error',
    verbose=2,
    n_jobs=-1
)

print("Starting Grid Search for Random Forest...")
grid_search.fit(X_train, y_train)

# 최적 모델
best_rf = grid_search.best_estimator_
print(f"\nBest parameters: {grid_search.best_params_}")

# 예측 및 평가
y_pred_rf = best_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest MSE: {mse_rf:.2f}")
print(f"Random Forest MAE: {mae_rf:.2f}")
print(f"Random Forest R²: {r2_rf:.3f}")

# 모델 저장
import joblib
joblib.dump(best_rf, 'best_random_forest.pkl')

# 결과 저장
pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred_rf
}, index=y_test.index).to_csv('rf_predictions.csv')
```

### Step 3.3: XGBoost 학습 (2일)

**목표**: XGBoost 최적화 및 학습

```python
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

# 하이퍼파라미터 그리드
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5]
}

# Grid Search
xgb = XGBRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    xgb, param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error',
    verbose=2,
    n_jobs=-1
)

print("Starting Grid Search for XGBoost...")
grid_search.fit(X_train, y_train)

# 최적 모델
best_xgb = grid_search.best_estimator_
print(f"\nBest parameters: {grid_search.best_params_}")

# 예측 및 평가
y_pred_xgb = best_xgb.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"XGBoost MSE: {mse_xgb:.2f}")
print(f"XGBoost MAE: {mae_xgb:.2f}")
print(f"XGBoost R²: {r2_xgb:.3f}")

# 모델 저장
joblib.dump(best_xgb, 'best_xgboost.pkl')

# 결과 저장
pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred_xgb
}, index=y_test.index).to_csv('xgb_predictions.csv')
```

### Step 3.4: 다양한 예측 기간 실험 (3일)

**목표**: 1, 7, 30일 예측 각각 수행

```python
horizons = [1, 7, 30]
results = {}

for h in horizons:
    print(f"\n{'='*50}")
    print(f"Training models for {h}-day prediction")
    print(f"{'='*50}")

    y = master_df[f'target_{h}d']

    # 데이터 분할
    y_train = y[train_mask]
    y_test = y[test_mask]

    # Random Forest
    rf = RandomForestRegressor(**grid_search.best_params_, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # XGBoost
    xgb = XGBRegressor(**grid_search.best_params_, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)

    # 평가
    results[h] = {
        'rf_mse': mean_squared_error(y_test, y_pred_rf),
        'rf_mae': mean_absolute_error(y_test, y_pred_rf),
        'rf_r2': r2_score(y_test, y_pred_rf),
        'xgb_mse': mean_squared_error(y_test, y_pred_xgb),
        'xgb_mae': mean_absolute_error(y_test, y_pred_xgb),
        'xgb_r2': r2_score(y_test, y_pred_xgb),
    }

    # 저장
    joblib.dump(rf, f'rf_{h}day.pkl')
    joblib.dump(xgb, f'xgb_{h}day.pkl')

# 결과 정리
results_df = pd.DataFrame(results).T
results_df.to_csv('multi_horizon_results.csv')
print("\n" + results_df.to_string())
```

### Step 3.5: ETF 전후 비교 (2일)

**목표**: 2024.01.10 전후로 모델 성능 비교

```python
# ETF 이전 데이터로만 학습
pre_etf_mask = master_df.index < etf_date
X_pre = X[pre_etf_mask]
y_pre = y[pre_etf_mask]

# 80/20 분할 (ETF 이전만)
split_idx = int(len(X_pre) * 0.8)
X_train_pre, X_val_pre = X_pre[:split_idx], X_pre[split_idx:]
y_train_pre, y_val_pre = y_pre[:split_idx], y_pre[split_idx:]

# 모델 학습
rf_pre = RandomForestRegressor(**best_params, random_state=42)
rf_pre.fit(X_train_pre, y_train_pre)

# ETF 이후 데이터로 테스트
post_etf_mask = master_df.index >= etf_date
X_post = X[post_etf_mask]
y_post = y[post_etf_mask]

y_pred_post = rf_pre.predict(X_post)
mse_post = mean_squared_error(y_post, y_pred_post)

# 비교
print(f"Pre-ETF validation MSE: {mean_squared_error(y_val_pre, rf_pre.predict(X_val_pre)):.2f}")
print(f"Post-ETF test MSE: {mse_post:.2f}")
print(f"Performance degradation: {((mse_post / mean_squared_error(y_val_pre, rf_pre.predict(X_val_pre))) - 1) * 100:.1f}%")

# ETF 이후 데이터로 재학습
X_all = X[pre_etf_mask | post_etf_mask]
y_all = y[pre_etf_mask | post_etf_mask]

rf_all = RandomForestRegressor(**best_params, random_state=42)
rf_all.fit(X_all, y_all)

# 비교
comparison = pd.DataFrame({
    'period': ['Pre-ETF only', 'With Post-ETF'],
    'mse': [mse_post, mean_squared_error(y_post, rf_all.predict(X_post))]
})
comparison.to_csv('etf_comparison.csv', index=False)
```

### Step 3.6: 모델 해석 (1일)

**목표**: Feature Importance 분석 및 시각화

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Feature Importance 추출
rf_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

xgb_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': best_xgb.feature_importances_
}).sort_values('importance', ascending=False)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

rf_importance.head(20).plot(x='feature', y='importance', kind='barh', ax=axes[0])
axes[0].set_title('Random Forest - Top 20 Features')
axes[0].set_xlabel('Importance')

xgb_importance.head(20).plot(x='feature', y='importance', kind='barh', ax=axes[1])
axes[1].set_title('XGBoost - Top 20 Features')
axes[1].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('final_feature_importance.png', dpi=300)

# 카테고리별 중요도
categories = {
    'Technical': ['EMA', 'SMA', 'RSI', 'MACD', 'BB', 'ATR'],
    'Traditional': ['QQQ', 'GSPC', 'UUP', 'EURUSD', 'GC', 'CL', 'BSV'],
    'Macro': ['FEDFUNDS', 'DGS10', 'CPIAUCSL', 'UNRATE', 'M2SL'],
    'Sentiment': ['fear', 'greed', 'gt_'],
    'Onchain': ['hash', 'difficulty', 'transactions']
}

def categorize_feature(feature):
    for category, keywords in categories.items():
        if any(kw in feature for kw in keywords):
            return category
    return 'Other'

rf_importance['category'] = rf_importance['feature'].apply(categorize_feature)
category_importance = rf_importance.groupby('category')['importance'].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
category_importance.plot(kind='bar')
plt.title('Feature Importance by Category')
plt.ylabel('Total Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('category_importance.png', dpi=300)
```

---

## Phase 4: 결과 분석 및 리포트 (3일)

### Step 4.1: 예측 성능 시각화 (1일)

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 실제값 vs 예측값
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# Random Forest
axes[0].plot(y_test.index, y_test.values, label='Actual', alpha=0.7)
axes[0].plot(y_test.index, y_pred_rf, label='RF Predicted', alpha=0.7)
axes[0].set_title('Random Forest: Actual vs Predicted')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# XGBoost
axes[1].plot(y_test.index, y_test.values, label='Actual', alpha=0.7)
axes[1].plot(y_test.index, y_pred_xgb, label='XGB Predicted', alpha=0.7)
axes[1].set_title('XGBoost: Actual vs Predicted')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('predictions_comparison.png', dpi=300)

# 오차 분석
errors_rf = y_test - y_pred_rf
errors_xgb = y_test - y_pred_xgb

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].hist(errors_rf, bins=50, alpha=0.7, edgecolor='black')
axes[0].set_title('Random Forest Error Distribution')
axes[0].set_xlabel('Prediction Error')
axes[0].axvline(0, color='red', linestyle='--', linewidth=2)

axes[1].hist(errors_xgb, bins=50, alpha=0.7, edgecolor='black')
axes[1].set_title('XGBoost Error Distribution')
axes[1].set_xlabel('Prediction Error')
axes[1].axvline(0, color='red', linestyle='--', linewidth=2)

plt.tight_layout()
plt.savefig('error_distribution.png', dpi=300)
```

### Step 4.2: 성능 메트릭 정리 (0.5일)

```python
# 종합 성능 표
performance = pd.DataFrame({
    'Model': ['Naive', 'Moving Average', 'Linear Regression', 'Random Forest', 'XGBoost'],
    'MSE': [mse_naive, mse_ma, mse_lr, mse_rf, mse_xgb],
    'MAE': [np.nan, np.nan, mae_lr, mae_rf, mae_xgb],
    'R²': [np.nan, np.nan, r2_score(y_test, y_pred_lr), r2_rf, r2_xgb]
})

# 개선율 계산
performance['MSE_improvement_%'] = ((performance['MSE'].iloc[0] - performance['MSE']) / performance['MSE'].iloc[0]) * 100

print(performance.to_string(index=False))
performance.to_csv('performance_summary.csv', index=False)

# 예측 기간별 성능
multi_horizon_results.to_csv('multi_horizon_performance.csv')
```

### Step 4.3: 최종 리포트 작성 (1.5일)

**목표**: 마크다운으로 상세 리포트 작성

```markdown
# Test1: BTC 종가 예측 프로젝트 결과 리포트

## 1. 프로젝트 개요
- 목표: 논문1 방법론 기반 BTC 종가 예측
- 기간: 2021-2025 (ETF 전후 포함)
- 데이터: 80개 변수 (5개 카테고리)

## 2. 데이터 수집
### 2.1 데이터 소스
- 가격 데이터: Yahoo Finance
- 거시경제: FRED API
- 감정: Fear & Greed Index, Google Trends
- (총 1748일 데이터)

### 2.2 Feature Engineering
- 기술적 지표: 50개
- 전통 시장: 8개
- 거시경제: 6개
- 감정: 5개
- 온체인: 4개 (기본)

## 3. Feature Selection
### 3.1 FRA 알고리즘
- 초기: 80개 변수
- 최종: 50개 변수 (FRA 적용)

### 3.2 주요 변수 (Top 10)
1. EMA200_close
2. RSI
3. QQQ_Close
4. ...

## 4. 모델 성능
### 4.1 1일 예측
| Model | MSE | MAE | R² |
|-------|-----|-----|-----|
| Random Forest | XXX | XXX | 0.XX |
| XGBoost | XXX | XXX | 0.XX |

### 4.2 ETF 전후 비교
- ETF 이전 MSE: XXX
- ETF 이후 MSE: XXX
- 성능 변화: +XX%

## 5. 주요 발견
1. 가장 중요한 변수: ...
2. ETF 이후 변화: ...
3. 카테고리별 기여도: ...

## 6. 한계 및 개선 방향
1. 온체인 데이터 제한적
2. 장기 예측 성능 낮음
3. ...

## 7. 결론
...
```

---

## Phase 5: 개선 및 확장 (선택)

### 추가 실험 아이디어

1. **다른 타겟 변수**:
   - 수익률 예측
   - 방향 분류 (상승/하락)
   - 변동성 예측

2. **앙상블 모델**:
   - RF + XGBoost Stacking
   - Voting Regressor

3. **딥러닝 모델 (논문2)**:
   - Helformer 구현
   - LSTM, GRU

4. **백테스팅**:
   - 간단한 트레이딩 전략
   - 수익률 계산

5. **실시간 예측**:
   - 최신 데이터 자동 업데이트
   - 일일 예측 파이프라인

---

## 예상 타임라인

| Phase | 작업 | 예상 시간 |
|-------|------|----------|
| 0 | 현재 상태 확인 | 완료 ✓ |
| 1 | 데이터 준비 | 1주 (5일) |
| 2 | Feature Engineering | 3일 |
| 3 | 모델 학습 및 평가 | 2주 (10일) |
| 4 | 결과 분석 및 리포트 | 3일 |
| **총** | | **약 3주** |

---

## 필요 라이브러리

```bash
# 데이터 수집
pip install yfinance fredapi pytrends glassnode

# 기술적 지표
pip install ta pandas-ta

# 머신러닝
pip install scikit-learn xgboost lightgbm

# 해석
pip install shap

# 시각화
pip install matplotlib seaborn plotly

# 유틸
pip install joblib tqdm
```

---

## 주요 산출물

### 데이터 파일
- `master_dataset.csv` - 통합 데이터셋
- `master_dataset_with_targets.csv` - 타겟 포함
- `selected_features_fra.csv` - 선택된 변수

### 모델 파일
- `best_random_forest.pkl`
- `best_xgboost.pkl`
- `rf_1day.pkl`, `rf_7day.pkl`, `rf_30day.pkl`
- `xgb_1day.pkl`, `xgb_7day.pkl`, `xgb_30day.pkl`

### 분석 결과
- `performance_summary.csv` - 성능 요약
- `multi_horizon_results.csv` - 다기간 예측 결과
- `etf_comparison.csv` - ETF 전후 비교
- `feature_importance.csv` - 변수 중요도

### 시각화
- `correlation_analysis.png`
- `tree_importance.png`
- `shap_summary.png`
- `predictions_comparison.png`
- `error_distribution.png`
- `category_importance.png`

### 리포트
- `test1_final_report.md` - 최종 리포트

---

## 성공 기준

### 최소 목표
- [ ] 80개 변수 수집 완료
- [ ] FRA로 50개 변수 선택
- [ ] Random Forest & XGBoost 학습 완료
- [ ] ETF 전후 비교 분석
- [ ] 베이스라인 대비 성능 개선 확인

### 도전 목표
- [ ] R² > 0.7 달성
- [ ] 논문1과 유사한 성능 개선율 (400%+)
- [ ] 7일/30일 예측도 안정적
- [ ] ETF 이후에도 강건한 성능
- [ ] 실제 트레이딩 전략 백테스팅

---

## 다음 단계 (Test2)

Test1 완료 후:
- **Test2**: Helformer 구현 (논문2)
- **Test3**: 하이브리드 모델 (논문1 데이터 + 논문2 모델)
- **Test4**: 실전 트레이딩 전략
- **Test5**: 포트폴리오 최적화

---

**문서 작성일**: 2025년
**프로젝트 기간**: 약 3주
**난이도**: 중급~고급
