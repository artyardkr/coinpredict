# 신규 변수 추가 문서 (New Variables Documentation)

## 📋 목차
1. [개요](#개요)
2. [추가 전통시장 데이터 (9개)](#1-추가-전통시장-데이터-9개)
3. [Fed 유동성 지표 (8개)](#2-fed-유동성-지표-8개)
4. [고급 온체인 지표 (21개)](#3-고급-온체인-지표-21개)
5. [Bitcoin ETF 데이터 (12개)](#4-bitcoin-etf-데이터-12개)
6. [실행 방법](#실행-방법)
7. [파일 구조](#파일-구조)
8. [참고 자료](#참고-자료)

---

## 개요

**작업 날짜**: 2025년 11월 2일
**목적**: Bitcoin 가격 예측 모델 성능 향상을 위한 외부 검색 기반 중요 변수 추가
**기존 변수**: 88개 → **신규 변수**: 138개 (+50개)
**데이터 기간**: 2021-02-03 ~ 2025-10-14 (1,715일)

### 변수 추가 근거

외부 웹 검색을 통해 2025년 기준 Bitcoin 가격 예측에 중요한 변수들을 조사:
- **학술 논문**: PMC, Springer, Frontiers in AI
- **업계 분석**: CoinGlass, Bitcoin Magazine, Bloomberg
- **전문가 의견**: JPMorgan, S&P Global

---

## 1. 추가 전통시장 데이터 (9개)

### 📄 생성 스크립트
`step2b_additional_markets.py`

### 📊 변수 목록

| 변수명 | 설명 | 데이터 소스 | 중요도 |
|--------|------|-------------|--------|
| `DIA` | 다우존스 산업평균 (Dow Jones Industrial Average) | Yahoo Finance | ⭐⭐⭐ |
| `IWM` | 러셀 2000 소형주 지수 (Russell 2000 Small-Cap) | Yahoo Finance | ⭐⭐ |
| `TLT` | 20년 장기 국채 ETF (iShares 20+ Year Treasury Bond) | Yahoo Finance | ⭐⭐⭐⭐ |
| `DXY` | 달러 지수 (US Dollar Index) | Yahoo Finance | ⭐⭐⭐⭐⭐ |
| `ETH` | 이더리움 가격 (Ethereum) | Yahoo Finance | ⭐⭐⭐⭐⭐ |
| `GLD` | 금 ETF (SPDR Gold Shares) | Yahoo Finance | ⭐⭐⭐⭐ |
| `HYG` | 하이일드 회사채 ETF (High Yield Corporate Bond) | Yahoo Finance | ⭐⭐⭐ |
| `LQD` | 투자등급 회사채 ETF (Investment Grade Corporate Bond) | Yahoo Finance | ⭐⭐⭐ |
| `VIX` | VIX 변동성 지수 (CBOE Volatility Index) | Yahoo Finance | ⭐⭐⭐ |

### 🔧 수집 방법

```python
import yfinance as yf

# 티커 리스트
tickers = {
    'DIA': '다우존스 산업평균',
    'IWM': '러셀 2000 소형주',
    'TLT': '20년 장기 국채 ETF',
    'DX-Y.NYB': '달러 지수',  # → DXY로 저장
    'ETH-USD': '이더리움',    # → ETH로 저장
    'GLD': '금 ETF',
    'HYG': '하이일드 회사채 ETF',
    'LQD': '투자등급 회사채 ETF',
    '^VIX': 'VIX 지수'        # → VIX로 저장
}

# 데이터 수집
for ticker in tickers:
    data = yf.download(ticker, start='2021-01-01', end='2025-11-01')
    market_data[ticker] = data['Close']
```

### 💡 중요도 이유

1. **DXY (달러 지수)**: 기존 UUP보다 표준적인 달러 강도 측정 지표
2. **ETH (이더리움)**: 암호화폐 시장 전반 건강도 측정 (연구에서 강조)
3. **TLT (장기 국채)**: 금리 정책과 리스크 선호도 측정
4. **GLD (금 ETF)**: Gold-BTC 자본 순환 분석 (가설3 검증용)
5. **HYG/LQD**: 신용 스프레드를 통한 시장 리스크 측정

### 📈 기본 통계

```
변수          평균         표준편차      최소         최대
DIA        353.93       51.14       272.30      477.15
IWM        198.03       20.92       156.83      250.33
TLT        --           --          --          --
DXY        --           --          --          --
ETH        2515.69      919.94      730.37      4831.35
GLD        --           --          --          --
HYG        69.48        5.30        59.34       81.28
LQD        103.49       7.13        87.26       116.27
VIX        19.40        5.34        11.86       52.33
```

### ⚠️ 주의사항

- **DST 문제**: `end_date = datetime.now() - timedelta(days=1)` 사용하여 일광절약시간 오류 회피
- **티커 명명**: Yahoo Finance 티커가 다를 수 있음 (DX-Y.NYB → DXY, ^VIX → VIX)
- **ETH 데이터**: 1,765개로 BTC보다 많음 (24/7 거래)

---

## 2. Fed 유동성 지표 (8개)

### 📄 생성 스크립트
`step3b_fed_liquidity.py`

### 📊 변수 목록

| 변수명 | 설명 | FRED Code | 중요도 | 갱신 빈도 |
|--------|------|-----------|--------|-----------|
| `WALCL` | Fed 총자산 (Total Assets) | WALCL | ⭐⭐⭐⭐⭐ | 주간 |
| `RRPONTSYD` | 역레포 (Reverse Repo) | RRPONTSYD | ⭐⭐⭐⭐⭐ | 일간 |
| `WTREGEN` | 재무부 계정 TGA (Treasury General Account) | WTREGEN | ⭐⭐⭐⭐ | 주간 |
| `T10Y3M` | 10년-3개월 스프레드 (10Y-3M Spread) | T10Y3M | ⭐⭐⭐⭐ | 일간 |
| `SOFR` | 담보부 익일물 금리 (Secured Overnight Financing Rate) | SOFR | ⭐⭐⭐ | 일간 |
| `BAMLH0A0HYM2` | 하이일드 스프레드 (High Yield Spread) | BAMLH0A0HYM2 | ⭐⭐⭐⭐ | 일간 |
| `BAMLC0A0CM` | 투자등급 스프레드 (Investment Grade Spread) | BAMLC0A0CM | ⭐⭐⭐ | 일간 |
| `FED_NET_LIQUIDITY` | Fed 순유동성 (계산값) | - | ⭐⭐⭐⭐⭐ | 주간 |

### 🔧 수집 방법

```python
from fredapi import Fred
import ssl

# SSL 인증서 설정 (필수!)
ssl._create_default_https_context = ssl._create_unverified_context

# FRED API 초기화
fred = Fred(api_key='YOUR_API_KEY')

# 데이터 수집
indicators = {
    'WALCL': 'Fed 총자산',
    'RRPONTSYD': '역레포',
    'WTREGEN': '재무부 계정',
    'T10Y3M': '10년-3개월 스프레드',
    'SOFR': '담보부 익일물 금리',
    'BAMLH0A0HYM2': '하이일드 스프레드',
    'BAMLC0A0CM': '투자등급 스프레드'
}

for code, name in indicators.items():
    data = fred.get_series(code, observation_start='2021-01-01')
    fed_data[code] = data
```

### 🧮 계산 지표

**FED_NET_LIQUIDITY (Fed 순유동성)**:
```python
FED_NET_LIQUIDITY = WALCL - RRPONTSYD - WTREGEN
```

이 지표가 **BTC 가격과 가장 높은 상관관계**를 보임 (연구 결과):
- 2018년 BTC 폭락 = Fed 자산 축소 (QT)
- 2020년 BTC +1000% = Fed QE 시작
- Fed Balance Sheet - RRP 피크 = BTC 피크

### 💡 중요도 이유

1. **WALCL (Fed 총자산)**: QE/QT 정책 직접 측정
2. **RRPONTSYD (역레포)**: 실제 시장 유동성 흡수량
3. **FED_NET_LIQUIDITY**: 실제 시장에 풀린 유동성 (가장 중요!)
4. **T10Y3M**: 경기침체 신호 (역전 시 침체 가능성 ↑)
5. **BAMLH0A0HYM2**: 신용 리스크와 위험 선호도

### 📈 최근 데이터 (2025년 10월 기준)

```
Fed 순유동성 변화 (최근 30일):
현재:     $0.00T
30일 전:  $0.00T
변화:     -$0.00T (-2.09%)
📉 유동성 감소 (BTC 약세 요인)
```

### ⚠️ 주의사항

- **API 키 필요**: FRED API 키 발급 필요 (https://fred.stlouisfed.org/)
- **SSL 인증서**: `ssl._create_unverified_context` 설정 필수
- **주간 데이터**: WALCL, WTREGEN은 주간 데이터 (252개)
- **결측치**: Forward fill로 일간 데이터로 변환 필요

---

## 3. 고급 온체인 지표 (21개)

### 📄 생성 스크립트
`step6b_advanced_onchain.py`

### 📊 변수 목록

#### 3.1 NVT Ratio (Network Value to Transactions) - 2개

| 변수명 | 설명 | 계산식 | 의미 |
|--------|------|--------|------|
| `NVT_Ratio` | NVT 비율 | `bc_market_cap / (bc_n_transactions + 1)` | 밸류에이션 지표 |
| `NVT_Ratio_MA90` | NVT 90일 이동평균 | `NVT_Ratio.rolling(90).mean()` | 추세 파악 |

**해석**:
- 높은 NVT = 과대평가 (거래량 대비 시총 높음)
- 낮은 NVT = 과소평가 (활발한 사용)

#### 3.2 Puell Multiple (채굴 수익성) - 1개

| 변수명 | 계산식 | 의미 |
|--------|--------|------|
| `Puell_Multiple` | `bc_miners_revenue / bc_miners_revenue.rolling(365).mean()` | 채굴자 수익성 |

**해석**:
- Puell > 2.0: 채굴자 과도한 수익 → 매도 압력 가능성
- Puell < 0.5: 채굴자 항복 → 바닥 신호

#### 3.3 Hash Ribbon (해시레이트 추세) - 3개

| 변수명 | 계산식 | 의미 |
|--------|--------|------|
| `Hash_Ribbon_MA30` | `bc_hash_rate.rolling(30).mean()` | 단기 해시레이트 |
| `Hash_Ribbon_MA60` | `bc_hash_rate.rolling(60).mean()` | 장기 해시레이트 |
| `Hash_Ribbon_Spread` | `(MA30 - MA60) / MA60 * 100` | 추세 강도 |

**해석**:
- MA30이 MA60 위로 교차: 채굴 회복 (매수 신호)
- MA30이 MA60 아래로 교차: 채굴자 항복 (바닥 근접)

#### 3.4 Difficulty Ribbon (난이도 추세) - 4개

| 변수명 | 계산식 |
|--------|--------|
| `Difficulty_MA30/60/90` | 30/60/90일 난이도 이동평균 |
| `Difficulty_Compression` | `(MA30 - MA90) / MA90 * 100` |

#### 3.5 Miner Position Index (채굴자 포지션) - 4개

| 변수명 | 계산식 | 의미 |
|--------|--------|------|
| `Miner_Revenue_to_Cap` | `bc_miners_revenue / bc_market_cap * 100` | 채굴 수익 비중 |
| `Miner_Revenue_to_Cap_MA30` | 30일 이동평균 | 추세 |
| `Hash_Price` | `bc_miners_revenue / bc_hash_rate` | 해시당 수익 (USD/TH/s) |
| `Hash_Price_MA90` | 90일 이동평균 | 장기 수익성 |

#### 3.6 Network Activity (네트워크 활동) - 4개

| 변수명 | 계산식 | 의미 |
|--------|--------|------|
| `Active_Addresses_Change` | `bc_n_unique_addresses.pct_change(30) * 100` | 활성 주소 변화율 |
| `Active_Addresses_MA90` | 90일 이동평균 | 네트워크 성장 추세 |
| `Avg_Fee_Per_Tx` | `bc_transaction_fees / bc_n_transactions` | 평균 거래 수수료 |
| `Avg_Fee_Per_Tx_MA30` | 30일 이동평균 | 수수료 추세 |

#### 3.7 Mempool Stress (멤풀 압력) - 1개

| 변수명 | 계산식 | 의미 |
|--------|--------|------|
| `Mempool_Stress` | `bc_mempool_size / bc_mempool_size.rolling(30).mean()` | 네트워크 혼잡도 |

#### 3.8 Simplified MVRV (간이 MVRV) - 2개

| 변수명 | 계산식 | 의미 |
|--------|--------|------|
| `Price_to_MA200` | `Close / Close.rolling(200).mean()` | 간이 MVRV |
| `Price_MA200` | 200일 이동평균 | "Fair Value" |

**해석**:
- Price/MA200 > 1.5: 과열
- Price/MA200 < 0.8: 과냉각

### 🔧 생성 방법

```python
import pandas as pd

# 기존 데이터 로드
btc_data = pd.read_csv('btc_technical_indicators.csv', index_col=0, parse_dates=True)
onchain_data = pd.read_csv('onchain_data.csv', index_col=0, parse_dates=True)

# 타임존 제거 (필수!)
btc_data.index = pd.to_datetime(btc_data.index).tz_localize(None)
onchain_data.index = pd.to_datetime(onchain_data.index).tz_localize(None)

# 데이터 결합
df = btc_data[['Close', 'Volume']].join(onchain_data, how='left')

# NVT Ratio 계산 예시
advanced_onchain['NVT_Ratio'] = df['bc_market_cap'] / (df['bc_n_transactions'] + 1)
advanced_onchain['NVT_Ratio_MA90'] = advanced_onchain['NVT_Ratio'].rolling(90).mean()

# Puell Multiple 계산 예시
revenue_ma365 = df['bc_miners_revenue'].rolling(365).mean()
advanced_onchain['Puell_Multiple'] = df['bc_miners_revenue'] / (revenue_ma365 + 1)
```

### 💡 중요도 이유

**웹 검색 결과 기반**:
1. **MVRV, SOPR, NVT**는 전문가들이 가장 많이 사용하는 온체인 지표
2. 기관들이 **머신러닝 모델**에 MVRV Z-score, SOPR 사용
3. **Puell Multiple**은 채굴자 항복 시점 예측에 효과적
4. **Hash Ribbon**은 바닥 타이밍에 높은 정확도

### ⚠️ 주의사항

- **실제 MVRV 불가**: Realized Price 데이터 필요 (유료 API)
- **대안**: Price/MA200을 간이 MVRV로 사용
- **결측치**: 이동평균 계산으로 초기 데이터 결측 발생 (정상)
- **+1 처리**: Division by zero 방지

---

## 4. Bitcoin ETF 데이터 (12개)

### 📄 생성 스크립트
`step8_btc_etf_data.py`

### 📊 변수 목록

#### 4.1 ETF 가격 (5개)

| 변수명 | ETF 이름 | 티커 | 출시일 | AUM (2025.10) |
|--------|----------|------|--------|---------------|
| `IBIT_Price` | BlackRock iShares Bitcoin Trust | IBIT | 2024-01-11 | ~$100B |
| `FBTC_Price` | Fidelity Wise Origin Bitcoin Fund | FBTC | 2024-01-11 | 2위 |
| `GBTC_Price` | Grayscale Bitcoin Trust | GBTC | 전환 | 고비용 |
| `ARKB_Price` | ARK 21Shares Bitcoin ETF | ARKB | 2024-01-11 | - |
| `BITB_Price` | Bitwise Bitcoin ETF | BITB | 2024-01-11 | - |

#### 4.2 프리미엄/디스카운트 (1개)

| 변수명 | 계산식 | 의미 | 현재값 (2025.10) |
|--------|--------|------|------------------|
| `GBTC_Premium` | `((GBTC/BTC_per_share - BTC) / BTC) * 100` | GBTC 프리미엄 | **-14.66%** |

#### 4.3 거래량 관련 (6개)

| 변수명 | 의미 |
|--------|------|
| `Total_BTC_ETF_Volume` | 전체 BTC ETF 거래량 |
| `IBIT_Volume_Change_7d` | IBIT 7일 거래량 변화율 |
| `FBTC_Volume_Change_7d` | FBTC 7일 거래량 변화율 |
| `GBTC_Volume_Change_7d` | GBTC 7일 거래량 변화율 |
| `ARKB_Volume_Change_7d` | ARKB 7일 거래량 변화율 |
| `BITB_Volume_Change_7d` | BITB 7일 거래량 변화율 |

### 🔧 생성 방법

```python
import yfinance as yf
import pandas as pd

# 데이터 기간: Bitcoin Spot ETF는 2024년 1월부터
start_date = datetime(2024, 1, 1)

# BTC 가격 수집 (Premium 계산용)
btc_data = yf.download('BTC-USD', start=start_date, end=end_date)
btc_price = btc_data['Close'].squeeze()  # Series로 변환!

# ETF 데이터 수집
etf_tickers = ['IBIT', 'FBTC', 'GBTC', 'ARKB', 'BITB']
for ticker in etf_tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    etf_data[f'{ticker}_Price'] = data['Close']
    etf_volume[f'{ticker}_Volume'] = data['Volume']

# GBTC Premium 계산
GBTC_BTC_PER_SHARE = 0.00092  # 근사치
implied_btc_price = etf_data['GBTC_Price'] / GBTC_BTC_PER_SHARE
etf_data['GBTC_Premium'] = ((implied_btc_price - btc_price) / btc_price * 100)

# 총 거래량
etf_data['Total_BTC_ETF_Volume'] = etf_volume.sum(axis=1)

# 거래량 변화율
for col in etf_volume.columns:
    ticker = col.replace('_Volume', '')
    etf_data[f'{ticker}_Volume_Change_7d'] = etf_volume[col].pct_change(7) * 100
```

### 💡 중요도 이유

**웹 검색 결과**:
1. **IBIT 우위**: BlackRock IBIT가 $100B AUM 돌파 직전 (역대 최빠른 성장)
2. **GBTC Premium**: 역사적으로 BTC 수요 측정에 사용
   - 2017년: +50% premium → BTC 과열
   - 2024년: -14.66% discount → ETF 경쟁 심화
3. **ETF Flows**: JPMorgan 분석가들이 유동성 지표로 사용
4. **Capital Rotation**: Gold ETF outflow → Bitcoin ETF inflow (2025년 패턴)

### 📈 최근 데이터 (2025년 10월 31일)

**ETF 가격**:
```
IBIT: $62.30
FBTC: $95.69
GBTC: $86.02
ARKB: $36.48
BITB: $59.67
```

**거래량 변화 (7일)**:
```
IBIT: -34.83%
FBTC: -46.57%
GBTC: -50.32%
ARKB: +3.55%
BITB: -4.09%
```

### ⚠️ 주의사항

- **데이터 기간**: 2024년 1월부터만 존재 (74.3% 결측치)
- **GBTC_BTC_PER_SHARE**: 0.00092는 근사치, 정확한 값은 Grayscale 웹사이트 확인 필요
- **Series 변환**: `btc_price.squeeze()` 필수 (DataFrame 에러 방지)
- **Forward Fill**: 결측치는 forward fill로 처리

---

## 실행 방법

### 전체 파이프라인 실행

```bash
cd "/Users/songhyowon/코인데이터분석"

# 1단계: 추가 전통시장 데이터 수집
python3 step2b_additional_markets.py

# 2단계: Fed 유동성 지표 수집
python3 step3b_fed_liquidity.py

# 3단계: 고급 온체인 지표 계산
python3 step6b_advanced_onchain.py

# 4단계: Bitcoin ETF 데이터 수집
python3 step8_btc_etf_data.py

# 5단계: 모든 데이터 통합
python3 step5b_integrate_all_new_data.py
```

### 개별 실행

특정 데이터만 업데이트하고 싶은 경우:

```bash
# Fed 유동성만 업데이트
python3 step3b_fed_liquidity.py
python3 step5b_integrate_all_new_data.py
```

### 결과 확인

```bash
# 생성된 CSV 파일 확인
ls -lh *.csv | grep -E "(additional|fed|advanced|bitcoin_etf|integrated_data_full_v2)"

# 변수 개수 확인
head -1 integrated_data_full_v2.csv | tr ',' '\n' | wc -l
```

---

## 파일 구조

### 생성된 파일

```
코인데이터분석/
├── step2b_additional_markets.py      # 추가 전통시장 수집 스크립트
├── step3b_fed_liquidity.py           # Fed 유동성 수집 스크립트
├── step6b_advanced_onchain.py        # 고급 온체인 계산 스크립트
├── step8_btc_etf_data.py             # Bitcoin ETF 수집 스크립트
├── step5b_integrate_all_new_data.py  # 통합 스크립트
│
├── additional_market_data.csv        # 추가 전통시장 데이터 (9개 변수)
├── fed_liquidity_data.csv            # Fed 유동성 데이터 (8개 변수)
├── advanced_onchain_data.csv         # 고급 온체인 데이터 (21개 변수)
├── bitcoin_etf_data.csv              # Bitcoin ETF 데이터 (12개 변수)
│
├── integrated_data_full.csv          # 기존 통합 데이터 (88개 변수)
└── integrated_data_full_v2.csv       # ⭐ 최종 통합 데이터 (138개 변수)
```

### 데이터 흐름

```
┌─────────────────────────┐
│  Yahoo Finance API      │
│  - DIA, IWM, TLT, etc.  │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  FRED API               │
│  - WALCL, RRP, etc.     │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  기존 온체인 데이터      │
│  - bc_*, cm_*           │
│  → 고급 지표 계산        │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Bitcoin ETF            │
│  - IBIT, FBTC, GBTC     │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  step5b_integrate       │
│  → integrated_v2.csv    │
└─────────────────────────┘
```

---

## 참고 자료

### 학술 논문

1. **"Predicting Bitcoin Prices Using Machine Learning"** (PMC, 2023)
   - 중요 변수: 온체인 지표, 거시경제 변수, 기술적 지표
   - URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC10216962/

2. **"Time-series forecasting of Bitcoin prices using high-dimensional features"** (Springer, 2020)
   - 67개 특성 사용, ElasticNet 효과적
   - URL: https://link.springer.com/article/10.1007/s00521-020-05129-6

3. **"Deep learning for Bitcoin price direction prediction"** (Financial Innovation, 2024)
   - 감정 지표, Google Trends 중요
   - URL: https://jfin-swufe.springeropen.com/articles/10.1186/s40854-024-00643-1

### 산업 자료

4. **"Bitcoin's On-Chain Signals"** (AInvest, 2025)
   - MVRV Z-Score, SOPR, 기관 매수 신호
   - URL: https://www.ainvest.com/news/bitcoin-chain-signals-decoding-institutional-buying-market-sentiment-2025-2509/

5. **"Viewing Bitcoin's Price in the Lens of Macro Liquidity Modelling"** (HTX Research, Medium)
   - Fed Balance Sheet, RRP, TGA 분석
   - URL: https://medium.com/huobi-research/viewing-bitcoins-price-in-the-lens-of-macro-liquidity-modelling-15b64e0972d7

6. **"Bitcoin ETF Overview"** (CoinGlass, 2025)
   - ETF Flows, GBTC Premium 추적
   - URL: https://www.coinglass.com/bitcoin-etf

### API 문서

7. **FRED API Documentation**
   - URL: https://fred.stlouisfed.org/docs/api/
   - API Key 발급: https://fred.stlouisfed.org/docs/api/api_key.html

8. **Yahoo Finance (yfinance) Documentation**
   - URL: https://pypi.org/project/yfinance/

### 온체인 지표 참고

9. **"Four Bitcoin On-Chain Indicators Worth Watching"** (Compass Mining)
   - Puell Multiple, Hash Ribbon, Difficulty Ribbon
   - URL: https://education.compassmining.io/education/bitcoin-on-chain-indicators-puell-multiple/

10. **"Breaking up On–Chain Metrics for Short and Long Term Investors"** (Glassnode)
    - LTH-SOPR, STH-SOPR, LTH-MVRV, STH-MVRV
    - URL: https://insights.glassnode.com/sth-lth-sopr-mvrv/

---

## 버전 히스토리

### v2.0 (2025-11-02)
- **추가**: 50개 신규 변수
- **총 변수**: 88개 → 138개
- **파일**: integrated_data_full_v2.csv

### v1.0 (이전)
- **기존**: 88개 변수
- **파일**: integrated_data_full.csv

---

## 라이선스 및 면책

이 데이터는 연구 및 교육 목적으로만 사용되어야 합니다.
- Yahoo Finance API: 개인 비상업적 사용
- FRED API: 공개 데이터, 출처 명시 필요

**투자 주의사항**: 이 데이터를 기반으로 한 예측 모델은 투자 조언이 아닙니다.

---

## 문의 및 기여

데이터 수집 오류 또는 개선 제안이 있으시면 이슈를 생성해주세요.

**작성자**: Claude Code
**날짜**: 2025년 11월 2일
**버전**: 2.0
