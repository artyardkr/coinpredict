# ElasticNet 변수 선정 및 중요 변수 분석

## 📊 1. 변수 선정 방법 (step25, step31 공통)

### 1.1 제외된 변수 (Exclude Columns)

```python
# step25_next_day_price_prediction.py:74-87
exclude_cols = [
    'Date',           # 날짜 (시계열 인덱스)
    'Close',          # 오늘 종가 (Data Leakage 방지)
    'High',           # 오늘 고가
    'Low',            # 오늘 저가
    'Open',           # 오늘 시가
    'target',         # 내일 종가 (예측 대상)
    'cumulative_return',  # 누적 수익률
    'bc_market_price',    # 시장가격 (Close와 중복)
    'bc_market_cap',      # 시가총액 (직접 계산 가능)
]

# EMA/SMA 중 'close' 포함된 것 제외 (Data Leakage)
ema_sma_cols = [col for col in df.columns
                if ('EMA' in col or 'SMA' in col) and 'close' in col.lower()]
# 예: EMA12_close, SMA20_close 등

# Bollinger Bands 제외 (Data Leakage)
bb_cols = [col for col in df.columns if col.startswith('BB_')]
# 예: BB_upper, BB_middle, BB_lower
```

### 1.2 제외 이유

| 변수 유형 | 제외 이유 |
|----------|----------|
| **Close, High, Low, Open** | 오늘 가격 → 내일 가격 예측 시 Data Leakage |
| **EMA/SMA_close** | 오늘 Close로 계산됨 → 미래 정보 사용 |
| **Bollinger Bands (BB_*)** | 오늘 Close 기반 → 미래 정보 사용 |
| **bc_market_price** | Close와 거의 동일 (중복) |
| **bc_market_cap** | Close × Total BTC로 계산 가능 (중복) |

### 1.3 최종 사용 변수

**총 67개 변수** (89개 중 22개 제외)

#### 카테고리별 변수:

1. **온체인 데이터** (블록체인 직접 지표)
   - bc_miners_revenue (채굴자 수익)
   - bc_hash_rate (해시레이트)
   - bc_difficulty (난이도)
   - bc_mempool_size (멤풀 크기)
   - bc_n_transactions (거래 수)
   - bc_n_unique_addresses (고유 주소 수)
   - bc_transaction_fees (거래 수수료)
   - bc_total_bitcoins (총 비트코인 수)

2. **전통 시장** (주식, 금, 환율)
   - SPX (S&P 500)
   - QQQ (나스닥 100)
   - GOLD, SILVER (귀금속)
   - OIL (유가)
   - UUP (달러 인덱스)
   - EURUSD, DEXUSEU (유로/달러)
   - DTWEXBGS (달러 무역가중지수)
   - BSV (Bitcoin SV - 대체재)

3. **거시경제 지표** (FRED 데이터)
   - DFF (연방기금금리)
   - DGS10 (10년 국채 수익률)
   - T10Y2Y (10년-2년 스프레드)
   - M2SL (통화량 M2)
   - GDP (국내총생산)
   - CPIAUCSL (소비자물가지수)
   - UNRATE (실업률)
   - VIXCLS (VIX 변동성 지수)

4. **기술적 지표** (계산된 지표)
   - Volume, OBV (거래량 지표)
   - RSI (상대강도지수)
   - MACD, MACD_signal, MACD_diff
   - ATR (평균 진폭)
   - MFI (자금흐름지수)
   - ADX (방향성 지수)
   - CCI (상품채널지수)
   - Stoch_K, Stoch_D (스토캐스틱)
   - Williams_R
   - ROC (변화율)
   - volatility_20d (20일 변동성)
   - daily_return, volume_change

5. **파생 지표** (EMA/SMA - volume/marketcap만)
   - EMA5/10/14/20/30/100/200_volume
   - EMA5/10/14/20/30/100/200_marketcap
   - SMA5/10/20/30_marketcap
   - market_cap_approx (추정 시가총액)

6. **감성 지표**
   - fear_greed_index (공포&탐욕 지수)
   - google_trends_btc (구글 트렌드)

---

## 🎯 2. ElasticNet 모델 설정

### 2.1 하이퍼파라미터

```python
# step25:138, step31:115, 165
ElasticNet(
    alpha=1.0,          # Regularization 강도
    l1_ratio=0.5,       # L1(Lasso):L2(Ridge) = 50:50
    max_iter=10000,     # 최대 반복 횟수
    random_state=42     # 재현성
)
```

### 2.2 파라미터 설명

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| **alpha** | 1.0 | Regularization 강도 (높을수록 계수 작아짐) |
| **l1_ratio** | 0.5 | L1 50% + L2 50% (균형) |
| **max_iter** | 10000 | 수렴 보장 (긴 학습) |

### 2.3 L1/L2 Regularization 효과

```
Loss = MSE + alpha * (l1_ratio * |coef| + (1-l1_ratio) * coef²)
     = MSE + 1.0 * (0.5 * |coef| + 0.5 * coef²)
```

**L1 (Lasso) 효과**:
- 불필요한 변수 계수 → 0
- 자동 변수 선택 (Feature Selection)

**L2 (Ridge) 효과**:
- 모든 계수 작게 만듦
- 다중공선성 문제 해결

---

## 📈 3. 변수 중요도 (ElasticNet Coefficients)

### 3.1 Pre-ETF (2021-2023) Top 15

| 순위 | 변수 | Coefficient | 해석 |
|-----|------|-------------|------|
| 1 | bc_miners_revenue | **+1848.43** | 채굴자 수익 ↑ → BTC 가격 ↑ |
| 2 | SPX | **+1321.11** | S&P 500 ↑ → BTC 가격 ↑ |
| 3 | QQQ | **+1194.94** | 나스닥 ↑ → BTC 가격 ↑ |
| 4 | OBV | **+1113.69** | 거래량 누적 ↑ → BTC 가격 ↑ |
| 5 | DFF | **-880.95** | 금리 ↑ → BTC 가격 ↓ |
| 6 | fear_greed_index | **+863.54** | 탐욕 ↑ → BTC 가격 ↑ |
| 7 | M2SL | **+794.89** | 통화량 ↑ → BTC 가격 ↑ |
| 8 | T10Y2Y | **+789.08** | 장단기 금리차 ↑ → BTC 가격 ↑ |
| 9 | SMA30_marketcap | **+777.46** | 시가총액 추세 ↑ → BTC 가격 ↑ |
| 10 | SMA20_marketcap | **+724.48** | 시가총액 추세 ↑ → BTC 가격 ↑ |
| 11 | bc_transaction_fees | **-719.79** | 거래 수수료 ↑ → BTC 가격 ↓ (과열) |
| 12 | EMA30_marketcap | **+690.72** | 시가총액 추세 ↑ → BTC 가격 ↑ |
| 13 | volatility_20d | **-678.54** | 변동성 ↑ → BTC 가격 ↓ (불안) |
| 14 | EMA20_marketcap | **+643.96** | 시가총액 추세 ↑ → BTC 가격 ↑ |
| 15 | ATR | **+601.50** | 변동 폭 ↑ → BTC 가격 ↑ (추세) |

**핵심 특징**:
- ✅ **전통 시장 의존** (SPX, QQQ가 2, 3위)
- ✅ **채굴자 수익 최우선** (온체인 지표)
- ✅ **금리 민감** (DFF, T10Y2Y)
- ✅ **감성 지표 중요** (fear_greed_index)

---

### 3.2 Post-ETF (2024-2025) Top 15

| 순위 | 변수 | Coefficient | 해석 |
|-----|------|-------------|------|
| 1 | OBV | **+1397.60** | 거래량 누적 ↑ → BTC 가격 ↑ |
| 2 | EMA200_volume | **+888.96** | 장기 거래량 추세 ↑ → BTC 가격 ↑ |
| 3 | EMA100_volume | **+864.95** | 중기 거래량 추세 ↑ → BTC 가격 ↑ |
| 4 | EMA100_marketcap | **+850.76** | 시가총액 추세 ↑ → BTC 가격 ↑ |
| 5 | bc_mempool_size | **-837.10** | 멤풀 크기 ↑ → BTC 가격 ↓ (혼잡) |
| 6 | EMA200_marketcap | **+833.00** | 장기 시가총액 추세 ↑ → BTC 가격 ↑ |
| 7 | SMA30_marketcap | **+726.32** | 시가총액 추세 ↑ → BTC 가격 ↑ |
| 8 | CPIAUCSL | **+724.13** | 인플레이션 ↑ → BTC 가격 ↑ (인플레 헤지) |
| 9 | bc_difficulty | **+714.76** | 채굴 난이도 ↑ → BTC 가격 ↑ (안정성) |
| 10 | fear_greed_index | **+703.76** | 탐욕 ↑ → BTC 가격 ↑ |
| 11 | SILVER | **+686.23** | 은 가격 ↑ → BTC 가격 ↑ (대체재) |
| 12 | bc_total_bitcoins | **+674.26** | 총 BTC 수 ↑ → 가격 ↑ (희소성) |
| 13 | UUP | **+668.81** | 달러 강세 → BTC 가격 ↑ (역관계 약화) |
| 14 | DGS10 | **+657.91** | 10년 국채 수익률 ↑ → BTC 가격 ↑ |
| 15 | ATR | **+615.16** | 변동 폭 ↑ → BTC 가격 ↑ (추세) |

**핵심 특징**:
- 🔥 **거래량 중심** (OBV, EMA200/100_volume이 1, 2, 3위)
- 🔥 **온체인 지표 강화** (bc_mempool_size, bc_difficulty)
- 🔥 **SPX/QQQ 하락** (2, 3위 → 20, 21위)
- 🔥 **독립적 생태계** (전통 시장 의존도 감소)

---

## 🔄 4. Pre-ETF vs Post-ETF 비교

### 4.1 순위 변화

| 변수 | Pre-ETF | Post-ETF | 변화 |
|------|---------|----------|------|
| **SPX** | 🥈 2위 | 21위 | ⬇️ -19 |
| **QQQ** | 🥉 3위 | 20위 | ⬇️ -17 |
| **OBV** | 4위 | 🥇 1위 | ⬆️ +3 |
| **bc_miners_revenue** | 🥇 1위 | 27위 | ⬇️ -26 |
| **bc_mempool_size** | 56위 | 🎖️ 5위 | ⬆️ +51 |
| **EMA200_volume** | 34위 | 🥈 2위 | ⬆️ +32 |
| **EMA100_volume** | 22위 | 🥉 3위 | ⬆️ +19 |
| **bc_difficulty** | 60위 | 🎖️ 9위 | ⬆️ +51 |

### 4.2 주요 발견

#### ❌ Pre-ETF 중요 → Post-ETF 하락
- **SPX (S&P 500)**: 2위 → 21위
- **QQQ (나스닥)**: 3위 → 20위
- **bc_miners_revenue**: 1위 → 27위
- **DFF (금리)**: 5위 → 34위

#### ✅ Post-ETF 중요도 급상승
- **bc_mempool_size**: 56위 → 5위 (+51)
- **bc_difficulty**: 60위 → 9위 (+51)
- **EMA200_volume**: 34위 → 2위 (+32)
- **EMA100_volume**: 22위 → 3위 (+19)

#### 🔄 지속적 중요 (공통 Top 10)
- fear_greed_index (6위 → 10위)
- ATR (15위 → 15위)
- SMA30_marketcap (9위 → 7위)

---

## 📊 5. 변수 사용 통계

### 5.1 ElasticNet 자동 변수 선택 결과

| 기간 | 사용 변수 | 전체 변수 | 사용 비율 |
|------|----------|----------|----------|
| **Pre-ETF** | 59개 | 67개 | **88.1%** |
| **Post-ETF** | 59개 | 67개 | **88.1%** |

**분석**:
- ElasticNet L1 regularization으로 **8개 변수 계수 = 0** 처리
- 거의 모든 변수 활용 (88%)
- 하지만 **계수 크기가 중요도를 결정**

### 5.2 제거된 변수 (Coefficient = 0)

#### Pre-ETF에서 제거된 8개:
(CSV에 없음 = 계수 0인 변수)

#### Post-ETF에서 제거된 8개:
- google_trends_btc (Pre-ETF 20위 → 제거)

---

## 💡 6. 핵심 인사이트

### 6.1 시장 구조 변화

```
Pre-ETF (2021-2023):
┌─────────────────────────────┐
│ 전통 시장 의존              │
│ - SPX (2위), QQQ (3위)     │
│ - 주식시장 따라감           │
│ - 리스크 자산 취급          │
└─────────────────────────────┘

Post-ETF (2024-2025):
┌─────────────────────────────┐
│ 독립 생태계 형성            │
│ - 거래량 중심 (1, 2, 3위)  │
│ - 온체인 지표 강화          │
│ - SPX/QQQ 영향력 감소      │
└─────────────────────────────┘
```

### 6.2 변수 중요도 해석

#### 🔴 Negative Coefficient (-)
- **DFF (금리)**: -880 (Pre) → -353 (Post)
  - 금리 ↑ → BTC 가격 ↓ (기회비용)

- **bc_mempool_size**: -20 (Pre) → **-837 (Post)**
  - 멤풀 혼잡 ↑ → 가격 ↓ (네트워크 스트레스)

- **bc_transaction_fees**: -719 (Pre) → -177 (Post)
  - 수수료 ↑ → 가격 ↓ (과열 신호)

#### 🟢 Positive Coefficient (+)
- **OBV (거래량)**: +1113 (Pre) → **+1397 (Post)**
  - 누적 거래량 ↑ → 가격 ↑ (수요)

- **fear_greed_index**: +863 (Pre) → +703 (Post)
  - 탐욕 ↑ → 가격 ↑ (시장 심리)

- **bc_difficulty**: +1.9 (Pre) → **+714 (Post)**
  - 난이도 ↑ → 가격 ↑ (네트워크 안정성)

### 6.3 실전 활용

**Pre-ETF 전략**:
1. SPX/QQQ 모니터링 필수
2. 채굴자 수익 추적
3. 금리 정책 민감

**Post-ETF 전략**:
1. **거래량 지표 최우선** (OBV, EMA_volume)
2. **온체인 지표 감시** (mempool, difficulty)
3. SPX/QQQ 영향력 감소 (덜 중요)

---

## 🎯 7. 모델 성능 (step31 결과)

### 7.1 Pre-ETF 성능
```
Train R²: 0.9581
Test R²:  0.0560
RMSE: $5,697
방향 정확도: 50.93%
```

### 7.2 Post-ETF 성능
```
Train R²: 0.9657
Test R²: -0.0455
RMSE: $10,692
방향 정확도: 47.15%
```

### 7.3 결론
- ❌ Post-ETF에서 **예측 더 어려워짐** (R² 0.056 → -0.045)
- ❌ RMSE 거의 **2배 증가** ($5,697 → $10,692)
- ❌ 방향 정확도 **감소** (50.93% → 47.15%)

**해석**:
- ETF 도입 → 시장 효율성 증가
- 기관 투자자 유입 → 예측 난이도 상승
- 시장 성숙 = 머신러닝 예측 한계

---

## 📋 요약

### ElasticNet 변수 선정
1. **67개 변수** 사용 (89개 중 22개 제외)
2. Data Leakage 방지 (EMA/SMA_close, BB, 오늘 가격 제외)
3. **L1+L2 Regularization** (alpha=1.0, l1_ratio=0.5)
4. **88%** 변수 활용 (L1이 8개만 제거)

### 중요 변수 (Pre-ETF)
1. 채굴자 수익 (+1848)
2. S&P 500 (+1321)
3. 나스닥 (+1194)
4. 거래량 (+1113)
5. 금리 (-880)

### 중요 변수 (Post-ETF)
1. 거래량 (+1397) 🥇
2. EMA200_volume (+888) 🥈
3. EMA100_volume (+864) 🥉
4. EMA100_marketcap (+850)
5. 멤풀 크기 (-837)

### 핵심 변화
- 전통 시장 의존 → 독립 생태계
- SPX/QQQ (2, 3위) → (20, 21위)
- 거래량 지표 급상승
- 온체인 지표 강화
- 예측 난이도 증가 (시장 성숙)
