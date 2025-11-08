# ETF 전후 분석 올바른 방법론

## 🎯 핵심 질문
**ETF 전후로 모델을 따로 만들고 변수도 따로 선택해야 하는가?**

**답: 예! 반드시 그래야 합니다.**

---

## 📊 두 가지 접근 방식 비교

### ❌ 방법 1: 현재 방식 (잘못됨)

```
전체 데이터 (2021-2025)
├─ Train: 2021-2024 (70%)
│   ├─ ETF 이전: 2021.02 ~ 2024.01.09
│   └─ ETF 이후: 2024.01.10 ~ 2024.10
└─ Test: 2024-2025 (30%)  ← 전부 ETF 이후!!
    └─ ETF 이후: 2024.10 ~ 2025.10
```

**문제점**:
1. ETF 이전 기간을 Test로 검증 못함
2. 24~25년이 전부 ETF 이후 → ETF 이전 모델 성능 알 수 없음
3. 동일 변수/계수로 다른 시장 구조 설명 시도
4. 변수 관계 변화를 반영 못함 (예: DFF 상관관계 -0.27 → -0.87)

---

### ✅ 방법 2: 올바른 접근 (Period-Specific Models)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📅 ETF 이전 기간 (2021.02 ~ 2024.01.09)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1️⃣ Lasso로 변수 선택 (α=0.1)
   → 예: 60개 변수 선택됨

2️⃣ Ridge로 변수 선택 (α=10)
   → 예: 55개 변수 선택됨

3️⃣ 합집합 사용 (Lasso ∪ Ridge)
   → 예: 70개 변수

4️⃣ ElasticNet으로 모델 학습
   - Train/Test: 7:3 split (Time Series)
   - Train: 2021.02 ~ 2023.03
   - Test:  2023.04 ~ 2024.01.09

5️⃣ Walk-Forward Validation으로 검증
   - Window: 252일 (1년)
   - Test: 30일

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📅 ETF 이후 기간 (2024.01.10 ~ 2025.10)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1️⃣ Lasso로 변수 선택 (α=0.1)
   → 예: 75개 변수 선택됨 (ETF 변수 포함!)

2️⃣ Ridge로 변수 선택 (α=10)
   → 예: 68개 변수 선택됨

3️⃣ 합집합 사용 (Lasso ∪ Ridge)
   → 예: 85개 변수 (ETF 이전과 다름!)

4️⃣ ElasticNet으로 모델 학습
   - Train/Test: 7:3 split (Time Series)
   - Train: 2024.01 ~ 2024.11
   - Test:  2024.12 ~ 2025.10

5️⃣ Walk-Forward Validation으로 검증
   - Window: 252일
   - Test: 30일
```

---

## 🔍 왜 이렇게 해야 하는가?

### 1. 시장 구조 변화

| 특성 | ETF 이전 | ETF 이후 |
|------|----------|----------|
| **주요 투자자** | 개인, 암호화폐 펀드 | 기관 투자자 추가 |
| **거래 채널** | 거래소 직접 매매 | ETF 통한 간접 투자 |
| **가격 결정 요인** | 온체인 지표, Fear & Greed | ETF 플로우, 기관 수요 |
| **변동성** | 높음 (평균 45%) | 낮아짐 (평균 32%) |
| **평균 가격** | $37,000 | $89,000 (+137%) |

### 2. 변수 중요도 변화

**ETF 이전 TOP 10 예상**:
1. bc_miners_revenue
2. Hash_Ribbon_Spread
3. NVT_Ratio
4. fear_greed_index
5. google_trends_btc
6. RSI
7. MACD
8. OBV
9. SPX
10. M2SL

**ETF 이후 TOP 10 예상**:
1. **IBIT_Price** ← 새로운 1위!
2. **Total_BTC_ETF_Volume** ← 새로운!
3. **FBTC_Price** ← 새로운!
4. bc_miners_revenue
5. **FED_NET_LIQUIDITY** ← 중요도 상승!
6. fear_greed_index
7. NVT_Ratio
8. SPX
9. **GBTC_Premium** ← 의미 변화!
10. M2SL

### 3. 변수 관계 변화 (재확인)

```python
# 변수       ETF 이전 상관   ETF 이후 상관   변화
DFF (금리)      -0.27          -0.87        -0.60 ⚠️
M2SL            +0.54          +0.18        -0.36
fear_greed      +0.58          +0.33        -0.25
OBV             +0.69          +0.94        +0.25
```

**동일한 변수도 가격에 대한 영향력이 완전히 바뀜!**

---

## 📝 구체적 실행 계획

### Step 1: ETF 이전 모델 구축

```python
# 1. 데이터 분리
df_pre_etf = df[df['Date'] < '2024-01-10'].copy()

# 2. Time Series Split
train_size = int(len(df_pre_etf) * 0.7)
train_pre = df_pre_etf.iloc[:train_size]
test_pre = df_pre_etf.iloc[train_size:]

# 3. Lasso로 변수 선택
lasso = LassoCV(alphas=np.logspace(-3, 1, 50), cv=TimeSeriesSplit(5))
lasso.fit(X_train, y_train)
lasso_selected = X_train.columns[lasso.coef_ != 0]

# 4. Ridge로 변수 선택
ridge = RidgeCV(alphas=np.logspace(-2, 3, 50), cv=TimeSeriesSplit(5))
ridge.fit(X_train, y_train)
ridge_selected = X_train.columns[np.abs(ridge.coef_) > 0.1]

# 5. 합집합
selected_vars_pre = set(lasso_selected) | set(ridge_selected)

# 6. ElasticNet 학습
elasticnet_pre = ElasticNetCV(
    alphas=np.logspace(-3, 2, 50),
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
    cv=TimeSeriesSplit(5)
)
elasticnet_pre.fit(X_train[selected_vars_pre], y_train)

# 7. Walk-Forward Validation
wf_results_pre = walk_forward_validation(elasticnet_pre, df_pre_etf, selected_vars_pre)
```

### Step 2: ETF 이후 모델 구축

```python
# 1. 데이터 분리
df_post_etf = df[df['Date'] >= '2024-01-10'].copy()

# 2. Time Series Split
train_size = int(len(df_post_etf) * 0.7)
train_post = df_post_etf.iloc[:train_size]
test_post = df_post_etf.iloc[train_size:]

# 3~7. 위와 동일한 과정 반복
# ⚠️ 주의: ETF 변수들(IBIT_Price, FBTC_Price 등)이 선택될 것!
```

### Step 3: 두 모델 비교

```python
comparison = pd.DataFrame({
    'Period': ['Pre-ETF', 'Post-ETF'],
    'Train R²': [train_r2_pre, train_r2_post],
    'Test R²': [test_r2_pre, test_r2_post],
    'Walk-Forward R²': [wf_r2_pre, wf_r2_post],
    'Selected Variables': [len(selected_vars_pre), len(selected_vars_post)],
    'Top 5 Features': [top5_pre, top5_post]
})
```

---

## 🎯 기대 결과

### 1. 변수 구성 차이 발견

```
ETF 이전 모델: 70개 변수
- bc_miners_revenue ✅
- Hash_Ribbon_Spread ✅
- NVT_Ratio ✅
- IBIT_Price ❌ (없음)
- FBTC_Price ❌ (없음)

ETF 이후 모델: 85개 변수
- IBIT_Price ✅✅✅ (1위!)
- Total_BTC_ETF_Volume ✅✅
- FBTC_Price ✅✅
- bc_miners_revenue ✅ (중요도 하락)
- Hash_Ribbon_Spread ✅ (중요도 하락)
```

### 2. 모델 성능 개선

| 지표 | 현재 방식 | 올바른 방식 |
|------|-----------|-------------|
| ETF 이전 Test R² | ❌ 측정 불가 | ✅ 예상 0.65 |
| ETF 이후 Test R² | 0.82 | ✅ 예상 0.78 |
| Walk-Forward R² | -2.43 | ✅ 예상 0.45 |

### 3. 해석력 증대

- ETF 도입이 **어떤 변수**의 중요도를 바꿨는지 정량화
- IBIT, FBTC 같은 ETF 변수가 **얼마나** 예측력을 높였는지 측정
- 온체인 지표(채굴자 수익, 해시리본)가 ETF 이후 **왜 덜 중요**한지 설명

---

## ⚠️ 주의사항

### 1. 데이터 양 문제

```
ETF 이전: 2021.02 ~ 2024.01.09 (1,072일) ✅ 충분
ETF 이후: 2024.01.10 ~ 2025.10 (643일)  ⚠️ 부족할 수 있음
```

**해결책**:
- ETF 이후는 최소 Train 450일, Test 193일 확보
- Walk-Forward Window를 126일(6개월)로 축소
- Cross-Validation fold를 3개로 축소

### 2. ETF 변수 결측치

```
IBIT_Price: 2024.01.11 시작
FBTC_Price: 2024.01.15 시작
```

**해결책**:
- ETF 이전 모델: ETF 변수 전부 제외
- ETF 이후 모델: 2024.01.15 이후 데이터만 사용 (628일)

### 3. 과적합 위험

ETF 이후 데이터가 적으므로:
- ElasticNet alpha를 더 높게 (5.0 ~ 50.0)
- L1 ratio를 높게 (0.7 ~ 0.9, 변수 더 제거)
- Cross-Validation 엄격히 적용

---

## 🏆 최종 권장 사항

### ✅ 해야 할 것

1. **ETF 전후로 완전히 독립적인 모델 구축**
   - 데이터 분리
   - 변수 선택 (Lasso + Ridge)
   - ElasticNet 학습
   - Time Series CV
   - Walk-Forward Validation

2. **ETF 이후 모델에서 ETF 변수 활용**
   - IBIT_Price, FBTC_Price, Total_BTC_ETF_Volume
   - GBTC_Premium (의미 변화 반영)
   - ETF Volume Change 7d

3. **두 모델 성능 비교**
   - 변수 중요도 변화 분석
   - 예측 정확도 비교
   - 백테스팅 수익률 비교

### ❌ 하지 말아야 할 것

1. **전체 기간에서 변수 선택 후 ETF 전후에 동일 적용**
   - 시장 구조 변화 무시
   - 비정상성 문제 해결 안 됨

2. **21~24년 Train, 24~25년 Test로 단일 분할**
   - ETF 이전 기간 검증 불가
   - Test가 전부 ETF 이후

3. **Random K-Fold CV 사용**
   - 시계열 순서 무시
   - 미래 정보로 과거 예측 (Data Leakage)

---

## 📊 예상 결과물

### 1. 변수 중요도 비교 테이블

| 변수 | ETF 이전 계수 | ETF 이후 계수 | 변화 |
|------|---------------|---------------|------|
| bc_miners_revenue | 1064.93 | 523.45 | -50.8% |
| IBIT_Price | 0 (없음) | 1523.67 | NEW! |
| fear_greed_index | 750.40 | 412.33 | -45.0% |
| FED_NET_LIQUIDITY | -40.81 | -567.89 | +1291% |

### 2. 모델 성능 비교 차트

```
                  ETF 이전        ETF 이후
Train R²          0.87           0.84
Test R²           0.63           0.76
Walk-Forward R²   0.41           0.52
RMSE              $2,345         $4,123
```

### 3. 백테스팅 비교

```
                  ETF 이전 모델   ETF 이후 모델
Total Return      +32.45%        +67.89%
Sharpe Ratio      1.23           1.87
Max Drawdown      -18.2%         -11.4%
```

---

## 🚀 다음 단계

1. **step26_elasticnet_backtesting_v2.py 수정**
   - ETF 전후 데이터 분리
   - 각각 독립적으로 변수 선택 + 모델 학습

2. **step27_etf_pre_post_comparison.py 신규 작성**
   - 두 모델 비교 분석
   - 변수 중요도 변화 시각화

3. **전체_분석_종합_정리.md 업데이트**
   - ETF 전후 모델 차이 설명
   - 변수 중요도 변화 추가

---

**결론**: 예, 반드시 ETF 전후로 **별도의 ElasticNet 모델**을 만들고, **Lasso + Ridge로 각각 변수 선택**을 해야 합니다. 현재 방식은 시장 구조 변화를 반영하지 못합니다.
