# Bitcoin 구조 변화 검정 계획서

## 목적
ETF 승인(2024-01-10)을 전후로 비트코인 가격 결정 요인의 구조적 변화를 통계적으로 검정

---

## 1. 검정 방법론

### 1.1 Chow Test (초우 검정)
**목적**: ETF 승인일을 기준으로 회귀계수가 유의미하게 변했는지 검정

**방법**:
```
H0: ETF 전후 회귀계수가 동일하다
H1: ETF 전후 회귀계수가 다르다

F-statistic = [(SSR_pooled - (SSR1 + SSR2)) / k] / [(SSR1 + SSR2) / (n1 + n2 - 2k)]
```

**주의사항**:
- Look-ahead bias 방지:
  - ETF 날짜는 사전에 알려진 이벤트이므로 허용
  - 하지만 out-of-sample 검증으로 과적합 여부 확인
- HAC 표준오차 사용 (Newey-West)
- 동분산성 가정 완화

**구현**:
1. 전체 기간 회귀 (pooled)
2. ETF 전 회귀
3. ETF 후 회귀
4. F-통계량 계산 및 p-value

### 1.2 Quandt-Andrews Test (콴트-앤드류스 검정)
**목적**: 알려지지 않은 변화점을 데이터에서 찾아냄

**방법**:
```
모든 가능한 분할점에서 Chow Test 수행
sup F-statistic = max{F(τ) : τ ∈ [τ_min, τ_max]}
```

**주의사항**:
- Look-ahead bias 방지:
  - 전체 기간의 15%-85% 구간에서만 탐색 (양 끝 제외)
  - Critical values는 Andrews (1993) 표 사용
- 다중 검정 문제:
  - Bonferroni 보정 불필요 (sup F-statistic 자체가 보정됨)
  - 하지만 변수별 검정은 FDR 통제 필요

**구현**:
1. 각 시점 τ에서 Chow F-통계량 계산
2. 최대값(sup F) 찾기
3. Andrews critical values와 비교
4. 변화점 날짜 및 신뢰구간 추정

### 1.3 CUSUM Test (누적합 검정)
**목적**: 점진적 변화 vs 급격한 변화 구분

**방법**:
```
CUSUM = Σ(e_t / σ_e)
점진적 변화: CUSUM이 천천히 경계 이탈
급격한 변화: CUSUM이 급격하게 경계 이탈
```

**주의사항**:
- Recursive residuals 사용
- 5% 유의수준 경계선: ±0.948√n
- CUSUM-SQ (제곱합)도 함께 검정

**구현**:
1. Recursive residuals 계산
2. CUSUM, CUSUM-SQ 통계량
3. 경계선 이탈 시점 탐지
4. 시각화

---

## 2. 통계적 보정 및 검증

### 2.1 Look-ahead Bias 방지

#### Out-of-Sample 검증 전략
```python
# 시나리오 1: Rolling Window
for t in range(T_train, T_total):
    - t 시점까지 데이터로 모델 학습
    - t+1 시점 예측
    - Chow Test를 t 시점까지만으로 수행

# 시나리오 2: Walk-Forward
- Train: 2021-02-03 ~ 2023-12-31
- Test: 2024-01-01 ~ 2025-10-14
- ETF 날짜(2024-01-10)는 Test 기간 내
- Train 데이터만으로 변수 선택 및 검정
```

#### 검증 절차
1. **In-sample**: 전체 기간으로 검정 (탐색적 분석)
2. **Out-of-sample**: ETF 이전 데이터만으로 모델 학습
3. **Post-hoc 검증**: ETF 이후 데이터로 구조 변화 확인

### 2.2 다중 검정 보정

#### 변수별 검정 (138개 변수)
```python
# 방법 1: Bonferroni 보정 (보수적)
α_corrected = α / n_tests
α_corrected = 0.05 / 138 = 0.00036

# 방법 2: FDR (False Discovery Rate) 통제 (추천)
# Benjamini-Hochberg 절차
p_values_sorted = sort(p_values)
for i, p in enumerate(p_values_sorted):
    if p <= (i+1)/n * α:
        reject H0
```

#### 시계열 검정 (여러 날짜)
```python
# Quandt-Andrews는 자체적으로 보정됨 (sup F)
# 하지만 여러 변수에 대해 수행하면 FDR 필요
```

#### 적용 전략
- **Chow Test (ETF 날짜 고정)**: Bonferroni 보정
- **Quandt-Andrews (날짜 탐색)**: FDR 통제
- **유의변수만 보고**: 전체 검정 수 명시

### 2.3 자기상관 처리

#### HAC 표준오차 (Newey-West)
```python
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac

# lag 선택: Newey-West 권장
lag = floor(4 * (T/100)^(2/9))

# HAC 표준오차로 t-통계량 재계산
model = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': lag})
```

#### Durbin-Watson 검정
```python
# 자기상관 확인
from statsmodels.stats.stattools import durbin_watson
dw_stat = durbin_watson(residuals)
# DW ≈ 2: 자기상관 없음
# DW < 1 or > 3: 강한 자기상관
```

#### AR 오차항 모델링 (대안)
```python
# ARMA 오차 회귀
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(y, exog=X, order=(1,0,0)).fit()
```

### 2.4 이상치 처리

#### Winsorization (추천)
```python
from scipy.stats.mstats import winsorize

# 상하위 1% 절사
y_winsorized = winsorize(y, limits=[0.01, 0.01])
X_winsorized = winsorize(X, limits=[0.01, 0.01], axis=0)
```

#### Robust Regression
```python
from statsmodels.robust.robust_linear_model import RLM

# Huber M-estimator
model = RLM(y, X, M=sm.robust.norms.HuberT()).fit()

# LAD (Least Absolute Deviation)
model = sm.QuantReg(y, X).fit(q=0.5)
```

#### 이상치 탐지 및 제거
```python
# Cook's Distance
from statsmodels.stats.outliers_influence import OLSInfluence

influence = OLSInfluence(model)
cooks_d = influence.cooks_distance[0]
outlier_threshold = 4 / len(y)
outliers = cooks_d > outlier_threshold

# DBSCAN (시계열용)
from sklearn.cluster import DBSCAN
# 이상치 제거 후 재검정
```

---

## 3. 구현 계획

### 3.1 데이터 준비
```python
1. 데이터 로드 및 전처리
   - 118 변수 (데이터 누수 제거 버전)
   - 2021-02-03 ~ 2025-10-14

2. Winsorization (상하위 1%)
   - 급등/급락 영향 완화

3. Train/Test 분할
   - Train: 2021-02-03 ~ 2023-12-31
   - Test: 2024-01-01 ~ 2025-10-14
```

### 3.2 변수 선택 (Pre-ETF 데이터만 사용)
```python
1. 상관분석 (Train 데이터만)
   - |corr| > 0.3 변수 선택

2. VIF (다중공선성) 검사
   - VIF < 10 변수만 유지

3. Forward/Backward Selection
   - AIC/BIC 기준

4. 최종 변수 세트 고정
```

### 3.3 Chow Test
```python
for var in selected_variables:
    # 1. Full sample regression (Train data only)
    model_full = OLS(y_train, X_train[[var]]).fit(cov_type='HAC')

    # 2. Pre-ETF regression
    model_pre = OLS(y_pre, X_pre[[var]]).fit(cov_type='HAC')

    # 3. Post-ETF regression (in-sample 검증용)
    model_post = OLS(y_post, X_post[[var]]).fit(cov_type='HAC')

    # 4. Chow F-test
    F_stat = compute_chow_f(model_full, model_pre, model_post)
    p_value = f.sf(F_stat, df1, df2)

    # 5. Bonferroni correction
    reject_h0 = p_value < (0.05 / n_variables)
```

### 3.4 Quandt-Andrews Test
```python
# 변화점 탐색 범위: 15% ~ 85% (look-ahead bias 방지)
tau_min = int(0.15 * T)
tau_max = int(0.85 * T)

sup_f_stats = []
for tau in range(tau_min, tau_max):
    # Split at tau
    model1 = OLS(y[:tau], X[:tau]).fit()
    model2 = OLS(y[tau:], X[tau:]).fit()

    # Chow F-test
    f_stat = compute_chow_f_at_tau(model1, model2)
    sup_f_stats.append((tau, f_stat))

# Find maximum
tau_star, sup_f = max(sup_f_stats, key=lambda x: x[1])

# Compare with Andrews critical values
critical_value = get_andrews_critical_value(alpha=0.05)
reject_h0 = sup_f > critical_value

# Bootstrap confidence interval for tau_star
```

### 3.5 CUSUM Test
```python
from statsmodels.stats.diagnostic import recursive_olsresiduals

# Recursive residuals
rr = recursive_olsresiduals(model, skip=30)

# CUSUM
cusum = np.cumsum(rr[0] / np.std(rr[0]))

# CUSUM-SQ
cusum_sq = np.cumsum((rr[0] / np.std(rr[0]))**2)

# Boundaries (5% significance)
n = len(cusum)
boundary = 0.948 * np.sqrt(n)

# Detect boundary crossings
crossing_points = np.where(np.abs(cusum) > boundary)[0]
```

### 3.6 Out-of-Sample 검증
```python
# 시나리오 1: Pre-ETF 데이터로 변수 선택
significant_vars_pretrain = select_variables(X_train, y_train)

# 시나리오 2: Post-ETF 데이터로 검정 (독립 검증)
for var in significant_vars_pretrain:
    # ETF 기준 Chow Test
    chow_result = chow_test(var, breakpoint='2024-01-10', data=y_test, X_test)

    # 결과 비교
    if chow_result.p_value < 0.05:
        print(f"{var}: 구조 변화 확인 (out-of-sample)")
```

---

## 4. 결과 보고 형식

### 4.1 Chow Test 결과표
```
| Variable | F-stat | p-value | p-adj (Bonf) | Significant | Coef_pre | Coef_post | Change |
|----------|--------|---------|--------------|-------------|----------|-----------|--------|
| IWM      | 45.23  | 0.0001  | 0.0138       | ***         | 0.32     | 0.57      | +0.25  |
| ...      | ...    | ...     | ...          | ...         | ...      | ...       | ...    |
```

### 4.2 Quandt-Andrews 결과
```
변수: IWM
sup F-statistic: 52.3
Critical value (5%): 12.5
변화점 추정: 2024-01-15 (95% CI: 2024-01-08 ~ 2024-01-22)
결론: ETF 승인일 근처에서 구조 변화 검출 ***
```

### 4.3 CUSUM 플롯
```
- CUSUM 시계열 그래프
- 경계선 (±0.948√n)
- ETF 날짜 수직선
- 이탈 시점 표시
```

### 4.4 요약 보고서
```markdown
## 주요 발견

1. **Chow Test (Bonferroni 보정)**
   - 138개 변수 중 **23개**에서 유의한 구조 변화 (p < 0.00036)
   - 주요 변수: IWM, SPX, T10Y3M, ...

2. **Quandt-Andrews Test (FDR 통제)**
   - 추정된 변화점: **2024-01-15** (95% CI: 2024-01-08 ~ 2024-01-22)
   - ETF 승인일(2024-01-10)과 일치
   - 45개 변수에서 변화점 검출 (FDR < 0.05)

3. **CUSUM Test**
   - 급격한 변화 패턴 관찰 (2024-01-10 전후)
   - 점진적 변화 아닌 단절적 변화
   - 12개 변수에서 경계 이탈

4. **Out-of-Sample 검증**
   - Pre-ETF 데이터로 선택된 18개 변수
   - Post-ETF 데이터에서 15개 변수 구조 변화 재확인
   - 과적합 위험 낮음

5. **Robustness Checks**
   - Winsorization 적용: 결과 일관성 유지
   - HAC 표준오차: 자기상관 보정
   - Robust regression: 이상치 영향 미미
```

---

## 5. 코드 구조

```
structural_change_tests.py
├── 1. Data Preparation
│   ├── load_data()
│   ├── winsorize_outliers()
│   └── train_test_split()
│
├── 2. Variable Selection (Pre-ETF only)
│   ├── correlation_filter()
│   ├── vif_filter()
│   └── stepwise_selection()
│
├── 3. Chow Test
│   ├── chow_test()
│   ├── bonferroni_correction()
│   └── plot_chow_results()
│
├── 4. Quandt-Andrews Test
│   ├── quandt_andrews_test()
│   ├── bootstrap_confidence_interval()
│   └── plot_sup_f()
│
├── 5. CUSUM Test
│   ├── cusum_test()
│   ├── cusum_sq_test()
│   └── plot_cusum()
│
├── 6. Robustness Checks
│   ├── hac_standard_errors()
│   ├── robust_regression()
│   └── outlier_sensitivity()
│
└── 7. Reporting
    ├── generate_summary_table()
    ├── export_results()
    └── create_visualizations()
```

---

## 6. 시간 계획

- **Phase 1** (1시간): 데이터 준비 및 변수 선택
- **Phase 2** (1시간): Chow Test 구현 및 실행
- **Phase 3** (1시간): Quandt-Andrews Test 구현 및 실행
- **Phase 4** (30분): CUSUM Test 구현 및 실행
- **Phase 5** (1시간): 통계적 보정 및 검증
- **Phase 6** (30분): 결과 정리 및 시각화

**총 소요 시간**: 약 5시간

---

## 7. 참고문헌

1. Chow, G. C. (1960). "Tests of Equality Between Sets of Coefficients in Two Linear Regressions"
2. Andrews, D. W. K. (1993). "Tests for Parameter Instability and Structural Change with Unknown Change Point"
3. Brown, R. L., Durbin, J., & Evans, J. M. (1975). "Techniques for Testing the Constancy of Regression Relationships over Time"
4. Newey, W. K., & West, K. D. (1987). "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"
5. Benjamini, Y., & Hochberg, Y. (1995). "Controlling the False Discovery Rate"

---

## 8. 예상 결과

ETF 승인일을 기준으로:
- **전통 금융시장 변수** (SPX, IWM, QQQ): 구조 변화 명확
- **금리 변수** (T10Y3M, T10Y2Y): 관계 반전
- **온체인 변수** (Active_Addresses, bc_n_transactions): 영향력 감소
- **Fed 유동성** (FED_NET_LIQUIDITY, WALCL): 역상관 전환

변화점: 2024-01-10 ± 7일 이내로 추정
검정력: 충분한 샘플 수 (ETF 전: 1,073일, ETF 후: 642일)
