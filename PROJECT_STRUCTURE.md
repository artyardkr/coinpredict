# 프로젝트 파일 구조 가이드

**작성일**: 2025-11-10
**총 파일 수**: 291개
**작성자**: Song Hyo Won

---

## 📁 폴더 구조 개요

```
코인데이터분석/
├── volatility_analysis/          # 변동성 동시분석 (최신)
│   ├── scripts/                  # 7개 Python 분석 스크립트
│   ├── results/
│   │   ├── csv/                 # 17개 CSV 결과 파일
│   │   └── images/              # 12개 PNG 시각화
│   └── reports/                 # 2개 종합 보고서
│
├── *.py                          # 82개 Python 스크립트 (루트)
├── *.csv                         # 106개 CSV 데이터 파일
├── *.png                         # 52개 PNG 시각화
└── *.md                          # 35개 Markdown 문서
```

---

## 📊 1. 변동성 분석 (volatility_analysis/)

### 📂 Scripts (volatility_analysis/scripts/)

| 파일명 | 목적 | 주요 기능 |
|--------|------|-----------|
| `volatility_comovement_analysis.py` | Step 1: 데이터 준비 | 실현 변동성(RV) 계산, ETF 전후 기초 통계 |
| `volatility_step2_garch_correlation.py` | Step 2: GARCH 모델 | 7개 핵심 자산 GARCH(1,1) 추정, 60일 롤링 상관관계 |
| `volatility_step3_spillover.py` | Step 3: VAR Spillover | Diebold-Yilmaz Spillover Index, BTC 변동성 분해 |
| `volatility_step4_granger_causality.py` | Step 4: 인과관계 | 그레인저 인과성 검정 (ETF 전후 비교) |
| `volatility_step5_dcc_garch_hierarchical.py` | Step 5: DCC-GARCH | 카테고리별 동적 조건부 상관관계 (4개 그룹) |
| `volatility_step6_markov_switching.py` | Step 6: 레짐 분석 | Markov Switching 모델, 변동성 레짐 전환 확률 |
| `volatility_step3_extended_category_analysis.py` | 확장 분석 | 카테고리별 상세 분석 |

### 📂 Results - CSV (volatility_analysis/results/csv/)

**GARCH & Correlation (2개)**
- `garch_volatility.csv` - GARCH(1,1) 조건부 변동성
- `rolling_correlation_60d.csv` - 60일 롤링 상관계수

**Spillover Analysis (2개)**
- `volatility_spillover_index.csv` - Spillover TO/FROM/NET 지수
- `volatility_btc_variance_decomposition.csv` - BTC 변동성 기여도 분해

**Granger Causality (3개)**
- `volatility_granger_causality_pre.csv` - ETF 이전 인과관계
- `volatility_granger_causality_post.csv` - ETF 이후 인과관계
- `volatility_btc_granger_causality.csv` - BTC 중심 인과관계

**DCC-GARCH (5개)**
- `dcc_garch_전통자산_dynamic_corr.csv` - SPX, GOLD, DXY, VIX, US10Y, SOFR
- `dcc_garch_거시경제_dynamic_corr.csv` - M2, CPI, EFFR, WALCL 등
- `dcc_garch_온체인_dynamic_corr.csv` - Hash Rate, Active Addresses 등
- `dcc_garch_밸류에이션_dynamic_corr.csv` - MVRV, NVT, Puell Multiple 등
- `dcc_garch_comparison_summary.csv` - 카테고리별 비교 요약

**Markov Switching (5개)**
- `markov_switching_regime_characteristics.csv` - 레짐별 특성 (수익률, 변동성)
- `markov_switching_regime_classification.csv` - 일별 레짐 분류
- `markov_switching_filtered_probabilities.csv` - 실시간 레짐 확률
- `markov_switching_smoothed_probabilities.csv` - 스무딩 레짐 확률
- `markov_switching_regime_performance.csv` - 레짐별 성과 통계

### 📂 Results - Images (volatility_analysis/results/images/)

**GARCH & Correlation (1개)**
- `volatility_rv_vs_garch.png` - 실현 변동성 vs GARCH 변동성 비교

**Spillover Analysis (1개)**
- `volatility_spillover_analysis.png` - Spillover Index 시계열, 네트워크 맵

**Granger Causality (2개)**
- `volatility_btc_granger_causality.png` - BTC 인과관계 히트맵
- `volatility_granger_causality_network.png` - 인과관계 네트워크 그래프

**DCC-GARCH (5개)**
- `dcc_garch_전통자산_plot.png` - 전통자산 동적 상관관계
- `dcc_garch_거시경제_plot.png` - 거시경제 동적 상관관계
- `dcc_garch_온체인_plot.png` - 온체인 동적 상관관계
- `dcc_garch_밸류에이션_plot.png` - 밸류에이션 동적 상관관계
- `dcc_garch_category_comparison.png` - 카테고리별 비교 차트

**Markov Switching (3개)**
- `markov_switching_regime_probabilities.png` - 레짐 전환 확률 시계열
- `markov_switching_return_distributions.png` - 레짐별 수익률 분포
- `markov_switching_etf_comparison.png` - ETF 전후 레짐 비교

### 📂 Reports (volatility_analysis/reports/)

| 파일명 | 내용 | 페이지 수 |
|--------|------|-----------|
| `변동성_동시분석_종합보고서.md` | 전체 5단계 분석 통합 보고서 | 200+ 페이지 |
| `변동성분석_결과해석가이드.md` | 실전 투자 전략 및 해석 가이드 | 50+ 페이지 |

**주요 발견사항**:
- BTC → GOLD 인과관계 출현 (p=0.158 → 0.046)
- 금리 관계 역전 (SOFR: -0.039 → +0.034)
- VIX가 BTC 최강 선행지표
- 저변동성 레짐 67.5%로 증가
- Spillover Index 37% 증가 (27.85% → 38.20%)

---

## 📊 2. 데이터 수집 스크립트

### 2.1 암호화폐 기본 데이터

| 파일명 | 데이터 소스 | 수집 항목 |
|--------|------------|-----------|
| `fetch_btc_data.py` | CoinGecko | BTC 가격/거래량 (2021-2025) |
| `fetch_crypto_data.py` | CoinGecko | BTC, ETH, SOL, DOGE, XRP |
| `collect_kimchi_premium.py` | Upbit + 환율 | 김치프리미엄 |
| `collect_long_short_ratio.py` | Binance Futures | 롱숏비율 |

**결과 파일**:
- `btc_data_2021_2025.csv`, `eth_data_2021_2025.csv`, `sol_data_2021_2025.csv`
- `doge_data_2021_2025.csv`, `xrp_data_2021_2025.csv`
- `crypto_close_prices_2021_2025.csv`, `crypto_volumes_2021_2025.csv`
- `upbit_krw_btc.csv`, `usd_krw_exchange_rate.csv`

### 2.2 전통 금융 시장

| 파일명 | 데이터 소스 | 수집 항목 |
|--------|------------|-----------|
| `step2_traditional_markets.py` | Yahoo Finance | SPX, GOLD, DXY, VIX, US10Y |
| `step2b_additional_markets.py` | Yahoo Finance | NASDAQ, S&P500, Russell2000 등 |
| `step8_btc_etf_data.py` | Yahoo Finance | IBIT, FBTC 등 현물 ETF |

**결과 파일**:
- `traditional_market_indices.csv`
- `additional_market_data.csv`
- `bitcoin_etf_data.csv`

### 2.3 거시경제 데이터

| 파일명 | 데이터 소스 | 수집 항목 |
|--------|------------|-----------|
| `step3_macro_data.py` | FRED API | CPI, M2, EFFR, US10Y 등 |
| `step3b_fed_liquidity.py` | FRED API | WALCL, RRPONTSYD (Fed 유동성) |

**결과 파일**:
- `fred_macro_data.csv`
- `fed_liquidity_data.csv`

### 2.4 온체인 & 파생상품

| 파일명 | 데이터 소스 | 수집 항목 |
|--------|------------|-----------|
| `step6_onchain_data.py` | Glassnode API | Hash Rate, Active Addresses 등 |
| `step6b_advanced_onchain.py` | Glassnode API | MVRV, NVT, Puell Multiple 등 |
| `collect_cryptoquant_all_data.py` | CryptoQuant | Exchange Flow, Whale 데이터 |
| `collect_binance_derivatives_2020_2025.py` | Binance | OI, Funding Rate, LS Ratio |
| `collect_derivatives_1hour.py` | Binance | 1시간 단위 파생상품 |

**결과 파일**:
- `onchain_data.csv`
- `advanced_onchain_data.csv`
- `binance_derivatives_2020_2025.csv`
- `binance_derivatives_2020_2025_full.csv`

### 2.5 감성 데이터

| 파일명 | 데이터 소스 | 수집 항목 |
|--------|------------|-----------|
| `step4_sentiment_data.py` | Alternative.me | Fear & Greed Index |

**결과 파일**:
- `sentiment_data.csv`

---

## 📊 3. 데이터 통합 & 전처리

| 파일명 | 목적 | 출력 |
|--------|------|------|
| `step5_integrate_data.py` | 일별 데이터 통합 (초기 버전) | `integrated_data_full.csv` |
| `step5b_integrate_all_new_data.py` | 신규 변수 포함 통합 (최종) | `integrated_data_full_v2.csv` |
| `create_4hour_data.py` | 4시간 데이터 생성 | `integrated_data_4hour.csv` |
| `create_4hour_data_simple.py` | 단순화 버전 | - |
| `create_4hour_data_recent.py` | 최근 데이터 전용 | - |

**핵심 통합 파일**:
- `integrated_data_full.csv` (137개 변수, 일별)
- `integrated_data_full_v2.csv` (신규 변수 포함)
- `integrated_data_4hour.csv` (4시간봉)

---

## 📊 4. 특성 엔지니어링 & 선택

### 4.1 특성 선택

| 파일명 | 방법론 | 결과 |
|--------|--------|------|
| `step7_feature_reduction.py` | FRA, RF, XGB | `feature_ranking_fra.csv`, `feature_scores_all_methods.csv` |
| `find_high_corr_features.py` | 상관관계 필터링 | `correlation_matrix.csv` |
| `calculate_and_save_correlations.py` | 상관계수 계산 | `correlation_analysis.csv`, `crypto_correlation.csv` |

### 4.2 기술적 지표

| 파일명 | 목적 |
|--------|------|
| `step1_technical_indicators.py` | RSI, MACD, Bollinger Bands 등 계산 |

**결과 파일**:
- `btc_technical_indicators.csv`

---

## 📊 5. 머신러닝 모델 (예측 분석)

### 5.1 초기 모델 실험

| 파일명 | 모델 | 목적 |
|--------|------|------|
| `step8_model_training.py` | RF, XGB, LSTM | 전체 기간 예측 |
| `step9_model_2021_only.py` | RF, XGB | 2021 데이터만 |
| `step10_model_2024_latest.py` | RF, XGB | 2024-2025 데이터 |
| `model_2024_2025.py` | 다양한 모델 | 최신 기간 검증 |
| `lstm_model_with_top10_features.py` | LSTM | Top 10 변수만 사용 |

**결과 파일**:
- `model_results.csv`
- `model_results_2021.csv`
- `model_performance_comparison.png`
- `lstm_performance.png`

### 5.2 변수 제거 실험

| 파일명 | 제거 변수 | 목적 |
|--------|----------|------|
| `step11_no_technical.py` | 기술적 지표 제거 | 과적합 검증 |
| `step16_no_technical_indicators.py` | 기술적 지표 제거 (v2) | 재검증 |
| `step12_macro_onchain_sentiment_volume.py` | 거시+온체인만 | 핵심 변수 효과 |

**결과 파일**:
- `model_results_no_technical.csv`
- `no_technical_indicators_results.csv`
- `model_results_macro_onchain_sentiment.csv`

### 5.3 다중 시계열 예측

| 파일명 | 예측 기간 |
|--------|----------|
| `step13_multi_horizon_prediction.py` | 1일, 7일, 30일 |
| `step22_multi_horizon_prediction.py` | 다중 기간 v2 |

**결과 파일**:
- `model_results_multi_horizon.csv`
- `multi_horizon_results.csv`
- `multi_horizon_etf_comparison.csv`
- Feature importance files (1d/7d/30d)

### 5.4 과적합 분석

| 파일명 | 목적 |
|--------|------|
| `step14_overfitting_analysis.py` | Train/Test 성능 비교 |
| `step17_fix_extrapolation.py` | 미래 데이터 누수 수정 |

**결과 파일**:
- `overfitting_analysis_results.csv`
- `data_leakage_suspects.csv`
- `extrapolation_fix_results.csv`

### 5.5 수익률 & 방향성 예측

| 파일명 | 예측 대상 |
|--------|----------|
| `step15_return_and_direction.py` | 수익률 + 방향 |
| `step19_direction_and_magnitude.py` | 방향 + 크기 |
| `step20_direction_prediction_all_features.py` | 방향 (전체 변수) |
| `step28_direction_backtesting.py` | 방향 예측 백테스팅 |

**결과 파일**:
- `return_prediction_results.csv`
- `direction_prediction_results.csv`
- `direction_magnitude_results.csv`
- `direction_backtesting_results.png`

### 5.6 가격 예측 (정규화)

| 파일명 | 방법 |
|--------|------|
| `step23_price_prediction_normalized.py` | Min-Max 정규화 |
| `step25_next_day_price_prediction.py` | 익일 가격 예측 |
| `step25_v2_next_day_prediction.py` | 익일 예측 v2 |

**결과 파일**:
- `price_prediction_normalized_results.csv`
- `next_day_price_prediction_results.csv`

---

## 📊 6. 회귀 모델 & ElasticNet

### 6.1 전통 회귀 모델

| 파일명 | 모델 |
|--------|------|
| `step24_all_regression_models.py` | Ridge, Lasso, ElasticNet, SVR 등 |
| `step29_stepwise_regression_proper.py` | Stepwise Regression |

**결과 파일**:
- `all_regression_models_results.csv`
- `stepwise_regression_results.png`

### 6.2 ElasticNet 심화 분석

| 파일명 | 목적 |
|--------|------|
| `step26_elasticnet_backtesting.py` | 백테스팅 (v1) |
| `step26_elasticnet_backtesting_v2.py` | 백테스팅 (v2, 신규 변수) |
| `step27_elasticnet_2025_only.py` | 2025년만 |
| `elasticnet_xai_analysis.py` | XAI (SHAP) v1 |
| `elasticnet_xai_analysis_v2.py` | XAI (SHAP) v2 |
| `elasticnet_daily_predictions.py` | 일별 예측값 |

**결과 파일**:
- `elasticnet_backtesting_results.csv`
- `elasticnet_backtesting_results_v2.csv`
- `elasticnet_2025_only_results.csv`
- `elasticnet_coefficients.csv`, `elasticnet_coefficients_v2.csv`
- `elasticnet_shap_importance.csv`, `elasticnet_shap_importance_v2.csv`
- `elasticnet_daily_predictions.csv`

### 6.3 4시간봉 분석

| 파일명 | 목적 |
|--------|------|
| `step25_4hour_version.py` | 4시간 단위 예측 |
| `verify_lasso_4hour.py` | Lasso 검증 |

**결과 파일**:
- `4hour_price_prediction_results.csv`
- `lasso_4hour_verification.png`

---

## 📊 7. ETF 영향 분석

### 7.1 기본 ETF 분석

| 파일명 | 목적 |
|--------|------|
| `analyze_etf_impact.py` | ETF 승인 전후 비교 |
| `step21_etf_comparison.py` | 모델 성능 비교 |
| `step30_dual_test_comparison.py` | Dual Test |

**결과 파일**:
- `etf_impact_summary.csv`
- `correlation_change_etf.csv`
- `etf_comparison_results.csv`
- `dual_test_comparison_results.csv`

### 7.2 ETF 전후 ElasticNet

| 파일명 | 기간 |
|--------|------|
| `step27_etf_pre_elasticnet.py` | ETF 이전 (2021-02-04 ~ 2024-01-09) |
| `step28_etf_post_elasticnet.py` | ETF 이후 (2024-01-10 ~ 2025-10-14) |
| `step31_etf_elasticnet_comparison.py` | 전후 통합 비교 |

**결과 파일**:
- `etf_pre_selected_variables.csv`, `etf_post_selected_variables.csv`
- `etf_pre_model_performance.csv`, `etf_post_model_performance.csv`
- `etf_pre_backtesting_results.csv`, `etf_post_backtesting_results.csv`
- `etf_elasticnet_performance.csv`

---

## 📊 8. 전략 백테스팅

### 8.1 백테스팅 스크립트

| 파일명 | 전략 |
|--------|------|
| `step18_backtesting.py` | 기본 백테스팅 |
| `predict_change_3methods.py` | 3가지 방법 비교 |
| `step30_trade_count.py` | 거래 횟수 분석 |
| `전략별_상세_성과_비교.py` | V1 vs V2 상세 비교 |

**결과 파일**:
- `backtesting_results.csv`
- `method1_regression_results.csv`
- `method2_classification_results.csv`
- `method3_multiclass_results.csv`
- `step30_trade_summary.csv`
- `step30_all_trades.csv`
- `strategy_detailed_comparison_table.csv`

### 8.2 베이스라인 비교

| 파일명 | 목적 |
|--------|------|
| `naive_baseline_comparison.py` | Naive 전략 vs 모델 |

**결과 파일**:
- `naive_baseline_results_pre.csv`
- `naive_baseline_results_post.csv`
- `naive_baseline_backtest_pre.csv`
- `naive_baseline_backtest_post.csv`

---

## 📊 9. 상관관계 & 리드-랙 분석

### 9.1 롤링 상관관계

| 파일명 | 목적 |
|--------|------|
| `rolling_correlation_analysis.py` | 60일 롤링 상관관계 |
| `multi_window_rolling_correlation.py` | 다중 윈도우 (30/60/90일) |

**결과 파일**:
- `rolling_correlation_summary.csv`
- `rolling_correlation_analysis.png`
- `rolling_correlation_60d.png` (루트)

### 9.2 BTC-GOLD 분석

| 파일명 | 목적 |
|--------|------|
| `btc_gold_similarity_analysis.py` | 유사성 분석 |
| `gold_btc_lead_lag_analysis.py` | 리드-랙 분석 |
| `gold_btc_three_hypotheses.py` | 3가지 가설 검증 |

**결과 파일**:
- `btc_gold_similarity_results.csv`
- `gold_btc_hypothesis_test.csv`
- `gold_btc_three_hypotheses_results.csv`

### 9.3 기간별 차이 분석

| 파일명 | 목적 |
|--------|------|
| `analyze_period_difference.py` | ETF 전후 통계 비교 |

---

## 📊 10. 구조 변화 분석 (Structural Change)

### 10.1 Chow Test & Quandt-Andrews

| 파일명 | 방법론 | 변수 수 |
|--------|--------|----------|
| `structural_change_tests.py` | Chow Test | 10개 변수 |
| `structural_change_tests_all_vars.py` | Chow + Q-A Test | 137개 전체 변수 |
| `zscore_structural_change_analysis.py` | Z-score 표준화 + Chow/Q-A | 137개 전체 변수 |

**결과 파일**:
- Chow Test 결과 (CSV, PNG)
- Q-A Test 결과 (CSV, PNG)
- Z-score 표준화 결과 (CSV, PNG)

**관련 보고서**:
- `비트코인_ETF_구조변화_분석_최종보고서.md`
- `비트코인_ETF_영향_분석_종합보고서.md`
- `Z-Score_표준화_구조변화분석_종합보고서.md`

---

## 📊 11. 시각화 & 유틸리티

### 11.1 시각화

| 파일명 | 목적 |
|--------|------|
| `visualize_trades_with_btc.py` | 거래 시각화 + BTC 가격 |
| `setup_korean_font.py` | 한글 폰트 설정 |

**결과 파일**:
- `trades_visualization_with_btc.png`
- `korean_font_test.png`

### 11.2 데이터 변환

| 파일명 | 목적 |
|--------|------|
| `convert_bitcoin_to_tafas.py` | Bitcoin → TAFAS 형식 변환 |
| `download_pdf.py` | PDF 다운로드 |

### 11.3 기타

| 파일명 | 목적 |
|--------|------|
| `규민tv.py`, `규민tv2.py` | 커스텀 분석 |
| `analyze_top20_categories.py` | 상위 20개 카테고리 분석 |

---

## 📊 12. 종합 보고서 (Markdown 문서)

### 12.1 핵심 보고서 (읽어야 할 순서)

1. **프로젝트 개요**
   - `README.md` - 프로젝트 전체 개요
   - `전체_분석_종합_정리.md` - 모든 분석 종합
   - `프로젝트_종합_평가.md` - 프로젝트 평가

2. **변동성 분석** ⭐ 최신
   - `volatility_analysis/reports/변동성_동시분석_종합보고서.md` (200+ 페이지)
   - `volatility_analysis/reports/변동성분석_결과해석가이드.md` (50+ 페이지)

3. **구조 변화 분석**
   - `비트코인_ETF_구조변화_분석_최종보고서.md`
   - `비트코인_ETF_영향_분석_종합보고서.md`
   - `Z-Score_표준화_구조변화분석_종합보고서.md` (45+ 페이지)

4. **ETF 분석**
   - `ETF_전후_분석_방법론.md`
   - `ETF_전후_ElasticNet_비교분석.md`

5. **백테스팅 & 전략**
   - `V1_vs_V2_백테스팅_비교분석.md`

### 12.2 방법론 & 가이드

- `structural_change_tests_plan.md` - 구조 변화 검정 계획
- `구조변화검정_쉬운_설명.md` - 쉬운 설명
- `구조변화검정_표준_프로토콜.md` - 표준 절차
- `다중공선성_문제_분석.md` - 다중공선성 해결
- `데이터_수집_가이드.md` - 데이터 수집 매뉴얼

### 12.3 변수 & 데이터 문서

- `NEW_VARIABLES_DOCUMENTATION.md` - 신규 변수 문서화
- `신규변수_추가_가이드.md` - 변수 추가 방법
- `ElasticNet_변수분석.md`
- `Step25_ElasticNet_변수분석.md`
- `TAFAS_통합_작업_정리.md`

### 12.4 논문 & 참고자료

- `논문1.md`, `논문1_요약.md`
- `논문2.md`, `논문2_요약.md`
- `논문3.md`, `논문3_상세.md`
- `논문_변수_정리.md`

### 12.5 발표 자료

- `PPT_구성안.md`
- `PPT_데이터수집_2장.md`
- `PPT_이미지_가이드.md`
- `발표자료_정리.md`
- `발표_추가자료.md`

### 12.6 초기 계획 & 문제점

- `test1_프로젝트_계획.md`
- `데이터분석계획초본.md`
- `multi_horizon_analysis_summary.md`
- `문제점차원의저주.md`

### 12.7 작업 이력

- `파일_작성_일지.md` - 파일 생성 이력

---

## 📊 13. 주요 분석 결과 요약

### 13.1 변동성 분석 주요 발견

1. **BTC → GOLD 인과관계 출현** (획기적 발견)
   - ETF 이전: p=0.1580 (비유의)
   - ETF 이후: p=0.0463 (유의)
   - 위치: `volatility_analysis/results/csv/volatility_btc_granger_causality.csv`

2. **금리 관계 역전**
   - SOFR 상관관계: -0.039 → +0.034
   - 위치: `volatility_analysis/results/csv/dcc_garch_거시경제_dynamic_corr.csv`

3. **VIX 선행성 강화**
   - VIX가 BTC 변동성의 최강 선행지표
   - Granger Causality: VIX → BTC (p<0.01)

4. **레짐 변화**
   - 저변동성 레짐: 55.3% → 67.5%
   - 위치: `volatility_analysis/results/csv/markov_switching_regime_characteristics.csv`

5. **Spillover 증가**
   - Spillover Index: 27.85% → 38.20% (+37%)
   - 위치: `volatility_analysis/results/csv/volatility_spillover_index.csv`

### 13.2 구조 변화 분석

- **Chow Test**: 137개 변수 중 62개 구조 변화 (45.3%)
- **Q-A Test**: 48개 변수 최대 F-통계량 시점 = ETF 승인일
- 위치: 루트 디렉토리 `*chow*.csv`, `*qa*.csv`

### 13.3 ElasticNet 예측 성능

| 기간 | R² | RMSE | MAE | Sharpe |
|------|-----|------|-----|--------|
| ETF 이전 | 0.7489 | 3,021 | 2,143 | 1.23 |
| ETF 이후 | 0.8124 | 2,847 | 1,982 | 1.56 |

위치: `etf_elasticnet_performance.csv`

---

## 📊 14. 파일 검색 가이드

### 14.1 목적별 빠른 검색

**변동성 분석 결과를 보고 싶다면?**
→ `volatility_analysis/reports/변동성_동시분석_종합보고서.md`

**구조 변화 분석 결과를 보고 싶다면?**
→ `비트코인_ETF_구조변화_분석_최종보고서.md`

**ElasticNet 백테스팅 결과를 보고 싶다면?**
→ `elasticnet_backtesting_results_v2.csv` + `elasticnet_backtesting_results_v2.png`

**ETF 전후 성능 비교를 보고 싶다면?**
→ `etf_elasticnet_performance.csv` + `etf_elasticnet_comparison.png`

**BTC-GOLD 관계를 보고 싶다면?**
→ `volatility_analysis/results/csv/volatility_btc_granger_causality.csv`
→ `gold_btc_three_hypotheses_results.csv`

**전체 프로젝트 요약을 보고 싶다면?**
→ `전체_분석_종합_정리.md`

### 14.2 파일명 패턴

| 패턴 | 의미 |
|------|------|
| `step*` | 순차적 분석 단계 |
| `*_v2.py` | 개선 버전 (v2) |
| `*_pre.csv` | ETF 이전 (2021-02-04 ~ 2024-01-09) |
| `*_post.csv` | ETF 이후 (2024-01-10 ~ 2025-10-14) |
| `*_4hour.*` | 4시간봉 데이터 |
| `elasticnet_*` | ElasticNet 관련 |
| `volatility_*` | 변동성 분석 관련 |
| `dcc_garch_*` | DCC-GARCH 관련 |
| `markov_switching_*` | Markov Switching 관련 |
| `zscore_*` | Z-score 표준화 관련 |

### 14.3 데이터 버전 관리

| 버전 | 파일명 | 변수 수 | 특징 |
|------|--------|---------|------|
| V1 (초기) | `integrated_data_full.csv` | 137개 | 기본 변수 |
| V2 (최종) | `integrated_data_full_v2.csv` | 137개+ | 신규 변수 포함 (ETF, Fed 유동성) |
| 4시간 | `integrated_data_4hour.csv` | 동일 | 4시간봉 변환 |

---

## 📊 15. 분석 파이프라인

```
1. 데이터 수집
   ├─ 암호화폐 (fetch_*.py, collect_*.py)
   ├─ 전통 시장 (step2*.py)
   ├─ 거시경제 (step3*.py)
   ├─ 온체인 (step6*.py)
   └─ 감성 (step4*.py)

2. 데이터 통합
   └─ step5b_integrate_all_new_data.py → integrated_data_full_v2.csv

3. 특성 엔지니어링
   ├─ step1_technical_indicators.py (기술적 지표)
   └─ step7_feature_reduction.py (특성 선택)

4. 모델 학습 & 예측
   ├─ ElasticNet (step24~31)
   ├─ Random Forest (step8~)
   ├─ XGBoost (step8~)
   └─ LSTM (lstm_*.py)

5. 백테스팅
   ├─ step18_backtesting.py
   ├─ step26_elasticnet_backtesting_v2.py
   └─ step28_direction_backtesting.py

6. 구조 변화 분석
   ├─ structural_change_tests_all_vars.py (Chow/Q-A)
   └─ zscore_structural_change_analysis.py (Z-score)

7. 변동성 분석 ⭐ 최신
   ├─ volatility_step2_garch_correlation.py (GARCH)
   ├─ volatility_step3_spillover.py (Spillover)
   ├─ volatility_step4_granger_causality.py (Granger)
   ├─ volatility_step5_dcc_garch_hierarchical.py (DCC-GARCH)
   └─ volatility_step6_markov_switching.py (Markov)
```

---

## 📊 16. 주요 성과 지표

### 16.1 모델 성능

| 모델 | 기간 | R² | RMSE | Sharpe |
|------|------|-----|------|--------|
| ElasticNet V2 | 전체 | 0.7806 | 2,934 | 1.39 |
| ElasticNet | ETF 이전 | 0.7489 | 3,021 | 1.23 |
| ElasticNet | ETF 이후 | 0.8124 | 2,847 | 1.56 |
| Random Forest | 전체 | 0.7234 | 3,156 | 1.12 |

### 16.2 백테스팅 수익률

| 전략 | 기간 | 총 수익률 | 연환산 | Sharpe | MDD |
|------|------|----------|--------|--------|-----|
| ElasticNet V2 | 2024-2025 | +78.3% | +52.1% | 1.56 | -18.2% |
| Naive Baseline | 2024-2025 | +45.2% | +28.4% | 0.89 | -31.5% |

### 16.3 구조 변화

- **Chow Test 유의 변수**: 62개 / 137개 (45.3%)
- **Q-A Test 최대 F 시점**: 2024-01-10 (ETF 승인일)
- **Z-score 표준화 후**: 동일한 F-통계량, 계수 비교 가능

---

## 📊 17. 다음 단계 제안

### 17.1 추가 분석 (미실행)

- [ ] Time-varying Spillover Index (롤링 윈도우)
- [ ] Rolling Beta Analysis
- [ ] Wavelet Coherence (시간-주파수 분석)
- [ ] Network Centrality Analysis
- [ ] Jump Detection (Bi-power Variation)

### 17.2 모델 개선

- [ ] Ensemble (RF + XGB + ElasticNet)
- [ ] Transformer 모델
- [ ] Quantile Regression
- [ ] Regime-dependent 모델

### 17.3 실전 적용

- [ ] 실시간 데이터 파이프라인
- [ ] 자동 매매 시스템
- [ ] 포트폴리오 최적화
- [ ] 리스크 관리 시스템

---

## 📊 18. 문의 & 참고

**작성자**: Song Hyo Won
**작성일**: 2025-11-10
**분석 기간**: 2021-02-04 ~ 2025-10-14
**ETF 승인일**: 2024-01-10

**주요 도구**:
- Python 3.x
- pandas, numpy, scipy
- statsmodels (GARCH, VAR, Markov Switching)
- arch (GARCH)
- scikit-learn (ElasticNet, RF)
- xgboost
- matplotlib, seaborn

**데이터 소스**:
- CoinGecko (암호화폐)
- Yahoo Finance (전통 시장, ETF)
- FRED (거시경제)
- Glassnode (온체인)
- Binance (파생상품)
- Alternative.me (Fear & Greed)

---

## 📊 부록: 전체 파일 수 통계

| 파일 유형 | 개수 |
|----------|------|
| Python 스크립트 (.py) | 82개 |
| CSV 데이터 (.csv) | 106개 |
| PNG 이미지 (.png) | 52개 |
| Markdown 문서 (.md) | 35개 |
| **총계** | **275개** |

*(volatility_analysis 폴더 내 16개 파일 별도)*

**Grand Total: 291개 파일**

---

**이 문서는 파일 이동 없이 프로젝트 구조를 문서화한 것입니다.**
**모든 Python 스크립트의 상대 경로는 유지되므로 정상 작동합니다.**
