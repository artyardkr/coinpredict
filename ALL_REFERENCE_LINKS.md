# 코인데이터분석 프로젝트 - 전체 참고 링크 목록

**작성일**: 2025-11-18
**프로젝트**: Bitcoin ETF 구조변화 분석
**분석 기간**: 2021-02-03 ~ 2025-10-14

---

## 📚 목차

1. [학술 논문 참고 자료](#1-학술-논문-참고-자료)
2. [데이터 소스 & API](#2-데이터-소스--api)
3. [프로젝트 문서 링크](#3-프로젝트-문서-링크)
4. [외부 참고 자료](#4-외부-참고-자료)
5. [관련 도구 & 라이브러리](#5-관련-도구--라이브러리)

---

## 1. 학술 논문 참고 자료

### 선행 연구 (비트코인 가격 예측 관련)

#### Kristoufek (2013)
**제목**: Bitcoin meets Google Trends and Wikipedia: Quantifying the relationship between phenomena of the Internet era

**저자**: Ladislav Kristoufek

**출처**: Scientific Reports 3, Article number: 3415 (2013)

**DOI**: https://doi.org/10.1038/srep03415

**핵심 기여**: 비트코인 가격 예측에 Google Trends 및 Wikipedia 조회수 사용 (소셜/웹 데이터 중심)

**비고**: 최초로 검색량과 비트코인 가격의 상관관계를 실증한 연구

---

#### Jang and Lee (2017)
**제목**: An Empirical Study on Modeling and Prediction of Bitcoin Prices With Bayesian Neural Networks Based on Blockchain Information

**저자**: Huisu Jang, Jaewook Lee

**출처**: IEEE Access, vol. 6, pp. 5427-5437, 2017

**DOI**: https://doi.org/10.1109/ACCESS.2017.2779181

**핵심 기여**: 비트코인 블록체인 정보와 거시 경제 요인 (S&P 500, DOW30) 활용 (복합적 요인 활용)

**비고**: 온체인 데이터와 전통 시장 지표를 결합한 초기 연구

---

#### Abraham et al. (2018)
**제목**: Cryptocurrency price prediction using tweet volumes and sentiment analysis

**저자**: Jethin Abraham, Daniel Higdon, John Nelson, Juan Ibarra

**출처**: SMU Data Science Review, vol. 1, no. 3, article 1, 2018

**URL**: https://scholar.smu.edu/datasciencereview/vol1/iss3/1

**핵심 기여**: Google Trends와 트윗량을 암호화폐 가격 변동 예측에 사용 (소셜 미디어 중심)

**비고**: Twitter 감성 분석을 활용한 초기 연구

---

#### Valencia et al. (2019)
**제목**: Price movement prediction of cryptocurrencies using sentiment analysis and machine learning

**저자**: Franco Valencia, Alfonso Gómez-Espinosa, Benjamín Valdés-Aguirre

**출처**: Entropy 21(6), 589 (2019)

**DOI**: https://doi.org/10.3390/e21060589

**핵심 기여**: 사용자 감정(Sentiment)을 가격 예측 요인으로 활용 (심리적 요인 중심)

**비고**: 감성 분석을 체계적으로 적용한 연구

---

#### Saad et al. (2019)
**제목**: Toward characterizing blockchain-based cryptocurrencies for highly accurate predictions

**저자**: Mohamed Saad, Zine Eddine Youness Chebouba, Aziz Mohaisen, Daehun Nyang

**출처**: IEEE Systems Journal, vol. 14, no. 1, pp. 321-332, 2020

**DOI**: https://doi.org/10.1109/JSYST.2019.2927707

**핵심 기여**: 채굴자 수익, 수수료, 해시율 등 비트코인 블록체인 정보 사용 (온체인 데이터 중심)

**비고**: 네트워크 지표의 예측력을 실증한 연구

---

#### Mallqui and Fernandes (2019)
**제목**: Predicting the direction, maximum, minimum and closing prices of daily Bitcoin exchange rate using machine learning techniques

**저자**: Darwin C.L. Mallqui, Ricardo A.S. Fernandes

**출처**: Applied Soft Computing, vol. 75, pp. 596-606, 2019

**DOI**: https://doi.org/10.1016/j.asoc.2018.11.038

**핵심 기여**: 블록체인, 거시 경제, Google 인기 지수 등을 조합하여 사용 (다차원 데이터 활용)

**비고**: 다양한 데이터 소스 통합의 중요성을 강조

---

#### Kahneman & Tversky (1979)
**제목**: Prospect Theory: An Analysis of Decision under Risk

**저자**: Daniel Kahneman, Amos Tversky

**출처**: Econometrica, vol. 47, no. 2, pp. 263-291, 1979

**DOI**: https://doi.org/10.2307/1914185

**핵심 기여**: 전망 이론(Prospect Theory) - 손실 회피 편향의 학술적 근거 (행동경제학 이론)

**비고**: 노벨 경제학상 수상 (2002, Kahneman), 투자자 심리 분석의 이론적 토대

---

### 본 연구 참고 논문 (3편)

#### 논문 1: From On-chain to Macro

**제목**: From On-chain to Macro: Assessing the Importance of Data Source Diversity in Cryptocurrency Market Forecasting

**저자**: Giorgos Demosthenous, Chryssis Georgiou, Eliada Polydorou (University of Cyprus)

**출처**: VLDB 2024 Workshop: Foundations and Applications of Blockchain (FAB)

**arXiv**: https://arxiv.org/abs/2506.21246v1 (arXiv:2506.21246v1 [q-fin.PM] 26 Jun 2025)

**GitHub**: https://github.com/gdemos01/FAB-2024

**주요 기여**:
- 데이터 소스 다양성의 중요성 실증
- Crypto100 지수 개발
- Feature Reduction Algorithm (FRA)
- 5개 데이터 카테고리 통합 (429개 → 100개 변수)

**프로젝트 참고 파일**:
- `/Users/songhyowon/코인데이터분석/논문1.md` (전체 논문)
- `/Users/songhyowon/코인데이터분석/논문1_요약.md` (요약본)

---

### 논문 2: Helformer

**제목**: Helformer: an attention-based deep learning model for cryptocurrency price forecasting

**저자**: T. O. Kehinde, et al. (The Hong Kong Polytechnic University)

**출처**: Journal of Big Data (2025) 12:81

**DOI**: https://doi.org/10.1186/s40537-025-01135-4

**주요 기여**:
- Holt-Winters + Transformer 결합 (Helformer)
- 계절성/비정상성 자동 처리
- 15개 코인 Transfer Learning 검증

**프로젝트 참고 파일**:
- `/Users/songhyowon/코인데이터분석/논문2.md` (전체 논문)
- `/Users/songhyowon/코인데이터분석/논문2_요약.md` (요약본)

---

### 논문 3: On-chain Data & CNN-LSTM

**제목**: Bitcoin price direction prediction using on-chain data and feature selection

**저자**: Ritwik Dubey, David Enke (Missouri University of Science and Technology)

**출처**: Machine Learning with Applications 20 (2025) 100674

**DOI**: https://doi.org/10.1016/j.mlwa.2025.100674

**주요 기여**:
- 온체인 데이터 5가지 카테고리 분류
- Boruta Feature Selection
- CNN-LSTM 모델 (정확도 82.03%)

**프로젝트 참고 파일**:
- `/Users/songhyowon/코인데이터분석/논문3.md` (전체 논문)
- `/Users/songhyowon/코인데이터분석/논문3_상세.md` (상세 분석)

---

### 논문 변수 통합 정리

**파일**: `/Users/songhyowon/코인데이터분석/논문_변수_정리.md`

**내용**: 3개 논문의 변수 통합 정리 및 비교

---

## 2. 데이터 소스 & API

### 2.1 암호화폐 가격 데이터

#### Yahoo Finance
- **URL**: https://finance.yahoo.com
- **라이브러리**: yfinance
- **수집 항목**: BTC, ETH, SOL, DOGE, XRP 가격/거래량
- **사용 코드**: `fetch_btc_data.py`, `fetch_crypto_data.py`

#### CoinGecko
- **URL**: https://www.coingecko.com
- **API Docs**: https://www.coingecko.com/en/api
- **수집 항목**: 암호화폐 가격, 시가총액, 거래량
- **카테고리**: https://www.coingecko.com/en/categories (논문1 참조)

---

### 2.2 거시경제 데이터

#### FRED (Federal Reserve Economic Data)
- **URL**: https://fred.stlouisfed.org
- **API Key**: https://fred.stlouisfed.org/docs/api/api_key.html (무료 등록)
- **라이브러리**: fredapi

**주요 지표**:
- **금리**: DFF (Federal Funds Rate), DGS10 (10-Year Treasury), SOFR
- **인플레이션**: CPIAUCSL (CPI)
- **통화량**: M2SL
- **경제**: GDP, UNRATE (실업률)
- **Fed 유동성**: WALCL (Fed 총자산), RRPONTSYD (역레포), WTREGEN (재무부 계정)

**사용 코드**: `step3_macro_data.py`, `step3b_fed_liquidity.py`

---

### 2.3 전통 시장 지수

#### Yahoo Finance (전통 시장)
**수집 항목**:
- **주식**: SPX (S&P 500), QQQ (Nasdaq-100), DIA (Dow Jones), IWM (Russell 2000)
- **채권**: TLT (20년 국채), LQD (투자등급 회사채), HYG (하이일드)
- **원자재**: GLD (Gold), SLV (Silver), USO (Oil)
- **환율**: DXY (Dollar Index), EURUSD, DTWEXBGS

**사용 코드**: `step2_traditional_markets.py`, `step2b_additional_markets.py`

#### Invesco (달러 인덱스 ETF)
- **URL**: https://www.invesco.com/us/financial-products/etfs/product-detail?ticker=UUP
- **티커**: UUP (Dollar Index Bullish Fund)

#### Vanguard (채권 ETF)
- **URL**: https://investor.vanguard.com/investment-products/etfs/profile/bsv
- **티커**: BSV (Short-Term Bond ETF)

---

### 2.4 온체인 데이터

#### Blockchain.com
- **URL**: https://blockchain.com
- **API**: https://www.blockchain.com/explorer/api/blockchain_api (무료)

**수집 항목**:
- n-transactions (거래 건수)
- hash-rate (해시레이트)
- difficulty (채굴 난이도)
- trade-volume (거래량)
- total-bitcoins (총 발행량)

**사용 코드**: `step6_onchain_data.py`

#### CoinMetrics
- **URL**: https://coinmetrics.io
- **Catalog**: https://studio.glassnode.com/catalog
- **API**: Community 버전 (제한적 무료)

**수집 항목** (논문1 참조):
- MVRV, NVT Ratio, Puell Multiple
- Hash Ribbon, Difficulty Ribbon
- 주소 밸런스, 공급 분포
- 채굴자 수익, 네트워크 활동

**사용 코드**: `step6b_advanced_onchain.py`

#### Glassnode
- **URL**: https://glassnode.com
- **Studio**: https://studio.glassnode.com/metrics
- **데이터 소스**: 논문1, 논문3 사용
- **무료 티어**: 기본 온체인 지표 제공

---

### 2.5 Bitcoin ETF 데이터

#### Yahoo Finance (ETF)
**수집 항목**:
- IBIT (BlackRock Bitcoin ETF)
- FBTC (Fidelity Bitcoin ETF)
- GBTC (Grayscale Bitcoin Trust)
- ARKB (Ark Bitcoin ETF)
- BITB (Bitwise Bitcoin ETF)

**사용 코드**: `step8_btc_etf_data.py`

**SEC 승인 뉴스** (2024-01-11):
- "美증권위, 11개 비트코인 현물 ETF 상장 승인"
- BlackRock, Fidelity, Ark, Grayscale 포함

---

### 2.6 감성 지표

#### Alternative.me (Fear & Greed Index)
- **URL**: https://alternative.me
- **API**: https://api.alternative.me/fng/?limit={days} (무료)
- **범위**: 0-100 (0=극도 공포, 100=극도 탐욕)

**사용 코드**: `step4_sentiment_data.py`

#### Google Trends
- **URL**: https://trends.google.com
- **라이브러리**: pytrends
- **검색어**: Bitcoin, Cryptocurrency, Ethereum

**사용 코드**: `step4_sentiment_data.py`

---

### 2.7 파생상품 데이터

#### Binance Futures
- **API Docs**: https://binance-docs.github.io/apidocs/futures/en/
- **수집 항목**: Open Interest, Funding Rate, Long/Short Ratio

**사용 코드**: `collect_binance_derivatives_2020_2025.py`

#### CryptoQuant
- **URL**: https://cryptoquant.com
- **수집 항목**: Exchange Flow, Whale 데이터

**사용 코드**: `collect_cryptoquant_all_data.py`

---

## 3. 프로젝트 문서 링크

### 3.1 종합 보고서

#### 변동성 동시분석 종합보고서 (최신!)
- **파일**: `/Users/songhyowon/코인데이터분석/volatility_analysis/reports/변동성_동시분석_종합보고서.md`
- **페이지**: 200+
- **내용**: GARCH, DCC-GARCH, Granger, Markov 분석

#### 비트코인 ETF 영향 분석 종합보고서
- **파일**: `/Users/songhyowon/코인데이터분석/비트코인_ETF_영향_분석_종합보고서.md`
- **페이지**: 85
- **내용**: 전체 연구 최종 보고서 (V1 vs V2 포함)

#### Z-Score 표준화 구조변화분석 종합보고서
- **파일**: `/Users/songhyowon/코인데이터분석/Z-Score_표준화_구조변화분석_종합보고서.md`
- **페이지**: 45
- **내용**: 표준화 계수 비교를 통한 영향력 변화 분석

#### 비트코인 ETF 구조변화 분석 최종보고서
- **파일**: `/Users/songhyowon/코인데이터분석/비트코인_ETF_구조변화_분석_최종보고서.md`
- **페이지**: 50
- **내용**: 구조변화 검정 중심 보고서

---

### 3.2 프로젝트 구조 & README

#### PROJECT_STRUCTURE.md
- **파일**: `/Users/songhyowon/코인데이터분석/PROJECT_STRUCTURE.md`
- **내용**: 전체 291개 파일 구조 가이드 (카테고리별 정리)

#### README.md
- **파일**: `/Users/songhyowon/코인데이터분석/README.md`
- **내용**: 프로젝트 전체 개요, 설치, 사용법

#### 전체 분석 종합 정리
- **파일**: `/Users/songhyowon/코인데이터분석/전체_분석_종합_정리.md`
- **내용**: 프로젝트 전체 흐름 요약

---

### 3.3 방법론 & 가이드

#### 구조변화검정 쉬운 설명
- **파일**: `/Users/songhyowon/코인데이터분석/구조변화검정_쉬운_설명.md`
- **내용**: Chow/QA/CUSUM 비전공자용 설명

#### 구조변화검정 표준 프로토콜
- **파일**: `/Users/songhyowon/코인데이터분석/구조변화검정_표준_프로토콜.md`
- **내용**: 표준 절차 및 프로토콜

#### 다중공선성 문제 분석
- **파일**: `/Users/songhyowon/코인데이터분석/다중공선성_문제_분석.md`
- **내용**: "119개 변수 검정해도 문제없는 이유"

#### ETF 전후 분석 방법론
- **파일**: `/Users/songhyowon/코인데이터분석/ETF_전후_분석_방법론.md`
- **내용**: 통계 검정 상세 설명

#### structural_change_tests_plan
- **파일**: `/Users/songhyowon/코인데이터분석/structural_change_tests_plan.md`
- **내용**: 구조 변화 검정 계획

---

### 3.4 변수 & 데이터 문서

#### NEW_VARIABLES_DOCUMENTATION
- **파일**: `/Users/songhyowon/코인데이터분석/NEW_VARIABLES_DOCUMENTATION.md`
- **내용**: V2 신규 변수 50개 상세 설명

#### 신규변수 추가 가이드
- **파일**: `/Users/songhyowon/코인데이터분석/신규변수_추가_가이드.md`
- **내용**: 변수 추가 방법 및 절차

#### Step25 ElasticNet 변수분석
- **파일**: `/Users/songhyowon/코인데이터분석/Step25_ElasticNet_변수분석.md`
- **내용**: ElasticNet 계수 해석

#### ElasticNet 변수분석
- **파일**: `/Users/songhyowon/코인데이터분석/ElasticNet_변수분석.md`
- **내용**: ElasticNet 변수 분석

#### 데이터 수집 가이드
- **파일**: `/Users/songhyowon/코인데이터분석/데이터_수집_가이드.md`
- **내용**: 데이터 수집 매뉴얼

---

### 3.5 백테스팅 & 전략

#### V1 vs V2 백테스팅 비교분석
- **파일**: `/Users/songhyowon/코인데이터분석/V1_vs_V2_백테스팅_비교분석.md`
- **내용**: V2 모델이 V1보다 우수한 이유

#### ETF 전후 ElasticNet 비교분석
- **파일**: `/Users/songhyowon/코인데이터분석/ETF_전후_ElasticNet_비교분석.md`
- **내용**: ETF 전후 변수 중요도 변화

#### 거래비용 제거 분석
- **파일**: `/Users/songhyowon/코인데이터분석/거래비용_제거_분석.md`
- **내용**: 거래비용 제거 후 성과 분석

#### 전략별 거래방식 상세설명
- **파일**: `/Users/songhyowon/코인데이터분석/전략별_거래방식_상세설명.md`
- **내용**: 6가지 전략 상세 설명

---

### 3.6 발표 자료

#### PPT 구성안
- **파일**: `/Users/songhyowon/코인데이터분석/PPT_구성안.md`
- **내용**: 발표 자료 구성 (25슬라이드, 10-15분)

#### PPT 데이터수집 2장
- **파일**: `/Users/songhyowon/코인데이터분석/PPT_데이터수집_2장.md`
- **내용**: 데이터 수집 슬라이드 (91개 변수 설명)

#### PPT 이미지 가이드
- **파일**: `/Users/songhyowon/코인데이터분석/PPT_이미지_가이드.md`
- **내용**: 발표 이미지 가이드

#### 발표자료 정리
- **파일**: `/Users/songhyowon/코인데이터분석/발표자료_정리.md`
- **내용**: 10-15분 발표 자료 (스크립트 포함)

#### 발표 추가자료
- **파일**: `/Users/songhyowon/코인데이터분석/발표_추가자료.md`
- **내용**: 발표 추가 자료

---

## 4. 외부 참고 자료

### 4.1 통계 방법론

#### Chow Test
- **위키백과**: https://en.wikipedia.org/wiki/Chow_test
- **설명**: 구조 변화 검정

#### Quandt-Andrews Test
- **논문**: Andrews, D. W. K. (1993). "Tests for Parameter Instability and Structural Change With Unknown Change Point"
- **Econometrica**: https://www.jstor.org/stable/2951764

#### CUSUM Test
- **위키백과**: https://en.wikipedia.org/wiki/CUSUM
- **설명**: 누적 합 검정

#### HAC 표준오차 (Newey-West)
- **논문**: Newey, W. K., & West, K. D. (1987). "A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"
- **Econometrica**: https://www.jstor.org/stable/1913610

---

### 4.2 머신러닝 방법론

#### ElasticNet
- **논문**: Zou, H., & Hastie, T. (2005). "Regularization and variable selection via the elastic net"
- **Journal**: Journal of the Royal Statistical Society: Series B
- **위키백과**: https://en.wikipedia.org/wiki/Elastic_net_regularization

#### Random Forest
- **논문**: Breiman, L. (2001). "Random Forests"
- **위키백과**: https://en.wikipedia.org/wiki/Random_forest

#### XGBoost
- **논문**: Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"
- **GitHub**: https://github.com/dmlc/xgboost
- **Docs**: https://xgboost.readthedocs.io

#### SHAP (Shapley Additive exPlanations)
- **논문**: Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions"
- **GitHub**: https://github.com/slundberg/shap
- **Docs**: https://shap.readthedocs.io

---

### 4.3 시계열 분석

#### Holt-Winters Exponential Smoothing
- **위키백과**: https://en.wikipedia.org/wiki/Exponential_smoothing
- **설명**: 계절성 시계열 예측

#### ARIMA
- **위키백과**: https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
- **설명**: 자기회귀 누적 이동평균 모델

#### LSTM (Long Short-Term Memory)
- **논문**: Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory"
- **DOI**: https://doi.org/10.1162/neco.1997.9.8.1735

#### Transformer
- **논문**: Vaswani, A., et al. (2017). "Attention is All You Need"
- **arXiv**: https://arxiv.org/abs/1706.03762

---

### 4.4 암호화폐 관련

#### Bitcoin Whitepaper
- **제목**: Bitcoin: A Peer-to-Peer Electronic Cash System
- **저자**: Satoshi Nakamoto (2008)
- **URL**: https://bitcoin.org/bitcoin.pdf

#### Stock-to-Flow Model
- **논문**: PlanB (2019). "Modeling Bitcoin's Value with Scarcity"
- **Medium**: https://medium.com/@100trillionUSD/modeling-bitcoins-value-with-scarcity-91fa0fc03e25

#### NVT Ratio
- **설명**: Network Value to Transactions Ratio
- **소개자**: Willy Woo (2017)
- **URL**: https://woobull.com/introducing-nvt-ratio-bitcoins-pe-ratio-use-it-to-detect-bubbles/

#### MVRV Ratio
- **설명**: Market Value to Realized Value Ratio
- **Glassnode**: https://academy.glassnode.com/indicators/market-indicators/mvrv-ratio

---

## 5. 관련 도구 & 라이브러리

### 5.1 Python 패키지

#### 데이터 수집
- **yfinance**: https://github.com/ranaroussi/yfinance
- **fredapi**: https://github.com/mortada/fredapi
- **requests**: https://requests.readthedocs.io
- **pytrends**: https://github.com/GeneralMills/pytrends

#### 데이터 처리
- **pandas**: https://pandas.pydata.org
- **numpy**: https://numpy.org

#### 기술적 지표
- **ta**: https://github.com/bukosabino/ta (Technical Analysis Library in Python)
- **pandas-ta**: https://github.com/twopirllc/pandas-ta

#### 머신러닝
- **scikit-learn**: https://scikit-learn.org
- **xgboost**: https://xgboost.readthedocs.io
- **lightgbm**: https://lightgbm.readthedocs.io

#### 통계 분석
- **statsmodels**: https://www.statsmodels.org
- **scipy**: https://scipy.org

#### XAI (설명가능한 AI)
- **shap**: https://shap.readthedocs.io

#### 시각화
- **matplotlib**: https://matplotlib.org
- **seaborn**: https://seaborn.pydata.org
- **plotly**: https://plotly.com/python

---

### 5.2 개발 도구

#### Version Control
- **Git**: https://git-scm.com
- **GitHub**: https://github.com

#### Package Manager
- **pip**: https://pip.pypa.io
- **requirements.txt**: `/Users/songhyowon/코인데이터분석/requirements.txt`

#### Jupyter Notebook
- **Jupyter**: https://jupyter.org
- **JupyterLab**: https://jupyterlab.readthedocs.io

---

### 5.3 참고 사이트

#### 기술 블로그
- **Towards Data Science**: https://towardsdatascience.com
- **Analytics Vidhya**: https://www.analyticsvidhya.com
- **KDnuggets**: https://www.kdnuggets.com

#### Stack Overflow
- **URL**: https://stackoverflow.com
- **태그**: python, machine-learning, scikit-learn, pandas

#### arXiv (머신러닝 논문)
- **URL**: https://arxiv.org
- **카테고리**: cs.LG (Machine Learning), stat.ML (Statistics), q-fin.PM (Portfolio Management)

---

## 6. 프로젝트 특정 링크

### 6.1 GitHub 저장소 (예상)
- **URL**: (프로젝트가 GitHub에 업로드되면 여기에 링크 추가)
- **Branch**: main / master
- **README**: `/Users/songhyowon/코인데이터분석/README.md`

### 6.2 주요 스크립트 위치

#### 데이터 수집
- `step1_technical_indicators.py`
- `step2_traditional_markets.py`
- `step3_macro_data.py`
- `step4_sentiment_data.py`
- `step5b_integrate_all_new_data.py` (최종 통합)
- `step6_onchain_data.py`
- `step8_btc_etf_data.py`

#### 모델 학습
- `step24_all_regression_models.py` (10개 모델 비교)
- `step25_v2_next_day_prediction.py` (V2 예측)

#### 백테스팅
- `step26_elasticnet_backtesting_v2.py` (V2 백테스팅)
- `step28_direction_backtesting.py` (방향 예측)

#### 구조변화 분석
- `structural_change_tests_all_vars.py`
- `zscore_structural_change_analysis.py`

#### XAI 분석
- `elasticnet_xai_analysis_v2.py`

#### 변동성 분석
- `volatility_analysis/scripts/volatility_step2_garch_correlation.py`
- `volatility_analysis/scripts/volatility_step5_dcc_garch_hierarchical.py`

---

## 7. 추가 참고 자료

### 7.1 Bitcoin ETF 관련 뉴스 및 시장 보고서

#### SEC 승인 (2024-01-10)
- **제목**: "美증권위, 11개 비트코인 현물 ETF 상장 승인"
- **날짜**: 2024-01-11 송고
- **내용**: BlackRock, Fidelity, Ark, Grayscale 등 11개 ETF 승인
- **의미**: 비트코인이 전통 금융 시장에 공식 편입된 역사적 시점

#### Bloomberg: Bitcoin ETF 초기 자금 유입 (2024.02.09)
- **제목**: "Bitcoin ETF's First Month Saw $6 Billion in Net Inflows"
- **날짜**: 2024년 2월 9일
- **URL**: https://www.bloomberg.com/news/articles/2024-02-09/bitcoin-etf-s-first-month-saw-5-8-billion-in-net-inflows
- **내용**: 비트코인 현물 ETF 출시 첫 달 순유입액 60억 달러 기록
- **의미**: 역대 ETF 출시 중 가장 성공적인 초기 성과
- **Google 검색**: https://www.google.com/search?q=bloomberg+bitcoin+etf+first+month+6+billion

#### Reuters: BlackRock IBIT 역대 최단 기록 (2024.03.01)
- **제목**: "BlackRock's spot Bitcoin ETF crosses $10 billion in assets"
- **날짜**: 2024년 3월 1일
- **URL**: https://www.reuters.com/business/finance/blackrocks-spot-bitcoin-etf-crosses-10-billion-assets-2024-03-01/
- **내용**: BlackRock의 iShares Bitcoin Trust(IBIT)가 출시 7주 만에 운용자산 100억 달러 돌파
- **의미**: 역대 ETF 중 가장 빠른 속도로 100억 달러 달성 (기존 기록 5년 → 7주)
- **Google 검색**: https://www.google.com/search?q=reuters+blackrock+bitcoin+etf+10+billion

#### Bloomberg: Cathie Wood의 비트코인 전망 (2024.03.20)
- **제목**: "Cathie Wood Says Bitcoin Still Has Big Upside Potential"
- **날짜**: 2024년 3월 20일
- **URL**: https://www.bloomberg.com/news/articles/2024-03-20/cathie-wood-says-bitcoin-still-has-big-upside-potential
- **내용**: Ark Invest CEO Cathie Wood, 비트코인을 "디지털 금(Digital Gold)"으로 규정하며 장기 상승 전망
- **핵심 논점**:
  - 기관 투자자 유입으로 시장 성숙도 증가
  - 비트코인의 희소성과 분산화된 특성이 장기 가치 창출
  - ETF 승인이 접근성을 높여 대중화 가속
- **Google 검색**: https://www.google.com/search?q=bloomberg+cathie+wood+bitcoin+upside+potential+2024

#### BlackRock 소개
- **URL**: https://www.blackrock.com
- **운용 자산**: 약 12조 5,000억 달러 (약 1경 7,000조 원, 2024년 기준)
- **설명**: 세계 최대 자산운용사
- **CEO**: Larry Fink (비트코인 ETF 적극 지지자)
- **비트코인 ETF**: iShares Bitcoin Trust (IBIT)

#### Ark Invest 소개
- **URL**: https://ark-invest.com
- **CEO**: Cathie Wood (캐시 우드)
- **설명**: 혁신 기술 중심 투자사, 비트코인 장기 투자자
- **비트코인 ETF**: ARK 21Shares Bitcoin ETF (ARKB)
- **목표가**: 2030년까지 비트코인 $1,000,000 예측 (강세 시나리오)

---

### 7.2 암호화폐 정보 사이트

#### CoinMarketCap
- **URL**: https://coinmarketcap.com
- **내용**: 암호화폐 시가총액, 가격, 거래량

#### CoinGecko
- **URL**: https://www.coingecko.com
- **내용**: 암호화폐 정보, API 제공

#### Glassnode Academy
- **URL**: https://academy.glassnode.com
- **내용**: 온체인 지표 설명

#### Bitcoin Magazine
- **URL**: https://bitcoinmagazine.com
- **내용**: 비트코인 뉴스, 분석

---

### 7.3 금융 데이터 사이트

#### Investing.com
- **URL**: https://www.investing.com
- **내용**: 전통 시장 데이터, 뉴스

#### TradingView
- **URL**: https://www.tradingview.com
- **내용**: 차트, 기술적 분석

#### Bloomberg
- **URL**: https://www.bloomberg.com
- **내용**: 금융 뉴스, 데이터

---

## 8. 라이선스 & 크레딧

### 8.1 오픈소스 라이선스

**프로젝트 라이선스**:
- **코드**: MIT License
- **문서**: CC BY 4.0

**사용 라이브러리 라이선스**:
- pandas: BSD 3-Clause
- scikit-learn: BSD 3-Clause
- matplotlib: PSF License
- statsmodels: BSD 3-Clause

---

### 8.2 크레딧

**논문 참고**:
1. Demosthenous et al. (2025) - VLDB Workshop
2. Kehinde et al. (2025) - Journal of Big Data
3. Dubey & Enke (2025) - Machine Learning with Applications

**데이터 소스**:
- Yahoo Finance
- FRED (St. Louis Fed)
- Blockchain.com
- CoinMetrics
- Glassnode
- Alternative.me

**AI 지원**:
- Claude Code (Anthropic Sonnet 4.5)

---

## 9. 연락처 & 피드백

### 프로젝트 정보
- **작성자**: 송성원 (Song Hyowon)
- **프로젝트 기간**: 2025년 10월 - 11월 (약 4주)
- **분석 기간**: 2021-02-03 ~ 2025-10-14 (4.7년)

### GitHub Issues
- (프로젝트가 GitHub에 업로드되면 Issue 링크 추가)

---

## 10. 업데이트 로그

**2025-11-18**: 초기 문서 생성
- 47개 마크다운 파일에서 참고 링크 추출
- 5개 카테고리로 분류 (학술 논문, 데이터 소스, 프로젝트 문서, 외부 자료, 도구)
- 총 100+ 링크 정리

---

**문서 끝**

**총 링크 수**: 100+ 개
**총 참고 문서**: 47개 마크다운 파일
**프로젝트 규모**: 291개 파일 (Python 82개, CSV 106개, PNG 52개, MD 35개)
