# PPT 레퍼런스 페이지

**발표 제목**: Bitcoin ETF 유인원에서 지식인까지
**발표일**: 2025.11.21
**팀**: 송성원(팀장), 김규민, 손서연, 하동욱, 찐

---

## 📄 슬라이드 구성 (3-4장 권장)

---

## **슬라이드 1: 참고문헌 - 학술 논문**

### 📚 선행 연구 (비트코인 가격 예측)

**행동경제학 이론적 토대**
- **Kahneman & Tversky (1979)** - "Prospect Theory: An Analysis of Decision under Risk"
  *Econometrica*, vol. 47, no. 2, pp. 263-291
  ★ **2002 노벨 경제학상** 수상 (Kahneman)
  → 손실 회피 편향, 투자자 심리 분석의 이론적 근거

**초기 연구 (2013-2019): 다양한 데이터 소스 탐색**
- **Kristoufek (2013)** - Google Trends, Wikipedia 조회수로 가격 예측
  *Scientific Reports* 3, Article 3415

- **Jang & Lee (2017)** - 블록체인 정보 + 거시경제 지표 (S&P 500, DOW30)
  *IEEE Access*, vol. 6, pp. 5427-5437

- **Abraham et al. (2018)** - Google Trends + 트윗량 활용
  *SMU Data Science Review*, vol. 1, no. 3

- **Valencia et al. (2019)** - 감정(Sentiment) 분석 중심
  *Entropy* 21(6), 589

- **Saad et al. (2019)** - 채굴자 수익, 해시율 등 온체인 데이터
  *IEEE Systems Journal*, vol. 14, no. 1, pp. 321-332

- **Mallqui & Fernandes (2019)** - **다차원 데이터 통합** (블록체인 + 거시경제 + 검색량)
  *Applied Soft Computing*, vol. 75, pp. 596-606

**💡 선행 연구의 공통 결론**: 다양한 데이터 소스를 통합할수록 예측력 향상

---

### 📖 본 연구 참고 논문 (2025년 최신)

1. **Demosthenous et al. (2025)** - "From On-chain to Macro: Data Source Diversity"
   *VLDB 2024 Workshop* | arXiv:2506.21246v1
   → 5개 데이터 카테고리 통합 (429개→100개 변수)

2. **Kehinde et al. (2025)** - "Helformer: Attention-based Deep Learning"
   *Journal of Big Data* 12:81
   → Holt-Winters + Transformer 결합

3. **Dubey & Enke (2025)** - "Bitcoin Price Direction using On-chain Data"
   *Machine Learning with Applications* 20:100674
   → 온체인 데이터 5개 카테고리, Boruta Feature Selection

---

## **슬라이드 2: 참고문헌 - 시장 보고서 & 뉴스**

### 📰 Bitcoin ETF 승인 및 시장 변화 (2024년)

**SEC 승인 (2024-01-10)**
- 미국 증권거래위원회, 11개 비트코인 현물 ETF 동시 승인
  BlackRock (IBIT), Fidelity (FBTC), Ark Invest (ARKB), Grayscale (GBTC) 등
- **역사적 의미**: 비트코인이 전통 금융 시장에 공식 편입된 시점

**시장 반응: 전례없는 자금 유입**

1. **Bloomberg (2024.02.09)** - "Bitcoin ETF's First Month Saw $6 Billion in Net Inflows"
   → 출시 첫 달 순유입 **$6 Billion** (역대 ETF 중 최고 기록)

2. **Reuters (2024.03.01)** - "BlackRock's Bitcoin ETF crosses $10 billion in assets"
   → BlackRock IBIT, **7주 만에 $10B 돌파** (기존 기록: 5년 → 7주로 단축)

3. **Bloomberg (2024.03.20)** - "Cathie Wood: Bitcoin Has Big Upside Potential"
   → Ark Invest CEO, 비트코인을 **"디지털 금(Digital Gold)"**으로 규정
   → 2030년 목표가: $1,000,000 (강세 시나리오)

**주요 기관 소개**
- **BlackRock**: 세계 최대 자산운용사 ($12.5조 운용, 2024년 기준)
- **Ark Invest**: 혁신 기술 투자 전문, Cathie Wood CEO

---

## **슬라이드 3: 데이터 소스 & 방법론**

### 📊 데이터 수집 (5개 플랫폼, 138개 변수)

| 데이터 소스 | 수집 항목 | 신뢰도 근거 |
|------------|---------|-----------|
| **Yahoo Finance** | BTC/ETH/SOL 가격, S&P 500, 금, 국채 | 세계 표준 금융 데이터 |
| **FRED (연준)** | 금리, CPI, M2, Fed 유동성 (27개) | 미국 중앙은행 공식 통계 |
| **Blockchain.com** | 거래 건수, 해시레이트, 난이도 (31개) | 비트코인 공식 익스플로러 |
| **CoinMetrics** | MVRV, NVT, Puell Multiple | 기관 투자자용 온체인 분석 |
| **Glassnode** | 공급 분포, 주소 밸런스, Hash Ribbon | 기관급 온체인 데이터 |

**수집 기간**: 2021년 2월 ~ 2025년 10월 (1,715일)
**분석 기준**: 2024-01-10 (ETF 승인일) 전후 2년

---

### 🔬 통계 방법론

**구조변화 검정 (5가지)**

1. **Chow Test** - Chow (1960)
   → 특정 시점(ETF 승인일) 전후 회귀계수 변화 검증

2. **Quandt-Andrews Test** - Andrews (1993)
   *"Tests for Parameter Instability with Unknown Change Point"*, Econometrica
   → 데이터가 스스로 변화 시점 탐지

3. **CUSUM Test** - Brown, Durbin & Evans (1975)
   → 누적합 검정, 점진적 구조 변화 감지

4. **Bai-Perron Test** - Bai & Perron (1998, 2003)
   → 다중 구조변화점 동시 검정

5. **QLR Test** - Quandt Likelihood Ratio
   → 최대 우도비 검정

**표준오차 보정**: Newey-West (1987) HAC (Heteroskedasticity and Autocorrelation Consistent)

---

### 🤖 머신러닝 모델

**ElasticNet** - Zou & Hastie (2005)
*"Regularization and variable selection via the elastic net"*
Journal of the Royal Statistical Society, Series B
- Lasso (변수 선택) + Ridge (다중공선성 해결)
- 138개 변수 → 핵심 변수 자동 선택
- 외삽(Extrapolation) 문제 해결 (선형 함수)

**비교 모델**: Random Forest, XGBoost, Gradient Boosting
**XAI (설명가능 AI)**: SHAP (Lundberg & Lee, 2017)

---

### 💹 백테스팅 검증

**방법론**
- 실시간 롤링 윈도우: 730일 학습 → 1일 예측
- 거래비용 0.1% 반영 (현실적 조건)
- statsmodels, scipy 활용

**성과 측정**
- 연평균 수익률, 샤프비율, 최대낙폭(MDD)
- 5가지 전략 비교 (Buy-and-Hold, Long-Only, Long-Short, Threshold 1%, 2%)

---

## **슬라이드 4: 프로젝트 규모 & 전체 레퍼런스**

### 📈 연구 범위 및 깊이

```
📚 참고 학술논문        10편 (1979-2025)
📰 시장 보고서          Bloomberg, Reuters (2024)
📊 데이터 플랫폼        5개 (Yahoo, FRED, Blockchain 등)
🔬 통계 검정           5가지 (Chow, QA, CUSUM, BP, QLR)
🤖 머신러닝 모델        10개 비교 (ElasticNet 최종 선정)
💹 백테스팅 전략        6가지
📝 작성 문서           45+ 개 (800+ 페이지)
💾 Python 스크립트     30+ 개
📅 분석 기간           4.7년 (1,715일)
🔢 최종 변수           138개
📊 샘플 수            1,715개
```

---

### 🔗 전체 레퍼런스 (100+ 링크)

**상세 레퍼런스 문서 (100개 이상 링크 포함)**

```
[QR 코드]
↓
전체 레퍼런스 목록
• 학술 논문 10편 (DOI 포함)
• 시장 보고서 & 뉴스 (URL 포함)
• 데이터 소스 5개 (API 문서)
• 방법론 참고자료 15+
• Python 라이브러리 문서
```

**파일**: `ALL_REFERENCE_LINKS.md` (GitHub/구글 드라이브 링크)

---

### 📧 문의 & 피드백

**팀 구성**
- **팀장**: 송성원
- **팀원**: 김규민, 손서연, 하동욱, 찐

**연락처**: [이메일 추가]
**GitHub**: [저장소 링크 추가]
**전체 보고서**: [QR 코드 또는 링크]

---

### 🙏 감사의 말

본 연구는 다음 자료들을 참고하여 수행되었습니다:
- 10편의 학술 논문 (1979-2025)
- Bloomberg, Reuters 시장 보고서
- 5개 데이터 제공 기관 (Yahoo Finance, FRED, Blockchain.com, CoinMetrics, Glassnode)
- 15개 이상의 Python 오픈소스 라이브러리

**특별히 감사드립니다**:
- statsmodels, scikit-learn, pandas 개발팀
- 데이터를 무료로 제공해주신 기관들
- 선행 연구자들의 학술적 기여

---

## 📌 슬라이드 디자인 팁

### **슬라이드 1: 학술 논문**
- 2단 레이아웃 (선행 연구 | 본 연구 참고)
- 노벨상 아이콘(★) 강조
- 연도 타임라인 형식 가능

### **슬라이드 2: 시장 보고서**
- 큰 숫자 강조 ($6B, 7주, $10B)
- Bloomberg/Reuters 로고 삽입
- BlackRock 규모 ($12.5T) 강조

### **슬라이드 3: 데이터 & 방법론**
- 표 형식 (깔끔하게)
- 5가지 검정 → 5개 박스로 시각화
- 데이터 소스 로고 삽입 가능

### **슬라이드 4: 프로젝트 규모**
- 인포그래픽 형식 (숫자 크게)
- QR 코드 중앙 배치
- 감사의 말 (신뢰도 향상)

---

## 📝 QR 코드 생성 방법

1. `ALL_REFERENCE_LINKS.md` 파일을 GitHub에 업로드
2. 파일 URL 복사
3. https://www.qr-code-generator.com/ 접속
4. URL 입력 → QR 코드 생성
5. PPT에 삽입

**또는**
- Google Drive에 업로드 → 공유 링크 생성 → QR 코드화

---

## 🎯 핵심 메시지

### **청중에게 전달할 것**

1. **학술적 깊이** → 10편 논문 참고, 5가지 통계 검정
2. **시장 영향력** → $6B 유입, 7주 $10B (역대 최단)
3. **데이터 신뢰도** → 5개 기관급 플랫폼
4. **연구 노력** → 138개 변수, 1,715일, 45+ 문서
5. **실용성** → 백테스팅 188% 수익률

### **한 문장 요약**
> "10편의 학술 논문과 5개 데이터 플랫폼을 기반으로,
> ETF 승인이 비트코인 시장을 투기에서 투자 자산으로 전환시켰음을
> 5가지 통계 검정과 실전 백테스팅으로 입증한 연구"

---

**문서 작성일**: 2025-11-18
**최종 수정**: 2025-11-18
**버전**: 1.0
