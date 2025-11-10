# Bitcoin ETF 변동성 동시분석

## 📁 폴더 구조

```
volatility_analysis/
├── scripts/              # Python 분석 스크립트
├── results/
│   ├── csv/             # 분석 결과 CSV 파일
│   └── images/          # 시각화 PNG 파일
└── reports/             # 종합 보고서
```

## 📊 분석 방법론

1. **GARCH(1,1)** - 조건부 변동성 추정
2. **VAR & Spillover Index** - 변동성 전이 분석
3. **Granger Causality** - 인과관계 테스트
4. **DCC-GARCH** - 동적 조건부 상관관계
5. **Markov Switching** - 레짐 분석

## 🔍 주요 발견

- BTC → GOLD 인과관계 출현 (획기적!)
- 금리 관계 역전 (음→양)
- VIX가 BTC 최강 선행지표
- 저변동성 레짐 67.5%로 증가
- 시스템 Spillover 37% 증가

## 📖 보고서

- `reports/변동성_동시분석_종합보고서.md` - 종합 분석 보고서
- `reports/변동성분석_결과해석가이드.md` - 결과 해석 가이드

## 📅 분석 기간

- 전체: 2021-02-04 ~ 2025-10-14
- ETF 이전: 2021-02-04 ~ 2024-01-09
- ETF 이후: 2024-01-10 ~ 2025-10-14
