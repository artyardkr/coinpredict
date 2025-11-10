"""
변동성 동시분석 (Volatility Co-Movement Analysis)
ETF 승인 전후 비트코인과 주요 자산 간 변동성 상호작용 분석

작성일: 2025-11-10
목적:
1. 변동성 연동성 변화 분석
2. 변동성 전이(spillover) 패턴 탐지
3. 동적 상관관계 추정
4. 인과관계 구조 변화 파악
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("변동성 동시분석 (Volatility Co-Movement Analysis)")
print("=" * 80)

# ============================================================================
# Step 1: 데이터 로드 및 준비
# ============================================================================

print("\n[Step 1] 데이터 로드 및 수익률 계산")
print("-" * 80)

# 데이터 로드
df = pd.read_csv('integrated_data_full_v2.csv', index_col='Date', parse_dates=True)
print(f"원본 데이터: {df.shape}")

# ETF 승인일
ETF_DATE = '2024-01-10'
etf_date = pd.to_datetime(ETF_DATE)

# 분석 대상 변수 선택
ASSET_GROUPS = {
    '비트코인': ['Close'],
    '주식시장': ['SPX', 'QQQ', 'DIA', 'IWM'],
    '변동성지수': ['VIX', 'VIXCLS'],
    '채권': ['TLT', 'LQD', 'HYG'],
    '금리': ['DFF', 'T10Y2Y'],
    '대체자산': ['GOLD', 'GLD', 'SILVER', 'DXY'],
    '원자재': ['OIL'],
    '비트코인온체인': ['bc_hash_rate', 'bc_difficulty', 'Volume', 'OBV']
}

# 전체 분석 변수 리스트
all_vars = []
for group, vars_list in ASSET_GROUPS.items():
    all_vars.extend(vars_list)

# 사용 가능한 변수만 선택
available_vars = [v for v in all_vars if v in df.columns]
df_selected = df[available_vars].copy()

print(f"\n분석 대상 변수: {len(available_vars)}개")
for group, vars_list in ASSET_GROUPS.items():
    available = [v for v in vars_list if v in df.columns]
    print(f"  {group}: {available}")

# 결측치 처리
print(f"\n결측치 처리 전: {df_selected.isnull().sum().sum()}개")
df_selected = df_selected.fillna(method='ffill').fillna(method='bfill')
print(f"결측치 처리 후: {df_selected.isnull().sum().sum()}개")

# ============================================================================
# Step 1.2: 수익률 계산
# ============================================================================

print("\n[Step 1.2] 수익률 계산")
print("-" * 80)

# 로그 수익률 계산
returns = pd.DataFrame(index=df_selected.index)

for col in df_selected.columns:
    # 가격 시계열은 로그 수익률
    if df_selected[col].min() > 0:  # 양수 확인
        returns[col] = np.log(df_selected[col] / df_selected[col].shift(1))
    else:
        # 음수 포함 시계열은 단순 차분
        returns[col] = df_selected[col].diff()

# 첫 행(NaN) 제거
returns = returns.dropna()

print(f"수익률 데이터: {returns.shape}")
print(f"기간: {returns.index[0]} ~ {returns.index[-1]}")

# 기초 통계량
print("\n수익률 기초 통계:")
print(returns[['Close', 'SPX', 'VIX', 'GOLD']].describe())

# ETF 전후 분할
returns_pre = returns[returns.index < etf_date]
returns_post = returns[returns.index >= etf_date]

print(f"\nETF 이전: {len(returns_pre)}일 ({returns_pre.index[0]} ~ {returns_pre.index[-1]})")
print(f"ETF 이후: {len(returns_post)}일 ({returns_post.index[0]} ~ {returns_post.index[-1]})")

# ============================================================================
# Step 1.3: 기초 변동성 계산 (Realized Volatility)
# ============================================================================

print("\n[Step 1.3] 실현 변동성(Realized Volatility) 계산")
print("-" * 80)

def calculate_realized_volatility(returns_df, window=20):
    """
    실현 변동성 계산 (이동창 표준편차)

    Parameters:
    -----------
    returns_df : DataFrame
        수익률 데이터
    window : int
        이동창 크기 (기본값: 20일)

    Returns:
    --------
    volatility : DataFrame
        실현 변동성 (연율화)
    """
    # 이동창 표준편차 계산
    rolling_std = returns_df.rolling(window=window).std()

    # 연율화 (√252)
    volatility = rolling_std * np.sqrt(252)

    return volatility

# 20일 실현 변동성 계산
RV_20 = calculate_realized_volatility(returns, window=20)
RV_20 = RV_20.dropna()

print(f"실현 변동성 (20일): {RV_20.shape}")

# 주요 자산 변동성 통계
print("\n주요 자산 변동성 (연율화):")
for col in ['Close', 'SPX', 'VIX', 'GOLD']:
    if col in RV_20.columns:
        print(f"  {col:15s}: 평균={RV_20[col].mean():.4f}, 표준편차={RV_20[col].std():.4f}")

# ETF 전후 변동성 비교
print("\nETF 전후 변동성 비교:")
rv_pre = RV_20[RV_20.index < etf_date]
rv_post = RV_20[RV_20.index >= etf_date]

for col in ['Close', 'SPX', 'VIX', 'GOLD']:
    if col in RV_20.columns:
        mean_pre = rv_pre[col].mean()
        mean_post = rv_post[col].mean()
        change_pct = ((mean_post - mean_pre) / mean_pre) * 100
        print(f"  {col:15s}: {mean_pre:.4f} → {mean_post:.4f} ({change_pct:+.1f}%)")

# 결과 저장
print("\n데이터 저장 중...")
returns.to_csv('returns_data.csv')
RV_20.to_csv('realized_volatility_20d.csv')
print("저장 완료!")

print("\n" + "=" * 80)
print("Step 1 완료: 데이터 준비 및 기초 변동성 계산")
print("=" * 80)
print("\n생성된 파일:")
print("  - returns_data.csv")
print("  - realized_volatility_20d.csv")
