"""
Bitcoin 데이터를 TAFAS 형식으로 변환하는 스크립트

목적: integrated_data_full_v2.csv (138개 변수)를 TAFAS가 요구하는 형식으로 변환

TAFAS 요구 형식:
- CSV 파일: data/{dataset_name}/{dataset_name}.csv
- 컬럼 구성: date, feature_1, feature_2, ..., feature_N
- 첫 번째 컬럼은 반드시 'date'
- 나머지 컬럼은 모두 숫자형 feature
"""

import pandas as pd
import numpy as np
from pathlib import Path


def convert_bitcoin_to_tafas(
    input_file='integrated_data_full_v2.csv',
    output_dir='TAFAS/data/bitcoin',
    output_file='bitcoin.csv'
):
    """
    Bitcoin 데이터를 TAFAS 형식으로 변환

    Parameters:
    - input_file: 원본 데이터 파일 (integrated_data_full_v2.csv)
    - output_dir: 출력 디렉토리 (TAFAS/data/bitcoin/)
    - output_file: 출력 파일명 (bitcoin.csv)
    """

    print("=" * 80)
    print("Bitcoin 데이터를 TAFAS 형식으로 변환 시작")
    print("=" * 80)

    # 1. 원본 데이터 로드
    print(f"\n[1/6] 원본 데이터 로드 중: {input_file}")
    df = pd.read_csv(input_file, index_col=0, parse_dates=True)
    print(f"  - 원본 데이터 shape: {df.shape}")
    print(f"  - 날짜 범위: {df.index.min()} ~ {df.index.max()}")
    print(f"  - 총 변수 개수: {len(df.columns)}")

    # 2. Data Leakage 방지를 위한 변수 제외
    print(f"\n[2/6] Data Leakage 변수 제외 중...")

    # 제외할 변수 목록
    exclude_cols = []

    # (1) 당일 가격 정보 (Close, High, Low, Open)
    price_cols = ['Close', 'High', 'Low', 'Open']
    for col in price_cols:
        if col in df.columns:
            exclude_cols.append(col)

    # (2) Close 기반 EMA/SMA (당일 종가 포함)
    for col in df.columns:
        if 'close' in col.lower() and ('ema' in col.lower() or 'sma' in col.lower()):
            exclude_cols.append(col)

    # (3) Bollinger Bands (당일 종가 포함)
    bb_cols = [col for col in df.columns if col.startswith('BB_')]
    exclude_cols.extend(bb_cols)

    # (4) bc_market_price, bc_market_cap (Close와 동일한 정보)
    for col in ['bc_market_price', 'bc_market_cap']:
        if col in df.columns:
            exclude_cols.append(col)

    # 중복 제거
    exclude_cols = list(set(exclude_cols))

    print(f"  - 제외할 변수 {len(exclude_cols)}개:")
    for col in sorted(exclude_cols):
        print(f"    - {col}")

    # 변수 제외
    df_clean = df.drop(columns=exclude_cols, errors='ignore')
    print(f"  - 제외 후 변수 개수: {len(df_clean.columns)}")

    # 3. TAFAS 형식으로 변환
    print(f"\n[3/6] TAFAS 형식으로 변환 중...")

    # date 컬럼 생성 (인덱스를 컬럼으로)
    df_tafas = df_clean.copy()
    df_tafas.insert(0, 'date', df_tafas.index.strftime('%Y-%m-%d'))
    df_tafas = df_tafas.reset_index(drop=True)

    print(f"  - TAFAS 형식 shape: {df_tafas.shape}")
    print(f"  - 첫 컬럼: {df_tafas.columns[0]} (date 필수)")
    print(f"  - 변수 개수: {len(df_tafas.columns) - 1}")

    # 4. 결측치 처리
    print(f"\n[4/6] 결측치 처리 중...")

    # 결측치 확인
    missing_before = df_tafas.isnull().sum().sum()
    print(f"  - 처리 전 결측치: {missing_before}개")

    # Forward fill → Backward fill
    df_tafas = df_tafas.fillna(method='ffill').fillna(method='bfill')

    missing_after = df_tafas.isnull().sum().sum()
    print(f"  - 처리 후 결측치: {missing_after}개")

    # 여전히 결측치가 있으면 0으로 채우기
    if missing_after > 0:
        df_tafas = df_tafas.fillna(0)
        print(f"  - 남은 결측치를 0으로 채움")

    # 5. 출력 디렉토리 생성 및 저장
    print(f"\n[5/6] 데이터 저장 중...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    full_output_path = output_path / output_file
    df_tafas.to_csv(full_output_path, index=False)

    print(f"  - 저장 경로: {full_output_path}")
    print(f"  - 파일 크기: {full_output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # 6. 변환 결과 검증
    print(f"\n[6/6] 변환 결과 검증 중...")

    # 저장된 파일 다시 읽기
    df_verify = pd.read_csv(full_output_path)

    print(f"  - 검증 shape: {df_verify.shape}")
    print(f"  - 첫 컬럼: {df_verify.columns[0]} (date 필수)")
    print(f"  - date 형식 예시: {df_verify['date'].iloc[0]} ~ {df_verify['date'].iloc[-1]}")
    print(f"  - 변수 개수: {len(df_verify.columns) - 1}")

    # 통계 정보
    print(f"\n  - 변수 통계:")
    print(f"    - 최솟값: {df_verify.iloc[:, 1:].min().min():.6f}")
    print(f"    - 최댓값: {df_verify.iloc[:, 1:].max().max():.6f}")
    print(f"    - 평균: {df_verify.iloc[:, 1:].mean().mean():.6f}")
    print(f"    - 표준편차: {df_verify.iloc[:, 1:].std().mean():.6f}")

    # 샘플 데이터 출력
    print(f"\n  - 샘플 데이터 (처음 5행, 처음 5개 변수):")
    print(df_verify.iloc[:5, :6].to_string(index=False))

    print("\n" + "=" * 80)
    print("변환 완료!")
    print("=" * 80)

    # 변환 요약
    summary = {
        'input_file': input_file,
        'output_file': str(full_output_path),
        'original_variables': len(df.columns),
        'excluded_variables': len(exclude_cols),
        'final_variables': len(df_verify.columns) - 1,
        'n_samples': len(df_verify),
        'date_range': f"{df_verify['date'].iloc[0]} ~ {df_verify['date'].iloc[-1]}",
    }

    return summary


if __name__ == '__main__':
    summary = convert_bitcoin_to_tafas()

    print("\n" + "=" * 80)
    print("변환 요약")
    print("=" * 80)
    for key, value in summary.items():
        print(f"  - {key}: {value}")

    print("\n다음 단계:")
    print("  1. TAFAS/datasets/build.py에 Bitcoin 데이터셋 클래스 추가")
    print("  2. TAFAS 설정 파일 생성 (config)")
    print("  3. 사전 학습 실행 (ETF 이전 데이터)")
    print("  4. Test-time Adaptation (ETF 이후 데이터)")
