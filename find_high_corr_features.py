
import pandas as pd

# 상관관계 행렬 파일 읽기
try:
    corr_matrix = pd.read_csv('correlation_matrix.csv', index_col=0)
except FileNotFoundError:
    print("오류: correlation_matrix.csv 파일을 찾을 수 없습니다.")
    print("먼저 상관관계 분석을 실행해야 합니다.")
    exit()

# 'Close' 컬럼과 상관계수가 0.99 이상인 변수 찾기
high_corr_cols = corr_matrix[corr_matrix['Close'] >= 0.99].index.tolist()

# 'Close' 자신은 제외
if 'Close' in high_corr_cols:
    high_corr_cols.remove('Close')

print("'Close'와 상관계수가 0.99 이상인 변수 목록:")
if high_corr_cols:
    for col in high_corr_cols:
        print(f"- {col} (상관계수: {corr_matrix.loc[col, 'Close']:.4f})")
else:
    print("해당하는 변수를 찾지 못했습니다.")
