
import pandas as pd

print("상관관계 분석을 시작합니다...")

# 데이터 로드
try:
    df = pd.read_csv('integrated_data_full.csv')
except FileNotFoundError:
    print("오류: integrated_data_full.csv 파일을 찾을 수 없습니다.")
    exit()

# 숫자형 데이터만 선택
numeric_df = df.select_dtypes(include=['number'])

# 상관관계 행렬 계산
correlation_matrix = numeric_df.corr()

# 결과를 CSV 파일로 저장
output_filename = 'correlation_matrix.csv'
correlation_matrix.to_csv(output_filename)

print(f"모든 변수 간의 상관관계 행렬을 '{output_filename}' 파일로 성공적으로 저장했습니다.")
