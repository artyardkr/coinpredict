import pandas as pd
import statsmodels.api as sm
import numpy as np

def backward_elimination(X, y, significance_level=0.05):
    """
    후진 제거 방식(Backward Elimination)을 사용한 단계별 회귀분석 함수

    Args:
    X (pd.DataFrame): 예측 변수(독립 변수) 데이터프레임
    y (pd.Series): 목표 변수(종속 변수) 시리즈
    significance_level (float): 변수를 모델에 유지하기 위한 p-value 임계값 (예: 0.05)
    
    Returns:
    sm.regression.linear_model.RegressionResultsWrapper: 최종 모델의 OLS 요약
    """
    
    # 모든 변수 목록으로 시작
    features = list(X.columns)
    
    while len(features) > 0:
        # 모델에 상수항(intercept) 추가
        X_with_const = sm.add_constant(X[features])
        
        # OLS 모델 피팅
        try:
            model = sm.OLS(y, X_with_const, missing='drop').fit()
        except Exception as e:
            # 심각한 다중공선성 등으로 모델 피팅 실패 시
            print(f"모델 피팅 오류 발생: {e}")
            # 오류 발생 시 마지막으로 성공한 모델을 반환하거나 None 반환 (여기서는 단순화)
            # 실제로는 VIF 등을 확인하며 더 정교하게 제거 로직을 구현할 수 있음
            
            # 오류를 유발한 변수(주로 가장 높은 p-value)를 제거하는 로직 시도
            if 'pvalues' in locals():
                feature_to_remove = pvalues.idxmax()
                if feature_to_remove in features:
                    features.remove(feature_to_remove)
                    continue
            else:
                 # p-value 계산 전 오류 시, 마지막 변수 제거 시도
                 if len(features) > 0:
                    features.pop()
                    continue
                 else:
                    return None
            return None

        # 상수항을 제외한 변수들의 p-value
        pvalues = model.pvalues.drop('const')
        
        # 가장 높은 p-value 찾기
        max_pvalue = pvalues.max()
        
        if max_pvalue > significance_level:
            # p-value가 임계값보다 높으면 해당 변수 제거
            feature_to_remove = pvalues.idxmax()
            features.remove(feature_to_remove)
            # print(f"변수 제거: {feature_to_remove} (p-value: {max_pvalue:.4f})")
        else:
            # 모든 변수의 p-value가 임계값보다 낮으면 반복 중지
            break
            
    # 최종적으로 선택된 변수들로 모델 다시 피팅
    print(f"최종 선택된 변수 개수: {len(features)}")
    X_final = sm.add_constant(X[features])
    final_model = sm.OLS(y, X_final, missing='drop').fit()
    
    return final_model

# --- 메인 코드 실행 ---
try:
    # 1. 데이터 로드
    df = pd.read_csv('integrated_data_full.csv')

    # 2. 목표 변수 (y) 생성: 다음 날의 종가
    # 'Close' 컬럼을 한 칸 위로 시프트
    y = df['Close'].shift(-1)

    # 3. 예측 변수 (X) 생성
    
    # 'market_cap_approx' 관련 컬럼 (인덱스 21 ~ 32)
    # 이 변수들은 'Close'와 완벽히 동일한 정보라(Close * 상수) 누수 및 다중공선성을 유발
    market_cap_cols = df.columns[21:33]
    
    # 제거할 컬럼 목록
    cols_to_drop = ['Date', 'Close'] + list(market_cap_cols)
    
    X = df.drop(columns=cols_to_drop)
    
    # 4. X와 y 정렬
    # y의 마지막 값은 NaN이 되므로, X와 y 모두 마지막 행을 제거
    X = X.iloc[:-1]
    y = y.iloc[:-1]
    
    print(f"총 {X.shape[1]}개의 변수로 단계별 회귀분석 시작...")

    # 5. 단계별 회귀분석 실행
    final_model_results = backward_elimination(X, y, significance_level=0.05)

    # 6. 결과 출력
    if final_model_results:
        print("\n--- 최종 모델 요약 (Final Model Summary) ---")
        print(final_model_results.summary())
    else:
        print("최종 모델을 생성하지 못했습니다.")

except FileNotFoundError:
    print("오류: 'integrated_data_full.csv' 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"알 수 없는 오류가 발생했습니다: {e}")