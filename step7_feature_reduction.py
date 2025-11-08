import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Phase 2: Feature Reduction Algorithm (FRA)")
print("=" * 70)

# ===== 데이터 로드 =====
print("\n1. 데이터 로드 중...")
print("-" * 70)

df = pd.read_csv('integrated_data_full.csv', index_col=0, parse_dates=True)
print(f"✓ 데이터 로드: {df.shape}")
print(f"  - 기간: {df.index[0].date()} ~ {df.index[-1].date()}")
print(f"  - 특성: {len(df.columns)}개")

# ===== 타겟 변수 설정 =====
print("\n2. 타겟 변수 설정")
print("-" * 70)

# 타겟: Close price (다음 날 종가)
target_col = 'Close'

# 다음 날 종가 예측을 위해 shift
df['target'] = df[target_col].shift(-1)

# 마지막 행(target이 NaN) 제거
df = df.dropna(subset=['target'])

print(f"타겟: {target_col} (다음 날 종가)")
print(f"데이터: {df.shape}")

# ===== 특성과 타겟 분리 =====
# 데이터 누수를 일으키는 특성 제거
exclude_cols = [
    'Close',              # 타겟과 동일
    'target',             # 미래 정보
    'cumulative_return',  # Close로부터 직접 계산 (심각한 데이터 누수)
    'High', 'Low', 'Open',  # 같은 날의 가격 정보 (타겟과 너무 강한 상관관계)
    'bc_market_price',    # Close와 거의 동일한 온체인 가격
    'bc_market_cap',      # Close * Supply로 계산되어 누수 가능
]

print("\n제거할 특성 (데이터 누수 방지):")
for col in exclude_cols:
    if col in df.columns:
        print(f"  - {col}")

feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols].copy()
y = df['target'].copy()

print(f"\n최종 특성: {len(feature_cols)}개")
print(f"타겟: {len(y)}개 샘플")

# ===== 데이터 분할 =====
print("\n3. 데이터 분할")
print("-" * 70)

# 시계열이므로 순서 유지하며 분할
split_idx = int(len(X) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"훈련: {X_train.shape}")
print(f"테스트: {X_test.shape}")

# ===== 데이터 스케일링 =====
print("\n4. 데이터 스케일링")
print("-" * 70)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

print("✓ StandardScaler 적용 완료")

# ===== Method 1: Pearson Correlation =====
print("\n" + "=" * 70)
print("Method 1: Pearson Correlation")
print("=" * 70)

pearson_scores = {}

for col in feature_cols:
    try:
        # NaN 값 제거 후 상관계수 계산
        valid_idx = ~(X_train[col].isna() | y_train.isna())
        if valid_idx.sum() > 0:
            corr, pval = pearsonr(X_train[col][valid_idx], y_train[valid_idx])
            pearson_scores[col] = abs(corr)  # 절대값 사용
        else:
            pearson_scores[col] = 0
    except Exception as e:
        pearson_scores[col] = 0

pearson_df = pd.DataFrame.from_dict(pearson_scores, orient='index', columns=['pearson_corr'])
pearson_df = pearson_df.sort_values('pearson_corr', ascending=False)

print(f"\n상위 20개 특성:")
for i, (col, score) in enumerate(pearson_df.head(20).iterrows(), 1):
    print(f"  {i:2}. {col:40} : {score['pearson_corr']:.4f}")

# ===== Method 2: Mean Decrease in Impurity (MDI) =====
print("\n" + "=" * 70)
print("Method 2: Mean Decrease in Impurity (MDI)")
print("=" * 70)

print("Random Forest 모델 훈련 중...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,           # 더 얕게 (과적합 방지)
    min_samples_split=20,  # 분할 최소 샘플 수
    min_samples_leaf=10,   # 리프 노드 최소 샘플 수
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)

# Feature importance 추출
mdi_scores = pd.DataFrame({
    'feature': feature_cols,
    'mdi_importance': rf_model.feature_importances_
}).sort_values('mdi_importance', ascending=False)

print(f"\n✓ Random Forest 훈련 완료")
print(f"훈련 R²: {rf_model.score(X_train_scaled, y_train):.4f}")
print(f"테스트 R²: {rf_model.score(X_test_scaled, y_test):.4f}")

print(f"\n상위 20개 특성:")
for i, row in mdi_scores.head(20).iterrows():
    print(f"  {i+1:2}. {row['feature']:40} : {row['mdi_importance']:.4f}")

# ===== Method 3: Permutation Feature Importance (PFI) =====
print("\n" + "=" * 70)
print("Method 3: Permutation Feature Importance (PFI)")
print("=" * 70)

print("PFI 계산 중 (시간이 걸릴 수 있습니다)...")
pfi_result = permutation_importance(
    rf_model,
    X_test_scaled,
    y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

pfi_scores = pd.DataFrame({
    'feature': feature_cols,
    'pfi_importance': pfi_result.importances_mean
}).sort_values('pfi_importance', ascending=False)

print(f"\n✓ PFI 계산 완료")

print(f"\n상위 20개 특성:")
for i, row in pfi_scores.head(20).iterrows():
    print(f"  {i+1:2}. {row['feature']:40} : {row['pfi_importance']:.4f}")

# ===== Method 4: SHAP Values =====
print("\n" + "=" * 70)
print("Method 4: SHAP Values")
print("=" * 70)

print("SHAP 계산 중 (시간이 많이 걸릴 수 있습니다)...")

# 샘플 줄이기 (SHAP은 계산이 매우 오래 걸림)
sample_size = min(100, len(X_train_scaled))
X_train_sample = X_train_scaled.sample(n=sample_size, random_state=42)

# TreeExplainer 사용 (Random Forest에 최적화)
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_train_sample)

# SHAP 절대값 평균
shap_importance = np.abs(shap_values).mean(axis=0)

shap_scores = pd.DataFrame({
    'feature': feature_cols,
    'shap_importance': shap_importance
}).sort_values('shap_importance', ascending=False)

print(f"\n✓ SHAP 계산 완료 (샘플 {sample_size}개)")

print(f"\n상위 20개 특성:")
for i, row in shap_scores.head(20).iterrows():
    print(f"  {i+1:2}. {row['feature']:40} : {row['shap_importance']:.4f}")

# ===== FRA: 종합 순위 계산 =====
print("\n" + "=" * 70)
print("FRA: Feature Reduction Algorithm - 종합 순위")
print("=" * 70)

# 각 방법별 순위 계산 (1위가 가장 중요)
pearson_rank = pearson_df.reset_index().reset_index()
pearson_rank.columns = ['rank', 'feature', 'score']
pearson_rank['pearson_rank'] = pearson_rank['rank'] + 1

mdi_rank = mdi_scores.reset_index(drop=True).reset_index()
mdi_rank.columns = ['rank', 'feature', 'score']
mdi_rank['mdi_rank'] = mdi_rank['rank'] + 1

pfi_rank = pfi_scores.reset_index(drop=True).reset_index()
pfi_rank.columns = ['rank', 'feature', 'score']
pfi_rank['pfi_rank'] = pfi_rank['rank'] + 1

shap_rank = shap_scores.reset_index(drop=True).reset_index()
shap_rank.columns = ['rank', 'feature', 'score']
shap_rank['shap_rank'] = shap_rank['rank'] + 1

# 모든 순위 병합
final_ranking = pearson_rank[['feature', 'pearson_rank']].copy()
final_ranking = final_ranking.merge(mdi_rank[['feature', 'mdi_rank']], on='feature')
final_ranking = final_ranking.merge(pfi_rank[['feature', 'pfi_rank']], on='feature')
final_ranking = final_ranking.merge(shap_rank[['feature', 'shap_rank']], on='feature')

# 평균 순위 계산 (낮을수록 중요)
final_ranking['avg_rank'] = final_ranking[['pearson_rank', 'mdi_rank', 'pfi_rank', 'shap_rank']].mean(axis=1)
final_ranking = final_ranking.sort_values('avg_rank')

print(f"\n최종 순위 (상위 30개):")
print("-" * 70)
for i, row in final_ranking.head(30).iterrows():
    print(f"{row['avg_rank']:5.1f}위 | {row['feature']:35} | P:{row['pearson_rank']:3.0f} M:{row['mdi_rank']:3.0f} PFI:{row['pfi_rank']:3.0f} S:{row['shap_rank']:3.0f}")

# ===== 특성 선택 =====
print("\n" + "=" * 70)
print("특성 선택")
print("=" * 70)

# 상위 N개 특성 선택 (논문에 따라 조정 가능)
top_n_features = [10, 20, 30, 40, 50]

for n in top_n_features:
    selected = final_ranking.head(n)['feature'].tolist()
    print(f"\n상위 {n}개 특성 선택 완료")

# 결과 저장
print("\n6. 결과 저장")
print("-" * 70)

# 전체 순위 저장
final_ranking.to_csv('feature_ranking_fra.csv', index=False)
print("✓ feature_ranking_fra.csv")

# 각 방법별 점수 저장
all_scores = pearson_df.reset_index()
all_scores.columns = ['feature', 'pearson_corr']
all_scores = all_scores.merge(mdi_scores[['feature', 'mdi_importance']], on='feature')
all_scores = all_scores.merge(pfi_scores[['feature', 'pfi_importance']], on='feature')
all_scores = all_scores.merge(shap_scores[['feature', 'shap_importance']], on='feature')

all_scores.to_csv('feature_scores_all_methods.csv', index=False)
print("✓ feature_scores_all_methods.csv")

# 상위 특성 목록 저장
for n in top_n_features:
    selected = final_ranking.head(n)['feature'].tolist()
    with open(f'selected_features_top{n}.txt', 'w') as f:
        f.write(f"Top {n} Features (FRA)\n")
        f.write("=" * 70 + "\n\n")
        for i, feat in enumerate(selected, 1):
            f.write(f"{i:2}. {feat}\n")
    print(f"✓ selected_features_top{n}.txt")

print("\n" + "=" * 70)
print("Phase 2 완료!")
print("=" * 70)
print(f"\n✅ Feature Reduction Algorithm 완료!")
print(f"   - 총 {len(feature_cols)}개 특성 분석")
print(f"   - 4가지 방법 (Pearson, MDI, PFI, SHAP) 종합")
print(f"   - 상위 특성 목록 생성 (10, 20, 30, 40, 50개)")
print(f"\n다음 단계: Phase 3 - 모델 훈련 및 평가")
print("=" * 70)
