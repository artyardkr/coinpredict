import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Phase 3: 모델 훈련 및 평가")
print("=" * 70)

# ===== 1. 데이터 로드 =====
print("\n1. 데이터 로드")
print("-" * 70)

df = pd.read_csv('integrated_data_full.csv', index_col=0, parse_dates=True)
print(f"✓ 데이터: {df.shape}")

# 타겟 설정
df['target'] = df['Close'].shift(-1)
df = df.dropna(subset=['target'])

# 제거할 특성
exclude_cols = ['Close', 'target', 'cumulative_return', 'High', 'Low', 'Open',
                'bc_market_price', 'bc_market_cap']
all_features = [col for col in df.columns if col not in exclude_cols]

X = df[all_features].copy()
y = df['target'].copy()

print(f"전체 특성: {len(all_features)}개")
print(f"샘플: {len(X)}개")

# ===== 2. 데이터 분할 (시계열) =====
print("\n2. 데이터 분할")
print("-" * 70)

# ETF 승인일: 2024-01-10
etf_date = '2024-01-10'

# 전체 데이터를 80/20으로 분할
split_idx = int(len(X) * 0.8)

# ETF 전후 분할
etf_idx = df.index.searchsorted(etf_date)

print(f"전체 데이터: {len(X)}개")
print(f"훈련/테스트 분할 (80/20): {split_idx}개 / {len(X) - split_idx}개")
print(f"ETF 전후 분할 ({etf_date}): {etf_idx}개 / {len(X) - etf_idx}개")

# ===== 3. 특성 집합 로드 =====
print("\n3. 특성 집합 로드")
print("-" * 70)

feature_sets = {}

# 상위 N개 특성 로드
for n in [10, 20, 30, 40, 50]:
    with open(f'selected_features_top{n}.txt', 'r') as f:
        lines = f.readlines()
        features = [line.strip().split('. ')[1] for line in lines if line.strip() and '. ' in line]
        feature_sets[f'top{n}'] = features
        print(f"✓ Top {n}: {len(features)}개 특성")

# 전체 특성
feature_sets['all'] = all_features
print(f"✓ All: {len(all_features)}개 특성")

# ===== 4. 모델 훈련 함수 =====
def train_and_evaluate(X_train, X_test, y_train, y_test, model_name, model, scaler=None):
    """모델 훈련 및 평가"""

    # 스케일링
    if scaler:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    # 훈련
    model.fit(X_train_scaled, y_train)

    # 예측
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # 평가
    results = {
        'model': model_name,
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
    }

    return results, model

# ===== 5. 모델별, 특성 집합별 훈련 =====
print("\n" + "=" * 70)
print("5. 모델 훈련 및 평가")
print("=" * 70)

all_results = []

for feature_name, features in feature_sets.items():
    print(f"\n{'='*70}")
    print(f"특성 집합: {feature_name} ({len(features)}개 특성)")
    print(f"{'='*70}")

    # 해당 특성만 선택
    X_subset = X[features].copy()

    # 80/20 분할
    X_train = X_subset.iloc[:split_idx]
    X_test = X_subset.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    # ===== Random Forest =====
    print(f"\n[Random Forest]")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )

    scaler = StandardScaler()
    rf_results, rf_model = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        f'RF_{feature_name}', rf, scaler
    )
    rf_results['feature_set'] = feature_name
    rf_results['n_features'] = len(features)
    all_results.append(rf_results)

    print(f"  Train R²: {rf_results['train_r2']:.4f} | Test R²: {rf_results['test_r2']:.4f}")
    print(f"  Train RMSE: ${rf_results['train_rmse']:,.2f} | Test RMSE: ${rf_results['test_rmse']:,.2f}")
    print(f"  Train MAE: ${rf_results['train_mae']:,.2f} | Test MAE: ${rf_results['test_mae']:,.2f}")

    # ===== XGBoost =====
    print(f"\n[XGBoost]")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    scaler = StandardScaler()
    xgb_results, xgb_fitted = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        f'XGB_{feature_name}', xgb_model, scaler
    )
    xgb_results['feature_set'] = feature_name
    xgb_results['n_features'] = len(features)
    all_results.append(xgb_results)

    print(f"  Train R²: {xgb_results['train_r2']:.4f} | Test R²: {xgb_results['test_r2']:.4f}")
    print(f"  Train RMSE: ${xgb_results['train_rmse']:,.2f} | Test RMSE: ${xgb_results['test_rmse']:,.2f}")
    print(f"  Train MAE: ${xgb_results['train_mae']:,.2f} | Test MAE: ${xgb_results['test_mae']:,.2f}")

# ===== 6. ETF 전후 성능 비교 =====
print("\n" + "=" * 70)
print("6. ETF 전후 성능 비교 (2024-01-10)")
print("=" * 70)

# Top 30 특성으로 ETF 전후 비교
best_features = feature_sets['top30']
X_subset = X[best_features].copy()

# ETF 전 (훈련: 처음~ETF전 80%, 테스트: ETF전 20%)
pre_etf_split = int(etf_idx * 0.8)
X_pre_train = X_subset.iloc[:pre_etf_split]
X_pre_test = X_subset.iloc[pre_etf_split:etf_idx]
y_pre_train = y.iloc[:pre_etf_split]
y_pre_test = y.iloc[pre_etf_split:etf_idx]

print(f"\nETF 전 ({df.index[0].date()} ~ {df.index[etf_idx-1].date()})")
print(f"  훈련: {len(X_pre_train)}개 | 테스트: {len(X_pre_test)}개")

if len(X_pre_test) > 0:
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=20,
                                min_samples_leaf=10, random_state=42, n_jobs=-1)
    scaler = StandardScaler()
    pre_rf_results, _ = train_and_evaluate(X_pre_train, X_pre_test, y_pre_train, y_pre_test,
                                           'RF_pre_etf', rf, scaler)
    print(f"  [RF] Test R²: {pre_rf_results['test_r2']:.4f} | RMSE: ${pre_rf_results['test_rmse']:,.2f}")

    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    scaler = StandardScaler()
    pre_xgb_results, _ = train_and_evaluate(X_pre_train, X_pre_test, y_pre_train, y_pre_test,
                                            'XGB_pre_etf', xgb_model, scaler)
    print(f"  [XGB] Test R²: {pre_xgb_results['test_r2']:.4f} | RMSE: ${pre_xgb_results['test_rmse']:,.2f}")

# ETF 후 (훈련: ETF후~전체 80%, 테스트: 전체 80%~끝)
X_post_train = X_subset.iloc[etf_idx:split_idx]
X_post_test = X_subset.iloc[split_idx:]
y_post_train = y.iloc[etf_idx:split_idx]
y_post_test = y.iloc[split_idx:]

print(f"\nETF 후 ({df.index[etf_idx].date()} ~ {df.index[-1].date()})")
print(f"  훈련: {len(X_post_train)}개 | 테스트: {len(X_post_test)}개")

if len(X_post_test) > 0 and len(X_post_train) > 0:
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=20,
                                min_samples_leaf=10, random_state=42, n_jobs=-1)
    scaler = StandardScaler()
    post_rf_results, _ = train_and_evaluate(X_post_train, X_post_test, y_post_train, y_post_test,
                                            'RF_post_etf', rf, scaler)
    print(f"  [RF] Test R²: {post_rf_results['test_r2']:.4f} | RMSE: ${post_rf_results['test_rmse']:,.2f}")

    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    scaler = StandardScaler()
    post_xgb_results, _ = train_and_evaluate(X_post_train, X_post_test, y_post_train, y_post_test,
                                             'XGB_post_etf', xgb_model, scaler)
    print(f"  [XGB] Test R²: {post_xgb_results['test_r2']:.4f} | RMSE: ${post_xgb_results['test_rmse']:,.2f}")

# ===== 7. 결과 저장 =====
print("\n" + "=" * 70)
print("7. 결과 저장")
print("=" * 70)

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('test_r2', ascending=False)

# CSV 저장
results_df.to_csv('model_results.csv', index=False)
print("✓ model_results.csv")

# 결과 요약
print("\n최고 성능 모델 (Test R² 기준):")
print("-" * 70)
for i, row in results_df.head(5).iterrows():
    print(f"{i+1}. {row['model']:20} | R²: {row['test_r2']:7.4f} | RMSE: ${row['test_rmse']:10,.2f} | {row['n_features']}개 특성")

# ===== 8. 시각화 =====
print("\n8. 결과 시각화")
print("-" * 70)

# 특성 수 vs 성능
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# R² vs Features
for model_type in ['RF', 'XGB']:
    model_results = results_df[results_df['model'].str.contains(model_type)]
    axes[0, 0].plot(model_results['n_features'], model_results['train_r2'],
                    marker='o', label=f'{model_type} Train')
    axes[0, 0].plot(model_results['n_features'], model_results['test_r2'],
                    marker='s', label=f'{model_type} Test')

axes[0, 0].set_xlabel('Number of Features')
axes[0, 0].set_ylabel('R² Score')
axes[0, 0].set_title('R² Score vs Number of Features')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# RMSE vs Features
for model_type in ['RF', 'XGB']:
    model_results = results_df[results_df['model'].str.contains(model_type)]
    axes[0, 1].plot(model_results['n_features'], model_results['train_rmse'],
                    marker='o', label=f'{model_type} Train')
    axes[0, 1].plot(model_results['n_features'], model_results['test_rmse'],
                    marker='s', label=f'{model_type} Test')

axes[0, 1].set_xlabel('Number of Features')
axes[0, 1].set_ylabel('RMSE ($)')
axes[0, 1].set_title('RMSE vs Number of Features')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Train vs Test R² (과적합 확인)
axes[1, 0].scatter(results_df['train_r2'], results_df['test_r2'], alpha=0.6)
axes[1, 0].plot([0, 1], [0, 1], 'r--', label='Perfect Fit')
axes[1, 0].set_xlabel('Train R²')
axes[1, 0].set_ylabel('Test R²')
axes[1, 0].set_title('Train vs Test R² (Overfitting Check)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Model Comparison
numeric_cols = ['train_r2', 'test_r2', 'train_rmse', 'test_rmse', 'train_mae', 'test_mae', 'n_features']
models = results_df.groupby('model')[numeric_cols].mean()[['test_r2', 'test_rmse']]
x_pos = np.arange(len(models))
axes[1, 1].bar(x_pos, models['test_r2'], alpha=0.7)
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(models.index, rotation=45, ha='right')
axes[1, 1].set_ylabel('Test R²')
axes[1, 1].set_title('Model Comparison (Test R²)')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
print("✓ model_performance_comparison.png")

plt.close()

print("\n" + "=" * 70)
print("Phase 3 완료!")
print("=" * 70)
print(f"\n✅ 모델 훈련 및 평가 완료!")
print(f"   - {len(feature_sets)}개 특성 집합 테스트")
print(f"   - Random Forest & XGBoost 비교")
print(f"   - ETF 전후 성능 분석")
print(f"\n최고 성능: {results_df.iloc[0]['model']}")
print(f"  Test R²: {results_df.iloc[0]['test_r2']:.4f}")
print(f"  Test RMSE: ${results_df.iloc[0]['test_rmse']:,.2f}")
print(f"  특성 수: {int(results_df.iloc[0]['n_features'])}개")
print("=" * 70)
