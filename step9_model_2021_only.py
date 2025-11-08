import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Phase 3: 모델 훈련 (2021년 데이터만, 7:3 분할)")
print("=" * 70)

# ===== 1. 데이터 로드 =====
print("\n1. 데이터 로드 및 필터링")
print("-" * 70)

df = pd.read_csv('integrated_data_full.csv', index_col=0, parse_dates=True)
print(f"전체 데이터: {df.shape} ({df.index[0].date()} ~ {df.index[-1].date()})")

# 2021년 데이터만 필터링
df_2021 = df[df.index.year == 2021].copy()
print(f"2021년 데이터: {df_2021.shape} ({df_2021.index[0].date()} ~ {df_2021.index[-1].date()})")

# 타겟 설정
df_2021['target'] = df_2021['Close'].shift(-1)
df_2021 = df_2021.dropna(subset=['target'])

print(f"타겟 생성 후: {df_2021.shape}")

# 제거할 특성
exclude_cols = ['Close', 'target', 'cumulative_return', 'High', 'Low', 'Open',
                'bc_market_price', 'bc_market_cap']
all_features = [col for col in df_2021.columns if col not in exclude_cols]

X = df_2021[all_features].copy()
y = df_2021['target'].copy()

print(f"특성: {len(all_features)}개")
print(f"샘플: {len(X)}개")

# ===== 2. 7:3 분할 =====
print("\n2. 7:3 데이터 분할")
print("-" * 70)

split_idx = int(len(X) * 0.7)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"훈련 데이터: {X_train.shape[0]}개 ({X_train.index[0].date()} ~ {X_train.index[-1].date()})")
print(f"테스트 데이터: {X_test.shape[0]}개 ({X_test.index[0].date()} ~ {X_test.index[-1].date()})")

# ===== 3. 특성 집합 로드 =====
print("\n3. 특성 집합 로드")
print("-" * 70)

feature_sets = {}

for n in [10, 20, 30, 40, 50]:
    with open(f'selected_features_top{n}.txt', 'r') as f:
        lines = f.readlines()
        features = [line.strip().split('. ')[1] for line in lines if line.strip() and '. ' in line]
        feature_sets[f'top{n}'] = features
        print(f"✓ Top {n}: {len(features)}개")

feature_sets['all'] = all_features
print(f"✓ All: {len(all_features)}개")

# ===== 4. 모델 훈련 함수 =====
def train_and_evaluate(X_train, X_test, y_train, y_test, model_name, model):
    """모델 훈련 및 평가"""

    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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
        'train_mape': np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100,
        'test_mape': np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100,
    }

    return results, model, y_test_pred, scaler

# ===== 5. 모델별, 특성 집합별 훈련 =====
print("\n" + "=" * 70)
print("4. 모델 훈련 및 평가 (2021년 데이터)")
print("=" * 70)

all_results = []
predictions = {}

for feature_name, features in feature_sets.items():
    print(f"\n{'='*70}")
    print(f"특성 집합: {feature_name} ({len(features)}개)")
    print(f"{'='*70}")

    # 해당 특성만 선택
    X_subset = X[features].copy()
    X_train_subset = X_train[features].copy()
    X_test_subset = X_test[features].copy()

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

    rf_results, rf_model, rf_pred, rf_scaler = train_and_evaluate(
        X_train_subset, X_test_subset, y_train, y_test,
        f'RF_{feature_name}', rf
    )
    rf_results['feature_set'] = feature_name
    rf_results['n_features'] = len(features)
    all_results.append(rf_results)
    predictions[f'RF_{feature_name}'] = rf_pred

    print(f"  Train R²: {rf_results['train_r2']:.4f} | Test R²: {rf_results['test_r2']:.4f}")
    print(f"  Train RMSE: ${rf_results['train_rmse']:,.2f} | Test RMSE: ${rf_results['test_rmse']:,.2f}")
    print(f"  Train MAE: ${rf_results['train_mae']:,.2f} | Test MAE: ${rf_results['test_mae']:,.2f}")
    print(f"  Train MAPE: {rf_results['train_mape']:.2f}% | Test MAPE: {rf_results['test_mape']:.2f}%")

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

    xgb_results, xgb_fitted, xgb_pred, xgb_scaler = train_and_evaluate(
        X_train_subset, X_test_subset, y_train, y_test,
        f'XGB_{feature_name}', xgb_model
    )
    xgb_results['feature_set'] = feature_name
    xgb_results['n_features'] = len(features)
    all_results.append(xgb_results)
    predictions[f'XGB_{feature_name}'] = xgb_pred

    print(f"  Train R²: {xgb_results['train_r2']:.4f} | Test R²: {xgb_results['test_r2']:.4f}")
    print(f"  Train RMSE: ${xgb_results['train_rmse']:,.2f} | Test RMSE: ${xgb_results['test_rmse']:,.2f}")
    print(f"  Train MAE: ${xgb_results['train_mae']:,.2f} | Test MAE: ${xgb_results['test_mae']:,.2f}")
    print(f"  Train MAPE: {xgb_results['train_mape']:.2f}% | Test MAPE: {xgb_results['test_mape']:.2f}%")

# ===== 6. 결과 저장 =====
print("\n" + "=" * 70)
print("5. 결과 저장")
print("=" * 70)

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('test_r2', ascending=False)

results_df.to_csv('model_results_2021.csv', index=False)
print("✓ model_results_2021.csv")

print("\n최고 성능 모델 (Test R² 기준):")
print("-" * 70)
for i, row in results_df.head(5).iterrows():
    print(f"{i+1}. {row['model']:20} | R²: {row['test_r2']:7.4f} | RMSE: ${row['test_rmse']:8,.2f} | MAPE: {row['test_mape']:5.2f}% | {int(row['n_features'])}개")

# 최고 성능 모델
best_model = results_df.iloc[0]
print(f"\n🏆 최고 성능: {best_model['model']}")
print(f"   Test R²: {best_model['test_r2']:.4f}")
print(f"   Test RMSE: ${best_model['test_rmse']:,.2f}")
print(f"   Test MAPE: {best_model['test_mape']:.2f}%")

# ===== 7. 시각화 =====
print("\n6. 결과 시각화")
print("-" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. 특성 수 vs R²
for model_type in ['RF', 'XGB']:
    model_results = results_df[results_df['model'].str.contains(model_type)]
    axes[0, 0].plot(model_results['n_features'], model_results['train_r2'],
                    marker='o', label=f'{model_type} Train', alpha=0.7)
    axes[0, 0].plot(model_results['n_features'], model_results['test_r2'],
                    marker='s', label=f'{model_type} Test', linewidth=2)

axes[0, 0].set_xlabel('Number of Features', fontsize=11)
axes[0, 0].set_ylabel('R² Score', fontsize=11)
axes[0, 0].set_title('R² Score vs Number of Features (2021 Data)', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)

# 2. 특성 수 vs RMSE
for model_type in ['RF', 'XGB']:
    model_results = results_df[results_df['model'].str.contains(model_type)]
    axes[0, 1].plot(model_results['n_features'], model_results['test_rmse'],
                    marker='s', label=f'{model_type} Test', linewidth=2)

axes[0, 1].set_xlabel('Number of Features', fontsize=11)
axes[0, 1].set_ylabel('RMSE ($)', fontsize=11)
axes[0, 1].set_title('Test RMSE vs Number of Features', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. 특성 수 vs MAPE
for model_type in ['RF', 'XGB']:
    model_results = results_df[results_df['model'].str.contains(model_type)]
    axes[0, 2].plot(model_results['n_features'], model_results['test_mape'],
                    marker='s', label=f'{model_type} Test', linewidth=2)

axes[0, 2].set_xlabel('Number of Features', fontsize=11)
axes[0, 2].set_ylabel('MAPE (%)', fontsize=11)
axes[0, 2].set_title('Test MAPE vs Number of Features', fontsize=12, fontweight='bold')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. Train vs Test R² (과적합 체크)
axes[1, 0].scatter(results_df['train_r2'], results_df['test_r2'],
                   c=results_df['n_features'], cmap='viridis', s=100, alpha=0.6)
axes[1, 0].plot([0, 1], [0, 1], 'r--', label='Perfect Fit', linewidth=2)
axes[1, 0].set_xlabel('Train R²', fontsize=11)
axes[1, 0].set_ylabel('Test R²', fontsize=11)
axes[1, 0].set_title('Train vs Test R² (Overfitting Check)', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
cbar.set_label('# Features')

# 5. 실제값 vs 예측값 (최고 성능 모델)
best_pred = predictions[best_model['model']]
axes[1, 1].scatter(y_test, best_pred, alpha=0.5, s=30)
axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                'r--', linewidth=2, label='Perfect Prediction')
axes[1, 1].set_xlabel('Actual Price ($)', fontsize=11)
axes[1, 1].set_ylabel('Predicted Price ($)', fontsize=11)
axes[1, 1].set_title(f'Actual vs Predicted ({best_model["model"]})', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. 시계열 예측 플롯 (최고 성능 모델)
test_dates = y_test.index
axes[1, 2].plot(test_dates, y_test.values, label='Actual', linewidth=2, color='blue')
axes[1, 2].plot(test_dates, best_pred, label='Predicted', linewidth=2,
                color='red', alpha=0.7, linestyle='--')
axes[1, 2].set_xlabel('Date', fontsize=11)
axes[1, 2].set_ylabel('BTC Price ($)', fontsize=11)
axes[1, 2].set_title(f'Time Series Prediction ({best_model["model"]})', fontsize=12, fontweight='bold')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('model_performance_2021.png', dpi=300, bbox_inches='tight')
print("✓ model_performance_2021.png")

plt.close()

# ===== 8. 상세 예측 결과 =====
print("\n7. 예측 상세 분석 (최고 성능 모델)")
print("-" * 70)

# 예측 오차 분석
errors = y_test - best_pred
abs_errors = np.abs(errors)
pct_errors = (errors / y_test) * 100

print(f"\n예측 오차 통계:")
print(f"  평균 오차: ${errors.mean():,.2f}")
print(f"  평균 절대 오차 (MAE): ${abs_errors.mean():,.2f}")
print(f"  평균 절대 백분율 오차 (MAPE): {np.abs(pct_errors).mean():.2f}%")
print(f"  최대 오차: ${abs_errors.max():,.2f}")
print(f"  최소 오차: ${abs_errors.min():,.2f}")

# 예측 정확도 분포
within_1pct = (np.abs(pct_errors) < 1).sum() / len(pct_errors) * 100
within_2pct = (np.abs(pct_errors) < 2).sum() / len(pct_errors) * 100
within_5pct = (np.abs(pct_errors) < 5).sum() / len(pct_errors) * 100

print(f"\n예측 정확도 분포:")
print(f"  ±1% 이내: {within_1pct:.1f}%")
print(f"  ±2% 이내: {within_2pct:.1f}%")
print(f"  ±5% 이내: {within_5pct:.1f}%")

print("\n" + "=" * 70)
print("2021년 데이터 모델 훈련 완료!")
print("=" * 70)
