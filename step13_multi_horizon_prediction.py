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
print("Phase 4: 다중 예측 기간 모델 (7일, 30일)")
print("=" * 70)

# ===== 1. 데이터 로드 =====
print("\n1. 데이터 로드 및 필터링")
print("-" * 70)

df = pd.read_csv('integrated_data_full.csv', index_col=0, parse_dates=True)
print(f"전체 데이터: {df.shape} ({df.index[0].date()} ~ {df.index[-1].date()})")

# 2021년 데이터만 필터링
df_2021 = df[df.index.year == 2021].copy()
print(f"2021년 데이터: {df_2021.shape} ({df_2021.index[0].date()} ~ {df_2021.index[-1].date()})")

# 제거할 특성
exclude_cols = ['Close', 'cumulative_return', 'High', 'Low', 'Open',
                'bc_market_price', 'bc_market_cap']
all_features = [col for col in df_2021.columns if col not in exclude_cols]

print(f"특성: {len(all_features)}개")

# ===== 2. 다중 타겟 생성 =====
print("\n2. 다중 예측 기간 타겟 생성")
print("-" * 70)

# 1일, 7일, 30일 후 종가
df_2021['target_1d'] = df_2021['Close'].shift(-1)
df_2021['target_7d'] = df_2021['Close'].shift(-7)
df_2021['target_30d'] = df_2021['Close'].shift(-30)

# 각 타겟별 유효한 데이터
df_1d = df_2021.dropna(subset=['target_1d']).copy()
df_7d = df_2021.dropna(subset=['target_7d']).copy()
df_30d = df_2021.dropna(subset=['target_30d']).copy()

print(f"1일 예측: {len(df_1d)}개 샘플 (마지막 1일 제외)")
print(f"7일 예측: {len(df_7d)}개 샘플 (마지막 7일 제외)")
print(f"30일 예측: {len(df_30d)}개 샘플 (마지막 30일 제외)")

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
def train_and_evaluate_horizon(df_horizon, target_col, horizon_name, features, model_name, model):
    """특정 예측 기간에 대한 모델 훈련 및 평가"""

    X = df_horizon[features].copy()
    y = df_horizon[target_col].copy()

    # 7:3 분할
    split_idx = int(len(X) * 0.7)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

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
        'horizon': horizon_name,
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

    return results, y_test, y_test_pred, X_test.index

# ===== 5. 모든 조합 훈련 =====
print("\n" + "=" * 70)
print("4. 다중 기간 모델 훈련 (1일, 7일, 30일)")
print("=" * 70)

all_results = []
all_predictions = {}

horizons = [
    (df_1d, 'target_1d', '1-day', 1),
    (df_7d, 'target_7d', '7-day', 7),
    (df_30d, 'target_30d', '30-day', 30)
]

for df_horizon, target_col, horizon_name, days in horizons:
    print(f"\n{'='*70}")
    print(f"예측 기간: {horizon_name} ({len(df_horizon)}개 샘플)")
    print(f"{'='*70}")

    for feature_name, features in feature_sets.items():
        print(f"\n특성 집합: {feature_name} ({len(features)}개)")
        print("-" * 70)

        # Random Forest
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )

        rf_results, y_test, rf_pred, test_dates = train_and_evaluate_horizon(
            df_horizon, target_col, horizon_name, features,
            f'RF_{feature_name}', rf
        )
        rf_results['feature_set'] = feature_name
        rf_results['n_features'] = len(features)
        rf_results['days'] = days
        all_results.append(rf_results)
        all_predictions[f'{horizon_name}_RF_{feature_name}'] = (y_test, rf_pred, test_dates)

        print(f"  [RF] Test R²: {rf_results['test_r2']:.4f} | RMSE: ${rf_results['test_rmse']:,.2f} | MAPE: {rf_results['test_mape']:.2f}%")

        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        xgb_results, y_test, xgb_pred, test_dates = train_and_evaluate_horizon(
            df_horizon, target_col, horizon_name, features,
            f'XGB_{feature_name}', xgb_model
        )
        xgb_results['feature_set'] = feature_name
        xgb_results['n_features'] = len(features)
        xgb_results['days'] = days
        all_results.append(xgb_results)
        all_predictions[f'{horizon_name}_XGB_{feature_name}'] = (y_test, xgb_pred, test_dates)

        print(f"  [XGB] Test R²: {xgb_results['test_r2']:.4f} | RMSE: ${xgb_results['test_rmse']:,.2f} | MAPE: {xgb_results['test_mape']:.2f}%")

# ===== 6. 결과 저장 및 분석 =====
print("\n" + "=" * 70)
print("5. 결과 분석 및 저장")
print("=" * 70)

results_df = pd.DataFrame(all_results)
results_df.to_csv('model_results_multi_horizon.csv', index=False)
print("✓ model_results_multi_horizon.csv")

# 각 예측 기간별 최고 성능
print("\n예측 기간별 최고 성능 모델:")
print("-" * 70)
for horizon in ['1-day', '7-day', '30-day']:
    horizon_results = results_df[results_df['horizon'] == horizon].sort_values('test_r2', ascending=False)
    best = horizon_results.iloc[0]
    print(f"\n{horizon} 예측:")
    print(f"  최고 모델: {best['model']}")
    print(f"  Test R²: {best['test_r2']:.4f}")
    print(f"  Test RMSE: ${best['test_rmse']:,.2f}")
    print(f"  Test MAPE: {best['test_mape']:.2f}%")
    print(f"  특성 수: {int(best['n_features'])}개")

    # 상위 3개 모델
    print(f"\n  상위 3개 모델:")
    for i, row in horizon_results.head(3).iterrows():
        print(f"    {i+1}. {row['model']:20} | R²: {row['test_r2']:7.4f} | RMSE: ${row['test_rmse']:8,.2f} | MAPE: {row['test_mape']:5.2f}%")

# ===== 7. 비교 시각화 =====
print("\n6. 결과 시각화")
print("-" * 70)

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. 예측 기간별 R² 비교 (특성 수별)
ax1 = fig.add_subplot(gs[0, 0])
for horizon in ['1-day', '7-day', '30-day']:
    horizon_data = results_df[results_df['horizon'] == horizon]
    for model_type in ['RF', 'XGB']:
        model_data = horizon_data[horizon_data['model'].str.contains(model_type)]
        ax1.plot(model_data['n_features'], model_data['test_r2'],
                marker='o', label=f'{horizon} {model_type}', linewidth=2)

ax1.set_xlabel('Number of Features', fontsize=11)
ax1.set_ylabel('Test R² Score', fontsize=11)
ax1.set_title('Test R² vs Features (Multi-Horizon)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=8, ncol=2)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)

# 2. 예측 기간별 RMSE 비교
ax2 = fig.add_subplot(gs[0, 1])
for horizon in ['1-day', '7-day', '30-day']:
    horizon_data = results_df[results_df['horizon'] == horizon]
    xgb_data = horizon_data[horizon_data['model'].str.contains('XGB')]
    ax2.plot(xgb_data['n_features'], xgb_data['test_rmse'],
            marker='s', label=f'{horizon}', linewidth=2)

ax2.set_xlabel('Number of Features', fontsize=11)
ax2.set_ylabel('Test RMSE ($)', fontsize=11)
ax2.set_title('Test RMSE vs Features (XGBoost)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# 3. 예측 기간별 MAPE 비교
ax3 = fig.add_subplot(gs[0, 2])
for horizon in ['1-day', '7-day', '30-day']:
    horizon_data = results_df[results_df['horizon'] == horizon]
    xgb_data = horizon_data[horizon_data['model'].str.contains('XGB')]
    ax3.plot(xgb_data['n_features'], xgb_data['test_mape'],
            marker='d', label=f'{horizon}', linewidth=2)

ax3.set_xlabel('Number of Features', fontsize=11)
ax3.set_ylabel('Test MAPE (%)', fontsize=11)
ax3.set_title('Test MAPE vs Features (XGBoost)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# 4-6. 각 기간별 최고 성능 모델 예측 결과
for idx, horizon in enumerate(['1-day', '7-day', '30-day']):
    ax = fig.add_subplot(gs[1, idx])
    horizon_results = results_df[results_df['horizon'] == horizon].sort_values('test_r2', ascending=False)
    best = horizon_results.iloc[0]

    y_test, y_pred, test_dates = all_predictions[f'{horizon}_{best["model"]}']

    ax.scatter(y_test, y_pred, alpha=0.5, s=30)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
           'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Price ($)', fontsize=11)
    ax.set_ylabel('Predicted Price ($)', fontsize=11)
    ax.set_title(f'{horizon} Prediction\n{best["model"]} (R²={best["test_r2"]:.4f})',
                fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# 7-9. 각 기간별 시계열 예측
for idx, horizon in enumerate(['1-day', '7-day', '30-day']):
    ax = fig.add_subplot(gs[2, idx])
    horizon_results = results_df[results_df['horizon'] == horizon].sort_values('test_r2', ascending=False)
    best = horizon_results.iloc[0]

    y_test, y_pred, test_dates = all_predictions[f'{horizon}_{best["model"]}']

    ax.plot(test_dates, y_test.values, label='Actual', linewidth=2, color='blue')
    ax.plot(test_dates, y_pred, label='Predicted', linewidth=2,
           color='red', alpha=0.7, linestyle='--')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('BTC Price ($)', fontsize=11)
    ax.set_title(f'{horizon} Time Series\n{best["model"]} (MAPE={best["test_mape"]:.2f}%)',
                fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

plt.savefig('multi_horizon_prediction_results.png', dpi=300, bbox_inches='tight')
print("✓ multi_horizon_prediction_results.png")

plt.close()

# ===== 8. 예측 기간별 성능 하락 분석 =====
print("\n7. 예측 기간별 성능 변화 분석")
print("-" * 70)

# XGBoost Top 10 기준으로 비교
print("\nXGBoost + Top 10 특성 기준:")
print("-" * 70)

for horizon in ['1-day', '7-day', '30-day']:
    row = results_df[(results_df['horizon'] == horizon) &
                     (results_df['model'] == 'XGB_top10')].iloc[0]
    print(f"\n{horizon} 예측:")
    print(f"  Test R²: {row['test_r2']:.4f}")
    print(f"  Test RMSE: ${row['test_rmse']:,.2f}")
    print(f"  Test MAE: ${row['test_mae']:,.2f}")
    print(f"  Test MAPE: {row['test_mape']:.2f}%")

# 성능 변화율 계산
baseline_1d = results_df[(results_df['horizon'] == '1-day') &
                         (results_df['model'] == 'XGB_top10')].iloc[0]
for horizon in ['7-day', '30-day']:
    row = results_df[(results_df['horizon'] == horizon) &
                     (results_df['model'] == 'XGB_top10')].iloc[0]
    r2_change = ((row['test_r2'] - baseline_1d['test_r2']) / baseline_1d['test_r2']) * 100
    rmse_change = ((row['test_rmse'] - baseline_1d['test_rmse']) / baseline_1d['test_rmse']) * 100
    mape_change = ((row['test_mape'] - baseline_1d['test_mape']) / baseline_1d['test_mape']) * 100

    print(f"\n1-day → {horizon} 성능 변화:")
    print(f"  R² 변화: {r2_change:+.1f}%")
    print(f"  RMSE 변화: {rmse_change:+.1f}%")
    print(f"  MAPE 변화: {mape_change:+.1f}%")

# ===== 9. 예측 정확도 분포 =====
print("\n8. 예측 정확도 분포 (XGBoost Top 10)")
print("-" * 70)

for horizon in ['1-day', '7-day', '30-day']:
    best_key = f'{horizon}_XGB_top10'
    if best_key in all_predictions:
        y_test, y_pred, _ = all_predictions[best_key]

        errors = y_test - y_pred
        pct_errors = (errors / y_test) * 100

        within_1pct = (np.abs(pct_errors) < 1).sum() / len(pct_errors) * 100
        within_2pct = (np.abs(pct_errors) < 2).sum() / len(pct_errors) * 100
        within_5pct = (np.abs(pct_errors) < 5).sum() / len(pct_errors) * 100
        within_10pct = (np.abs(pct_errors) < 10).sum() / len(pct_errors) * 100

        print(f"\n{horizon} 예측 정확도:")
        print(f"  ±1% 이내: {within_1pct:.1f}%")
        print(f"  ±2% 이내: {within_2pct:.1f}%")
        print(f"  ±5% 이내: {within_5pct:.1f}%")
        print(f"  ±10% 이내: {within_10pct:.1f}%")

print("\n" + "=" * 70)
print("다중 예측 기간 모델 훈련 완료!")
print("=" * 70)

# 최종 요약
print("\n📊 최종 요약:")
print("-" * 70)
for horizon in ['1-day', '7-day', '30-day']:
    horizon_results = results_df[results_df['horizon'] == horizon].sort_values('test_r2', ascending=False)
    best = horizon_results.iloc[0]
    print(f"\n{horizon} 예측 최고 성능:")
    print(f"  모델: {best['model']}")
    print(f"  R²: {best['test_r2']:.4f} | RMSE: ${best['test_rmse']:,.2f} | MAPE: {best['test_mape']:.2f}%")
    print(f"  특성: {int(best['n_features'])}개 ({best['feature_set']})")

print("\n" + "=" * 70)
