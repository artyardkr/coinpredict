import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("과적합 분석 (Overfitting Analysis)")
print("=" * 70)

# ===== 1. 데이터 로드 =====
print("\n1. 데이터 로드")
print("-" * 70)

df = pd.read_csv('integrated_data_full.csv', index_col=0, parse_dates=True)
print(f"전체 데이터: {df.shape} ({df.index[0].date()} ~ {df.index[-1].date()})")

# ===== 2. 데이터 누수 가능성 있는 특성 분석 =====
print("\n2. 데이터 누수 위험 특성 탐지")
print("-" * 70)

# Close와의 상관계수 계산
correlations = df.corr()['Close'].abs().sort_values(ascending=False)

print("\nClose와 상관계수가 높은 상위 20개 특성:")
print("-" * 70)
for i, (col, corr) in enumerate(correlations.head(20).items(), 1):
    risk = ""
    if corr > 0.99:
        risk = "⚠️ 매우 높음 (데이터 누수 의심)"
    elif corr > 0.95:
        risk = "⚠️ 높음"
    elif corr > 0.90:
        risk = "⚠️ 주의"
    print(f"{i:2d}. {col:30s} : {corr:.6f} {risk}")

# 데이터 누수 의심 특성 (상관계수 > 0.95)
leakage_features = correlations[correlations > 0.95].index.tolist()
leakage_features = [f for f in leakage_features if f != 'Close']

print(f"\n데이터 누수 의심 특성 ({len(leakage_features)}개):")
for feat in leakage_features:
    print(f"  - {feat} (상관계수: {correlations[feat]:.6f})")

# ===== 3. 제거할 특성 정의 =====
print("\n3. 제거할 특성 정의")
print("-" * 70)

# 명시적으로 제거할 특성
explicit_exclude = [
    'Close',           # 타겟과 동일
    'cumulative_return',  # Close로부터 직접 계산
    'High', 'Low', 'Open',  # 같은 날의 가격 정보
    'bc_market_price',   # Close와 거의 동일
    'bc_market_cap',     # Close * Supply로 계산
]

# 데이터 누수 의심 특성도 제거
all_exclude = list(set(explicit_exclude + leakage_features))

print(f"제거할 특성 ({len(all_exclude)}개):")
for feat in sorted(all_exclude):
    if feat in df.columns:
        print(f"  - {feat}")

# ===== 4. 타겟 생성 및 특성 선택 =====
print("\n4. 타겟 생성 및 특성 선택")
print("-" * 70)

# 타겟: 다음 날 종가
df['target'] = df['Close'].shift(-1)
df_clean = df.dropna(subset=['target']).copy()

# 특성 선택
all_features = [col for col in df_clean.columns if col not in all_exclude and col != 'target']
X = df_clean[all_features].copy()
y = df_clean['target'].copy()

print(f"특성 수: {len(all_features)}개")
print(f"샘플 수: {len(X)}개")
print(f"기간: {X.index[0].date()} ~ {X.index[-1].date()}")

# ===== 5. 데이터 분할 =====
print("\n5. 데이터 분할 (시계열 80:20)")
print("-" * 70)

split_idx = int(len(X) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"훈련: {X_train.shape[0]}개 ({X_train.index[0].date()} ~ {X_train.index[-1].date()})")
print(f"테스트: {X_test.shape[0]}개 ({X_test.index[0].date()} ~ {X_test.index[-1].date()})")

# ===== 6. 모델 훈련 및 과적합 분석 =====
print("\n" + "=" * 70)
print("6. 모델 훈련 및 과적합 분석")
print("=" * 70)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = []

# Random Forest
print("\n[Random Forest]")
print("-" * 70)
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_scaled, y_train)

y_train_pred_rf = rf.predict(X_train_scaled)
y_test_pred_rf = rf.predict(X_test_scaled)

rf_train_r2 = r2_score(y_train, y_train_pred_rf)
rf_test_r2 = r2_score(y_test, y_test_pred_rf)
rf_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_rf))
rf_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_rf))

print(f"Train R²: {rf_train_r2:.4f} | RMSE: ${rf_train_rmse:,.2f}")
print(f"Test R²:  {rf_test_r2:.4f} | RMSE: ${rf_test_rmse:,.2f}")
print(f"R² 차이:  {rf_train_r2 - rf_test_r2:.4f} {'⚠️ 과적합 의심' if rf_train_r2 - rf_test_r2 > 0.1 else '✓'}")

results.append({
    'model': 'Random Forest',
    'train_r2': rf_train_r2,
    'test_r2': rf_test_r2,
    'r2_gap': rf_train_r2 - rf_test_r2,
    'train_rmse': rf_train_rmse,
    'test_rmse': rf_test_rmse
})

# Feature Importance
rf_importance = pd.DataFrame({
    'feature': all_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n상위 10개 중요 특성:")
for i, row in rf_importance.head(10).iterrows():
    print(f"  {i+1}. {row['feature']:30s} : {row['importance']:.4f}")

# 특정 특성이 과도하게 높은지 확인
top1_importance = rf_importance.iloc[0]['importance']
if top1_importance > 0.5:
    print(f"\n⚠️ 경고: '{rf_importance.iloc[0]['feature']}' 특성이 {top1_importance:.1%}의 중요도를 가짐 (과적합 가능성)")

# XGBoost
print("\n[XGBoost]")
print("-" * 70)
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train_scaled, y_train)

y_train_pred_xgb = xgb_model.predict(X_train_scaled)
y_test_pred_xgb = xgb_model.predict(X_test_scaled)

xgb_train_r2 = r2_score(y_train, y_train_pred_xgb)
xgb_test_r2 = r2_score(y_test, y_test_pred_xgb)
xgb_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_xgb))
xgb_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_xgb))

print(f"Train R²: {xgb_train_r2:.4f} | RMSE: ${xgb_train_rmse:,.2f}")
print(f"Test R²:  {xgb_test_r2:.4f} | RMSE: ${xgb_test_rmse:,.2f}")
print(f"R² 차이:  {xgb_train_r2 - xgb_test_r2:.4f} {'⚠️ 과적합 의심' if xgb_train_r2 - xgb_test_r2 > 0.1 else '✓'}")

results.append({
    'model': 'XGBoost',
    'train_r2': xgb_train_r2,
    'test_r2': xgb_test_r2,
    'r2_gap': xgb_train_r2 - xgb_test_r2,
    'train_rmse': xgb_train_rmse,
    'test_rmse': xgb_test_rmse
})

# Feature Importance
xgb_importance = pd.DataFrame({
    'feature': all_features,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n상위 10개 중요 특성:")
for i, row in xgb_importance.head(10).iterrows():
    print(f"  {i+1}. {row['feature']:30s} : {row['importance']:.4f}")

# 특정 특성이 과도하게 높은지 확인
top1_importance = xgb_importance.iloc[0]['importance']
if top1_importance > 0.5:
    print(f"\n⚠️ 경고: '{xgb_importance.iloc[0]['feature']}' 특성이 {top1_importance:.1%}의 중요도를 가짐 (과적합 가능성)")

# ===== 7. 과적합 진단 =====
print("\n" + "=" * 70)
print("7. 과적합 진단")
print("=" * 70)

results_df = pd.DataFrame(results)

print("\n모델별 과적합 분석:")
print("-" * 70)
for _, row in results_df.iterrows():
    print(f"\n{row['model']}:")
    print(f"  Train R²: {row['train_r2']:.4f}")
    print(f"  Test R²:  {row['test_r2']:.4f}")
    print(f"  R² 차이:  {row['r2_gap']:.4f}")

    # 과적합 진단
    if row['r2_gap'] < 0.05:
        diagnosis = "✅ 과적합 없음 (양호)"
    elif row['r2_gap'] < 0.1:
        diagnosis = "⚠️ 경미한 과적합 (허용 범위)"
    elif row['r2_gap'] < 0.2:
        diagnosis = "⚠️ 중간 수준 과적합 (개선 필요)"
    else:
        diagnosis = "❌ 심각한 과적합 (모델 재설계 필요)"

    print(f"  진단:    {diagnosis}")

    # Test R² 진단
    if row['test_r2'] < 0:
        print(f"  ❌ Test R²가 음수 - 모델이 평균보다 못함")
    elif row['test_r2'] < 0.3:
        print(f"  ⚠️ Test R²가 낮음 - 예측력 부족")
    elif row['test_r2'] < 0.7:
        print(f"  ✓ Test R²가 보통 수준")
    else:
        print(f"  ✅ Test R²가 높음 - 우수한 예측력")

# ===== 8. 시각화 =====
print("\n8. 과적합 분석 시각화")
print("-" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Train vs Test R² 비교
ax1 = axes[0, 0]
models = results_df['model']
x = np.arange(len(models))
width = 0.35

ax1.bar(x - width/2, results_df['train_r2'], width, label='Train R²', alpha=0.8)
ax1.bar(x + width/2, results_df['test_r2'], width, label='Test R²', alpha=0.8)
ax1.set_xlabel('Model', fontsize=11)
ax1.set_ylabel('R² Score', fontsize=11)
ax1.set_title('Train vs Test R² (Overfitting Check)', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)

# 2. R² Gap (과적합 정도)
ax2 = axes[0, 1]
colors = ['orange' if gap > 0.1 else 'green' for gap in results_df['r2_gap']]
ax2.bar(models, results_df['r2_gap'], color=colors, alpha=0.7)
ax2.set_xlabel('Model', fontsize=11)
ax2.set_ylabel('R² Gap (Train - Test)', fontsize=11)
ax2.set_title('Overfitting Severity (R² Gap)', fontsize=12, fontweight='bold')
ax2.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Threshold (0.1)')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# 3. Feature Importance (Random Forest)
ax3 = axes[0, 2]
top_features_rf = rf_importance.head(15)
ax3.barh(range(len(top_features_rf)), top_features_rf['importance'], alpha=0.7)
ax3.set_yticks(range(len(top_features_rf)))
ax3.set_yticklabels(top_features_rf['feature'], fontsize=9)
ax3.set_xlabel('Importance', fontsize=11)
ax3.set_title('Feature Importance (Random Forest)', fontsize=12, fontweight='bold')
ax3.invert_yaxis()
ax3.grid(True, alpha=0.3, axis='x')

# 4. Feature Importance (XGBoost)
ax4 = axes[1, 0]
top_features_xgb = xgb_importance.head(15)
ax4.barh(range(len(top_features_xgb)), top_features_xgb['importance'], alpha=0.7, color='orange')
ax4.set_yticks(range(len(top_features_xgb)))
ax4.set_yticklabels(top_features_xgb['feature'], fontsize=9)
ax4.set_xlabel('Importance', fontsize=11)
ax4.set_title('Feature Importance (XGBoost)', fontsize=12, fontweight='bold')
ax4.invert_yaxis()
ax4.grid(True, alpha=0.3, axis='x')

# 5. Actual vs Predicted (Random Forest)
ax5 = axes[1, 1]
ax5.scatter(y_test, y_test_pred_rf, alpha=0.5, s=30)
ax5.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        'r--', linewidth=2, label='Perfect Prediction')
ax5.set_xlabel('Actual Price ($)', fontsize=11)
ax5.set_ylabel('Predicted Price ($)', fontsize=11)
ax5.set_title(f'Random Forest (Test R²={rf_test_r2:.4f})', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Actual vs Predicted (XGBoost)
ax6 = axes[1, 2]
ax6.scatter(y_test, y_test_pred_xgb, alpha=0.5, s=30, color='orange')
ax6.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        'r--', linewidth=2, label='Perfect Prediction')
ax6.set_xlabel('Actual Price ($)', fontsize=11)
ax6.set_ylabel('Predicted Price ($)', fontsize=11)
ax6.set_title(f'XGBoost (Test R²={xgb_test_r2:.4f})', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('overfitting_analysis.png', dpi=300, bbox_inches='tight')
print("✓ overfitting_analysis.png")

plt.close()

# ===== 9. 상관관계 히트맵 (상위 특성) =====
print("\n9. 상위 특성 간 상관관계 분석")
print("-" * 70)

# RF와 XGB의 상위 특성 합치기
top_features_combined = list(set(
    rf_importance.head(10)['feature'].tolist() +
    xgb_importance.head(10)['feature'].tolist()
))

if len(top_features_combined) > 0:
    corr_matrix = X[top_features_combined].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                fmt='.2f', square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Top Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('top_features_correlation.png', dpi=300, bbox_inches='tight')
    print("✓ top_features_correlation.png")
    plt.close()

    # 높은 상관관계 탐지 (다중공선성)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))

    if high_corr_pairs:
        print(f"\n⚠️ 높은 상관관계 특성 쌍 ({len(high_corr_pairs)}개, |r| > 0.8):")
        for feat1, feat2, corr in high_corr_pairs:
            print(f"  - {feat1} ↔ {feat2}: {corr:.3f}")
        print("\n→ 다중공선성 가능성: 하나의 특성 제거 고려")
    else:
        print("\n✓ 높은 상관관계 특성 쌍 없음 (다중공선성 문제 없음)")

# ===== 10. 결과 저장 =====
print("\n10. 결과 저장")
print("-" * 70)

# 모델 결과 저장
results_df.to_csv('overfitting_analysis_results.csv', index=False)
print("✓ overfitting_analysis_results.csv")

# Feature Importance 저장
rf_importance.to_csv('feature_importance_rf.csv', index=False)
xgb_importance.to_csv('feature_importance_xgb.csv', index=False)
print("✓ feature_importance_rf.csv")
print("✓ feature_importance_xgb.csv")

# 데이터 누수 의심 특성 저장
leakage_df = pd.DataFrame({
    'feature': correlations[correlations > 0.95].index.tolist(),
    'correlation_with_close': correlations[correlations > 0.95].values
})
leakage_df.to_csv('data_leakage_suspects.csv', index=False)
print("✓ data_leakage_suspects.csv")

print("\n" + "=" * 70)
print("과적합 분석 완료!")
print("=" * 70)

print("\n📊 최종 요약:")
print("-" * 70)
print(f"제거된 특성 수: {len(all_exclude)}개")
print(f"사용된 특성 수: {len(all_features)}개")
print(f"\n모델 성능:")
for _, row in results_df.iterrows():
    print(f"  {row['model']:15s}: Test R² = {row['test_r2']:7.4f} (Gap: {row['r2_gap']:.4f})")

# 최종 권장사항
print("\n💡 권장사항:")
print("-" * 70)
best_model = results_df.loc[results_df['test_r2'].idxmax()]
if best_model['r2_gap'] > 0.2:
    print("⚠️ 모든 모델에서 과적합 발견 - 정규화 강화 또는 특성 추가 제거 필요")
elif best_model['test_r2'] < 0:
    print("❌ 모든 모델의 Test R²가 음수 - 데이터 기간 또는 특성 재검토 필요")
else:
    print(f"✅ {best_model['model']} 모델 사용 권장 (Test R²: {best_model['test_r2']:.4f})")

print("=" * 70)
