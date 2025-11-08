import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("변화율 & 방향 예측 모델 (Return & Direction Prediction)")
print("=" * 70)

# ===== 1. 데이터 로드 =====
print("\n1. 데이터 로드")
print("-" * 70)

df = pd.read_csv('integrated_data_full.csv', index_col=0, parse_dates=True)
print(f"전체 데이터: {df.shape} ({df.index[0].date()} ~ {df.index[-1].date()})")

# 2021년 데이터만 필터링 (성공했던 접근)
df_2021 = df[df.index.year == 2021].copy()
print(f"2021년 데이터: {df_2021.shape} ({df_2021.index[0].date()} ~ {df_2021.index[-1].date()})")

# ===== 2. 타겟 생성 =====
print("\n2. 타겟 생성 (변화율 & 방향)")
print("-" * 70)

# 1) 변화율 타겟 (Regression)
df_2021['target_return'] = df_2021['Close'].pct_change().shift(-1) * 100  # 내일 수익률(%)

# 2) 방향 타겟 (Classification)
# 임계값 설정: ±0.5% 이내는 횡보로 간주
threshold = 0.5

def classify_direction(return_pct):
    if pd.isna(return_pct):
        return np.nan
    elif return_pct > threshold:
        return 1  # 상승
    elif return_pct < -threshold:
        return -1  # 하락
    else:
        return 0  # 횡보

df_2021['target_direction'] = df_2021['target_return'].apply(classify_direction)

# NaN 제거
df_2021 = df_2021.dropna(subset=['target_return', 'target_direction'])

print(f"타겟 생성 후 샘플 수: {len(df_2021)}")
print(f"\n변화율 통계:")
print(f"  평균: {df_2021['target_return'].mean():.2f}%")
print(f"  표준편차: {df_2021['target_return'].std():.2f}%")
print(f"  최소: {df_2021['target_return'].min():.2f}%")
print(f"  최대: {df_2021['target_return'].max():.2f}%")

print(f"\n방향 분포 (임계값: ±{threshold}%):")
direction_counts = df_2021['target_direction'].value_counts().sort_index()
total = len(df_2021)
for direction, count in direction_counts.items():
    dir_int = int(direction)
    label = {-1: '하락', 0: '횡보', 1: '상승'}[dir_int]
    print(f"  {label:4s} ({dir_int:2d}): {count:3d}개 ({count/total*100:5.1f}%)")

# ===== 3. 특성 선택 =====
print("\n3. 특성 선택")
print("-" * 70)

# 데이터 누수 제거
exclude_cols = [
    'Close', 'target_return', 'target_direction',
    'cumulative_return', 'High', 'Low', 'Open',
    'bc_market_price', 'bc_market_cap'
]

all_features = [col for col in df_2021.columns if col not in exclude_cols]

# Top 10 특성 로드 (step9에서 성공했던 특성들)
try:
    with open('selected_features_top10.txt', 'r') as f:
        lines = f.readlines()
        top10_features = [line.strip().split('. ')[1] for line in lines if line.strip() and '. ' in line]
    print(f"✓ Top 10 특성 로드: {len(top10_features)}개")
    use_features = top10_features
except:
    print("⚠️ Top 10 특성 파일 없음 - 모든 특성 사용")
    use_features = all_features

X = df_2021[use_features].copy()
y_return = df_2021['target_return'].copy()
y_direction = df_2021['target_direction'].copy()

print(f"특성 수: {len(use_features)}개")
print(f"샘플 수: {len(X)}개")

# ===== 4. 데이터 분할 =====
print("\n4. 데이터 분할 (7:3)")
print("-" * 70)

split_idx = int(len(X) * 0.7)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_return_train = y_return.iloc[:split_idx]
y_return_test = y_return.iloc[split_idx:]
y_direction_train = y_direction.iloc[:split_idx]
y_direction_test = y_direction.iloc[split_idx:]

print(f"훈련: {X_train.shape[0]}개 ({X_train.index[0].date()} ~ {X_train.index[-1].date()})")
print(f"테스트: {X_test.shape[0]}개 ({X_test.index[0].date()} ~ {X_test.index[-1].date()})")

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== 5. 변화율 예측 (Regression) =====
print("\n" + "=" * 70)
print("5. 변화율 예측 모델 (Regression)")
print("=" * 70)

regression_results = []

# Random Forest Regressor
print("\n[Random Forest - Regression]")
print("-" * 70)
rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

rf_reg.fit(X_train_scaled, y_return_train)
y_return_pred_rf = rf_reg.predict(X_test_scaled)

rf_r2 = r2_score(y_return_test, y_return_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_return_test, y_return_pred_rf))
rf_mae = mean_absolute_error(y_return_test, y_return_pred_rf)

print(f"Test R²: {rf_r2:.4f}")
print(f"Test RMSE: {rf_rmse:.4f}%")
print(f"Test MAE: {rf_mae:.4f}%")

regression_results.append({
    'model': 'Random Forest',
    'r2': rf_r2,
    'rmse': rf_rmse,
    'mae': rf_mae
})

# XGBoost Regressor
print("\n[XGBoost - Regression]")
print("-" * 70)
xgb_reg = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_reg.fit(X_train_scaled, y_return_train)
y_return_pred_xgb = xgb_reg.predict(X_test_scaled)

xgb_r2 = r2_score(y_return_test, y_return_pred_xgb)
xgb_rmse = np.sqrt(mean_squared_error(y_return_test, y_return_pred_xgb))
xgb_mae = mean_absolute_error(y_return_test, y_return_pred_xgb)

print(f"Test R²: {xgb_r2:.4f}")
print(f"Test RMSE: {xgb_rmse:.4f}%")
print(f"Test MAE: {xgb_mae:.4f}%")

regression_results.append({
    'model': 'XGBoost',
    'r2': xgb_r2,
    'rmse': xgb_rmse,
    'mae': xgb_mae
})

# ===== 6. 방향 예측 (Classification) =====
print("\n" + "=" * 70)
print("6. 방향 예측 모델 (Classification)")
print("=" * 70)

classification_results = []

# Random Forest Classifier
print("\n[Random Forest - Classification]")
print("-" * 70)
rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # 불균형 클래스 처리
)

rf_clf.fit(X_train_scaled, y_direction_train)
y_direction_pred_rf = rf_clf.predict(X_test_scaled)
y_direction_proba_rf = rf_clf.predict_proba(X_test_scaled)

rf_acc = accuracy_score(y_direction_test, y_direction_pred_rf)
rf_prec, rf_rec, rf_f1, _ = precision_recall_fscore_support(
    y_direction_test, y_direction_pred_rf, average='weighted', zero_division=0
)

print(f"Accuracy: {rf_acc:.4f} ({rf_acc*100:.1f}%)")
print(f"Precision: {rf_prec:.4f}")
print(f"Recall: {rf_rec:.4f}")
print(f"F1-Score: {rf_f1:.4f}")

print(f"\n클래스별 성능:")
print(classification_report(y_direction_test, y_direction_pred_rf,
                          target_names=['하락', '횡보', '상승'], zero_division=0))

classification_results.append({
    'model': 'Random Forest',
    'accuracy': rf_acc,
    'precision': rf_prec,
    'recall': rf_rec,
    'f1': rf_f1
})

# XGBoost Classifier
print("\n[XGBoost - Classification]")
print("-" * 70)

# XGBoost는 0, 1, 2로 레이블 필요
y_direction_train_xgb = y_direction_train + 1  # -1,0,1 → 0,1,2
y_direction_test_xgb = y_direction_test + 1

xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_clf.fit(X_train_scaled, y_direction_train_xgb)
y_direction_pred_xgb_raw = xgb_clf.predict(X_test_scaled)
y_direction_pred_xgb = y_direction_pred_xgb_raw - 1  # 0,1,2 → -1,0,1
y_direction_proba_xgb = xgb_clf.predict_proba(X_test_scaled)

xgb_acc = accuracy_score(y_direction_test, y_direction_pred_xgb)
xgb_prec, xgb_rec, xgb_f1, _ = precision_recall_fscore_support(
    y_direction_test, y_direction_pred_xgb, average='weighted', zero_division=0
)

print(f"Accuracy: {xgb_acc:.4f} ({xgb_acc*100:.1f}%)")
print(f"Precision: {xgb_prec:.4f}")
print(f"Recall: {xgb_rec:.4f}")
print(f"F1-Score: {xgb_f1:.4f}")

print(f"\n클래스별 성능:")
print(classification_report(y_direction_test, y_direction_pred_xgb,
                          target_names=['하락', '횡보', '상승'], zero_division=0))

classification_results.append({
    'model': 'XGBoost',
    'accuracy': xgb_acc,
    'precision': xgb_prec,
    'recall': xgb_rec,
    'f1': xgb_f1
})

# ===== 7. 결합 분석 (변화율 + 방향) =====
print("\n" + "=" * 70)
print("7. 결합 분석: 변화율과 방향 동시 평가")
print("=" * 70)

# 최고 성능 모델 선택
best_reg = 'XGBoost' if xgb_r2 > rf_r2 else 'Random Forest'
best_clf = 'XGBoost' if xgb_acc > rf_acc else 'Random Forest'

print(f"\n최고 성능 모델:")
print(f"  변화율 예측: {best_reg} (R² {max(xgb_r2, rf_r2):.4f})")
print(f"  방향 예측: {best_clf} (Accuracy {max(xgb_acc, rf_acc):.4f})")

# XGBoost 결과로 통합 분석
y_return_pred = y_return_pred_xgb
y_direction_pred = y_direction_pred_xgb

# 실제 가격 복원
actual_prices = []
predicted_prices = []
current_price = df_2021['Close'].iloc[split_idx - 1]  # 테스트 시작 전 마지막 가격

for i, (actual_return, pred_return) in enumerate(zip(y_return_test, y_return_pred)):
    actual_price = current_price * (1 + actual_return / 100)
    pred_price = current_price * (1 + pred_return / 100)

    actual_prices.append(actual_price)
    predicted_prices.append(pred_price)

    current_price = actual_price  # 실제 가격으로 업데이트

actual_prices = np.array(actual_prices)
predicted_prices = np.array(predicted_prices)

# 가격 예측 성능
price_r2 = r2_score(actual_prices, predicted_prices)
price_rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
price_mae = mean_absolute_error(actual_prices, predicted_prices)

print(f"\n복원된 가격 예측 성능:")
print(f"  R²: {price_r2:.4f}")
print(f"  RMSE: ${price_rmse:,.2f}")
print(f"  MAE: ${price_mae:,.2f}")

# 트레이딩 시뮬레이션
print(f"\n트레이딩 시뮬레이션 (방향 예측 기반):")
print("-" * 70)

initial_capital = 10000  # $10,000 초기 자본
capital = initial_capital
position = 0  # 0: 현금, 1: 매수
trades = []

for i, (actual_return, pred_direction) in enumerate(zip(y_return_test.values, y_direction_pred)):
    if pred_direction == 1 and position == 0:  # 상승 예측 & 현금 보유 → 매수
        position = 1
        trades.append(('BUY', i, actual_return))
    elif pred_direction == -1 and position == 1:  # 하락 예측 & 주식 보유 → 매도
        capital *= (1 + actual_return / 100)
        position = 0
        trades.append(('SELL', i, actual_return))
    elif position == 1:  # 보유 중
        capital *= (1 + actual_return / 100)

# 마지막에 포지션 정리
if position == 1:
    capital *= (1 + y_return_test.values[-1] / 100)

total_return = (capital - initial_capital) / initial_capital * 100
buy_hold_capital = initial_capital * (1 + y_return_test.sum() / 100)
buy_hold_return = (buy_hold_capital - initial_capital) / initial_capital * 100

print(f"초기 자본: ${initial_capital:,.2f}")
print(f"최종 자본 (전략): ${capital:,.2f}")
print(f"수익률 (전략): {total_return:+.2f}%")
print(f"\n비교: Buy & Hold")
print(f"최종 자본: ${buy_hold_capital:,.2f}")
print(f"수익률: {buy_hold_return:+.2f}%")
print(f"\n전략 우위: {total_return - buy_hold_return:+.2f}%p")
print(f"거래 횟수: {len(trades)}회")

# ===== 8. 시각화 =====
print("\n8. 시각화")
print("-" * 70)

fig, axes = plt.subplots(3, 3, figsize=(18, 15))

# 1. 변화율 예측 (Regression)
ax1 = axes[0, 0]
ax1.scatter(y_return_test, y_return_pred_rf, alpha=0.5, s=30, label='RF')
ax1.scatter(y_return_test, y_return_pred_xgb, alpha=0.5, s=30, label='XGB')
ax1.plot([y_return_test.min(), y_return_test.max()],
         [y_return_test.min(), y_return_test.max()], 'r--', linewidth=2)
ax1.set_xlabel('Actual Return (%)', fontsize=11)
ax1.set_ylabel('Predicted Return (%)', fontsize=11)
ax1.set_title('Return Prediction (Regression)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 변화율 분포
ax2 = axes[0, 1]
ax2.hist(y_return_test, bins=30, alpha=0.5, label='Actual', color='blue')
ax2.hist(y_return_pred_xgb, bins=30, alpha=0.5, label='Predicted (XGB)', color='orange')
ax2.set_xlabel('Return (%)', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('Return Distribution', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# 3. 모델 성능 비교 (Regression)
ax3 = axes[0, 2]
models_reg = ['RF', 'XGB']
r2_scores = [rf_r2, xgb_r2]
colors_reg = ['blue', 'orange']
ax3.bar(models_reg, r2_scores, color=colors_reg, alpha=0.7)
ax3.set_ylabel('R² Score', fontsize=11)
ax3.set_title('Regression Model Comparison', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
for i, score in enumerate(r2_scores):
    ax3.text(i, score + 0.01, f'{score:.3f}', ha='center', fontweight='bold')

# 4. Confusion Matrix (RF)
ax4 = axes[1, 0]
cm_rf = confusion_matrix(y_direction_test, y_direction_pred_rf, labels=[-1, 0, 1])
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=['하락', '횡보', '상승'],
            yticklabels=['하락', '횡보', '상승'])
ax4.set_xlabel('Predicted', fontsize=11)
ax4.set_ylabel('Actual', fontsize=11)
ax4.set_title('Confusion Matrix (RF)', fontsize=12, fontweight='bold')

# 5. Confusion Matrix (XGB)
ax5 = axes[1, 1]
cm_xgb = confusion_matrix(y_direction_test, y_direction_pred_xgb, labels=[-1, 0, 1])
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges', ax=ax5,
            xticklabels=['하락', '횡보', '상승'],
            yticklabels=['하락', '횡보', '상승'])
ax5.set_xlabel('Predicted', fontsize=11)
ax5.set_ylabel('Actual', fontsize=11)
ax5.set_title('Confusion Matrix (XGB)', fontsize=12, fontweight='bold')

# 6. 방향 예측 정확도 비교
ax6 = axes[1, 2]
models_clf = ['RF', 'XGB']
accuracies = [rf_acc, xgb_acc]
ax6.bar(models_clf, accuracies, color=['blue', 'orange'], alpha=0.7)
ax6.set_ylabel('Accuracy', fontsize=11)
ax6.set_title('Classification Accuracy', fontsize=12, fontweight='bold')
ax6.set_ylim([0, 1])
ax6.grid(True, alpha=0.3, axis='y')
for i, acc in enumerate(accuracies):
    ax6.text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontweight='bold')

# 7. 복원된 가격 예측
ax7 = axes[2, 0]
test_dates = y_return_test.index
ax7.plot(test_dates, actual_prices, label='Actual', linewidth=2, color='blue')
ax7.plot(test_dates, predicted_prices, label='Predicted (XGB)',
         linewidth=2, color='red', alpha=0.7, linestyle='--')
ax7.set_xlabel('Date', fontsize=11)
ax7.set_ylabel('BTC Price ($)', fontsize=11)
ax7.set_title('Price Prediction (Reconstructed from Returns)', fontsize=12, fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)
ax7.tick_params(axis='x', rotation=45)

# 8. 트레이딩 시뮬레이션
ax8 = axes[2, 1]
strategy_equity = [initial_capital]
hold_equity = [initial_capital]
current_capital = initial_capital
hold_capital = initial_capital

for actual_return, pred_dir in zip(y_return_test.values, y_direction_pred):
    if pred_dir == 1:  # 상승 예측
        current_capital *= (1 + actual_return / 100)
    # 하락 예측 시 현금 보유 (변화 없음)
    hold_capital *= (1 + actual_return / 100)

    strategy_equity.append(current_capital)
    hold_equity.append(hold_capital)

ax8.plot(range(len(strategy_equity)), strategy_equity, label='Strategy', linewidth=2)
ax8.plot(range(len(hold_equity)), hold_equity, label='Buy & Hold', linewidth=2, linestyle='--')
ax8.set_xlabel('Trading Days', fontsize=11)
ax8.set_ylabel('Portfolio Value ($)', fontsize=11)
ax8.set_title('Trading Strategy Performance', fontsize=12, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 9. Feature Importance (Classification)
ax9 = axes[2, 2]
feature_importance_clf = pd.DataFrame({
    'feature': use_features,
    'importance': xgb_clf.feature_importances_
}).sort_values('importance', ascending=False).head(10)

ax9.barh(range(len(feature_importance_clf)), feature_importance_clf['importance'], alpha=0.7)
ax9.set_yticks(range(len(feature_importance_clf)))
ax9.set_yticklabels(feature_importance_clf['feature'], fontsize=9)
ax9.set_xlabel('Importance', fontsize=11)
ax9.set_title('Feature Importance (XGB Classifier)', fontsize=12, fontweight='bold')
ax9.invert_yaxis()
ax9.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('return_and_direction_prediction.png', dpi=300, bbox_inches='tight')
print("✓ return_and_direction_prediction.png")

plt.close()

# ===== 9. 결과 저장 =====
print("\n9. 결과 저장")
print("-" * 70)

# Regression 결과
regression_df = pd.DataFrame(regression_results)
regression_df.to_csv('return_prediction_results.csv', index=False)
print("✓ return_prediction_results.csv")

# Classification 결과
classification_df = pd.DataFrame(classification_results)
classification_df.to_csv('direction_prediction_results.csv', index=False)
print("✓ direction_prediction_results.csv")

# 예측 결과
predictions_df = pd.DataFrame({
    'date': y_return_test.index,
    'actual_return': y_return_test.values,
    'predicted_return': y_return_pred,
    'actual_direction': y_direction_test.values,
    'predicted_direction': y_direction_pred,
    'actual_price': actual_prices,
    'predicted_price': predicted_prices
})
predictions_df.to_csv('predictions_return_direction.csv', index=False)
print("✓ predictions_return_direction.csv")

print("\n" + "=" * 70)
print("변화율 & 방향 예측 완료!")
print("=" * 70)

print("\n📊 최종 요약:")
print("-" * 70)
print(f"변화율 예측 (Regression):")
print(f"  최고 모델: {best_reg}")
print(f"  R²: {max(xgb_r2, rf_r2):.4f}")
print(f"  RMSE: {min(xgb_rmse, rf_rmse):.4f}%")

print(f"\n방향 예측 (Classification):")
print(f"  최고 모델: {best_clf}")
print(f"  Accuracy: {max(xgb_acc, rf_acc):.4f} ({max(xgb_acc, rf_acc)*100:.1f}%)")
print(f"  F1-Score: {max(xgb_f1, rf_f1):.4f}")

print(f"\n트레이딩 성과:")
print(f"  전략 수익률: {total_return:+.2f}%")
print(f"  Buy & Hold: {buy_hold_return:+.2f}%")
print(f"  초과 수익: {total_return - buy_hold_return:+.2f}%p")

print("=" * 70)
