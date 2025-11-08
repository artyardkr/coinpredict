import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("방향 + 변화율 예측 (Direction + Magnitude Prediction)")
print("=" * 70)

# ===== 1. 데이터 로드 =====
print("\n1. 데이터 로드 및 전처리")
print("-" * 70)

df = pd.read_csv('integrated_data_full.csv', index_col=0, parse_dates=True)
print(f"전체 데이터: {df.shape} ({df.index[0].date()} ~ {df.index[-1].date()})")

# 제거할 특성
exclude_features = [
    'Close', 'High', 'Low', 'Open', 'cumulative_return',
    'bc_market_price', 'bc_market_cap',
    'EMA5_close', 'EMA10_close', 'EMA14_close', 'EMA20_close', 'EMA30_close', 'EMA100_close',
    'SMA5_close', 'SMA10_close', 'SMA20_close', 'SMA30_close',
    'BB_high', 'BB_mid', 'BB_low',
]

# 타겟 1: 일별 수익률 (%)
df['target_return'] = (df['Close'].shift(-1) / df['Close'] - 1) * 100

# 타겟 2: 방향 (상승/중립/하락)
# 0.5% 이상 상승 = 2 (Up)
# -0.5% 이상 하락 = 0 (Down)
# 그 사이 = 1 (Neutral)
def classify_direction(return_pct, threshold=0.5):
    if return_pct > threshold:
        return 2  # 상승 (Up)
    elif return_pct < -threshold:
        return 0  # 하락 (Down)
    else:
        return 1  # 중립 (Neutral)

df['target_direction'] = df['target_return'].apply(classify_direction)

print(f"\n타겟 분포:")
print(f"  하락 (Down):    {(df['target_direction'] == 0).sum()}개 ({(df['target_direction'] == 0).sum()/len(df)*100:.1f}%)")
print(f"  중립 (Neutral): {(df['target_direction'] == 1).sum()}개 ({(df['target_direction'] == 1).sum()/len(df)*100:.1f}%)")
print(f"  상승 (Up):      {(df['target_direction'] == 2).sum()}개 ({(df['target_direction'] == 2).sum()/len(df)*100:.1f}%)")

# Feature Engineering
if 'bc_miners_revenue' in df.columns:
    df['miners_revenue_normalized'] = df['bc_miners_revenue'] / df['Close']

for col in ['RSI', 'MACD', 'ATR', 'OBV', 'ADX', 'CCI', 'MFI']:
    if col in df.columns:
        df[f'{col}_change'] = df[col].pct_change() * 100

for col in ['DGS10', 'CPIAUCSL', 'UNRATE', 'M2SL']:
    if col in df.columns and df[col].notna().sum() > 0:
        df[f'{col}_change'] = df[col].pct_change() * 100

df_clean = df.dropna(subset=['target_return', 'target_direction']).copy()

# 특성 선택
all_features = [col for col in df_clean.columns
                if col not in exclude_features
                and col not in ['target_return', 'target_direction']
                and not col.endswith('_change')]

change_features = [col for col in df_clean.columns if col.endswith('_change')]
all_features.extend(change_features)

if 'miners_revenue_normalized' in df_clean.columns:
    all_features.append('miners_revenue_normalized')
    if 'bc_miners_revenue' in all_features:
        all_features.remove('bc_miners_revenue')

X = df_clean[all_features].copy()
y_return = df_clean['target_return'].copy()
y_direction = df_clean['target_direction'].copy()

mask = X.notna().all(axis=1) & y_return.notna() & y_direction.notna()
X = X[mask]
y_return = y_return[mask]
y_direction = y_direction[mask]

print(f"\n최종 데이터:")
print(f"  특성 수: {len(all_features)}개")
print(f"  샘플 수: {len(X)}개")

# ===== 2. 데이터 분할 =====
print("\n2. 데이터 분할")
print("-" * 70)

split_idx = int(len(X) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_return_train = y_return.iloc[:split_idx]
y_return_test = y_return.iloc[split_idx:]
y_direction_train = y_direction.iloc[:split_idx]
y_direction_test = y_direction.iloc[split_idx:]

print(f"Train: {len(X_train)}개 ({X_train.index[0].date()} ~ {X_train.index[-1].date()})")
print(f"Test:  {len(X_test)}개 ({X_test.index[0].date()} ~ {X_test.index[-1].date()})")

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== 3. 모델 1: 방향 예측 (Classification) =====
print("\n" + "=" * 70)
print("3. 모델 1: 방향 예측 (Classification)")
print("=" * 70)

# Random Forest Classifier
print("\n[Random Forest Classifier]")
print("-" * 70)
rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

rf_clf.fit(X_train_scaled, y_direction_train)

# 예측
y_direction_pred_rf = rf_clf.predict(X_test_scaled)
y_direction_proba_rf = rf_clf.predict_proba(X_test_scaled)  # 확률

# 정확도
rf_clf_accuracy = accuracy_score(y_direction_test, y_direction_pred_rf)
print(f"정확도: {rf_clf_accuracy:.4f} ({rf_clf_accuracy*100:.2f}%)")

# 클래스별 정확도
print(f"\n클래스별 성능:")
report = classification_report(y_direction_test, y_direction_pred_rf,
                               target_names=['Down', 'Neutral', 'Up'],
                               output_dict=True)
for label in ['Down', 'Neutral', 'Up']:
    print(f"  {label:8s}: Precision={report[label]['precision']:.3f}, "
          f"Recall={report[label]['recall']:.3f}, "
          f"F1={report[label]['f1-score']:.3f}")

# XGBoost Classifier
print("\n[XGBoost Classifier]")
print("-" * 70)
xgb_clf = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_clf.fit(X_train_scaled, y_direction_train)

y_direction_pred_xgb = xgb_clf.predict(X_test_scaled)
y_direction_proba_xgb = xgb_clf.predict_proba(X_test_scaled)

xgb_clf_accuracy = accuracy_score(y_direction_test, y_direction_pred_xgb)
print(f"정확도: {xgb_clf_accuracy:.4f} ({xgb_clf_accuracy*100:.2f}%)")

# ===== 4. 모델 2: 변화율 예측 (Regression) =====
print("\n" + "=" * 70)
print("4. 모델 2: 변화율 예측 (Regression)")
print("=" * 70)

# Random Forest Regressor
print("\n[Random Forest Regressor]")
print("-" * 70)
rf_reg = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

rf_reg.fit(X_train_scaled, y_return_train)

y_return_pred_rf = rf_reg.predict(X_test_scaled)
rf_reg_r2 = r2_score(y_return_test, y_return_pred_rf)
rf_reg_rmse = np.sqrt(mean_squared_error(y_return_test, y_return_pred_rf))

print(f"R²: {rf_reg_r2:.4f} | RMSE: {rf_reg_rmse:.3f}%")

# XGBoost Regressor
print("\n[XGBoost Regressor]")
print("-" * 70)
xgb_reg = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_reg.fit(X_train_scaled, y_return_train)

y_return_pred_xgb = xgb_reg.predict(X_test_scaled)
xgb_reg_r2 = r2_score(y_return_test, y_return_pred_xgb)
xgb_reg_rmse = np.sqrt(mean_squared_error(y_return_test, y_return_pred_xgb))

print(f"R²: {xgb_reg_r2:.4f} | RMSE: {xgb_reg_rmse:.3f}%")

# ===== 5. 결합 전략 백테스팅 =====
print("\n" + "=" * 70)
print("5. 결합 전략 백테스팅")
print("=" * 70)

def backtest_combined(direction_pred, direction_proba, magnitude_pred,
                     actual_returns, prices, confidence_threshold=0.5):
    """
    방향 + 변화율 결합 전략

    전략:
    1. 방향 예측 확신도 > threshold일 때만 거래
    2. 예측 변화율 크기에 따라 포지션 크기 조절
    """
    initial_capital = 10000
    capital = initial_capital
    position = None
    entry_price = None
    portfolio_values = [initial_capital]
    trades = []

    TRANSACTION_COST = 0.01

    for i in range(len(direction_pred) - 1):
        current_price = prices[i]
        next_price = prices[i + 1]

        pred_dir = direction_pred[i]
        pred_mag = magnitude_pred[i]

        # 확신도 계산 (최대 확률)
        confidence = np.max(direction_proba[i])

        # 확신도가 높을 때만 거래
        if confidence > confidence_threshold:
            if pred_dir == 2:  # 상승 예측 (Up)
                if position != 'long':
                    # 롱 진입
                    position = 'long'
                    entry_price = current_price
                    capital -= capital * TRANSACTION_COST
                    trades.append(('LONG', current_price, confidence, pred_mag))

                # 포지션 유지
                current_value = capital * (next_price / entry_price)

            elif pred_dir == 0:  # 하락 예측 (Down)
                if position == 'long':
                    # 포지션 청산
                    capital = capital * (current_price / entry_price)
                    capital -= capital * TRANSACTION_COST
                    position = None
                    trades.append(('CLOSE', current_price, confidence, pred_mag))

                # 현금 보유
                current_value = capital

            else:  # 중립 (Neutral, pred_dir == 1)
                if position == 'long':
                    capital = capital * (current_price / entry_price)
                    capital -= capital * TRANSACTION_COST
                    position = None

                current_value = capital
        else:
            # 확신도 낮으면 현금 보유
            if position == 'long':
                current_value = capital * (next_price / entry_price)
            else:
                current_value = capital

        portfolio_values.append(current_value)

    return np.array(portfolio_values), trades

# Test 기간 가격 데이터
test_dates = y_return_test.index
test_prices = df.loc[test_dates, 'Close'].values

# 전략 실행
print("\n전략 테스트:")
print("-" * 70)

# 1. Buy-and-Hold
bh_initial = 10000
bh_final = bh_initial * (test_prices[-1] / test_prices[0]) * (1 - 0.01)
bh_return = (bh_final / bh_initial - 1) * 100

print(f"Buy-and-Hold: ${bh_initial:,.0f} → ${bh_final:,.0f} ({bh_return:+.2f}%)")

# 2. 방향만 (RF)
portfolio_rf_dir, trades_rf_dir = backtest_combined(
    y_direction_pred_rf, y_direction_proba_rf, y_return_pred_rf,
    y_return_test.values, test_prices, confidence_threshold=0.4
)

rf_dir_return = (portfolio_rf_dir[-1] / portfolio_rf_dir[0] - 1) * 100
print(f"\n방향 예측 (RF, 확신도>40%): ${portfolio_rf_dir[0]:,.0f} → ${portfolio_rf_dir[-1]:,.0f} ({rf_dir_return:+.2f}%)")
print(f"  거래 횟수: {len(trades_rf_dir)}회")
print(f"  Buy-and-Hold 대비: {rf_dir_return - bh_return:+.2f}%p")

# 3. 방향만 (XGB)
portfolio_xgb_dir, trades_xgb_dir = backtest_combined(
    y_direction_pred_xgb, y_direction_proba_xgb, y_return_pred_xgb,
    y_return_test.values, test_prices, confidence_threshold=0.4
)

xgb_dir_return = (portfolio_xgb_dir[-1] / portfolio_xgb_dir[0] - 1) * 100
print(f"\n방향 예측 (XGB, 확신도>40%): ${portfolio_xgb_dir[0]:,.0f} → ${portfolio_xgb_dir[-1]:,.0f} ({xgb_dir_return:+.2f}%)")
print(f"  거래 횟수: {len(trades_xgb_dir)}회")
print(f"  Buy-and-Hold 대비: {xgb_dir_return - bh_return:+.2f}%p")

# 확신도 임계값별 테스트
print(f"\n확신도 임계값 최적화 (Random Forest):")
print("-" * 70)
thresholds = [0.35, 0.40, 0.45, 0.50, 0.55]
best_threshold = 0.4
best_return = rf_dir_return

for threshold in thresholds:
    portfolio, trades = backtest_combined(
        y_direction_pred_rf, y_direction_proba_rf, y_return_pred_rf,
        y_return_test.values, test_prices, confidence_threshold=threshold
    )
    ret = (portfolio[-1] / portfolio[0] - 1) * 100
    print(f"  확신도>{threshold*100:.0f}%: {ret:+7.2f}% (거래 {len(trades)}회)")

    if ret > best_return:
        best_return = ret
        best_threshold = threshold

print(f"\n최적 임계값: {best_threshold*100:.0f}% (수익률: {best_return:+.2f}%)")

# ===== 6. 시각화 =====
print("\n6. 시각화")
print("-" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Confusion Matrix (Random Forest)
from sklearn.metrics import confusion_matrix
ax1 = axes[0, 0]
cm = confusion_matrix(y_direction_test, y_direction_pred_rf)
im = ax1.imshow(cm, cmap='Blues')
ax1.set_xticks([0, 1, 2])
ax1.set_yticks([0, 1, 2])
ax1.set_xticklabels(['Down', 'Neutral', 'Up'])
ax1.set_yticklabels(['Down', 'Neutral', 'Up'])
ax1.set_xlabel('Predicted', fontsize=11)
ax1.set_ylabel('Actual', fontsize=11)
ax1.set_title(f'Direction Prediction (RF, Acc={rf_clf_accuracy:.1%})',
             fontsize=12, fontweight='bold')

# 숫자 표시
for i in range(3):
    for j in range(3):
        text = ax1.text(j, i, cm[i, j], ha="center", va="center", color="black")

plt.colorbar(im, ax=ax1)

# 2. 방향 예측 확률 분포
ax2 = axes[0, 1]
max_proba_rf = np.max(y_direction_proba_rf, axis=1)
ax2.hist(max_proba_rf, bins=30, alpha=0.7, edgecolor='black')
ax2.axvline(x=best_threshold, color='red', linestyle='--', linewidth=2,
           label=f'Best Threshold ({best_threshold*100:.0f}%)')
ax2.set_xlabel('Confidence (Max Probability)', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('Prediction Confidence Distribution', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# 3. 포트폴리오 가치 변화
ax3 = axes[1, 0]
dates = test_dates[:len(portfolio_rf_dir)]
bh_values = np.linspace(10000, bh_final, len(dates))
ax3.plot(dates, bh_values, label='Buy-and-Hold', linewidth=2, alpha=0.8)
ax3.plot(dates, portfolio_rf_dir, label=f'Direction (RF)', linewidth=2, alpha=0.8)
ax3.plot(dates, portfolio_xgb_dir, label=f'Direction (XGB)', linewidth=2, alpha=0.8)
ax3.set_xlabel('Date', fontsize=11)
ax3.set_ylabel('Portfolio Value ($)', fontsize=11)
ax3.set_title('Portfolio Performance', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

# 4. 변화율 예측 (Actual vs Predicted)
ax4 = axes[1, 1]
ax4.scatter(y_return_test, y_return_pred_rf, alpha=0.3, s=20)
ax4.plot([y_return_test.min(), y_return_test.max()],
        [y_return_test.min(), y_return_test.max()],
        'r--', linewidth=2, label='Perfect')
ax4.set_xlabel('Actual Return (%)', fontsize=11)
ax4.set_ylabel('Predicted Return (%)', fontsize=11)
ax4.set_title(f'Magnitude Prediction (RF, R²={rf_reg_r2:.4f})',
             fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('direction_magnitude_results.png', dpi=300, bbox_inches='tight')
print("✓ direction_magnitude_results.png")
plt.close()

# ===== 7. 결과 저장 =====
print("\n7. 결과 저장")
print("-" * 70)

results = pd.DataFrame({
    'Strategy': ['Buy-and-Hold', 'Direction (RF)', 'Direction (XGB)'],
    'Final_Value': [bh_final, portfolio_rf_dir[-1], portfolio_xgb_dir[-1]],
    'Return_Pct': [bh_return, rf_dir_return, xgb_dir_return],
    'Trades': [1, len(trades_rf_dir), len(trades_xgb_dir)],
    'Direction_Accuracy': [np.nan, rf_clf_accuracy, xgb_clf_accuracy],
    'Magnitude_R2': [np.nan, rf_reg_r2, xgb_reg_r2]
})

results.to_csv('direction_magnitude_results.csv', index=False)
print("✓ direction_magnitude_results.csv")

# ===== 8. 최종 평가 =====
print("\n" + "=" * 70)
print("8. 최종 평가")
print("=" * 70)

print(f"\n방향 예측 성능:")
print(f"  Random Forest:  {rf_clf_accuracy:.1%}")
print(f"  XGBoost:        {xgb_clf_accuracy:.1%}")

print(f"\n변화율 예측 성능:")
print(f"  Random Forest:  R² = {rf_reg_r2:.4f}")
print(f"  XGBoost:        R² = {xgb_reg_r2:.4f}")

print(f"\n백테스팅 결과:")
print(f"  Buy-and-Hold:    {bh_return:+.2f}%")
print(f"  Direction (RF):  {rf_dir_return:+.2f}% ({rf_dir_return - bh_return:+.2f}%p)")
print(f"  Direction (XGB): {xgb_dir_return:+.2f}% ({xgb_dir_return - bh_return:+.2f}%p)")

print(f"\n💡 결론:")
print("-" * 70)
if rf_dir_return > bh_return or xgb_dir_return > bh_return:
    best = 'RF' if rf_dir_return > xgb_dir_return else 'XGB'
    best_ret = max(rf_dir_return, xgb_dir_return)
    print(f"✅ {best} 방향 예측이 Buy-and-Hold보다 {best_ret - bh_return:+.2f}%p 높음")
    print(f"   → 방향 예측 + 확신도 필터링이 효과적!")
else:
    print(f"❌ 모든 전략이 Buy-and-Hold보다 낮음")
    print(f"   → 방향 예측 정확도 개선 필요 (현재 {max(rf_clf_accuracy, xgb_clf_accuracy):.1%})")
    print(f"   → 추가 개선 방안:")
    print(f"      1. 더 많은 온체인 데이터 수집")
    print(f"      2. 앙상블 모델 (여러 모델 투표)")
    print(f"      3. 딥러닝 모델 (LSTM, CNN-LSTM)")

print("\n" + "=" * 70)
print("방향 + 변화율 예측 완료!")
print("=" * 70)
