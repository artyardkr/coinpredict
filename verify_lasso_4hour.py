#!/usr/bin/env python3
"""
Lasso 4시간 예측 결과 철저 검증
R² 0.9933이 진짜인지 확인
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("Lasso 4시간 예측 철저 검증")
print("="*80)

# ========================================
# 1. 데이터 로드
# ========================================
print("\n[1/10] 데이터 로드...")
df = pd.read_csv('integrated_data_4hour.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Close 처리
if 'Close_x' in df.columns:
    df['Close'] = df['Close_x']
    df['Volume'] = df['Volume_x']

# Target 생성
df['target'] = df['Close'].shift(-1)
df = df[:-1].copy()

print(f"✅ 데이터: {len(df)} samples")
print(f"   기간: {df['Date'].min()} ~ {df['Date'].max()}")

# ========================================
# 2. Feature 준비
# ========================================
print("\n[2/10] Feature 준비...")

exclude_cols = [
    'Date', 'Close', 'High', 'Low', 'Open', 'target',
    'Close_x', 'High_x', 'Low_x', 'Open_x', 'Volume_x',
    'Close_y', 'High_y', 'Low_y', 'Open_y', 'Volume_y',
    'cumulative_return', 'bc_market_price', 'bc_market_cap',
]

ema_sma_cols = [col for col in df.columns
                if ('EMA' in col or 'SMA' in col) and 'close' in col.lower()]
exclude_cols.extend(ema_sma_cols)

bb_cols = [col for col in df.columns if col.startswith('BB_')]
exclude_cols.extend(bb_cols)

feature_cols = [col for col in df.columns
                if col not in exclude_cols and col in df.columns]

for col in feature_cols:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

print(f"✅ Features: {len(feature_cols)}개")

# ========================================
# 3. Train/Test Split
# ========================================
print("\n[3/10] Train/Test Split (70/30)...")

split_idx = int(len(df) * 0.7)
split_date = df['Date'].iloc[split_idx]

train_mask = df['Date'] < split_date
test_mask = df['Date'] >= split_date

X_train = df[train_mask][feature_cols].values
X_test = df[test_mask][feature_cols].values
y_train = df[train_mask]['target'].values
y_test = df[test_mask]['target'].values

dates_test = df[test_mask]['Date'].values
close_test = df[test_mask]['Close'].values

print(f"✅ Train: {len(X_train)} samples")
print(f"✅ Test: {len(X_test)} samples")

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========================================
# 4. Lasso 학습
# ========================================
print("\n[4/10] Lasso 모델 학습...")

model = Lasso(alpha=1.0, max_iter=10000, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

print("✅ 학습 완료")

# ========================================
# 5. 모든 성능 지표 계산
# ========================================
print("\n[5/10] 성능 지표 계산...")

# Train
r2_train = r2_score(y_train, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
mae_train = mean_absolute_error(y_train, y_pred_train)
mape_train = mean_absolute_percentage_error(y_train, y_pred_train) * 100

# Test
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
mape_test = mean_absolute_percentage_error(y_test, y_pred_test) * 100

# 추가 지표
errors_test = y_test - y_pred_test
relative_errors = (errors_test / y_test) * 100

print(f"""
📊 Train 성능:
   R²: {r2_train:.6f}
   RMSE: ${rmse_train:.2f}
   MAE: ${mae_train:.2f}
   MAPE: {mape_train:.2f}%

📊 Test 성능:
   R²: {r2_test:.6f} {'🔥 너무 높음!' if r2_test > 0.95 else ''}
   RMSE: ${rmse_test:.2f}
   MAE: ${mae_test:.2f}
   MAPE: {mape_test:.2f}%

   평균 오차: ${errors_test.mean():.2f}
   오차 std: ${errors_test.std():.2f}
   최대 오차: ${abs(errors_test).max():.2f}

   평균 상대오차: {relative_errors.mean():.2f}%
   상대오차 std: {relative_errors.std():.2f}%
""")

# ========================================
# 6. Data Leakage 체크
# ========================================
print("\n[6/10] Data Leakage 체크...")

# 예측값과 실제값의 상관관계
corr_pred_actual = np.corrcoef(y_pred_test, y_test)[0, 1]
print(f"   예측 vs 실제 상관관계: {corr_pred_actual:.6f}")

# 예측값과 현재 가격의 상관관계
corr_pred_current = np.corrcoef(y_pred_test, close_test)[0, 1]
print(f"   예측 vs 현재가격 상관관계: {corr_pred_current:.6f}")

# 실제값과 현재 가격의 상관관계
corr_actual_current = np.corrcoef(y_test, close_test)[0, 1]
print(f"   실제 vs 현재가격 상관관계: {corr_actual_current:.6f}")

if corr_actual_current > 0.99:
    print(f"   ⚠️ 실제값과 현재가격 상관관계 {corr_actual_current:.4f} = 4시간 변화가 매우 작음!")
    print(f"   → 높은 R²는 '가격 변화가 작아서'일 수 있음")

# ========================================
# 7. 예측 변화량 분석
# ========================================
print("\n[7/10] 가격 변화량 분석...")

# 실제 변화량
actual_changes = y_test - close_test
actual_pct_changes = ((y_test - close_test) / close_test) * 100

# 예측 변화량
pred_changes = y_pred_test - close_test
pred_pct_changes = ((y_pred_test - close_test) / close_test) * 100

print(f"""
실제 4시간 변화:
   평균: ${actual_changes.mean():.2f} ({actual_pct_changes.mean():.3f}%)
   std: ${actual_changes.std():.2f} ({actual_pct_changes.std():.3f}%)
   범위: ${actual_changes.min():.2f} ~ ${actual_changes.max():.2f}

예측 4시간 변화:
   평균: ${pred_changes.mean():.2f} ({pred_pct_changes.mean():.3f}%)
   std: ${pred_changes.std():.2f} ({pred_pct_changes.std():.3f}%)
   범위: ${pred_changes.min():.2f} ~ ${pred_changes.max():.2f}
""")

# ========================================
# 8. 방향 정확도 및 상세 분석
# ========================================
print("\n[8/10] 방향 예측 분석...")

actual_direction = (y_test > close_test).astype(int)
pred_direction = (y_pred_test > close_test).astype(int)
direction_correct = (actual_direction == pred_direction)

direction_acc = direction_correct.mean()
print(f"   방향 정확도: {direction_acc:.2%}")

# 상승/하락별 성능
up_mask = actual_direction == 1
down_mask = actual_direction == 0

if up_mask.sum() > 0:
    up_correct = direction_correct[up_mask].mean()
    print(f"   상승 예측 정확도: {up_correct:.2%} ({up_mask.sum()}개)")

if down_mask.sum() > 0:
    down_correct = direction_correct[down_mask].mean()
    print(f"   하락 예측 정확도: {down_correct:.2%} ({down_mask.sum()}개)")

# ========================================
# 9. Lasso 계수 분석
# ========================================
print("\n[9/10] Lasso 계수 분석...")

coef_df = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model.coef_
})
coef_df['Abs_Coef'] = np.abs(coef_df['Coefficient'])
coef_df = coef_df.sort_values('Abs_Coef', ascending=False)

non_zero = (coef_df['Abs_Coef'] > 0).sum()
print(f"   비영 계수: {non_zero} / {len(feature_cols)} ({non_zero/len(feature_cols)*100:.1f}%)")
print(f"   제거된 변수: {len(feature_cols) - non_zero}개")

print(f"\n   Top 10 중요 변수:")
for i, row in coef_df.head(10).iterrows():
    print(f"      {row['Feature']}: {row['Coefficient']:.2f}")

# Volume이 높으면 의심
if 'Volume' in coef_df.head(3)['Feature'].values:
    print(f"   ⚠️ Volume이 Top 3 - Data Leakage 가능성!")

# ========================================
# 10. 시각화
# ========================================
print("\n[10/10] 시각화 생성...")

fig = plt.figure(figsize=(20, 16))

# 1. Actual vs Predicted Scatter
ax1 = plt.subplot(3, 3, 1)
ax1.scatter(y_test, y_pred_test, alpha=0.3, s=10)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2)
ax1.set_xlabel('Actual 4h Later ($)', fontweight='bold')
ax1.set_ylabel('Predicted 4h Later ($)', fontweight='bold')
ax1.set_title(f'Actual vs Predicted (R²={r2_test:.4f})', fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2. Error Distribution
ax2 = plt.subplot(3, 3, 2)
ax2.hist(errors_test, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax2.axvline(0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Prediction Error ($)', fontweight='bold')
ax2.set_ylabel('Frequency', fontweight='bold')
ax2.set_title(f'Error Distribution (Mean=${errors_test.mean():.2f})', fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Relative Error Distribution
ax3 = plt.subplot(3, 3, 3)
ax3.hist(relative_errors, bins=50, color='coral', alpha=0.7, edgecolor='black')
ax3.axvline(0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('Relative Error (%)', fontweight='bold')
ax3.set_ylabel('Frequency', fontweight='bold')
ax3.set_title(f'Relative Error (Mean={relative_errors.mean():.2f}%)', fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Time Series (전체)
ax4 = plt.subplot(3, 3, 4)
ax4.plot(dates_test, y_test, label='Actual', linewidth=2, color='black', alpha=0.8)
ax4.plot(dates_test, y_pred_test, label='Predicted', linewidth=2, color='red', alpha=0.6, linestyle='--')
ax4.plot(dates_test, close_test, label='Current', linewidth=1, color='gray', alpha=0.5, linestyle=':')
ax4.set_xlabel('Date', fontweight='bold')
ax4.set_ylabel('Price ($)', fontweight='bold')
ax4.set_title('Time Series: All Test Period', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='x', rotation=45)

# 5. Time Series (처음 200개)
ax5 = plt.subplot(3, 3, 5)
n_show = min(200, len(dates_test))
ax5.plot(dates_test[:n_show], y_test[:n_show], label='Actual', linewidth=2, color='black', alpha=0.8)
ax5.plot(dates_test[:n_show], y_pred_test[:n_show], label='Predicted', linewidth=2, color='red', alpha=0.6, linestyle='--')
ax5.plot(dates_test[:n_show], close_test[:n_show], label='Current', linewidth=1, color='gray', alpha=0.5, linestyle=':')
ax5.set_xlabel('Date', fontweight='bold')
ax5.set_ylabel('Price ($)', fontweight='bold')
ax5.set_title(f'Time Series: First {n_show} samples (확대)', fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.tick_params(axis='x', rotation=45)

# 6. 실제 변화 vs 예측 변화
ax6 = plt.subplot(3, 3, 6)
ax6.scatter(actual_changes, pred_changes, alpha=0.3, s=10)
ax6.plot([actual_changes.min(), actual_changes.max()],
         [actual_changes.min(), actual_changes.max()], 'r--', linewidth=2)
ax6.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax6.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax6.set_xlabel('Actual Change ($)', fontweight='bold')
ax6.set_ylabel('Predicted Change ($)', fontweight='bold')
ax6.set_title('4h Change: Actual vs Predicted', fontweight='bold')
ax6.grid(True, alpha=0.3)

# 7. Error over time
ax7 = plt.subplot(3, 3, 7)
ax7.plot(dates_test, errors_test, linewidth=1, alpha=0.7)
ax7.axhline(0, color='red', linestyle='--', linewidth=2)
ax7.set_xlabel('Date', fontweight='bold')
ax7.set_ylabel('Error ($)', fontweight='bold')
ax7.set_title('Prediction Error Over Time', fontweight='bold')
ax7.grid(True, alpha=0.3)
ax7.tick_params(axis='x', rotation=45)

# 8. Top 10 Coefficients
ax8 = plt.subplot(3, 3, 8)
top10 = coef_df.head(10)
colors = ['green' if c > 0 else 'red' for c in top10['Coefficient']]
ax8.barh(range(len(top10)), top10['Coefficient'], color=colors, alpha=0.7)
ax8.set_yticks(range(len(top10)))
ax8.set_yticklabels(top10['Feature'], fontsize=9)
ax8.set_xlabel('Coefficient', fontweight='bold')
ax8.set_title('Top 10 Features (Lasso)', fontweight='bold')
ax8.axvline(0, color='black', linewidth=0.8)
ax8.grid(True, alpha=0.3, axis='x')
ax8.invert_yaxis()

# 9. Residual Plot
ax9 = plt.subplot(3, 3, 9)
ax9.scatter(y_pred_test, errors_test, alpha=0.3, s=10)
ax9.axhline(0, color='red', linestyle='--', linewidth=2)
ax9.set_xlabel('Predicted ($)', fontweight='bold')
ax9.set_ylabel('Residual ($)', fontweight='bold')
ax9.set_title('Residual Plot', fontweight='bold')
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lasso_4hour_verification.png', dpi=300, bbox_inches='tight')
print("✅ 저장: lasso_4hour_verification.png")

# ========================================
# 최종 진단
# ========================================
print("\n" + "="*80)
print("최종 진단")
print("="*80)

print(f"""
1. R² {r2_test:.4f}은 진짜인가?
   {'✅ YES - 하지만...' if r2_test > 0.99 else '❌ NO'}

2. 이유:
   - 4시간 가격 변화가 매우 작음 (평균 {actual_pct_changes.mean():.3f}%)
   - 현재 가격과 4시간 후 가격 상관관계: {corr_actual_current:.4f}
   - 즉, "현재 가격 ≈ 4시간 후 가격"이 거의 맞음

3. 모델이 잘 예측하는가?
   - 가격 자체: {'✅ YES (MAPE ' + f'{mape_test:.2f}%)' if mape_test < 2 else '❌ NO'}
   - 방향: {'✅ YES' if direction_acc > 0.6 else '⚠️ NO (' + f'{direction_acc:.1%})'}
   - 변화량: {'✅ YES' if abs(pred_pct_changes.mean() - actual_pct_changes.mean()) < 0.1 else '❌ NO'}

4. Data Leakage 가능성:
   {'⚠️ 낮음 - Volume이 Top 변수에 없음' if 'Volume' not in coef_df.head(3)['Feature'].values else '❌ 높음 - Volume이 Top 변수!'}

5. 실전 활용 가능성:
   {'✅ 가격 예측: 가능 (RMSE $' + f'{rmse_test:.0f})' if rmse_test < 2000 else '❌ 가격 예측: 불가능'}
   {'✅ 방향 예측: 가능' if direction_acc > 0.6 else '❌ 방향 예측: 불가능 (동전 던지기 수준)'}

6. 결론:
   {
   '✅ 모델은 정상 작동 - 4시간은 변화가 작아서 예측이 쉬움'
   if r2_test > 0.99 and mape_test < 2 and direction_acc < 0.6
   else '⚠️ 추가 검증 필요'
   }

   BUT: 방향 예측은 {direction_acc:.1%} = 실전 트레이딩 어려움

7. 권장 사항:
   - 가격 예측용으로만 사용 (변화량이 작음)
   - 방향 예측은 별도 분류 모델 필요
   - 백테스팅으로 실제 수익성 확인 필요
""")

# 샘플 예측 출력
print("\n" + "="*80)
print("샘플 예측 (처음 10개)")
print("="*80)

sample_df = pd.DataFrame({
    'Date': dates_test[:10],
    'Current': close_test[:10],
    'Actual 4h': y_test[:10],
    'Predicted 4h': y_pred_test[:10],
    'Error': errors_test[:10],
    'Actual Chg%': actual_pct_changes[:10],
    'Pred Chg%': pred_pct_changes[:10]
})

for col in ['Current', 'Actual 4h', 'Predicted 4h', 'Error']:
    sample_df[col] = sample_df[col].round(2)
for col in ['Actual Chg%', 'Pred Chg%']:
    sample_df[col] = sample_df[col].round(3)

print(sample_df.to_string(index=False))

print("\n" + "="*80)
print("검증 완료!")
print("="*80)
