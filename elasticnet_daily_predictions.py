#!/usr/bin/env python3
"""
ElasticNet 일별 예측값 생성
1일 후 가격 예측 (step25 방식)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ElasticNet 일별 예측 (1일 후 가격)")
print("="*80)

# 데이터 로드
print("\n[1/5] 데이터 로드...")
df = pd.read_csv('integrated_data_full.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"✅ 데이터: {len(df)} rows, {len(df.columns)} columns")
print(f"   기간: {df['Date'].min()} ~ {df['Date'].max()}")

# Feature 준비 (step25와 동일)
print("\n[2/5] Feature 준비...")
exclude_cols = [
    'Date', 'Close', 'High', 'Low', 'Open', 'target',
    'cumulative_return', 'bc_market_price', 'bc_market_cap',
]

# Data Leakage 방지: EMA/SMA_close, Bollinger Bands 제외
ema_sma_cols = [col for col in df.columns
                if ('EMA' in col or 'SMA' in col) and 'close' in col.lower()]
exclude_cols.extend(ema_sma_cols)

bb_cols = [col for col in df.columns if col.startswith('BB_')]
exclude_cols.extend(bb_cols)

feature_cols = [col for col in df.columns
                if col not in exclude_cols and col in df.columns]

print(f"✅ Features: {len(feature_cols)}개")
print(f"   제외된 컬럼: {len(exclude_cols)}개")

# NaN/Inf 처리
for col in feature_cols:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

# Target: 1일 후 종가
print("\n[3/5] Target 생성 (1일 후 종가)...")
df['target'] = df['Close'].shift(-1)
df = df[:-1].copy()

print(f"✅ 데이터: {len(df)} rows (마지막 1개 제거)")

# Train/Test Split (70/30)
print("\n[4/5] Train/Test Split (70/30)...")
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

print(f"Train: {len(X_train)} samples ({df[train_mask]['Date'].min()} ~ {df[train_mask]['Date'].max()})")
print(f"Test: {len(X_test)} samples ({df[test_mask]['Date'].min()} ~ {df[test_mask]['Date'].max()})")
print(f"Split date: {split_date}")

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ElasticNet 학습
print("\n[5/5] ElasticNet 학습 및 예측...")
model = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000, random_state=42)
model.fit(X_train_scaled, y_train)

# 예측
pred_train = model.predict(X_train_scaled)
pred_test = model.predict(X_test_scaled)

# 평가
r2_train = r2_score(y_train, pred_train)
r2_test = r2_score(y_test, pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
mae_test = mean_absolute_error(y_test, pred_test)

# 변화율 계산
actual_change = ((y_test - close_test) / close_test) * 100
pred_change = ((pred_test - close_test) / close_test) * 100

# 방향 정확도
actual_dir = (actual_change > 0).astype(int)
pred_dir = (pred_change > 0).astype(int)
dir_acc = (actual_dir == pred_dir).mean()

print(f"""
✅ ElasticNet 성능:
   Train R²: {r2_train:.4f}
   Test R²: {r2_test:.4f}
   RMSE: ${rmse_test:,.2f}
   MAE: ${mae_test:,.2f}
   방향 정확도: {dir_acc:.2%}
""")

# 상세 결과 저장
print("\n결과 저장 중...")
results_df = pd.DataFrame({
    'Date': dates_test,
    'Current_Close': close_test,
    'Actual_Next_Close': y_test,
    'Predicted_Next_Close': pred_test,
    'Prediction_Error_$': y_test - pred_test,
    'Actual_Change_%': actual_change,
    'Predicted_Change_%': pred_change,
    'Error_%': actual_change - pred_change,
    'Actual_Direction': actual_dir,
    'Predicted_Direction': pred_dir,
    'Direction_Correct': actual_dir == pred_dir
})

results_df.to_csv('elasticnet_daily_predictions.csv', index=False)

print("✅ 저장 완료: elasticnet_daily_predictions.csv")

# 통계 요약
print("\n" + "="*80)
print("결과 요약")
print("="*80)
print(f"""
파일: elasticnet_daily_predictions.csv
샘플 수: {len(results_df)}
기간: {results_df['Date'].min()} ~ {results_df['Date'].max()}

성능:
  R²: {r2_test:.4f}
  RMSE: ${rmse_test:,.2f}
  MAE: ${mae_test:,.2f}

변화율:
  실제 평균: {actual_change.mean():.3f}%
  예측 평균: {pred_change.mean():.3f}%

방향 정확도: {dir_acc:.2%}
  맞춘 개수: {(actual_dir == pred_dir).sum()} / {len(actual_dir)}
""")

# 샘플 출력
print("\n처음 5개:")
print(results_df[['Date', 'Current_Close', 'Actual_Next_Close', 'Predicted_Next_Close',
                   'Actual_Change_%', 'Predicted_Change_%']].head(5).to_string(index=False))

print("\n마지막 5개:")
print(results_df[['Date', 'Current_Close', 'Actual_Next_Close', 'Predicted_Next_Close',
                   'Actual_Change_%', 'Predicted_Change_%']].tail(5).to_string(index=False))

print("\n" + "="*80)
print("완료!")
print("="*80)
