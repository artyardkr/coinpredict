#!/usr/bin/env python3
"""
4시간 변화율 예측 - 3가지 방법 비교

방법 1: 변화율 회귀 (Regression) - ElasticNet, Lasso, Ridge
방법 2: 방향 분류 (Binary Classification) - Random Forest, XGBoost
방법 3: Multi-class 분류 (구간 예측) - Random Forest, XGBoost

각 방법의 최적 모델로 백테스팅까지 진행
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                            accuracy_score, classification_report, confusion_matrix,
                            roc_auc_score, precision_recall_fscore_support)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # Mac
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

print("="*80)
print("4시간 변화율 예측 - 3가지 방법 종합 비교")
print("="*80)

# ========================================
# 1. 데이터 준비
# ========================================
print("\n[1/9] 데이터 로드 및 준비...")

df = pd.read_csv('integrated_data_4hour.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Close 처리
if 'Close_x' in df.columns:
    df['Close'] = df['Close_x']

# Feature 준비
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
# 2. Target 생성 (3가지)
# ========================================
print("\n[2/9] Target 생성 (3가지 방법)...")

# 4시간 후 가격
df['Close_4h_later'] = df['Close'].shift(-1)
df = df[:-1].copy()

# 방법 1: 변화율 (%)
df['target_pct_change'] = ((df['Close_4h_later'] - df['Close']) / df['Close']) * 100

# 방법 2: 방향 (Binary)
df['target_direction'] = (df['Close_4h_later'] > df['Close']).astype(int)

# 방법 3: Multi-class (5개 구간)
def classify_change(pct_change):
    if pct_change < -1.0:
        return 0  # 큰 하락
    elif pct_change < -0.3:
        return 1  # 작은 하락
    elif pct_change < 0.3:
        return 2  # 횡보
    elif pct_change < 1.0:
        return 3  # 작은 상승
    else:
        return 4  # 큰 상승

df['target_multiclass'] = df['target_pct_change'].apply(classify_change)

# 통계
print(f"""
변화율 통계:
   평균: {df['target_pct_change'].mean():.3f}%
   std: {df['target_pct_change'].std():.3f}%
   범위: {df['target_pct_change'].min():.2f}% ~ {df['target_pct_change'].max():.2f}%

방향 분포:
   하락 (0): {(df['target_direction']==0).sum()} ({(df['target_direction']==0).sum()/len(df)*100:.1f}%)
   상승 (1): {(df['target_direction']==1).sum()} ({(df['target_direction']==1).sum()/len(df)*100:.1f}%)

Multi-class 분포:
   큰 하락 (<-1%): {(df['target_multiclass']==0).sum()} ({(df['target_multiclass']==0).sum()/len(df)*100:.1f}%)
   작은 하락 (-1~-0.3%): {(df['target_multiclass']==1).sum()} ({(df['target_multiclass']==1).sum()/len(df)*100:.1f}%)
   횡보 (-0.3~0.3%): {(df['target_multiclass']==2).sum()} ({(df['target_multiclass']==2).sum()/len(df)*100:.1f}%)
   작은 상승 (0.3~1%): {(df['target_multiclass']==3).sum()} ({(df['target_multiclass']==3).sum()/len(df)*100:.1f}%)
   큰 상승 (>1%): {(df['target_multiclass']==4).sum()} ({(df['target_multiclass']==4).sum()/len(df)*100:.1f}%)
""")

# ========================================
# 3. Train/Test Split
# ========================================
print("\n[3/9] Train/Test Split (70/30)...")

split_idx = int(len(df) * 0.7)
split_date = df['Date'].iloc[split_idx]

train_mask = df['Date'] < split_date
test_mask = df['Date'] >= split_date

X_train = df[train_mask][feature_cols].values
X_test = df[test_mask][feature_cols].values

dates_test = df[test_mask]['Date'].values
close_test = df[test_mask]['Close'].values

print(f"Train: {len(X_train)} samples")
print(f"Test: {len(X_test)} samples")
print(f"Split date: {split_date}")

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========================================
# 방법 1: 변화율 회귀
# ========================================
print("\n" + "="*80)
print("방법 1: 변화율 회귀 (Regression)")
print("="*80)

y_train_reg = df[train_mask]['target_pct_change'].values
y_test_reg = df[test_mask]['target_pct_change'].values

# 모델 테스트
reg_models = {
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
    'Lasso': Lasso(alpha=0.1, max_iter=10000),
    'Ridge': Ridge(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10,
                                          min_samples_split=20,
                                          random_state=42, n_jobs=-1)
}

reg_results = []

for name, model in reg_models.items():
    model.fit(X_train_scaled, y_train_reg)
    pred_train = model.predict(X_train_scaled)
    pred_test = model.predict(X_test_scaled)

    r2_train = r2_score(y_train_reg, pred_train)
    r2_test = r2_score(y_test_reg, pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test_reg, pred_test))
    mae_test = mean_absolute_error(y_test_reg, pred_test)

    # 방향 정확도
    actual_dir = (y_test_reg > 0).astype(int)
    pred_dir = (pred_test > 0).astype(int)
    dir_acc = (actual_dir == pred_dir).mean()

    reg_results.append({
        'Model': name,
        'Train R²': r2_train,
        'Test R²': r2_test,
        'RMSE': rmse_test,
        'MAE': mae_test,
        'Direction Acc': dir_acc
    })

    print(f"{name}: R²={r2_test:.4f}, RMSE={rmse_test:.3f}%, Dir={dir_acc:.2%}")

reg_results_df = pd.DataFrame(reg_results).sort_values('Test R²', ascending=False)
best_reg_model = reg_models[reg_results_df.iloc[0]['Model']]
best_reg_name = reg_results_df.iloc[0]['Model']
best_reg_pred = best_reg_model.predict(X_test_scaled)

print(f"\n🏆 Best Regression: {best_reg_name}")
print(f"   R²: {reg_results_df.iloc[0]['Test R²']:.4f}")
print(f"   RMSE: {reg_results_df.iloc[0]['RMSE']:.3f}%")

# ========================================
# 방법 2: 방향 분류 (Binary)
# ========================================
print("\n" + "="*80)
print("방법 2: 방향 분류 (Binary Classification)")
print("="*80)

y_train_dir = df[train_mask]['target_direction'].values
y_test_dir = df[test_mask]['target_direction'].values

# 모델 테스트
clf_models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15,
                                           min_samples_split=20,
                                           random_state=42, n_jobs=-1,
                                           class_weight='balanced'),
    'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=7,
                                learning_rate=0.05, subsample=0.8,
                                random_state=42, n_jobs=-1)
}

clf_results = []

for name, model in clf_models.items():
    model.fit(X_train_scaled, y_train_dir)
    pred_train = model.predict(X_train_scaled)
    pred_test = model.predict(X_test_scaled)
    pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc_train = accuracy_score(y_train_dir, pred_train)
    acc_test = accuracy_score(y_test_dir, pred_test)
    auc_test = roc_auc_score(y_test_dir, pred_proba)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_dir, pred_test, average='binary'
    )

    clf_results.append({
        'Model': name,
        'Train Acc': acc_train,
        'Test Acc': acc_test,
        'AUC': auc_test,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    })

    print(f"{name}: Acc={acc_test:.2%}, AUC={auc_test:.4f}, F1={f1:.4f}")

clf_results_df = pd.DataFrame(clf_results).sort_values('Test Acc', ascending=False)
best_clf_model = clf_models[clf_results_df.iloc[0]['Model']]
best_clf_name = clf_results_df.iloc[0]['Model']
best_clf_pred = best_clf_model.predict(X_test_scaled)
best_clf_proba = best_clf_model.predict_proba(X_test_scaled)[:, 1]

print(f"\n🏆 Best Binary Classifier: {best_clf_name}")
print(f"   Accuracy: {clf_results_df.iloc[0]['Test Acc']:.2%}")
print(f"   AUC: {clf_results_df.iloc[0]['AUC']:.4f}")

# ========================================
# 방법 3: Multi-class 분류
# ========================================
print("\n" + "="*80)
print("방법 3: Multi-class 분류 (5개 구간)")
print("="*80)

y_train_multi = df[train_mask]['target_multiclass'].values
y_test_multi = df[test_mask]['target_multiclass'].values

# 모델 테스트
multi_models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15,
                                           min_samples_split=20,
                                           random_state=42, n_jobs=-1,
                                           class_weight='balanced'),
    'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=7,
                                learning_rate=0.05, subsample=0.8,
                                random_state=42, n_jobs=-1)
}

multi_results = []

for name, model in multi_models.items():
    model.fit(X_train_scaled, y_train_multi)
    pred_train = model.predict(X_train_scaled)
    pred_test = model.predict(X_test_scaled)

    acc_train = accuracy_score(y_train_multi, pred_train)
    acc_test = accuracy_score(y_test_multi, pred_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_multi, pred_test, average='weighted'
    )

    multi_results.append({
        'Model': name,
        'Train Acc': acc_train,
        'Test Acc': acc_test,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    })

    print(f"{name}: Acc={acc_test:.2%}, F1={f1:.4f}")

multi_results_df = pd.DataFrame(multi_results).sort_values('Test Acc', ascending=False)
best_multi_model = multi_models[multi_results_df.iloc[0]['Model']]
best_multi_name = multi_results_df.iloc[0]['Model']
best_multi_pred = best_multi_model.predict(X_test_scaled)

print(f"\n🏆 Best Multi-class Classifier: {best_multi_name}")
print(f"   Accuracy: {multi_results_df.iloc[0]['Test Acc']:.2%}")
print(f"   F1: {multi_results_df.iloc[0]['F1']:.4f}")

# ========================================
# 백테스팅 (3가지 방법 비교)
# ========================================
print("\n" + "="*80)
print("백테스팅 (초기 자본 $10,000)")
print("="*80)

initial_capital = 10000
test_close = df[test_mask]['Close'].values
test_close_4h = df[test_mask]['Close_4h_later'].values

# 방법 1: 회귀 (Threshold ±0.5%)
print("\n[방법 1] 변화율 회귀 + Threshold")
portfolio_reg = initial_capital
cash_reg = initial_capital
btc_reg = 0
positions_reg = []

for i in range(len(best_reg_pred)):
    pred_change = best_reg_pred[i]
    current_price = test_close[i]
    next_price = test_close_4h[i]

    if pred_change > 0.5:  # 매수
        if cash_reg > 0:
            btc_reg = cash_reg / current_price
            cash_reg = 0
            positions_reg.append('Long')
        else:
            positions_reg.append('Hold Long')
    elif pred_change < -0.5:  # 매도
        if btc_reg > 0:
            cash_reg = btc_reg * current_price
            btc_reg = 0
            positions_reg.append('Close')
        else:
            positions_reg.append('Hold Cash')
    else:  # 관망
        positions_reg.append('Hold')

    # 포트폴리오 가치
    portfolio_value = cash_reg + btc_reg * next_price
    portfolio_reg = portfolio_value

return_reg = (portfolio_reg - initial_capital) / initial_capital * 100

# 방법 2: 이진 분류 (확률 > 0.55)
print("\n[방법 2] 방향 분류 + 확률 Threshold")
portfolio_clf = initial_capital
cash_clf = initial_capital
btc_clf = 0

for i in range(len(best_clf_proba)):
    pred_proba = best_clf_proba[i]
    current_price = test_close[i]
    next_price = test_close_4h[i]

    if pred_proba > 0.55:  # 상승 확률 높음
        if cash_clf > 0:
            btc_clf = cash_clf / current_price
            cash_clf = 0
    elif pred_proba < 0.45:  # 하락 확률 높음
        if btc_clf > 0:
            cash_clf = btc_clf * current_price
            btc_clf = 0

    portfolio_value = cash_clf + btc_clf * next_price
    portfolio_clf = portfolio_value

return_clf = (portfolio_clf - initial_capital) / initial_capital * 100

# 방법 3: Multi-class (큰 상승/하락만)
print("\n[방법 3] Multi-class + 극단 신호만")
portfolio_multi = initial_capital
cash_multi = initial_capital
btc_multi = 0

for i in range(len(best_multi_pred)):
    pred_class = best_multi_pred[i]
    current_price = test_close[i]
    next_price = test_close_4h[i]

    if pred_class == 4:  # 큰 상승 (>1%)
        if cash_multi > 0:
            btc_multi = cash_multi / current_price
            cash_multi = 0
    elif pred_class == 0:  # 큰 하락 (<-1%)
        if btc_multi > 0:
            cash_multi = btc_multi * current_price
            btc_multi = 0

    portfolio_value = cash_multi + btc_multi * next_price
    portfolio_multi = portfolio_value

return_multi = (portfolio_multi - initial_capital) / initial_capital * 100

# Buy and Hold
buy_hold = (test_close_4h[-1] - test_close[0]) / test_close[0] * 100

print(f"""
백테스팅 결과:

방법 1 (변화율 회귀): {return_reg:+.2f}%
방법 2 (방향 분류): {return_clf:+.2f}%
방법 3 (Multi-class): {return_multi:+.2f}%
Buy-and-Hold: {buy_hold:+.2f}%

{'🏆 회귀가 최고!' if return_reg >= max(return_clf, return_multi, buy_hold) else ''}
{'🏆 이진분류가 최고!' if return_clf >= max(return_reg, return_multi, buy_hold) else ''}
{'🏆 Multi-class가 최고!' if return_multi >= max(return_reg, return_clf, buy_hold) else ''}
{'🏆 Buy-and-Hold가 최고!' if buy_hold >= max(return_reg, return_clf, return_multi) else ''}
""")

# ========================================
# 시각화
# ========================================
print("\n[9/9] 시각화 생성 중...")

fig = plt.figure(figsize=(20, 16))

# 1. 회귀 결과
ax1 = plt.subplot(4, 3, 1)
ax1.scatter(y_test_reg, best_reg_pred, alpha=0.3, s=10)
ax1.plot([y_test_reg.min(), y_test_reg.max()],
         [y_test_reg.min(), y_test_reg.max()], 'r--', linewidth=2)
ax1.set_xlabel('Actual Change (%)', fontweight='bold')
ax1.set_ylabel('Predicted Change (%)', fontweight='bold')
ax1.set_title(f'방법 1: {best_reg_name}\nR²={reg_results_df.iloc[0]["Test R²"]:.4f}',
             fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2. 이진분류 Confusion Matrix
ax2 = plt.subplot(4, 3, 2)
cm_clf = confusion_matrix(y_test_dir, best_clf_pred)
sns.heatmap(cm_clf, annot=True, fmt='d', cmap='Blues', ax=ax2)
ax2.set_xlabel('Predicted', fontweight='bold')
ax2.set_ylabel('Actual', fontweight='bold')
ax2.set_title(f'방법 2: {best_clf_name}\nAcc={clf_results_df.iloc[0]["Test Acc"]:.2%}',
             fontweight='bold')
ax2.set_xticklabels(['Down', 'Up'])
ax2.set_yticklabels(['Down', 'Up'])

# 3. Multi-class Confusion Matrix
ax3 = plt.subplot(4, 3, 3)
cm_multi = confusion_matrix(y_test_multi, best_multi_pred)
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Greens', ax=ax3, cbar=False)
ax3.set_xlabel('Predicted', fontweight='bold')
ax3.set_ylabel('Actual', fontweight='bold')
ax3.set_title(f'방법 3: {best_multi_name}\nAcc={multi_results_df.iloc[0]["Test Acc"]:.2%}',
             fontweight='bold')
ax3.set_xticklabels(['--', '-', '0', '+', '++'], rotation=0)
ax3.set_yticklabels(['--', '-', '0', '+', '++'], rotation=0)

# 4-6. 성능 비교
ax4 = plt.subplot(4, 3, 4)
methods = ['회귀\n(R²)', '이진분류\n(Acc)', 'Multi\n(Acc)']
scores = [reg_results_df.iloc[0]['Test R²'],
         clf_results_df.iloc[0]['Test Acc'],
         multi_results_df.iloc[0]['Test Acc']]
colors_bar = ['#2ecc71', '#3498db', '#e74c3c']
bars = ax4.bar(methods, scores, color=colors_bar, alpha=0.7)
ax4.set_ylabel('Score', fontweight='bold')
ax4.set_title('예측 성능 비교', fontweight='bold')
ax4.set_ylim([0, 1.0])
ax4.grid(True, alpha=0.3, axis='y')
for bar, score in zip(bars, scores):
    ax4.text(bar.get_x() + bar.get_width()/2, score,
            f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

# 5. 백테스팅 수익률
ax5 = plt.subplot(4, 3, 5)
methods_bt = ['회귀', '이진분류', 'Multi', 'Buy&Hold']
returns_bt = [return_reg, return_clf, return_multi, buy_hold]
colors_bt = ['green' if r > 0 else 'red' for r in returns_bt]
bars = ax5.bar(methods_bt, returns_bt, color=colors_bt, alpha=0.7)
ax5.set_ylabel('수익률 (%)', fontweight='bold')
ax5.set_title('백테스팅 수익률 비교', fontweight='bold')
ax5.axhline(0, color='black', linestyle='--', linewidth=1)
ax5.grid(True, alpha=0.3, axis='y')
for bar, ret in zip(bars, returns_bt):
    ax5.text(bar.get_x() + bar.get_width()/2, ret,
            f'{ret:+.1f}%', ha='center',
            va='bottom' if ret > 0 else 'top', fontweight='bold')

# 6. 회귀 Error Distribution
ax6 = plt.subplot(4, 3, 6)
errors_reg = y_test_reg - best_reg_pred
ax6.hist(errors_reg, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax6.axvline(0, color='red', linestyle='--', linewidth=2)
ax6.set_xlabel('Prediction Error (%)', fontweight='bold')
ax6.set_ylabel('Frequency', fontweight='bold')
ax6.set_title(f'회귀 오차 분포 (MAE={reg_results_df.iloc[0]["MAE"]:.3f}%)',
             fontweight='bold')
ax6.grid(True, alpha=0.3)

# 7-9. 시계열 (처음 200개)
n_show = min(200, len(dates_test))

# 회귀
ax7 = plt.subplot(4, 3, 7)
ax7.plot(dates_test[:n_show], y_test_reg[:n_show], label='Actual',
        linewidth=2, color='black', alpha=0.8)
ax7.plot(dates_test[:n_show], best_reg_pred[:n_show], label='Predicted',
        linewidth=2, color='red', alpha=0.6, linestyle='--')
ax7.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax7.set_xlabel('Date', fontweight='bold')
ax7.set_ylabel('Change (%)', fontweight='bold')
ax7.set_title('회귀: 변화율 예측', fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)
ax7.tick_params(axis='x', rotation=45)

# 이진분류
ax8 = plt.subplot(4, 3, 8)
ax8.plot(dates_test[:n_show], y_test_dir[:n_show], label='Actual',
        linewidth=2, color='black', alpha=0.8, marker='o', markersize=3)
ax8.plot(dates_test[:n_show], best_clf_pred[:n_show], label='Predicted',
        linewidth=2, color='blue', alpha=0.6, linestyle='--', marker='x', markersize=3)
ax8.set_xlabel('Date', fontweight='bold')
ax8.set_ylabel('Direction (0=Down, 1=Up)', fontweight='bold')
ax8.set_title('이진분류: 방향 예측', fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)
ax8.tick_params(axis='x', rotation=45)

# Multi-class
ax9 = plt.subplot(4, 3, 9)
ax9.plot(dates_test[:n_show], y_test_multi[:n_show], label='Actual',
        linewidth=2, color='black', alpha=0.8, marker='o', markersize=3)
ax9.plot(dates_test[:n_show], best_multi_pred[:n_show], label='Predicted',
        linewidth=2, color='green', alpha=0.6, linestyle='--', marker='x', markersize=3)
ax9.set_xlabel('Date', fontweight='bold')
ax9.set_ylabel('Class (0=--,1=-,2=0,3=+,4=++)', fontweight='bold')
ax9.set_title('Multi-class: 구간 예측', fontweight='bold')
ax9.legend()
ax9.grid(True, alpha=0.3)
ax9.tick_params(axis='x', rotation=45)

# 10. 결과 요약 테이블
ax10 = plt.subplot(4, 3, 10)
ax10.axis('off')
summary_text = f"""
【성능 요약】

방법 1 (변화율 회귀):
  모델: {best_reg_name}
  R²: {reg_results_df.iloc[0]['Test R²']:.4f}
  RMSE: {reg_results_df.iloc[0]['RMSE']:.3f}%
  방향 정확도: {reg_results_df.iloc[0]['Direction Acc']:.1%}
  백테스팅: {return_reg:+.2f}%

방법 2 (방향 분류):
  모델: {best_clf_name}
  정확도: {clf_results_df.iloc[0]['Test Acc']:.2%}
  AUC: {clf_results_df.iloc[0]['AUC']:.4f}
  F1: {clf_results_df.iloc[0]['F1']:.4f}
  백테스팅: {return_clf:+.2f}%

방법 3 (Multi-class):
  모델: {best_multi_name}
  정확도: {multi_results_df.iloc[0]['Test Acc']:.2%}
  F1: {multi_results_df.iloc[0]['F1']:.4f}
  백테스팅: {return_multi:+.2f}%

Buy-and-Hold: {buy_hold:+.2f}%
"""
ax10.text(0.1, 0.5, summary_text, fontsize=9,
         verticalalignment='center')
ax10.set_title('종합 요약', fontweight='bold', fontsize=12)

# 11-12. 추가 분석
ax11 = plt.subplot(4, 3, 11)
reg_results_df_plot = reg_results_df.sort_values('Test R²', ascending=True)
ax11.barh(range(len(reg_results_df_plot)), reg_results_df_plot['Test R²'],
         color='steelblue', alpha=0.7)
ax11.set_yticks(range(len(reg_results_df_plot)))
ax11.set_yticklabels(reg_results_df_plot['Model'], fontsize=9)
ax11.set_xlabel('Test R²', fontweight='bold')
ax11.set_title('회귀 모델 비교', fontweight='bold')
ax11.grid(True, alpha=0.3, axis='x')

ax12 = plt.subplot(4, 3, 12)
class_names = ['큰하락', '작은하락', '횡보', '작은상승', '큰상승']
class_counts = [(y_test_multi == i).sum() for i in range(5)]
ax12.bar(class_names, class_counts, color='coral', alpha=0.7)
ax12.set_ylabel('Count', fontweight='bold')
ax12.set_title('Multi-class 분포 (Test)', fontweight='bold')
ax12.grid(True, alpha=0.3, axis='y')
for i, (name, count) in enumerate(zip(class_names, class_counts)):
    ax12.text(i, count, f'{count}\n({count/len(y_test_multi)*100:.1f}%)',
             ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('change_prediction_3methods.png', dpi=300, bbox_inches='tight')
print("✅ 저장: change_prediction_3methods.png")

# ========================================
# 결과 저장
# ========================================
# CSV 저장
reg_results_df.to_csv('method1_regression_results.csv', index=False)
clf_results_df.to_csv('method2_classification_results.csv', index=False)
multi_results_df.to_csv('method3_multiclass_results.csv', index=False)

print("""
✅ 저장:
   - method1_regression_results.csv
   - method2_classification_results.csv
   - method3_multiclass_results.csv
""")

# ========================================
# 최종 결론
# ========================================
print("\n" + "="*80)
print("최종 결론")
print("="*80)

best_method = max([
    ('회귀', return_reg),
    ('이진분류', return_clf),
    ('Multi-class', return_multi)
], key=lambda x: x[1])

print(f"""
📊 3가지 방법 종합 평가

1. 예측 성능:
   회귀: R²={reg_results_df.iloc[0]['Test R²']:.4f}
   이진분류: Acc={clf_results_df.iloc[0]['Test Acc']:.2%}
   Multi: Acc={multi_results_df.iloc[0]['Test Acc']:.2%}

2. 백테스팅 수익률:
   회귀: {return_reg:+.2f}%
   이진분류: {return_clf:+.2f}%
   Multi-class: {return_multi:+.2f}%
   Buy-and-Hold: {buy_hold:+.2f}%

3. 최고 성능:
   🏆 {best_method[0]} ({best_method[1]:+.2f}%)

4. 권장 사항:
   {'✅ 회귀 추천: 변화량 정확도 높음' if return_reg == best_method[1] else ''}
   {'✅ 이진분류 추천: 단순하고 해석 쉬움' if return_clf == best_method[1] else ''}
   {'✅ Multi 추천: 극단 상황 포착' if return_multi == best_method[1] else ''}

   실전 사용:
   - 회귀: 정확한 변화율 예측 필요 시
   - 이진분류: 단순 방향만 필요 시
   - Multi: 큰 변화만 거래 시

5. 주의사항:
   - 4시간은 변화가 작음 (평균 {df['target_pct_change'].mean():.3f}%)
   - 백테스팅 수수료 미포함
   - 슬리피지 미고려
   - 과적합 주의
""")

print("="*80)
print("3가지 방법 비교 완료!")
print("="*80)
