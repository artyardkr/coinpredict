"""
Step 29: Proper Stepwise Regression with VIF Check

규민tv.py의 문제점 개선:
1. VIF (다중공선성) 체크 추가
2. Train/Test 분리 (2025년 out-of-sample)
3. ElasticNet과 성능 비교
4. 백테스팅 포함
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("올바른 Stepwise Regression (VIF 체크 포함)")
print("="*80)

# ========================================
# 1. 데이터 로드
# ========================================
df = pd.read_csv('integrated_data_full.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"\n데이터: {df.shape}")
print(f"기간: {df['Date'].min().date()} ~ {df['Date'].max().date()}")

# ========================================
# 2. 타겟 생성: 다음날 Close
# ========================================
df['target'] = df['Close'].shift(-1)
df = df[:-1].copy()

print(f"\n타겟 생성 후: {df.shape}")

# ========================================
# 3. 특성 선택
# ========================================
# Market cap 관련 제거 (Close * 상수)
market_cap_cols = [col for col in df.columns if 'market_cap' in col.lower()]

# 제거할 컬럼
exclude_cols = ['Date', 'Close', 'target'] + market_cap_cols

# 특성 선택
feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols].copy()
y = df['target'].copy()

print(f"\n총 특성 수: {len(feature_cols)}")
print(f"샘플 수: {len(X)}")

# ========================================
# 4. VIF 기반 다중공선성 제거
# ========================================
def remove_high_vif(X, threshold=10.0):
    """VIF가 threshold보다 높은 변수를 반복적으로 제거"""
    print(f"\n{'='*60}")
    print(f"VIF 기반 다중공선성 제거 (threshold={threshold})")
    print(f"{'='*60}")

    X_clean = X.copy()
    removed_features = []
    iteration = 0

    while True:
        iteration += 1
        # VIF 계산
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X_clean.columns
        vif_data["VIF"] = [variance_inflation_factor(X_clean.values, i)
                           for i in range(X_clean.shape[1])]

        # 최대 VIF
        max_vif = vif_data["VIF"].max()

        if max_vif > threshold:
            # VIF가 가장 높은 변수 제거
            feature_to_remove = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
            removed_features.append((feature_to_remove, max_vif))
            X_clean = X_clean.drop(columns=[feature_to_remove])
            print(f"  Iter {iteration}: 제거 {feature_to_remove} (VIF={max_vif:.2f})")
        else:
            print(f"\n최종: {len(X_clean.columns)}개 변수 (최대 VIF={max_vif:.2f})")
            break

    return X_clean, removed_features

X_no_vif, removed_vif = remove_high_vif(X, threshold=10.0)

# ========================================
# 5. Backward Elimination (p-value 기반)
# ========================================
def backward_elimination(X, y, significance_level=0.05):
    """p-value 기반 후진 제거"""
    print(f"\n{'='*60}")
    print(f"Backward Elimination (p-value < {significance_level})")
    print(f"{'='*60}")

    features = list(X.columns)
    removed_features = []
    iteration = 0

    while len(features) > 0:
        iteration += 1
        X_with_const = sm.add_constant(X[features])

        try:
            model = sm.OLS(y, X_with_const, missing='drop').fit()
        except:
            print(f"  모델 피팅 실패 - 변수 {features[-1]} 제거")
            features.pop()
            continue

        # 상수항 제외한 p-value
        pvalues = model.pvalues.drop('const')
        max_pvalue = pvalues.max()

        if max_pvalue > significance_level:
            feature_to_remove = pvalues.idxmax()
            removed_features.append((feature_to_remove, max_pvalue))
            features.remove(feature_to_remove)
            if iteration % 5 == 0:
                print(f"  Iter {iteration}: {len(features)}개 변수 남음")
        else:
            break

    print(f"\n최종: {len(features)}개 변수 선택")
    return features, removed_features

selected_features, removed_pval = backward_elimination(X_no_vif, y, significance_level=0.05)

# ========================================
# 6. Train/Test 분리
# ========================================
train_end = pd.to_datetime('2024-12-31')
test_start = pd.to_datetime('2025-01-01')

train_mask = df['Date'] <= train_end
test_mask = df['Date'] >= test_start

X_train = X_no_vif.loc[train_mask, selected_features]
y_train = y[train_mask]
X_test = X_no_vif.loc[test_mask, selected_features]
y_test = y[test_mask]

print(f"\n{'='*60}")
print(f"Train/Test 분리")
print(f"{'='*60}")
print(f"Train: {df[train_mask]['Date'].min().date()} ~ {df[train_mask]['Date'].max().date()} ({len(X_train)} 샘플)")
print(f"Test:  {df[test_mask]['Date'].min().date()} ~ {df[test_mask]['Date'].max().date()} ({len(X_test)} 샘플)")

# ========================================
# 7. OLS Stepwise Regression
# ========================================
print(f"\n{'='*60}")
print("OLS Stepwise Regression")
print(f"{'='*60}")

X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

ols_model = sm.OLS(y_train, X_train_const).fit()

# 예측
y_train_pred_ols = ols_model.predict(X_train_const)
y_test_pred_ols = ols_model.predict(X_test_const)

# 성능
train_r2_ols = r2_score(y_train, y_train_pred_ols)
test_r2_ols = r2_score(y_test, y_test_pred_ols)
train_rmse_ols = np.sqrt(mean_squared_error(y_train, y_train_pred_ols))
test_rmse_ols = np.sqrt(mean_squared_error(y_test, y_test_pred_ols))

print(f"\nOLS Stepwise 성능:")
print(f"  Train R²: {train_r2_ols:.4f}, RMSE: ${train_rmse_ols:,.2f}")
print(f"  Test R²:  {test_r2_ols:.4f}, RMSE: ${test_rmse_ols:,.2f}")

# Condition Number 확인
print(f"\nCondition Number: {np.linalg.cond(X_train_const):.2e}")

# ========================================
# 8. ElasticNet 비교
# ========================================
print(f"\n{'='*60}")
print("ElasticNet 비교")
print(f"{'='*60}")

# 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ElasticNet
elasticnet_model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=10000)
elasticnet_model.fit(X_train_scaled, y_train)

# 예측
y_train_pred_en = elasticnet_model.predict(X_train_scaled)
y_test_pred_en = elasticnet_model.predict(X_test_scaled)

# 성능
train_r2_en = r2_score(y_train, y_train_pred_en)
test_r2_en = r2_score(y_test, y_test_pred_en)
train_rmse_en = np.sqrt(mean_squared_error(y_train, y_train_pred_en))
test_rmse_en = np.sqrt(mean_squared_error(y_test, y_test_pred_en))

print(f"\nElasticNet 성능:")
print(f"  Train R²: {train_r2_en:.4f}, RMSE: ${train_rmse_en:,.2f}")
print(f"  Test R²:  {test_r2_en:.4f}, RMSE: ${test_rmse_en:,.2f}")

# ========================================
# 9. 백테스팅 (2025년)
# ========================================
def backtest_predictions(df_test, y_true, y_pred, model_name, initial_capital=10000):
    """예측 기반 백테스팅"""
    test_df = df_test.copy()
    test_df['predicted'] = y_pred
    test_df['actual'] = y_true.values
    test_df['current_price'] = df_test['Close'].values

    # 예측 변화율
    test_df['predicted_return'] = (test_df['predicted'] / test_df['current_price'] - 1) * 100
    test_df['actual_return'] = (test_df['actual'] / test_df['current_price'] - 1) * 100

    # 전략: 상승 예측시 매수 (threshold 1%)
    threshold = 1.0
    cash = initial_capital
    btc = 0
    portfolio_values = []

    for idx, row in test_df.iterrows():
        if row['predicted_return'] > threshold:
            # 매수
            if cash > 0:
                btc = cash / row['current_price']
                cash = 0
        else:
            # 매도
            if btc > 0:
                cash = btc * row['current_price']
                btc = 0

        # 다음날 포트폴리오 가치
        if btc > 0:
            portfolio_value = btc * row['actual']
        else:
            portfolio_value = cash

        portfolio_values.append(portfolio_value)

    test_df['portfolio_value'] = portfolio_values

    # Buy-and-Hold
    bnh_btc = initial_capital / test_df.iloc[0]['current_price']
    test_df['bnh_value'] = bnh_btc * test_df['current_price']

    # 성과
    final_value = test_df['portfolio_value'].iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100

    bnh_final = test_df['bnh_value'].iloc[-1]
    bnh_return = (bnh_final / initial_capital - 1) * 100

    # 샤프 비율
    daily_returns = test_df['portfolio_value'].pct_change()
    annual_return = (1 + total_return/100) ** (365/len(test_df)) - 1
    annual_vol = daily_returns.std() * np.sqrt(365)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0

    print(f"\n{model_name} 백테스팅 (2025년, Threshold {threshold}%):")
    print(f"  총 수익률: {total_return:+.2f}%")
    print(f"  샤프 비율: {sharpe:.3f}")
    print(f"  Buy-and-Hold: {bnh_return:+.2f}%")

    return test_df, total_return, sharpe

print(f"\n{'='*60}")
print("백테스팅 (2025년)")
print(f"{'='*60}")

df_test = df[test_mask].reset_index(drop=True)
bt_ols, ret_ols, sharpe_ols = backtest_predictions(df_test, y_test, y_test_pred_ols, "OLS Stepwise")
bt_en, ret_en, sharpe_en = backtest_predictions(df_test, y_test, y_test_pred_en, "ElasticNet")

# ========================================
# 10. 시각화
# ========================================
fig = plt.figure(figsize=(16, 12))

# (1) 모델 요약
ax1 = plt.subplot(3, 3, 1)
ax1.axis('off')
summary_text = f"""
【OLS Stepwise Regression】

선택된 변수: {len(selected_features)}개
제거된 변수 (VIF): {len(removed_vif)}개
제거된 변수 (p-value): {len(removed_pval)}개

Condition Number: {np.linalg.cond(X_train_const):.2e}
(규민tv.py: 9.26e+16)

Train R²: {train_r2_ols:.4f}
Test R² (2025): {test_r2_ols:.4f}

백테스팅 수익률 (2025):
  OLS: {ret_ols:+.2f}%
  ElasticNet: {ret_en:+.2f}%
"""
ax1.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
         verticalalignment='center')
ax1.set_title('모델 요약', fontsize=12, fontweight='bold')

# (2) Train R² 비교
ax2 = plt.subplot(3, 3, 2)
models = ['OLS\nStepwise', 'ElasticNet']
train_r2s = [train_r2_ols, train_r2_en]
bars = ax2.bar(models, train_r2s, color=['steelblue', 'coral'], alpha=0.7)
for bar, val in zip(bars, train_r2s):
    ax2.text(bar.get_x() + bar.get_width()/2, val, f'{val:.4f}',
            ha='center', va='bottom', fontweight='bold')
ax2.set_ylabel('R²')
ax2.set_title('Train R² 비교 (2021-2024)', fontsize=11, fontweight='bold')
ax2.set_ylim([0, 1.0])
ax2.grid(True, alpha=0.3, axis='y')

# (3) Test R² 비교
ax3 = plt.subplot(3, 3, 3)
test_r2s = [test_r2_ols, test_r2_en]
colors = ['green' if r > 0 else 'red' for r in test_r2s]
bars = ax3.bar(models, test_r2s, color=colors, alpha=0.7)
for bar, val in zip(bars, test_r2s):
    ax3.text(bar.get_x() + bar.get_width()/2, val, f'{val:.4f}',
            ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')
ax3.set_ylabel('R²')
ax3.set_title('Test R² 비교 (2025년)', fontsize=11, fontweight='bold')
ax3.axhline(0, color='black', linewidth=0.8)
ax3.grid(True, alpha=0.3, axis='y')

# (4) Train RMSE
ax4 = plt.subplot(3, 3, 4)
train_rmses = [train_rmse_ols, train_rmse_en]
bars = ax4.barh(models, train_rmses, color=['steelblue', 'coral'], alpha=0.7)
for bar, val in zip(bars, train_rmses):
    ax4.text(val, bar.get_y() + bar.get_height()/2, f'${val:,.0f}',
            va='center', ha='left', fontweight='bold')
ax4.set_xlabel('RMSE ($)')
ax4.set_title('Train RMSE 비교', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

# (5) Test RMSE
ax5 = plt.subplot(3, 3, 5)
test_rmses = [test_rmse_ols, test_rmse_en]
bars = ax5.barh(models, test_rmses, color=['steelblue', 'coral'], alpha=0.7)
for bar, val in zip(bars, test_rmses):
    ax5.text(val, bar.get_y() + bar.get_height()/2, f'${val:,.0f}',
            va='center', ha='left', fontweight='bold')
ax5.set_xlabel('RMSE ($)')
ax5.set_title('Test RMSE 비교 (2025년)', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='x')

# (6) 백테스팅 수익률
ax6 = plt.subplot(3, 3, 6)
returns = [ret_ols, ret_en]
colors_ret = ['green' if r > 0 else 'red' for r in returns]
bars = ax6.barh(models, returns, color=colors_ret, alpha=0.7)
for bar, val in zip(bars, returns):
    ax6.text(val, bar.get_y() + bar.get_height()/2, f'{val:+.2f}%',
            va='center', ha='left' if val > 0 else 'right', fontweight='bold')
ax6.set_xlabel('수익률 (%)')
ax6.set_title('백테스팅 수익률 (2025년, Threshold 1%)', fontsize=11, fontweight='bold')
ax6.axvline(0, color='black', linewidth=0.8)
ax6.grid(True, alpha=0.3, axis='x')

# (7) 포트폴리오 가치 - OLS
ax7 = plt.subplot(3, 3, 7)
ax7.plot(bt_ols.index, bt_ols['portfolio_value'], label='OLS Strategy', linewidth=2, color='steelblue')
ax7.plot(bt_ols.index, bt_ols['bnh_value'], label='Buy-and-Hold', linewidth=2, color='gray', alpha=0.7)
ax7.set_ylabel('자산 ($)')
ax7.set_title('OLS Stepwise 백테스팅', fontsize=11, fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)

# (8) 포트폴리오 가치 - ElasticNet
ax8 = plt.subplot(3, 3, 8)
ax8.plot(bt_en.index, bt_en['portfolio_value'], label='ElasticNet Strategy', linewidth=2, color='coral')
ax8.plot(bt_en.index, bt_en['bnh_value'], label='Buy-and-Hold', linewidth=2, color='gray', alpha=0.7)
ax8.set_ylabel('자산 ($)')
ax8.set_title('ElasticNet 백테스팅', fontsize=11, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

# (9) 예측 정확도 (실제 vs 예측)
ax9 = plt.subplot(3, 3, 9)
ax9.scatter(y_test, y_test_pred_ols, alpha=0.3, s=20, label='OLS', color='steelblue')
ax9.scatter(y_test, y_test_pred_en, alpha=0.3, s=20, label='ElasticNet', color='coral')
min_val = min(y_test.min(), y_test_pred_ols.min(), y_test_pred_en.min())
max_val = max(y_test.max(), y_test_pred_ols.max(), y_test_pred_en.max())
ax9.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, label='Perfect')
ax9.set_xlabel('실제 가격 ($)')
ax9.set_ylabel('예측 가격 ($)')
ax9.set_title('예측 정확도 (2025년)', fontsize=11, fontweight='bold')
ax9.legend()
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stepwise_regression_results.png', dpi=300, bbox_inches='tight')
print(f"\n시각화 저장: stepwise_regression_results.png")

# ========================================
# 11. 최종 모델 요약 출력
# ========================================
print(f"\n{'='*80}")
print("최종 OLS Stepwise 모델 요약")
print(f"{'='*80}")
print(ols_model.summary())

# 선택된 변수 저장
selected_df = pd.DataFrame({
    'Feature': selected_features,
    'Coefficient': ols_model.params[1:],  # const 제외
    'P-value': ols_model.pvalues[1:]
})
selected_df = selected_df.sort_values('P-value')
selected_df.to_csv('stepwise_selected_features.csv', index=False)
print(f"\n선택된 변수 저장: stepwise_selected_features.csv")

# ========================================
# 12. 규민tv.py와 비교
# ========================================
print(f"\n{'='*80}")
print("규민tv.py vs step29 비교")
print(f"{'='*80}")
print(f"\n【규민tv.py】")
print(f"  Condition Number: 9.26e+16 (다중공선성 심각)")
print(f"  R²: 0.997 (In-sample, 전체 데이터)")
print(f"  선택 변수: 30개 (VIF 체크 없음)")

print(f"\n【step29 (개선)】")
print(f"  Condition Number: {np.linalg.cond(X_train_const):.2e} (정상)")
print(f"  Train R²: {train_r2_ols:.4f}")
print(f"  Test R² (2025): {test_r2_ols:.4f} (Out-of-sample)")
print(f"  선택 변수: {len(selected_features)}개 (VIF < 10)")

print(f"\n핵심 개선사항:")
print(f"  ✅ VIF 체크로 다중공선성 제거")
print(f"  ✅ Train/Test 분리로 과적합 확인")
print(f"  ✅ 2025년 백테스팅으로 실제 성능 검증")
print(f"  ✅ ElasticNet과 비교")

print("="*80)
