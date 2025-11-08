import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("모델 훈련: Macro + OnChain + Sentiment + Volume")
print("=" * 70)

# ===== 1. 개별 데이터 로드 =====
print("\n1. 데이터 로드")
print("-" * 70)

# BTC 원본 데이터 (Volume만 사용)
btc = pd.read_csv('btc_data_2021_2025.csv', index_col=0, parse_dates=True)
btc.index = pd.to_datetime(btc.index).tz_localize(None)
print(f"✓ BTC 데이터: {btc.shape} (Close, Volume)")

# FRED 거시경제 데이터
macro = pd.read_csv('fred_macro_data.csv', index_col=0, parse_dates=True)
macro.index = pd.to_datetime(macro.index).tz_localize(None)
print(f"✓ FRED 거시경제: {macro.shape} - {list(macro.columns)}")

# 온체인 데이터 (bc_market_price 제외)
onchain = pd.read_csv('onchain_data.csv', index_col=0, parse_dates=True)
onchain.index = pd.to_datetime(onchain.index).tz_localize(None)
if 'bc_market_price' in onchain.columns:
    onchain = onchain.drop('bc_market_price', axis=1)
if 'bc_market_cap' in onchain.columns:
    onchain = onchain.drop('bc_market_cap', axis=1)
print(f"✓ 온체인 데이터 (가격 제외): {onchain.shape}")

# 감정 데이터
sentiment = pd.read_csv('sentiment_data.csv', index_col=0, parse_dates=True)
sentiment.index = pd.to_datetime(sentiment.index).tz_localize(None)
# 수치형만 선택
sentiment_numeric = sentiment[['fear_greed_index', 'google_trends_btc']]
print(f"✓ 감정 데이터: {sentiment_numeric.shape}")

# ===== 2. 데이터 통합 =====
print("\n2. 데이터 통합")
print("-" * 70)

# Close와 Volume만 선택
btc_subset = btc[['Close', 'Volume']].copy()

# 데이터 결합
integrated = btc_subset.copy()
integrated = integrated.join(macro, how='left')
integrated = integrated.join(onchain, how='left')
integrated = integrated.join(sentiment_numeric, how='left')

print(f"통합 후: {integrated.shape}")
print(f"기간: {integrated.index[0].date()} ~ {integrated.index[-1].date()}")

# 결측치 처리
print(f"\n결측치 처리 전: {integrated.isnull().sum().sum()}개")
integrated = integrated.ffill().bfill()
integrated = integrated.dropna()
print(f"결측치 처리 후: {integrated.shape}")

# ===== 3. 상관관계 분석 =====
print("\n" + "=" * 70)
print("3. Close와의 상관관계 분석")
print("=" * 70)

# Close와의 상관계수 계산
correlations = {}
for col in integrated.columns:
    if col != 'Close':
        try:
            corr, pval = pearsonr(integrated[col].dropna(), integrated.loc[integrated[col].notna(), 'Close'])
            correlations[col] = {'correlation': corr, 'p_value': pval}
        except:
            correlations[col] = {'correlation': 0, 'p_value': 1}

# DataFrame으로 변환
corr_df = pd.DataFrame(correlations).T
corr_df = corr_df.sort_values('correlation', key=abs, ascending=False)

# 카테고리 추가
def get_category(col):
    if col == 'Volume':
        return 'Volume'
    elif col in macro.columns:
        return 'Macro'
    elif col in onchain.columns:
        return 'OnChain'
    elif col in sentiment_numeric.columns:
        return 'Sentiment'
    else:
        return 'Other'

corr_df['category'] = corr_df.index.map(get_category)

# 결과 출력
print("\n상위 20개 특성 (절대값 기준):")
print("-" * 70)
for i, (idx, row) in enumerate(corr_df.head(20).iterrows(), 1):
    sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
    print(f"{i:2}. {idx:30} | Corr: {row['correlation']:7.4f} {sig} | Cat: {row['category']}")

# 카테고리별 평균 상관계수
print("\n카테고리별 평균 절대 상관계수:")
print("-" * 70)
for cat in ['Volume', 'Macro', 'OnChain', 'Sentiment']:
    cat_corr = corr_df[corr_df['category'] == cat]['correlation'].abs().mean()
    cat_count = (corr_df['category'] == cat).sum()
    print(f"{cat:15} : {cat_corr:.4f} (n={cat_count})")

# 상관관계 저장
corr_df.to_csv('correlation_analysis.csv')
print("\n✓ correlation_analysis.csv")

# ===== 4. 타겟 설정 및 데이터 분할 =====
print("\n4. 타겟 설정 및 데이터 분할")
print("-" * 70)

# 다음 날 Close 예측
integrated['target'] = integrated['Close'].shift(-1)
integrated = integrated.dropna(subset=['target'])

# 특성과 타겟 분리
X = integrated.drop(['Close', 'target'], axis=1)
y = integrated['target']

print(f"특성 수: {X.shape[1]}개")
print(f"샘플 수: {len(X)}개")
print(f"\n특성 목록:")
print(f"  - Volume: 1개")
print(f"  - Macro: {len([c for c in X.columns if c in macro.columns])}개")
print(f"  - OnChain: {len([c for c in X.columns if c in onchain.columns])}개")
print(f"  - Sentiment: {len([c for c in X.columns if c in sentiment_numeric.columns])}개")

# 7:3 분할
split_idx = int(len(X) * 0.7)
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"\n훈련: {X_train.shape[0]}개 ({X_train.index[0].date()} ~ {X_train.index[-1].date()})")
print(f"테스트: {X_test.shape[0]}개 ({X_test.index[0].date()} ~ {X_test.index[-1].date()})")

# ===== 5. 모델 훈련 =====
print("\n" + "=" * 70)
print("5. 모델 훈련 및 평가")
print("=" * 70)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

all_results = []

# ===== Random Forest =====
print("\n[Random Forest]")
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

rf_results = {
    'model': 'Random Forest',
    'train_r2': r2_score(y_train, y_train_pred_rf),
    'test_r2': r2_score(y_test, y_test_pred_rf),
    'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred_rf)),
    'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred_rf)),
    'train_mae': mean_absolute_error(y_train, y_train_pred_rf),
    'test_mae': mean_absolute_error(y_test, y_test_pred_rf),
    'train_mape': np.mean(np.abs((y_train - y_train_pred_rf) / y_train)) * 100,
    'test_mape': np.mean(np.abs((y_test - y_test_pred_rf) / y_test)) * 100,
}

all_results.append(rf_results)

print(f"  Train R²: {rf_results['train_r2']:.4f} | Test R²: {rf_results['test_r2']:.4f}")
print(f"  Train RMSE: ${rf_results['train_rmse']:,.2f} | Test RMSE: ${rf_results['test_rmse']:,.2f}")
print(f"  Train MAE: ${rf_results['train_mae']:,.2f} | Test MAE: ${rf_results['test_mae']:,.2f}")
print(f"  Train MAPE: {rf_results['train_mape']:.2f}% | Test MAPE: {rf_results['test_mape']:.2f}%")

# Feature Importance
rf_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_,
    'category': [get_category(c) for c in X.columns]
}).sort_values('importance', ascending=False)

print(f"\n  상위 10개 중요 특성:")
for i, row in rf_importance.head(10).iterrows():
    print(f"    {i+1:2}. {row['feature']:30} : {row['importance']:.4f} ({row['category']})")

# 카테고리별 중요도
print(f"\n  카테고리별 평균 중요도:")
for cat in ['Volume', 'Macro', 'OnChain', 'Sentiment']:
    cat_imp = rf_importance[rf_importance['category'] == cat]['importance'].mean()
    print(f"    {cat:15} : {cat_imp:.4f}")

# ===== XGBoost =====
print("\n[XGBoost]")
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

xgb_results = {
    'model': 'XGBoost',
    'train_r2': r2_score(y_train, y_train_pred_xgb),
    'test_r2': r2_score(y_test, y_test_pred_xgb),
    'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred_xgb)),
    'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred_xgb)),
    'train_mae': mean_absolute_error(y_train, y_train_pred_xgb),
    'test_mae': mean_absolute_error(y_test, y_test_pred_xgb),
    'train_mape': np.mean(np.abs((y_train - y_train_pred_xgb) / y_train)) * 100,
    'test_mape': np.mean(np.abs((y_test - y_test_pred_xgb) / y_test)) * 100,
}

all_results.append(xgb_results)

print(f"  Train R²: {xgb_results['train_r2']:.4f} | Test R²: {xgb_results['test_r2']:.4f}")
print(f"  Train RMSE: ${xgb_results['train_rmse']:,.2f} | Test RMSE: ${xgb_results['test_rmse']:,.2f}")
print(f"  Train MAE: ${xgb_results['train_mae']:,.2f} | Test MAE: ${xgb_results['test_mae']:,.2f}")
print(f"  Train MAPE: {xgb_results['train_mape']:.2f}% | Test MAPE: {xgb_results['test_mape']:.2f}%")

# Feature Importance
xgb_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_,
    'category': [get_category(c) for c in X.columns]
}).sort_values('importance', ascending=False)

print(f"\n  상위 10개 중요 특성:")
for i, row in xgb_importance.head(10).iterrows():
    print(f"    {i+1:2}. {row['feature']:30} : {row['importance']:.4f} ({row['category']})")

# 카테고리별 중요도
print(f"\n  카테고리별 평균 중요도:")
for cat in ['Volume', 'Macro', 'OnChain', 'Sentiment']:
    cat_imp = xgb_importance[xgb_importance['category'] == cat]['importance'].mean()
    print(f"    {cat:15} : {cat_imp:.4f}")

# ===== 6. 결과 저장 =====
print("\n" + "=" * 70)
print("6. 결과 저장")
print("=" * 70)

results_df = pd.DataFrame(all_results)
results_df.to_csv('model_results_macro_onchain_sentiment.csv', index=False)
print("✓ model_results_macro_onchain_sentiment.csv")

rf_importance.to_csv('feature_importance_rf.csv', index=False)
print("✓ feature_importance_rf.csv")

xgb_importance.to_csv('feature_importance_xgb.csv', index=False)
print("✓ feature_importance_xgb.csv")

# ===== 7. 시각화 =====
print("\n7. 시각화")
print("-" * 70)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. 상관관계 히트맵 (상위 30개)
ax1 = fig.add_subplot(gs[0, :2])
top_30_features = corr_df.head(30).index
corr_matrix = integrated[list(top_30_features) + ['Close']].corr()['Close'].drop('Close')
colors = ['red' if x < 0 else 'blue' for x in corr_matrix.values]
ax1.barh(range(len(corr_matrix)), corr_matrix.values, color=colors, alpha=0.6)
ax1.set_yticks(range(len(corr_matrix)))
ax1.set_yticklabels(corr_matrix.index, fontsize=8)
ax1.set_xlabel('Correlation with Close')
ax1.set_title('Top 30 Features: Correlation with Close', fontweight='bold', fontsize=12)
ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax1.grid(True, alpha=0.3, axis='x')
ax1.invert_yaxis()

# 2. 카테고리별 평균 상관계수
ax2 = fig.add_subplot(gs[0, 2])
cat_corr = corr_df.groupby('category')['correlation'].apply(lambda x: x.abs().mean())
ax2.bar(range(len(cat_corr)), cat_corr.values, alpha=0.7)
ax2.set_xticks(range(len(cat_corr)))
ax2.set_xticklabels(cat_corr.index, rotation=45, ha='right')
ax2.set_ylabel('Average |Correlation|')
ax2.set_title('Avg Correlation by Category', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# 3. RF Feature Importance (Top 20)
ax3 = fig.add_subplot(gs[1, :])
top_20_rf = rf_importance.head(20)
colors_rf = [{'Volume': 'red', 'Macro': 'blue', 'OnChain': 'green', 'Sentiment': 'orange'}[c]
             for c in top_20_rf['category']]
ax3.barh(range(len(top_20_rf)), top_20_rf['importance'], color=colors_rf, alpha=0.6)
ax3.set_yticks(range(len(top_20_rf)))
ax3.set_yticklabels(top_20_rf['feature'], fontsize=9)
ax3.set_xlabel('Importance')
ax3.set_title('Random Forest: Top 20 Feature Importance', fontweight='bold', fontsize=12)
ax3.invert_yaxis()
ax3.grid(True, alpha=0.3, axis='x')
# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='red', alpha=0.6, label='Volume'),
                   Patch(facecolor='blue', alpha=0.6, label='Macro'),
                   Patch(facecolor='green', alpha=0.6, label='OnChain'),
                   Patch(facecolor='orange', alpha=0.6, label='Sentiment')]
ax3.legend(handles=legend_elements, loc='lower right')

# 4. XGB Feature Importance (Top 20)
ax4 = fig.add_subplot(gs[2, :])
top_20_xgb = xgb_importance.head(20)
colors_xgb = [{'Volume': 'red', 'Macro': 'blue', 'OnChain': 'green', 'Sentiment': 'orange'}[c]
              for c in top_20_xgb['category']]
ax4.barh(range(len(top_20_xgb)), top_20_xgb['importance'], color=colors_xgb, alpha=0.6)
ax4.set_yticks(range(len(top_20_xgb)))
ax4.set_yticklabels(top_20_xgb['feature'], fontsize=9)
ax4.set_xlabel('Importance')
ax4.set_title('XGBoost: Top 20 Feature Importance', fontweight='bold', fontsize=12)
ax4.invert_yaxis()
ax4.grid(True, alpha=0.3, axis='x')
ax4.legend(handles=legend_elements, loc='lower right')

plt.savefig('correlation_and_importance_analysis.png', dpi=300, bbox_inches='tight')
print("✓ correlation_and_importance_analysis.png")

# 시계열 예측
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

test_dates = y_test.index

# Random Forest
axes[0].plot(test_dates, y_test.values, label='Actual', linewidth=2, color='blue')
axes[0].plot(test_dates, y_test_pred_rf, label='Predicted', linewidth=2,
            color='red', alpha=0.7, linestyle='--')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('BTC Price ($)')
axes[0].set_title(f'Random Forest: R²={rf_results["test_r2"]:.4f}, MAPE={rf_results["test_mape"]:.2f}%',
                 fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# XGBoost
axes[1].plot(test_dates, y_test.values, label='Actual', linewidth=2, color='blue')
axes[1].plot(test_dates, y_test_pred_xgb, label='Predicted', linewidth=2,
            color='red', alpha=0.7, linestyle='--')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('BTC Price ($)')
axes[1].set_title(f'XGBoost: R²={xgb_results["test_r2"]:.4f}, MAPE={xgb_results["test_mape"]:.2f}%',
                 fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('timeseries_prediction_macro_onchain.png', dpi=300, bbox_inches='tight')
print("✓ timeseries_prediction_macro_onchain.png")

plt.close('all')

print("\n" + "=" * 70)
print("분석 완료!")
print("=" * 70)
print(f"\n✅ 사용한 특성: Macro + OnChain (가격 제외) + Sentiment + Volume")
print(f"   - 총 {X.shape[1]}개 특성")
print(f"   - {len(X)}개 샘플")
print(f"\n🏆 최고 성능: {results_df.loc[results_df['test_r2'].idxmax(), 'model']}")
print(f"   - Test R²: {results_df['test_r2'].max():.4f}")
print(f"   - Test RMSE: ${results_df.loc[results_df['test_r2'].idxmax(), 'test_rmse']:,.2f}")
print(f"   - Test MAPE: {results_df.loc[results_df['test_r2'].idxmax(), 'test_mape']:.2f}%")
print("=" * 70)
