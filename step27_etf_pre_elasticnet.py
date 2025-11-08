#!/usr/bin/env python3
"""
Step 27: ETF 이전 기간 ElasticNet 모델 (2021.02 ~ 2024.01.09)

1. Lasso로 변수 선택
2. Ridge로 변수 선택
3. 합집합 사용 (Lasso ∪ Ridge)
4. ElasticNet으로 모델 학습 (Train/Test 7:3)
5. Time Series Cross-Validation
6. Walk-Forward Validation
7. 백테스팅
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ========================================
# 1. Load Data & ETF 이전 필터링
# ========================================
print("=" * 80)
print("ETF 이전 기간 ElasticNet 모델 (2021.02 ~ 2024.01.09)")
print("=" * 80)

df = pd.read_csv('integrated_data_full_v2.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# ETF 이전만 필터링 (2024년 1월 10일 이전)
ETF_DATE = '2024-01-10'
df_pre = df[df['Date'] < ETF_DATE].copy()

print(f"\n전체 데이터: {len(df)} samples ({df['Date'].min()} ~ {df['Date'].max()})")
print(f"ETF 이전 데이터: {len(df_pre)} samples ({df_pre['Date'].min()} ~ {df_pre['Date'].max()})")
print(f"ETF 이전 기간: {(df_pre['Date'].max() - df_pre['Date'].min()).days} days")

# Create target: next day Close
df_pre['target'] = df_pre['Close'].shift(-1)
df_pre = df_pre[:-1].copy()

print(f"타겟 생성 후: {len(df_pre)} samples")

# ========================================
# 2. Feature Preparation
# ========================================
print("\n" + "=" * 80)
print("변수 준비")
print("=" * 80)

# Exclude columns (data leakage)
exclude_cols = [
    'Date', 'Close', 'High', 'Low', 'Open', 'target',
    'cumulative_return', 'bc_market_price', 'bc_market_cap',
]

# EMA/SMA with 'close' (data leakage)
ema_sma_cols = [col for col in df_pre.columns if ('EMA' in col or 'SMA' in col) and 'close' in col.lower()]
exclude_cols.extend(ema_sma_cols)

# Bollinger Bands (data leakage)
bb_cols = [col for col in df_pre.columns if col.startswith('BB_')]
exclude_cols.extend(bb_cols)

# ETF 관련 변수 제거 (ETF 이전 기간이므로)
etf_cols = [col for col in df_pre.columns if any(x in col for x in ['IBIT', 'FBTC', 'ARKB', 'BITB', 'GBTC', 'Total_BTC_ETF'])]
exclude_cols.extend(etf_cols)

exclude_cols = list(set(exclude_cols))

feature_cols = [col for col in df_pre.columns if col not in exclude_cols]

print(f"전체 컬럼: {len(df_pre.columns)}")
print(f"제외 컬럼: {len(exclude_cols)}")
print(f"사용 변수: {len(feature_cols)}")
print(f"\n제외된 ETF 변수: {[col for col in etf_cols if col in df_pre.columns]}")

# Handle inf and NaN
for col in feature_cols:
    df_pre[col] = df_pre[col].replace([np.inf, -np.inf], np.nan)
    df_pre[col] = df_pre[col].fillna(method='ffill').fillna(method='bfill')

# ========================================
# 3. Train/Test Split (7:3 Time Series Split)
# ========================================
print("\n" + "=" * 80)
print("Train/Test Split (7:3)")
print("=" * 80)

split_idx = int(len(df_pre) * 0.7)
split_date = df_pre['Date'].iloc[split_idx]

train_mask = df_pre['Date'] < split_date
test_mask = df_pre['Date'] >= split_date

train_df = df_pre[train_mask].copy()
test_df = df_pre[test_mask].copy()

X_train = train_df[feature_cols].values
X_test = test_df[feature_cols].values
y_train = train_df['target'].values
y_test = test_df['target'].values

print(f"Train: {len(train_df)} samples ({train_df['Date'].min()} ~ {train_df['Date'].max()})")
print(f"Test:  {len(test_df)} samples ({test_df['Date'].min()} ~ {test_df['Date'].max()})")
print(f"Train period: {(train_df['Date'].max() - train_df['Date'].min()).days} days")
print(f"Test period:  {(test_df['Date'].max() - test_df['Date'].min()).days} days")

# Scale features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# ========================================
# 4. Variable Selection: Lasso
# ========================================
print("\n" + "=" * 80)
print("Step 1: Lasso로 변수 선택")
print("=" * 80)

# LassoCV with Time Series Split
tscv = TimeSeriesSplit(n_splits=5)
alphas_lasso = np.logspace(-3, 1, 50)

lasso_cv = LassoCV(alphas=alphas_lasso, cv=tscv, max_iter=10000, n_jobs=-1)
lasso_cv.fit(X_train_scaled, y_train)

print(f"Lasso 최적 alpha: {lasso_cv.alpha_:.4f}")

# Get selected variables
lasso_coef = lasso_cv.coef_
lasso_selected_idx = np.where(lasso_coef != 0)[0]
lasso_selected_vars = [feature_cols[i] for i in lasso_selected_idx]

print(f"Lasso 선택 변수: {len(lasso_selected_vars)}개 / {len(feature_cols)}개")
print(f"제거 비율: {(1 - len(lasso_selected_vars)/len(feature_cols))*100:.1f}%")

# Lasso 성능
y_pred_lasso_train = lasso_cv.predict(X_train_scaled)
y_pred_lasso_test = lasso_cv.predict(X_test_scaled)

lasso_train_r2 = r2_score(y_train, y_pred_lasso_train)
lasso_test_r2 = r2_score(y_test, y_pred_lasso_test)
lasso_train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_lasso_train))
lasso_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lasso_test))

print(f"\nLasso 성능:")
print(f"  Train R²: {lasso_train_r2:.4f}")
print(f"  Test R²:  {lasso_test_r2:.4f}")
print(f"  R² Gap:   {lasso_train_r2 - lasso_test_r2:.4f}")
print(f"  Train RMSE: ${lasso_train_rmse:,.2f}")
print(f"  Test RMSE:  ${lasso_test_rmse:,.2f}")

# ========================================
# 5. Variable Selection: Ridge
# ========================================
print("\n" + "=" * 80)
print("Step 2: Ridge로 변수 선택")
print("=" * 80)

alphas_ridge = np.logspace(-2, 3, 50)

ridge_cv = RidgeCV(alphas=alphas_ridge, cv=tscv)
ridge_cv.fit(X_train_scaled, y_train)

print(f"Ridge 최적 alpha: {ridge_cv.alpha_:.4f}")

# Get selected variables (계수 절댓값 > threshold)
ridge_coef = ridge_cv.coef_
ridge_coef_abs = np.abs(ridge_coef)
ridge_threshold = np.percentile(ridge_coef_abs, 25)  # 하위 25% 제거
ridge_selected_idx = np.where(ridge_coef_abs > ridge_threshold)[0]
ridge_selected_vars = [feature_cols[i] for i in ridge_selected_idx]

print(f"Ridge 선택 변수: {len(ridge_selected_vars)}개 / {len(feature_cols)}개")
print(f"제거 비율: {(1 - len(ridge_selected_vars)/len(feature_cols))*100:.1f}%")
print(f"Ridge 계수 threshold: {ridge_threshold:.4f}")

# Ridge 성능
y_pred_ridge_train = ridge_cv.predict(X_train_scaled)
y_pred_ridge_test = ridge_cv.predict(X_test_scaled)

ridge_train_r2 = r2_score(y_train, y_pred_ridge_train)
ridge_test_r2 = r2_score(y_test, y_pred_ridge_test)
ridge_train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_ridge_train))
ridge_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge_test))

print(f"\nRidge 성능:")
print(f"  Train R²: {ridge_train_r2:.4f}")
print(f"  Test R²:  {ridge_test_r2:.4f}")
print(f"  R² Gap:   {ridge_train_r2 - ridge_test_r2:.4f}")
print(f"  Train RMSE: ${ridge_train_rmse:,.2f}")
print(f"  Test RMSE:  ${ridge_test_rmse:,.2f}")

# ========================================
# 6. Variable Selection: Union
# ========================================
print("\n" + "=" * 80)
print("Step 3: Lasso ∪ Ridge 합집합")
print("=" * 80)

# Union of Lasso and Ridge
selected_vars_union = sorted(list(set(lasso_selected_vars) | set(ridge_selected_vars)))

print(f"Lasso 선택: {len(lasso_selected_vars)}개")
print(f"Ridge 선택: {len(ridge_selected_vars)}개")
print(f"합집합:     {len(selected_vars_union)}개")
print(f"교집합:     {len(set(lasso_selected_vars) & set(ridge_selected_vars))}개")

# Lasso에만 있는 변수
lasso_only = set(lasso_selected_vars) - set(ridge_selected_vars)
print(f"Lasso에만:  {len(lasso_only)}개")

# Ridge에만 있는 변수
ridge_only = set(ridge_selected_vars) - set(lasso_selected_vars)
print(f"Ridge에만:  {len(ridge_only)}개")

# 선택된 변수 TOP 20
print("\n선택된 변수 TOP 20 (Lasso 계수 절댓값 기준):")
lasso_coef_dict = {feature_cols[i]: abs(lasso_coef[i]) for i in lasso_selected_idx}
top20_vars = sorted(lasso_coef_dict.items(), key=lambda x: x[1], reverse=True)[:20]
for i, (var, coef) in enumerate(top20_vars, 1):
    print(f"  {i:2d}. {var:30s} {coef:10.4f}")

# ========================================
# 7. ElasticNet with Selected Variables
# ========================================
print("\n" + "=" * 80)
print("Step 4: ElasticNet 학습 (선택된 변수만)")
print("=" * 80)

# Get selected features
X_train_selected = train_df[selected_vars_union].values
X_test_selected = test_df[selected_vars_union].values

# Scale
scaler_selected = StandardScaler()
X_train_selected_scaled = scaler_selected.fit_transform(X_train_selected)
X_test_selected_scaled = scaler_selected.transform(X_test_selected)

# ElasticNetCV with Time Series Split
alphas_en = np.logspace(-3, 2, 50)
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

elasticnet_cv = ElasticNetCV(
    alphas=alphas_en,
    l1_ratio=l1_ratios,
    cv=tscv,
    max_iter=10000,
    n_jobs=-1
)
elasticnet_cv.fit(X_train_selected_scaled, y_train)

print(f"ElasticNet 최적 alpha: {elasticnet_cv.alpha_:.4f}")
print(f"ElasticNet 최적 l1_ratio: {elasticnet_cv.l1_ratio_:.4f}")

# ElasticNet 성능
y_pred_en_train = elasticnet_cv.predict(X_train_selected_scaled)
y_pred_en_test = elasticnet_cv.predict(X_test_selected_scaled)

en_train_r2 = r2_score(y_train, y_pred_en_train)
en_test_r2 = r2_score(y_test, y_pred_en_test)
en_train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_en_train))
en_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_en_test))
en_train_mae = mean_absolute_error(y_train, y_pred_en_train)
en_test_mae = mean_absolute_error(y_test, y_pred_en_test)

print(f"\nElasticNet 성능:")
print(f"  Train R²: {en_train_r2:.4f}")
print(f"  Test R²:  {en_test_r2:.4f}")
print(f"  R² Gap:   {en_train_r2 - en_test_r2:.4f} {'✅' if en_train_r2 - en_test_r2 < 0.15 else '⚠️ 과적합'}")
print(f"  Train RMSE: ${en_train_rmse:,.2f}")
print(f"  Test RMSE:  ${en_test_rmse:,.2f}")
print(f"  RMSE Ratio: {en_test_rmse / en_train_rmse:.4f}")
print(f"  Train MAE:  ${en_train_mae:,.2f}")
print(f"  Test MAE:   ${en_test_mae:,.2f}")

# ElasticNet coefficients
en_coef = elasticnet_cv.coef_
en_nonzero_idx = np.where(en_coef != 0)[0]
en_nonzero_vars = [selected_vars_union[i] for i in en_nonzero_idx]

print(f"\nElasticNet 최종 사용 변수: {len(en_nonzero_vars)}개 / {len(selected_vars_union)}개")
print(f"ElasticNet 제거 비율: {(1 - len(en_nonzero_vars)/len(selected_vars_union))*100:.1f}%")

# TOP 20 계수
print("\nElasticNet 계수 TOP 20:")
en_coef_dict = {selected_vars_union[i]: en_coef[i] for i in en_nonzero_idx}
top20_en = sorted(en_coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
for i, (var, coef) in enumerate(top20_en, 1):
    print(f"  {i:2d}. {var:30s} {coef:10.4f}")

# ========================================
# 8. Time Series Cross-Validation
# ========================================
print("\n" + "=" * 80)
print("Step 5: Time Series Cross-Validation (5-Fold)")
print("=" * 80)

# Full data (train + test)
X_full = df_pre[selected_vars_union].values
y_full = df_pre['target'].values
X_full_scaled = scaler_selected.fit_transform(X_full)

tscv_scores = []
fold_num = 1

for train_idx, val_idx in tscv.split(X_full):
    X_cv_train, X_cv_val = X_full_scaled[train_idx], X_full_scaled[val_idx]
    y_cv_train, y_cv_val = y_full[train_idx], y_full[val_idx]

    # Re-scale
    scaler_cv = StandardScaler()
    X_cv_train = scaler_cv.fit_transform(X_cv_train)
    X_cv_val = scaler_cv.transform(X_cv_val)

    # Train ElasticNet
    en_cv = ElasticNet(alpha=elasticnet_cv.alpha_, l1_ratio=elasticnet_cv.l1_ratio_, max_iter=10000)
    en_cv.fit(X_cv_train, y_cv_train)

    # Predict
    y_cv_pred = en_cv.predict(X_cv_val)
    cv_r2 = r2_score(y_cv_val, y_cv_pred)
    tscv_scores.append(cv_r2)

    print(f"  Fold {fold_num}: R² = {cv_r2:.4f}")
    fold_num += 1

tscv_mean = np.mean(tscv_scores)
tscv_std = np.std(tscv_scores)

print(f"\nTime Series CV 결과:")
print(f"  Mean R²: {tscv_mean:.4f}")
print(f"  Std R²:  {tscv_std:.4f}")
print(f"  CV Gap (Train R² - CV Mean): {en_train_r2 - tscv_mean:.4f}")

# ========================================
# 9. Walk-Forward Validation
# ========================================
print("\n" + "=" * 80)
print("Step 6: Walk-Forward Validation")
print("=" * 80)

# Walk-Forward parameters
window_size = 252  # 1 year
test_size = 30     # 30 days

wf_results = []
wf_dates = []

for i in range(window_size, len(df_pre) - test_size, test_size):
    # Train window
    train_start = i - window_size
    train_end = i

    # Test window
    test_start = i
    test_end = i + test_size

    # Get data
    X_wf_train = df_pre[selected_vars_union].iloc[train_start:train_end].values
    y_wf_train = df_pre['target'].iloc[train_start:train_end].values
    X_wf_test = df_pre[selected_vars_union].iloc[test_start:test_end].values
    y_wf_test = df_pre['target'].iloc[test_start:test_end].values

    # Scale
    scaler_wf = StandardScaler()
    X_wf_train_scaled = scaler_wf.fit_transform(X_wf_train)
    X_wf_test_scaled = scaler_wf.transform(X_wf_test)

    # Train ElasticNet
    en_wf = ElasticNet(alpha=elasticnet_cv.alpha_, l1_ratio=elasticnet_cv.l1_ratio_, max_iter=10000)
    en_wf.fit(X_wf_train_scaled, y_wf_train)

    # Predict
    y_wf_pred = en_wf.predict(X_wf_test_scaled)
    wf_r2 = r2_score(y_wf_test, y_wf_pred)
    wf_results.append(wf_r2)
    wf_dates.append(df_pre['Date'].iloc[test_start])

wf_mean = np.mean(wf_results)
wf_std = np.std(wf_results)

print(f"Walk-Forward 설정:")
print(f"  Window: {window_size}일 (1년)")
print(f"  Test: {test_size}일")
print(f"  테스트 기간: {len(wf_results)}개")
print(f"\nWalk-Forward 결과:")
print(f"  Mean R²: {wf_mean:.4f}")
print(f"  Std R²:  {wf_std:.4f}")
print(f"  Min R²:  {np.min(wf_results):.4f}")
print(f"  Max R²:  {np.max(wf_results):.4f}")

# ========================================
# 10. Model Performance Summary
# ========================================
print("\n" + "=" * 80)
print("모델 성능 종합")
print("=" * 80)

performance_summary = pd.DataFrame({
    'Metric': ['Train R²', 'Test R²', 'R² Gap', 'TSCV Mean R²', 'TSCV Std', 'WF Mean R²', 'WF Std'],
    'Value': [
        f"{en_train_r2:.4f}",
        f"{en_test_r2:.4f}",
        f"{en_train_r2 - en_test_r2:.4f}",
        f"{tscv_mean:.4f}",
        f"{tscv_std:.4f}",
        f"{wf_mean:.4f}",
        f"{wf_std:.4f}"
    ]
})

print("\n" + performance_summary.to_string(index=False))

# Overfitting score
overfitting_indicators = {
    'R² Gap': abs(en_train_r2 - en_test_r2),
    'RMSE Ratio': en_test_rmse / en_train_rmse,
    'CV Gap': abs(en_train_r2 - tscv_mean),
    'CV Std': tscv_std,
    'WF Mean': abs(wf_mean)
}

print("\n과적합 지표:")
for indicator, value in overfitting_indicators.items():
    print(f"  {indicator:15s}: {value:.4f}")

# ========================================
# 11. Backtesting
# ========================================
print("\n" + "=" * 80)
print("백테스팅")
print("=" * 80)

# Add predictions to test_df
test_df = test_df.copy()
test_df['predicted_price'] = y_pred_en_test
test_df['actual_price'] = y_test

def backtest_strategy(df, strategy_type='long_only', threshold=0.0, initial_capital=10000):
    """백테스팅 함수"""
    TRANSACTION_COST = 0.01
    SHORT_COST = 0.005

    capital = initial_capital
    position = None
    entry_price = 0

    portfolio_values = []
    returns = []
    trades = []

    for i in range(len(df)):
        current_price = df['Close'].iloc[i]
        predicted_next = df['predicted_price'].iloc[i]

        predicted_change_pct = (predicted_next - current_price) / current_price * 100

        if i < len(df) - 1:
            next_actual_price = df['Close'].iloc[i + 1]
        else:
            next_actual_price = current_price

        if strategy_type == 'long_only':
            if predicted_change_pct > 0:
                if position != 'long':
                    capital -= capital * TRANSACTION_COST
                    entry_price = current_price
                    position = 'long'
                    trades.append({'date': df['Date'].iloc[i], 'action': 'buy', 'price': current_price})

                if i < len(df) - 1:
                    capital = capital * (next_actual_price / entry_price)
                    entry_price = next_actual_price
            else:
                if position == 'long':
                    capital -= capital * TRANSACTION_COST
                    position = None
                    trades.append({'date': df['Date'].iloc[i], 'action': 'sell', 'price': current_price})

        elif strategy_type == 'threshold':
            if predicted_change_pct > threshold:
                if position != 'long':
                    capital -= capital * TRANSACTION_COST
                    entry_price = current_price
                    position = 'long'
                    trades.append({'date': df['Date'].iloc[i], 'action': 'buy', 'price': current_price})

                if i < len(df) - 1:
                    capital = capital * (next_actual_price / entry_price)
                    entry_price = next_actual_price
            elif predicted_change_pct < -threshold:
                if position == 'long':
                    capital -= capital * TRANSACTION_COST
                    position = None
                    trades.append({'date': df['Date'].iloc[i], 'action': 'sell', 'price': current_price})

        portfolio_values.append(capital)

        if i > 0:
            daily_return = (capital / portfolio_values[i-1] - 1) * 100
            returns.append(daily_return)
        else:
            returns.append(0)

    # Metrics
    final_value = portfolio_values[-1]
    total_return = (final_value / initial_capital - 1) * 100

    days = len(df)
    years = days / 365
    annual_return = (np.power(final_value / initial_capital, 1/years) - 1) * 100 if years > 0 else 0

    returns_array = np.array(returns)
    daily_volatility = np.std(returns_array)
    annual_volatility = daily_volatility * np.sqrt(252)

    sharpe_ratio = (annual_return / annual_volatility) if annual_volatility > 0 else 0

    cummax = np.maximum.accumulate(portfolio_values)
    drawdowns = (np.array(portfolio_values) - cummax) / cummax * 100
    max_drawdown = np.min(drawdowns)

    winning_days = np.sum(returns_array > 0)
    win_rate = winning_days / len(returns_array) * 100 if len(returns_array) > 0 else 0

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'final_value': final_value,
        'win_rate': win_rate,
        'num_trades': len(trades),
        'portfolio_values': portfolio_values
    }

# Run backtests
strategies = [
    ('Long-Only', 'long_only', 0.0),
    ('Threshold 1%', 'threshold', 1.0),
    ('Threshold 2%', 'threshold', 2.0),
]

backtest_results = []

# Buy-and-Hold
initial_price = test_df['Close'].iloc[0]
final_price = test_df['Close'].iloc[-1]
bnh_return = (final_price / initial_price - 1) * 100
days = len(test_df)
years = days / 365
bnh_annual = (np.power(final_price / initial_price, 1/years) - 1) * 100
bnh_returns = test_df['Close'].pct_change().fillna(0) * 100
bnh_vol = np.std(bnh_returns) * np.sqrt(252)
bnh_sharpe = (bnh_annual / bnh_vol) if bnh_vol > 0 else 0

bnh_values = (test_df['Close'] / initial_price * 10000).values
bnh_cummax = np.maximum.accumulate(bnh_values)
bnh_dd = (bnh_values - bnh_cummax) / bnh_cummax * 100
bnh_max_dd = np.min(bnh_dd)

backtest_results.append({
    'Strategy': 'Buy-and-Hold',
    'Total Return': bnh_return,
    'Annual Return': bnh_annual,
    'Sharpe Ratio': bnh_sharpe,
    'Max Drawdown': bnh_max_dd,
    'Win Rate': (bnh_returns > 0).sum() / len(bnh_returns) * 100,
    'Trades': 0
})

print(f"Buy-and-Hold: {bnh_return:.2f}% (Sharpe: {bnh_sharpe:.4f})")

# ElasticNet strategies
for name, strategy_type, threshold in strategies:
    result = backtest_strategy(test_df, strategy_type, threshold)
    backtest_results.append({
        'Strategy': name,
        'Total Return': result['total_return'],
        'Annual Return': result['annual_return'],
        'Sharpe Ratio': result['sharpe_ratio'],
        'Max Drawdown': result['max_drawdown'],
        'Win Rate': result['win_rate'],
        'Trades': result['num_trades']
    })
    print(f"{name}: {result['total_return']:.2f}% (Sharpe: {result['sharpe_ratio']:.4f})")

backtest_df = pd.DataFrame(backtest_results)

print("\n백테스팅 결과:")
print(backtest_df.to_string(index=False))

# ========================================
# 12. Save Results
# ========================================
print("\n" + "=" * 80)
print("결과 저장")
print("=" * 80)

# Selected variables
selected_vars_df = pd.DataFrame({
    'Variable': selected_vars_union,
    'Lasso_Coef': [lasso_coef_dict.get(var, 0) for var in selected_vars_union],
    'Ridge_Coef': [ridge_coef[feature_cols.index(var)] if var in feature_cols else 0 for var in selected_vars_union],
    'ElasticNet_Coef': [en_coef_dict.get(var, 0) for var in selected_vars_union]
})
selected_vars_df['Abs_ElasticNet_Coef'] = selected_vars_df['ElasticNet_Coef'].abs()
selected_vars_df = selected_vars_df.sort_values('Abs_ElasticNet_Coef', ascending=False)
selected_vars_df.to_csv('etf_pre_selected_variables.csv', index=False)
print("Saved: etf_pre_selected_variables.csv")

# Performance summary
performance_df = pd.DataFrame({
    'Model': ['Lasso', 'Ridge', 'ElasticNet'],
    'Train_R2': [lasso_train_r2, ridge_train_r2, en_train_r2],
    'Test_R2': [lasso_test_r2, ridge_test_r2, en_test_r2],
    'R2_Gap': [lasso_train_r2 - lasso_test_r2, ridge_train_r2 - ridge_test_r2, en_train_r2 - en_test_r2],
    'Train_RMSE': [lasso_train_rmse, ridge_train_rmse, en_train_rmse],
    'Test_RMSE': [lasso_test_rmse, ridge_test_rmse, en_test_rmse],
    'Selected_Vars': [len(lasso_selected_vars), len(ridge_selected_vars), len(en_nonzero_vars)]
})
performance_df.to_csv('etf_pre_model_performance.csv', index=False)
print("Saved: etf_pre_model_performance.csv")

# Backtesting results
backtest_df.to_csv('etf_pre_backtesting_results.csv', index=False)
print("Saved: etf_pre_backtesting_results.csv")

# Validation results
validation_df = pd.DataFrame({
    'Validation': ['Train/Test Split', 'Time Series CV', 'Walk-Forward'],
    'Mean_R2': [en_test_r2, tscv_mean, wf_mean],
    'Std_R2': [0, tscv_std, wf_std]
})
validation_df.to_csv('etf_pre_validation_results.csv', index=False)
print("Saved: etf_pre_validation_results.csv")

# ========================================
# 13. Visualization
# ========================================
print("\n" + "=" * 80)
print("시각화")
print("=" * 80)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Model Performance Comparison
ax1 = fig.add_subplot(gs[0, 0])
models = ['Lasso', 'Ridge', 'ElasticNet']
train_r2s = [lasso_train_r2, ridge_train_r2, en_train_r2]
test_r2s = [lasso_test_r2, ridge_test_r2, en_test_r2]

x = np.arange(len(models))
width = 0.35
ax1.bar(x - width/2, train_r2s, width, label='Train R²', alpha=0.8)
ax1.bar(x + width/2, test_r2s, width, label='Test R²', alpha=0.8)
ax1.set_ylabel('R² Score', fontweight='bold')
ax1.set_title('Model Performance (ETF 이전)', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# 2. Variable Selection Venn
ax2 = fig.add_subplot(gs[0, 1])
lasso_set = set(lasso_selected_vars)
ridge_set = set(ridge_selected_vars)
intersection = lasso_set & ridge_set
lasso_only_count = len(lasso_set - ridge_set)
ridge_only_count = len(ridge_set - lasso_set)
intersection_count = len(intersection)

data = [lasso_only_count, intersection_count, ridge_only_count]
labels = [f'Lasso만\n{lasso_only_count}', f'교집합\n{intersection_count}', f'Ridge만\n{ridge_only_count}']
colors = ['#3498db', '#2ecc71', '#e74c3c']
ax2.bar(range(3), data, color=colors, alpha=0.7)
ax2.set_xticks(range(3))
ax2.set_xticklabels(labels)
ax2.set_ylabel('변수 개수', fontweight='bold')
ax2.set_title('Lasso vs Ridge 변수 선택', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# 3. ElasticNet Coefficients TOP 15
ax3 = fig.add_subplot(gs[0, 2])
top15 = sorted(en_coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
vars_top15 = [v[0][:20] for v in top15]  # 변수명 20자로 제한
coefs_top15 = [v[1] for v in top15]
colors_top15 = ['#2ecc71' if c > 0 else '#e74c3c' for c in coefs_top15]
ax3.barh(range(len(vars_top15)), coefs_top15, color=colors_top15, alpha=0.7)
ax3.set_yticks(range(len(vars_top15)))
ax3.set_yticklabels(vars_top15, fontsize=8)
ax3.set_xlabel('Coefficient', fontweight='bold')
ax3.set_title('ElasticNet 계수 TOP 15', fontweight='bold')
ax3.axvline(x=0, color='black', linestyle='--', alpha=0.3)
ax3.invert_yaxis()
ax3.grid(True, alpha=0.3, axis='x')

# 4. Actual vs Predicted (Test)
ax4 = fig.add_subplot(gs[1, :])
ax4.plot(test_df['Date'], y_test, label='Actual', linewidth=2, alpha=0.8)
ax4.plot(test_df['Date'], y_pred_en_test, label='Predicted', linewidth=2, alpha=0.8)
ax4.set_xlabel('Date', fontweight='bold')
ax4.set_ylabel('Price ($)', fontweight='bold')
ax4.set_title(f'Actual vs Predicted (Test R²: {en_test_r2:.4f})', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='x', rotation=45)

# 5. Residuals
ax5 = fig.add_subplot(gs[2, 0])
residuals = y_test - y_pred_en_test
ax5.scatter(y_pred_en_test, residuals, alpha=0.5, s=10)
ax5.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax5.set_xlabel('Predicted Price ($)', fontweight='bold')
ax5.set_ylabel('Residuals ($)', fontweight='bold')
ax5.set_title('Residual Plot', fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. Walk-Forward R² over time
ax6 = fig.add_subplot(gs[2, 1])
ax6.plot(wf_dates, wf_results, marker='o', linewidth=2, markersize=4, alpha=0.7)
ax6.axhline(y=wf_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {wf_mean:.4f}')
ax6.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
ax6.set_xlabel('Date', fontweight='bold')
ax6.set_ylabel('R² Score', fontweight='bold')
ax6.set_title('Walk-Forward Validation R²', fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.tick_params(axis='x', rotation=45)

# 7. Backtesting Results
ax7 = fig.add_subplot(gs[2, 2])
strategies_bt = backtest_df['Strategy'].tolist()
returns_bt = backtest_df['Total Return'].tolist()
colors_bt = ['#95a5a6'] + ['#3498db', '#2ecc71', '#f39c12']
ax7.barh(range(len(strategies_bt)), returns_bt, color=colors_bt, alpha=0.7)
ax7.set_yticks(range(len(strategies_bt)))
ax7.set_yticklabels(strategies_bt, fontsize=9)
ax7.set_xlabel('Total Return (%)', fontweight='bold')
ax7.set_title('백테스팅 결과 (ETF 이전)', fontweight='bold')
ax7.axvline(x=0, color='black', linestyle='--', alpha=0.3)
ax7.invert_yaxis()
ax7.grid(True, alpha=0.3, axis='x')

for i, ret in enumerate(returns_bt):
    ax7.text(ret, i, f"  {ret:.1f}%", va='center', fontsize=9, fontweight='bold')

plt.savefig('etf_pre_elasticnet_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: etf_pre_elasticnet_analysis.png")

# ========================================
# 14. Final Summary
# ========================================
print("\n" + "=" * 80)
print("최종 요약 (ETF 이전 기간)")
print("=" * 80)

best_strategy = backtest_df.loc[backtest_df['Sharpe Ratio'].idxmax()]

print(f"""
📊 ETF 이전 ElasticNet 모델 분석 결과

🗓️ 데이터 기간:
   - 전체: {df_pre['Date'].min().date()} ~ {df_pre['Date'].max().date()} ({len(df_pre)} days)
   - Train: {train_df['Date'].min().date()} ~ {train_df['Date'].max().date()} ({len(train_df)} days)
   - Test:  {test_df['Date'].min().date()} ~ {test_df['Date'].max().date()} ({len(test_df)} days)

🔧 변수 선택:
   - 원본 변수: {len(feature_cols)}개 (ETF 변수 제외)
   - Lasso 선택: {len(lasso_selected_vars)}개
   - Ridge 선택: {len(ridge_selected_vars)}개
   - 합집합: {len(selected_vars_union)}개
   - ElasticNet 최종: {len(en_nonzero_vars)}개

📈 모델 성능:
   - Train R²: {en_train_r2:.4f}
   - Test R²:  {en_test_r2:.4f}
   - R² Gap:   {en_train_r2 - en_test_r2:.4f} {'✅' if en_train_r2 - en_test_r2 < 0.15 else '⚠️ 과적합'}
   - TSCV R²:  {tscv_mean:.4f} (±{tscv_std:.4f})
   - WF R²:    {wf_mean:.4f} (±{wf_std:.4f})

💰 백테스팅:
   - 최고 전략: {best_strategy['Strategy']}
   - Total Return: {best_strategy['Total Return']:.2f}%
   - Sharpe Ratio: {best_strategy['Sharpe Ratio']:.4f}
   - Max Drawdown: {best_strategy['Max Drawdown']:.2f}%
   - Win Rate: {best_strategy['Win Rate']:.2f}%
   - Trades: {best_strategy['Trades']}

🏆 결론:
   {'✅ ElasticNet이 Buy-and-Hold보다 우수!' if best_strategy['Sharpe Ratio'] > bnh_sharpe else '⚠️ Buy-and-Hold가 더 나음'}
   {'✅ 과적합 위험 낮음 (R² Gap < 0.15)' if en_train_r2 - en_test_r2 < 0.15 else '⚠️ 과적합 주의'}
   {'✅ Walk-Forward 검증 통과 (R² > 0)' if wf_mean > 0 else '⚠️ Walk-Forward 검증 실패'}
""")

print("=" * 80)
print("Step 27 (ETF 이전) Completed!")
print("=" * 80)
