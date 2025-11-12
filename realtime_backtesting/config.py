"""
실시간 백테스팅 시스템 설정 파일
"""

# API 설정
BINANCE_API_BASE = "https://api.binance.com/api/v3"
COINBASE_API_BASE = "https://api.coinbase.com/v2"

# 데이터 수집 설정
SYMBOL = "BTCUSDT"
INTERVAL = "1d"  # 일봉
DATA_LOOKBACK_DAYS = 365  # 과거 데이터 기간

# 모델 설정
MODEL_PATH = "../elasticnet_model_v2.pkl"
SCALER_PATH = "../scaler_v2.pkl"
FEATURE_COLUMNS_PATH = "../feature_columns_v2.txt"

# 백테스팅 설정
INITIAL_CAPITAL = 10000  # 초기 자본 ($)
TRANSACTION_COST = 0.001  # 거래 비용 (0.1%)
SHORT_COST = 0.0005  # 숏 포지션 유지 비용 (0.05%)

# 전략 설정
STRATEGIES = {
    "long_only": "상승 예측시 매수, 하락 예측시 현금",
    "long_short": "상승 예측시 매수, 하락 예측시 공매도",
    "threshold_0.5": "예측 수익률 > 0.5%면 매수",
    "threshold_1.0": "예측 수익률 > 1.0%면 매수",
    "threshold_2.0": "예측 수익률 > 2.0%면 매수",
}

DEFAULT_STRATEGY = "threshold_1.0"
DEFAULT_THRESHOLD = 1.0  # %

# UI 설정
UPDATE_INTERVAL = 60  # 업데이트 간격 (초)
MAX_DISPLAY_TRADES = 50  # 최대 표시 거래 수

# 차트 설정
CHART_HEIGHT = 400
CHART_WIDTH = 800

# 색상 설정
COLORS = {
    "profit": "#00CC96",
    "loss": "#EF553B",
    "neutral": "#636EFA",
    "buy": "#00CC96",
    "sell": "#EF553B",
    "short": "#AB63FA",
    "hold": "#FFA15A"
}
