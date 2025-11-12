"""
실시간 데이터 수집 모듈
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import yfinance as yf
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTimeDataFetcher:
    """실시간 암호화폐 및 시장 데이터 수집"""

    def __init__(self, symbol: str = "BTC-USD", lookback_days: int = 365):
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.last_update = None
        self.cached_data = None

    def fetch_btc_price(self) -> Optional[Dict]:
        """비트코인 현재 가격 가져오기 (Binance)"""
        try:
            url = "https://api.binance.com/api/v3/ticker/price"
            params = {"symbol": "BTCUSDT"}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return {
                "price": float(data["price"]),
                "timestamp": datetime.now()
            }
        except Exception as e:
            logger.error(f"비트코인 가격 가져오기 실패: {e}")
            return None

    def fetch_historical_data(self) -> pd.DataFrame:
        """과거 데이터 가져오기 (yfinance)"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)

            # 비트코인 데이터
            btc = yf.download(self.symbol, start=start_date, end=end_date, progress=False)
            btc = btc.reset_index()
            btc.columns = [col[0] if isinstance(col, tuple) else col for col in btc.columns]

            # 전통 시장 데이터
            symbols = {
                "^GSPC": "SPX",  # S&P 500
                "^IXIC": "NASDAQ",  # NASDAQ
                "^VIX": "VIX",  # VIX
                "GC=F": "GOLD",  # 금
                "^TNX": "US10Y"  # 10년물 국채
            }

            market_data = {}
            for ticker, name in symbols.items():
                try:
                    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    if not df.empty:
                        market_data[name] = df['Close'].values
                except:
                    logger.warning(f"{name} 데이터 가져오기 실패")
                    market_data[name] = np.nan

            # 통합
            for name, values in market_data.items():
                if isinstance(values, np.ndarray) and len(values) == len(btc):
                    btc[name] = values
                else:
                    btc[name] = np.nan

            # 기술적 지표 계산
            btc = self._calculate_technical_indicators(btc)

            # 결측치 처리
            btc = btc.fillna(method='ffill').fillna(method='bfill')

            self.cached_data = btc
            self.last_update = datetime.now()

            logger.info(f"데이터 로드 완료: {len(btc)}행, {btc['Date'].min()} ~ {btc['Date'].max()}")
            return btc

        except Exception as e:
            logger.error(f"과거 데이터 가져오기 실패: {e}")
            return pd.DataFrame()

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산"""
        close = df['Close']

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        sma = close.rolling(window=20).mean()
        std = close.rolling(window=20).std()
        df['BB_upper'] = sma + (std * 2)
        df['BB_lower'] = sma - (std * 2)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / sma

        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']

        # Price changes
        df['returns_1d'] = close.pct_change()
        df['returns_7d'] = close.pct_change(7)
        df['returns_30d'] = close.pct_change(30)

        # Volatility
        df['volatility_7d'] = close.pct_change().rolling(window=7).std()
        df['volatility_30d'] = close.pct_change().rolling(window=30).std()

        return df

    def get_latest_features(self) -> Optional[pd.Series]:
        """최신 특성 벡터 가져오기"""
        if self.cached_data is None or len(self.cached_data) == 0:
            self.fetch_historical_data()

        if self.cached_data is not None and len(self.cached_data) > 0:
            return self.cached_data.iloc[-1]
        return None

    def update_data(self) -> bool:
        """데이터 업데이트"""
        try:
            new_data = self.fetch_historical_data()
            if not new_data.empty:
                return True
            return False
        except Exception as e:
            logger.error(f"데이터 업데이트 실패: {e}")
            return False


if __name__ == "__main__":
    # 테스트
    fetcher = RealTimeDataFetcher()

    print("1. 현재 비트코인 가격 가져오기")
    current_price = fetcher.fetch_btc_price()
    if current_price:
        print(f"  가격: ${current_price['price']:,.2f}")
        print(f"  시간: {current_price['timestamp']}")

    print("\n2. 과거 데이터 가져오기")
    df = fetcher.fetch_historical_data()
    if not df.empty:
        print(f"  데이터: {len(df)}행 x {len(df.columns)}열")
        print(f"  기간: {df['Date'].min()} ~ {df['Date'].max()}")
        print(f"  컬럼: {list(df.columns)[:10]}...")

    print("\n3. 최신 특성 벡터 가져오기")
    latest = fetcher.get_latest_features()
    if latest is not None:
        print(f"  날짜: {latest['Date']}")
        print(f"  종가: ${latest['Close']:,.2f}")
        print(f"  RSI: {latest['RSI']:.2f}")
        print(f"  MACD: {latest['MACD']:.4f}")
