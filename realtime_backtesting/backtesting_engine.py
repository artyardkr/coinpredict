"""
실시간 백테스팅 엔진
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionType(Enum):
    """포지션 타입"""
    LONG = "LONG"
    SHORT = "SHORT"
    CASH = "CASH"


class OrderType(Enum):
    """주문 타입"""
    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"


class BacktestingEngine:
    """실시간 백테스팅 엔진"""

    def __init__(
        self,
        initial_capital: float = 10000,
        transaction_cost: float = 0.001,
        short_cost: float = 0.0005,
        strategy: str = "threshold",
        threshold: float = 1.0
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.short_cost = short_cost
        self.strategy = strategy
        self.threshold = threshold

        # 상태 변수
        self.cash = initial_capital
        self.position = PositionType.CASH
        self.position_size = 0.0  # BTC 보유량
        self.entry_price = 0.0
        self.current_price = 0.0

        # 거래 및 성과 추적
        self.trades: List[Dict] = []
        self.portfolio_history: List[Dict] = []
        self.equity_curve: List[float] = [initial_capital]

    def reset(self):
        """엔진 초기화"""
        self.cash = self.initial_capital
        self.position = PositionType.CASH
        self.position_size = 0.0
        self.entry_price = 0.0
        self.current_price = 0.0
        self.trades = []
        self.portfolio_history = []
        self.equity_curve = [self.initial_capital]

    def get_portfolio_value(self) -> float:
        """현재 포트폴리오 가치"""
        if self.position == PositionType.CASH:
            return self.cash
        elif self.position == PositionType.LONG:
            return self.position_size * self.current_price
        elif self.position == PositionType.SHORT:
            # 숏: 진입 시 받은 돈 + (진입가 - 현재가) * 수량
            return self.cash + (self.entry_price - self.current_price) * self.position_size
        return self.cash

    def execute_order(
        self,
        order_type: OrderType,
        price: float,
        timestamp: datetime,
        reason: str = ""
    ) -> bool:
        """주문 실행"""
        try:
            old_value = self.get_portfolio_value()

            if order_type == OrderType.BUY:
                # 매수
                if self.position == PositionType.SHORT:
                    # 숏 포지션 청산
                    pnl = (self.entry_price - price) * self.position_size
                    self.cash += pnl
                    self.cash -= self.cash * self.transaction_cost

                # 롱 포지션 진입
                self.position_size = self.cash / price
                self.entry_price = price
                self.position = PositionType.LONG
                self.cash = 0

            elif order_type == OrderType.SELL:
                # 매도
                if self.position == PositionType.LONG:
                    self.cash = self.position_size * price
                    self.cash -= self.cash * self.transaction_cost
                    self.position_size = 0
                    self.position = PositionType.CASH

            elif order_type == OrderType.SHORT:
                # 공매도
                if self.position == PositionType.LONG:
                    # 롱 포지션 청산
                    self.cash = self.position_size * price
                    self.cash -= self.cash * self.transaction_cost

                # 숏 포지션 진입
                self.position_size = self.cash / price
                self.entry_price = price
                self.position = PositionType.SHORT
                self.cash = self.cash  # 숏은 현금 보유

            elif order_type == OrderType.COVER:
                # 숏 커버
                if self.position == PositionType.SHORT:
                    pnl = (self.entry_price - price) * self.position_size
                    self.cash += pnl
                    self.cash -= self.cash * (self.transaction_cost + self.short_cost)
                    self.position_size = 0
                    self.position = PositionType.CASH

            # 거래 기록
            self.current_price = price
            new_value = self.get_portfolio_value()

            trade = {
                "timestamp": timestamp,
                "type": order_type.value,
                "price": price,
                "position_size": self.position_size,
                "portfolio_value": new_value,
                "pnl": new_value - old_value,
                "reason": reason
            }
            self.trades.append(trade)

            logger.info(f"{order_type.value} @ ${price:,.2f} | PnL: ${new_value - old_value:+,.2f}")
            return True

        except Exception as e:
            logger.error(f"주문 실행 실패: {e}")
            return False

    def update(
        self,
        current_price: float,
        predicted_return: float,
        timestamp: datetime
    ) -> Dict:
        """시장 업데이트 및 전략 실행"""
        self.current_price = current_price
        portfolio_value = self.get_portfolio_value()

        # 전략 실행
        signal = self._generate_signal(predicted_return)

        # 주문 실행
        if signal == "BUY" and self.position != PositionType.LONG:
            self.execute_order(
                OrderType.BUY,
                current_price,
                timestamp,
                f"예측 수익률: {predicted_return:+.2f}%"
            )
        elif signal == "SELL" and self.position == PositionType.LONG:
            self.execute_order(
                OrderType.SELL,
                current_price,
                timestamp,
                f"예측 수익률: {predicted_return:+.2f}%"
            )
        elif signal == "SHORT" and self.position != PositionType.SHORT:
            self.execute_order(
                OrderType.SHORT,
                current_price,
                timestamp,
                f"예측 수익률: {predicted_return:+.2f}%"
            )
        elif signal == "COVER" and self.position == PositionType.SHORT:
            self.execute_order(
                OrderType.COVER,
                current_price,
                timestamp,
                f"예측 수익률: {predicted_return:+.2f}%"
            )

        # 포트폴리오 업데이트
        portfolio_value = self.get_portfolio_value()
        self.equity_curve.append(portfolio_value)

        status = {
            "timestamp": timestamp,
            "price": current_price,
            "predicted_return": predicted_return,
            "position": self.position.value,
            "position_size": self.position_size,
            "cash": self.cash,
            "portfolio_value": portfolio_value,
            "total_return": (portfolio_value / self.initial_capital - 1) * 100,
            "num_trades": len(self.trades)
        }

        self.portfolio_history.append(status)
        return status

    def _generate_signal(self, predicted_return: float) -> str:
        """거래 신호 생성"""
        if self.strategy == "long_only":
            if predicted_return > 0:
                return "BUY"
            else:
                return "SELL"

        elif self.strategy == "long_short":
            if predicted_return > 0.5:
                return "BUY"
            elif predicted_return < -0.5:
                return "SHORT"
            else:
                if self.position == PositionType.LONG:
                    return "SELL"
                elif self.position == PositionType.SHORT:
                    return "COVER"
                return "HOLD"

        elif self.strategy == "threshold":
            if predicted_return > self.threshold:
                return "BUY"
            else:
                if self.position == PositionType.LONG:
                    return "SELL"
                return "HOLD"

        return "HOLD"

    def get_performance_metrics(self) -> Dict:
        """성과 지표 계산"""
        if len(self.equity_curve) < 2:
            return {}

        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]

        # 수익률
        total_return = (equity[-1] / equity[0] - 1) * 100

        # 연율화 (가정: 일별 데이터)
        days = len(equity)
        annual_return = ((equity[-1] / equity[0]) ** (365 / days) - 1) * 100 if days > 0 else 0

        # 변동성
        annual_volatility = returns.std() * np.sqrt(365) * 100 if len(returns) > 0 else 0

        # 샤프 비율
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0

        # 최대 낙폭
        cummax = np.maximum.accumulate(equity)
        drawdown = (equity - cummax) / cummax * 100
        max_drawdown = drawdown.min()

        # 승률
        winning_trades = sum(1 for t in self.trades if t['pnl'] > 0)
        win_rate = winning_trades / len(self.trades) * 100 if self.trades else 0

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "num_trades": len(self.trades),
            "final_value": equity[-1]
        }


if __name__ == "__main__":
    # 테스트
    print("백테스팅 엔진 테스트")
    print("=" * 60)

    # 더미 데이터 생성
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    prices = 50000 + np.cumsum(np.random.randn(100) * 500)
    predicted_returns = np.random.normal(0, 2, 100)

    # 엔진 초기화
    engine = BacktestingEngine(
        initial_capital=10000,
        strategy="threshold",
        threshold=1.0
    )

    # 시뮬레이션
    print("\n시뮬레이션 시작...")
    for i, (date, price, pred_ret) in enumerate(zip(dates, prices, predicted_returns)):
        status = engine.update(price, pred_ret, date)

        if i % 20 == 0:
            print(f"Day {i}: ${price:,.0f} | Pred: {pred_ret:+.1f}% | "
                  f"Portfolio: ${status['portfolio_value']:,.0f} | "
                  f"Return: {status['total_return']:+.1f}%")

    # 성과 지표
    print("\n" + "=" * 60)
    print("최종 성과")
    print("=" * 60)
    metrics = engine.get_performance_metrics()
    for key, value in metrics.items():
        if 'return' in key or 'volatility' in key or 'drawdown' in key or 'rate' in key:
            print(f"{key}: {value:.2f}%")
        elif 'ratio' in key:
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
