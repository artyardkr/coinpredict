"""
실시간 예측 엔진
"""

import pandas as pd
import numpy as np
import pickle
import logging
from typing import Optional, Dict
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionEngine:
    """ElasticNet 모델 기반 실시간 예측"""

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.model_loaded = False

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> bool:
        """모델 로드"""
        try:
            # 모델 파일 확인
            model_file = Path(model_path)
            if not model_file.exists():
                logger.warning(f"모델 파일 없음: {model_path}, 더미 모델 사용")
                return self._create_dummy_model()

            # 모델 로드
            with open(model_path, 'rb') as f:
                saved_data = pickle.load(f)

            if isinstance(saved_data, dict):
                self.model = saved_data.get('model')
                self.scaler = saved_data.get('scaler')
                self.feature_columns = saved_data.get('feature_columns')
            else:
                self.model = saved_data
                logger.warning("Scaler 및 feature_columns가 없습니다. 기본값 사용")

            self.model_loaded = True
            logger.info(f"모델 로드 완료: {model_path}")
            return True

        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            return self._create_dummy_model()

    def _create_dummy_model(self) -> bool:
        """더미 모델 생성 (테스트용)"""
        logger.info("더미 모델 생성 (랜덤 예측)")
        self.model_loaded = False
        return True

    def prepare_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """특성 준비"""
        try:
            # 제외할 컬럼
            exclude_cols = [
                'Date', 'Close', 'High', 'Low', 'Open', 'Volume',
                'Adj Close', 'target', 'target_return'
            ]

            # 특성 선택
            if self.feature_columns:
                available_cols = [col for col in self.feature_columns if col in data.columns]
                X = data[available_cols].values
            else:
                feature_cols = [col for col in data.columns if col not in exclude_cols]
                X = data[feature_cols].values

            # 스케일링
            if self.scaler:
                X = self.scaler.transform(X)

            return X

        except Exception as e:
            logger.error(f"특성 준비 실패: {e}")
            return None

    def predict(self, features: pd.DataFrame) -> Optional[Dict]:
        """가격 예측"""
        try:
            # 더미 모델 (테스트용)
            if not self.model_loaded or self.model is None:
                return self._dummy_predict(features)

            # 특성 준비
            X = self.prepare_features(features)
            if X is None:
                return None

            # 예측
            if len(X.shape) == 1:
                X = X.reshape(1, -1)

            predicted_return = self.model.predict(X)[0]

            # 현재 가격
            current_price = features['Close'].iloc[-1] if 'Close' in features.columns else 0

            # 예측 가격
            predicted_price = current_price * (1 + predicted_return / 100)

            return {
                "predicted_return": predicted_return,
                "predicted_price": predicted_price,
                "current_price": current_price,
                "confidence": self._calculate_confidence(predicted_return)
            }

        except Exception as e:
            logger.error(f"예측 실패: {e}")
            return None

    def _dummy_predict(self, features: pd.DataFrame) -> Dict:
        """더미 예측 (테스트용)"""
        current_price = features['Close'].iloc[-1] if 'Close' in features.columns else 50000
        predicted_return = np.random.normal(0, 2)  # 평균 0%, 표준편차 2%
        predicted_price = current_price * (1 + predicted_return / 100)

        return {
            "predicted_return": predicted_return,
            "predicted_price": predicted_price,
            "current_price": current_price,
            "confidence": 0.5
        }

    def _calculate_confidence(self, predicted_return: float) -> float:
        """예측 신뢰도 계산 (절댓값이 클수록 신뢰도 높음)"""
        abs_return = abs(predicted_return)
        # 시그모이드 함수 적용
        confidence = 2 / (1 + np.exp(-abs_return / 2)) - 1
        return min(confidence, 1.0)

    def batch_predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """배치 예측"""
        try:
            predictions = []

            for i in range(len(data)):
                if i < 30:  # 최소 30일 데이터 필요
                    predictions.append({
                        "predicted_return": np.nan,
                        "predicted_price": np.nan,
                        "confidence": np.nan
                    })
                    continue

                # i번째 행까지의 데이터로 예측
                window_data = data.iloc[:i + 1]
                pred = self.predict(window_data)

                if pred:
                    predictions.append(pred)
                else:
                    predictions.append({
                        "predicted_return": np.nan,
                        "predicted_price": np.nan,
                        "confidence": np.nan
                    })

            pred_df = pd.DataFrame(predictions)
            return pred_df

        except Exception as e:
            logger.error(f"배치 예측 실패: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    # 테스트
    print("예측 엔진 테스트")
    print("=" * 60)

    # 더미 데이터 생성
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)

    df = pd.DataFrame({
        'Date': dates,
        'Close': 50000 + np.cumsum(np.random.randn(len(dates)) * 500),
        'Volume': np.random.randint(1e9, 5e9, len(dates)),
        'RSI': np.random.uniform(30, 70, len(dates)),
        'MACD': np.random.randn(len(dates)) * 100,
        'SPX': 4500 + np.cumsum(np.random.randn(len(dates)) * 10),
        'VIX': np.random.uniform(10, 30, len(dates))
    })

    # 엔진 초기화
    engine = PredictionEngine()

    # 단일 예측
    print("\n1. 단일 예측")
    pred = engine.predict(df)
    if pred:
        print(f"  현재 가격: ${pred['current_price']:,.2f}")
        print(f"  예측 수익률: {pred['predicted_return']:+.2f}%")
        print(f"  예측 가격: ${pred['predicted_price']:,.2f}")
        print(f"  신뢰도: {pred['confidence']:.2%}")

    # 배치 예측
    print("\n2. 배치 예측 (최근 10일)")
    batch_pred = engine.batch_predict(df.tail(40))
    print(batch_pred.tail(10))
